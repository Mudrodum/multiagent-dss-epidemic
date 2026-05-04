"""
Evaluation-layer для бюллетеней ai4epi.

Модуль переносит evaluation/judge слой из исходного notebook в пакетную
архитектуру. Он не генерирует бюллетень и не редактирует текст. Его задача —
оценить уже собранный Bulletin по независимым критериям. Evaluation не является semantic gate: findings сохраняются в отчёте, но не блокируют pipeline:

- numeric: наличие обязательных числовых якорей из GlobalContext;
- factual: claim-level соответствие фактов контексту;
- logic: логическая согласованность прогноза и SHAP-утверждений;
- water: информационная плотность narrative-разделов;
- tautology: повторы мыслей;
- grammar: собственно языковая норма;
- orthotypography: орфо-типографическая норма;
- style: редакторская ясность.

LLM в этом модуле не выставляет итоговый балл. LLM используется только как
структурный judge: извлекает claims или проблемные фрагменты в JSON. Сами
оценки и штрафы считаются детерминированно в коде.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4epi.generation.bulletin import Bulletin, flatten_text, load_bulletin
from ai4epi.core.context import GlobalContext, load_global_context
from ai4epi.generation.narrator import ChatBackend


JsonObject = dict[str, Any]
Severity = Literal["error", "warn", "info"]
EvaluatorName = Literal[
    "numeric",
    "factual",
    "logic",
    "water",
    "tautology",
    "grammar",
    "orthotypography",
    "style",
]


DEFAULT_JUDGE_TEMPERATURE = 0.0
DEFAULT_ENABLED_EVALUATORS: tuple[EvaluatorName, ...] = (
    "numeric",
    "factual",
    "tautology",
    "grammar",
    "orthotypography",
    "style",
)
DEFAULT_EVAL_WEIGHTS: dict[str, float] = {
    "numeric": 8 / 30,
    "factual": 8 / 30,
    "logic": 0.0,
    "style": 6 / 30,
    "tautology": 2 / 30,
    "grammar": 3 / 30,
    "orthotypography": 2 / 30,
    "water": 1 / 30,
}
SEMANTIC_EVALUATORS: frozenset[str] = frozenset({"numeric", "factual", "logic"})
EDITORIAL_EVALUATORS: frozenset[str] = frozenset({"style", "grammar", "orthotypography", "tautology", "water"})


class EvaluationError(RuntimeError):
    """Ошибка выполнения evaluation-layer."""


class StrictModel(BaseModel):
    """Базовая модель с закрытым контрактом."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


class Issue(StrictModel):
    """Одно нарушение, найденное evaluator-ом."""

    evaluator: str = Field(min_length=1)
    severity: Severity
    section: str = Field(default="global", min_length=1)
    message: str = Field(min_length=1)
    detail: str | None = None

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, value: Any) -> Severity:
        text = str(value or "warn").strip().lower()
        if text not in {"error", "warn", "info"}:
            return "warn"
        return text  # type: ignore[return-value]


class EvalScore(StrictModel):
    """Оценка одного критерия."""

    name: str = Field(min_length=1)
    score: float = Field(ge=0.0, le=1.0)
    evaluated: bool
    issues: list[Issue] = Field(default_factory=list)
    checks_total: int = Field(default=0, ge=0)
    checks_passed: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_passed_not_above_total(self) -> "EvalScore":
        if self.checks_passed > self.checks_total and self.checks_total > 0:
            raise ValueError("checks_passed must not exceed checks_total.")
        return self


class EvalReport(StrictModel):
    """Итоговый отчёт evaluation-layer.

    Важный контракт: evaluator findings являются диагностикой, а не gate-ом.
    Даже если numeric/factual evaluators находят ошибки, это фиксируется в
    all_issues/semantic_findings, но сам EvalReport остаётся валидным JSON-
    отчётом и не должен блокировать pipeline.
    """

    scores: dict[str, EvalScore]
    aggregate: float = Field(ge=0.0, le=1.0)
    aggregate_pre_gate: float = Field(ge=0.0, le=1.0)
    status: Literal["completed", "completed_with_findings"] = "completed"
    semantic_integrity: Literal["ok", "warnings", "issues", "not_evaluated"] = "not_evaluated"
    editorial_quality: Literal["ok", "warnings", "issues", "not_evaluated"] = "not_evaluated"
    semantic_findings: list[JsonObject] = Field(default_factory=list)
    editorial_findings: list[JsonObject] = Field(default_factory=list)
    technical_warnings: list[JsonObject] = Field(default_factory=list)
    all_issues: list[JsonObject] = Field(default_factory=list)
    summary: str

    @property
    def errors(self) -> list[JsonObject]:
        """Диагностические severity='error' findings.

        Это не semantic gate и не инструкция остановить pipeline. Решение о
        блокировке, если оно вообще нужно, должно приниматься внешним слоем явно.
        """

        return [item for item in self.all_issues if item.get("severity") == "error"]

    @property
    def warnings(self) -> list[JsonObject]:
        """Диагностические предупреждения."""

        return [item for item in self.all_issues if item.get("severity") == "warn"]

    def raise_for_errors(self) -> None:
        """Legacy helper для ручных проверок.

        Production workflow не должен использовать этот метод как semantic gate.
        """

        if self.errors:
            messages = "; ".join(str(item.get("message")) for item in self.errors[:5])
            raise EvaluationError(f"Evaluation findings with severity='error': {messages}")

    def to_public_dict(self) -> JsonObject:
        """Вернуть JSON-сериализуемый отчёт."""

        return self.model_dump(mode="json")

    def to_feedback_text(self, *, include_warnings: bool = True, max_items: int = 20) -> str:
        """Сформировать компактный feedback для editor/repair слоя."""

        selected = [*self.errors]
        if include_warnings:
            selected.extend(self.warnings)
        if not selected:
            return "Evaluation не выявил критических замечаний."

        lines = ["Замечания evaluation-layer:"]
        for item in selected[:max_items]:
            criterion = item.get("criterion") or item.get("evaluator") or "unknown"
            section = item.get("section") or "global"
            detail = item.get("detail")
            line = f"- [{criterion}] {section}: {item.get('message')}"
            if detail:
                line += f" ({detail})"
            lines.append(line)
        return "\n".join(lines)


class EvalConfig(StrictModel):
    """Настройки evaluation-layer."""

    numeric_tolerance_pct: float = Field(default=1.5, ge=0.0)
    weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_EVAL_WEIGHTS))
    enabled_evaluators: tuple[str, ...] = Field(default_factory=lambda: tuple(DEFAULT_ENABLED_EVALUATORS))
    judge_temperature: float = DEFAULT_JUDGE_TEMPERATURE
    llm: Any = None
    request_timeout_sec: int | None = Field(default=None, gt=0)
    raw_preview_chars: int = Field(default=500, gt=0)

    @field_validator("enabled_evaluators", mode="before")
    @classmethod
    def normalize_enabled_evaluators(cls, value: Any) -> tuple[str, ...]:
        if value is None:
            value = DEFAULT_ENABLED_EVALUATORS
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in value:
            name = str(raw).strip()
            if not name:
                continue
            if name not in ALL_EVALUATOR_FACTORIES:
                raise ValueError(
                    f"Unknown evaluator: {name!r}. Allowed: {list(ALL_EVALUATOR_FACTORIES.keys())!r}."
                )
            if name not in seen:
                normalized.append(name)
                seen.add(name)
        if not normalized:
            raise ValueError("enabled_evaluators must contain at least one evaluator.")
        return tuple(normalized)

    @model_validator(mode="after")
    def validate_weights(self) -> "EvalConfig":
        missing = [name for name in self.enabled_evaluators if name not in self.weights]
        if missing:
            raise ValueError(f"weights missing entries for enabled evaluators: {missing!r}.")
        for name, weight in self.weights.items():
            if weight < 0:
                raise ValueError(f"Evaluator weight must be non-negative: {name!r}.")
        return self


class ExpectedNumber(StrictModel):
    """Один числовой якорь, который должен присутствовать в разделе."""

    label: str
    value: float
    section: str
    required: bool = True


class CanonicalClaim(StrictModel):
    """Каноническое утверждение, построенное из GlobalContext."""

    claim_id: str
    section: str
    predicate: str
    value: Any
    horizon_weeks: int | None = None
    required: bool = True
    weight: float = Field(default=1.0, gt=0.0)
    severity_if_missing: Severity = "warn"
    severity_if_contradicted: Severity = "error"


class ExtractedClaim(StrictModel):
    """Утверждение, извлечённое LLM-judge из текста."""

    section: str
    predicate: str
    value: Any
    horizon_weeks: int | None = None
    modality: str
    evidence: str = ""


class ClaimVerdict(StrictModel):
    """Результат сопоставления extracted claim с canonical claim."""

    claim_id: str
    status: Literal["supported", "missing", "contradicted", "unknown"]
    message: str
    detail: str | None = None


class BaseEvaluator(ABC):
    """Базовый класс evaluator-а."""

    name: str

    @abstractmethod
    def evaluate(self, ctx: Mapping[str, Any], bulletin: Mapping[str, Any], config: EvalConfig) -> EvalScore:
        """Оценить бюллетень по одному критерию."""


class NumericEvaluator(BaseEvaluator):
    """Проверка наличия числовых якорей из контекста."""

    name = "numeric"

    def build_expectations(self, ctx: Mapping[str, Any], bulletin: Mapping[str, Any]) -> list[ExpectedNumber]:
        expectations: list[ExpectedNumber] = []
        origin = _as_mapping(ctx.get("origin"))
        current = _as_mapping(ctx.get("current_situation"))
        forecast = _as_mapping(ctx.get("forecast"))
        model_quality = _as_mapping(ctx.get("model_quality"))
        model_info = _as_mapping(ctx.get("model_info"))

        _append_number(expectations, "номер эпидемиологической недели", origin.get("iso_week"), "current_situation")
        _append_number(expectations, "заболеваемость текущей недели", current.get("current_value"), "current_situation")
        _append_number(
            expectations,
            "заболеваемость предыдущей недели",
            current.get("previous_value"),
            "current_situation",
            required=False,
        )

        for horizon in _as_sequence(forecast.get("horizons")):
            h = _as_mapping(horizon)
            weeks = h.get("horizon_weeks")
            _append_number(
                expectations,
                f"точечный прогноз (горизонт {weeks} нед.)",
                h.get("point_forecast"),
                "forecast_risks",
            )

        metrics = [_as_mapping(item) for item in _as_sequence(model_quality.get("metrics"))]
        first_horizon = metrics[0].get("horizon_weeks") if metrics else None
        for metric in metrics:
            horizon = metric.get("horizon_weeks")
            for metric_name in ("mae", "rmse", "r2"):
                _append_number(
                    expectations,
                    f"{metric_name.upper()} (h={horizon})",
                    metric.get(metric_name),
                    "model_quality",
                    required=(horizon == first_horizon),
                )

        error_stats = _as_mapping(model_quality.get("error_stats_h1"))
        _append_number(
            expectations,
            "медиана ошибки (h=1)",
            error_stats.get("median_error_h1"),
            "model_quality",
            required=False,
        )
        _append_number(
            expectations,
            "макс. ошибка (h=1)",
            error_stats.get("max_error_h1"),
            "model_quality",
            required=False,
        )
        _append_number(
            expectations,
            "год начала калибровки",
            model_info.get("calibration_start_year"),
            "model_description",
        )

        if "age_group_season_table" in _sections_dict(bulletin):
            age_group = _as_mapping(ctx.get("age_group_season"))
            for row in _as_sequence(age_group.get("rows")):
                row_map = _as_mapping(row)
                label = row_map.get("age_group_label") or row_map.get("age_group_code") or "группа"
                _append_number(
                    expectations,
                    f"накопленная сезонная заболеваемость ({label})",
                    row_map.get("cumulative_incidence_pct"),
                    "age_group_season_table",
                )
                _append_number(
                    expectations,
                    f"неделя пика ({label})",
                    row_map.get("peak_week"),
                    "age_group_season_table",
                )
                _append_number(
                    expectations,
                    f"пик на 10 тыс. ({label})",
                    row_map.get("peak_inc_per_10k"),
                    "age_group_season_table",
                )
                _append_number(
                    expectations,
                    f"средняя недельная заболеваемость на 10 тыс. ({label})",
                    row_map.get("mean_weekly_inc_per_10k"),
                    "age_group_season_table",
                    required=False,
                )
                _append_number(
                    expectations,
                    f"ширина главного пика ({label})",
                    row_map.get("peak_width_weeks"),
                    "age_group_season_table",
                    required=False,
                )

        return expectations

    def evaluate(self, ctx: Mapping[str, Any], bulletin: Mapping[str, Any], config: EvalConfig) -> EvalScore:
        expectations = self.build_expectations(ctx, bulletin)
        issues: list[Issue] = []
        checks_total = 0
        checks_passed = 0

        for expected in expectations:
            text = _section_text(bulletin, expected.section)
            numbers = _extract_numbers(text)
            checks_total += 1
            if _number_present(expected.value, numbers, config.numeric_tolerance_pct):
                checks_passed += 1
                continue
            issues.append(
                Issue(
                    evaluator=self.name,
                    severity="error" if expected.required else "warn",
                    section=expected.section,
                    message=f"Не найдено: {expected.label} = {expected.value:g}",
                    detail=f"Числа в разделе: {sorted(set(numbers))[:20]}",
                )
            )

        score = checks_passed / checks_total if checks_total else 1.0
        return EvalScore(
            name=self.name,
            score=round(score, 4),
            evaluated=True,
            issues=issues,
            checks_total=checks_total,
            checks_passed=checks_passed,
        )


class FactualEvaluator(BaseEvaluator):
    """Claim-level проверка фактологии относительно GlobalContext."""

    name = "factual"

    SECTION_SPECS: dict[str, dict[str, Any]] = {
        "current_situation": {
            "predicates": ["direction_code", "trend_4w_code"],
            "notes": "Нормализуй только направление изменения текущей недели и характер 4-недельного тренда.",
        },
        "forecast_risks": {
            "predicates": [
                "forecast_shape_code",
                "uncertainty_level_code",
                "has_intermediate_rise",
                "has_intermediate_decline",
            ],
            "notes": "Нормализуй форму прогноза, уровень неопределённости и промежуточный рост/снижение.",
        },
        "shap_interpretation": {
            "predicates": ["h1_dominant_group", "h4_dominant_group", "transition_pattern"],
            "notes": "Нормализуй доминирующие группы факторов на h=1 и h=4 и переход между горизонтами.",
        },
        "model_quality": {
            "predicates": ["worst_case_regime", "peak_error_months"],
            "notes": "Нормализуй режим worst-case ошибок и месяцы основных ошибок.",
        },
        "model_description": {
            "predicates": ["family_code", "strategy_code", "calibration_start_year"],
            "notes": "Нормализуй тип модели, стратегию многошагового прогноза и стартовый год калибровки.",
        },
        "age_group_season_overview": {
            "predicates": [
                "highest_peak_group",
                "largest_cumulative_incidence_group",
                "widest_wave_group",
                "latest_peak_group",
            ],
            "notes": "Нормализуй только явные утверждения о лидирующих возрастных группах по сезонным показателям.",
        },
    }

    def build_canonical_claims(self, ctx: Mapping[str, Any]) -> list[CanonicalClaim]:
        claims: list[CanonicalClaim] = []
        current_sem = _semantic(ctx, "current_situation")
        forecast_sem = _semantic(ctx, "forecast")
        shap_sem = _semantic(ctx, "shap_summary")
        model_quality_sem = _semantic(ctx, "model_quality")
        model_info = _as_mapping(ctx.get("model_info"))
        model_sem = _as_mapping(model_info.get("semantic"))

        self._add_claim(claims, "cs_direction_code", "current_situation", "direction_code", current_sem.get("direction_code"), weight=2.0, severity_if_missing="error")
        self._add_claim(claims, "cs_trend_4w_code", "current_situation", "trend_4w_code", current_sem.get("trend_4w_code"), weight=1.5)
        self._add_claim(claims, "forecast_shape_code", "forecast_risks", "forecast_shape_code", forecast_sem.get("shape_code"), weight=2.5, severity_if_missing="error")
        self._add_claim(claims, "forecast_uncertainty_level_code", "forecast_risks", "uncertainty_level_code", forecast_sem.get("uncertainty_level_code"), weight=2.0)
        self._add_claim(claims, "shap_h1_dominant_group", "shap_interpretation", "h1_dominant_group", shap_sem.get("h1_dominant_group"), horizon_weeks=1, weight=1.5)
        self._add_claim(claims, "shap_h4_dominant_group", "shap_interpretation", "h4_dominant_group", shap_sem.get("h4_dominant_group"), horizon_weeks=4, weight=1.5)
        self._add_claim(claims, "shap_transition_pattern", "shap_interpretation", "transition_pattern", shap_sem.get("transition_pattern"), weight=2.0)
        self._add_claim(claims, "model_quality_worst_case_regime", "model_quality", "worst_case_regime", model_quality_sem.get("worst_case_regime"), weight=2.0)
        self._add_claim(claims, "model_quality_peak_error_months", "model_quality", "peak_error_months", model_quality_sem.get("peak_error_months"), weight=1.5)
        self._add_claim(claims, "model_family_code", "model_description", "family_code", model_sem.get("family_code"), weight=1.5)
        self._add_claim(claims, "model_strategy_code", "model_description", "strategy_code", model_sem.get("strategy_code"), weight=1.5)
        self._add_claim(claims, "model_calibration_start_year", "model_description", "calibration_start_year", model_info.get("calibration_start_year"), weight=1.0)

        if forecast_sem.get("has_intermediate_rise"):
            self._add_claim(claims, "forecast_has_intermediate_rise", "forecast_risks", "has_intermediate_rise", True, weight=1.0)
        if forecast_sem.get("has_intermediate_decline"):
            self._add_claim(claims, "forecast_has_intermediate_decline", "forecast_risks", "has_intermediate_decline", True, weight=1.0)

        age_sem = _semantic(ctx, "age_group_season")
        age_mapping = [
            ("highest_peak_group", "highest_peak_group_code", 1.5),
            ("largest_cumulative_incidence_group", "largest_cumulative_incidence_group_code", 1.5),
            ("latest_peak_group", "latest_peak_group_code", 1.25),
            ("widest_wave_group", "widest_wave_group_code", 1.0),
        ]
        for predicate, key, weight in age_mapping:
            value = age_sem.get(key)
            self._add_claim(
                claims,
                f"age_group_season_{predicate}",
                "age_group_season_overview",
                predicate,
                value,
                weight=weight,
                required=(predicate != "widest_wave_group"),
            )

        return claims

    @staticmethod
    def _add_claim(
        claims: list[CanonicalClaim],
        claim_id: str,
        section: str,
        predicate: str,
        value: Any,
        *,
        horizon_weeks: int | None = None,
        required: bool = True,
        weight: float = 1.0,
        severity_if_missing: Severity = "warn",
        severity_if_contradicted: Severity = "error",
    ) -> None:
        if value is None:
            return
        if isinstance(value, str) and not value.strip():
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and len(value) == 0:
            return
        claims.append(
            CanonicalClaim(
                claim_id=claim_id,
                section=section,
                predicate=predicate,
                value=value,
                horizon_weeks=horizon_weeks,
                required=required,
                weight=weight,
                severity_if_missing=severity_if_missing,
                severity_if_contradicted=severity_if_contradicted,
            )
        )

    def evaluate(self, ctx: Mapping[str, Any], bulletin: Mapping[str, Any], config: EvalConfig) -> EvalScore:
        if config.llm is None:
            return EvalScore(
                name=self.name,
                score=0.0,
                evaluated=False,
                issues=[Issue(evaluator=self.name, severity="info", section="global", message="LLM не предоставлен — factual judge пропущен")],
            )

        canonical_claims = self.build_canonical_claims(ctx)
        allowed_values = self._build_allowed_values(canonical_claims)
        extracted_claims: list[ExtractedClaim] = []
        active_claims: list[CanonicalClaim] = []
        failed_sections: set[str] = set()
        issues: list[Issue] = []

        for section in sorted({claim.section for claim in canonical_claims}):
            text = _section_text(bulletin, section)
            if not text.strip():
                continue
            try:
                extracted_claims.extend(self._extract_claims_for_section(section, text, ctx, config, allowed_values))
            except Exception as exc:
                failed_sections.add(section)
                issues.append(
                    Issue(
                        evaluator=self.name,
                        severity="warn",
                        section=section,
                        message="Claim extraction не выполнен",
                        detail=str(exc),
                    )
                )

        for claim in canonical_claims:
            if claim.section not in failed_sections and _section_text(bulletin, claim.section).strip():
                active_claims.append(claim)

        verdicts = self._adjudicate_claims(active_claims, extracted_claims)
        weighted_total = sum(claim.weight for claim in active_claims)
        claim_by_id = {claim.claim_id: claim for claim in active_claims}
        penalty = 0.0
        checks_passed = 0

        for verdict in verdicts:
            claim = claim_by_id[verdict.claim_id]
            if verdict.status == "supported":
                checks_passed += 1
                continue
            if verdict.status == "missing":
                penalty += 0.5 * claim.weight
                severity = claim.severity_if_missing
            elif verdict.status == "contradicted":
                penalty += 1.0 * claim.weight
                severity = claim.severity_if_contradicted
            else:
                penalty += 0.25 * claim.weight
                severity = "warn"
            issues.append(
                Issue(
                    evaluator=self.name,
                    severity=severity,
                    section=claim.section,
                    message=verdict.message,
                    detail=verdict.detail,
                )
            )

        score = max(0.0, 1.0 - penalty / weighted_total) if weighted_total > 0 else 1.0
        return EvalScore(
            name=self.name,
            score=round(score, 4),
            evaluated=weighted_total > 0,
            issues=issues,
            checks_total=len(active_claims),
            checks_passed=checks_passed,
        )

    def _build_allowed_values(self, canonical_claims: Sequence[CanonicalClaim]) -> dict[str, list[Any]]:
        allowed: dict[str, list[Any]] = {}
        for claim in canonical_claims:
            values = allowed.setdefault(claim.predicate, [])
            if claim.value not in values:
                values.append(claim.value)
        allowed.setdefault("has_intermediate_rise", [True, False])
        allowed.setdefault("has_intermediate_decline", [True, False])
        return allowed

    def _extract_claims_for_section(
        self,
        section: str,
        text: str,
        ctx: Mapping[str, Any],
        config: EvalConfig,
        allowed_values: Mapping[str, list[Any]],
    ) -> list[ExtractedClaim]:
        spec = self.SECTION_SPECS.get(section)
        if spec is None:
            return []
        predicates = [str(item) for item in spec["predicates"]]
        schema = _claim_extraction_schema(predicates)
        system = (
            "Твоя задача — не оценивать правильность текста, а только нормализовать "
            "явные содержательные утверждения в структурированные claims.\n"
            "Не придумывай claims, которых нет в тексте. Если утверждение не выражено явно, не возвращай его.\n"
            "predicate должен быть только из разрешённого списка. value выбирай из allowed_values, если значение подходит; "
            "если текст явно противоречит reference, верни фактически утверждаемое значение.\n"
            "modality: asserted, hedged или unclear. Верни только JSON заданной структуры."
        )
        payload = {
            "section": section,
            "section_notes": spec.get("notes", ""),
            "allowed_predicates": predicates,
            "allowed_values": {key: value for key, value in allowed_values.items() if key in predicates},
            "context_reference": self._context_reference_for_section(ctx, section),
            "section_text": text,
        }
        parsed = _call_json_llm(
            config.llm,
            system=system,
            payload=payload,
            schema=schema,
            config=config,
            max_tokens=1800,
            evaluator_name=self.name,
        )
        extracted: list[ExtractedClaim] = []
        for item in parsed.get("claims", []) or []:
            if not isinstance(item, Mapping):
                continue
            extracted.append(
                ExtractedClaim(
                    section=section,
                    predicate=str(item.get("predicate") or ""),
                    value=item.get("value"),
                    horizon_weeks=item.get("horizon_weeks"),
                    modality=str(item.get("modality") or "unclear"),
                    evidence=str(item.get("evidence") or ""),
                )
            )
        return extracted

    @staticmethod
    def _context_reference_for_section(ctx: Mapping[str, Any], section: str) -> JsonObject:
        if section == "current_situation":
            return _semantic(ctx, "current_situation")
        if section == "forecast_risks":
            return _semantic(ctx, "forecast")
        if section == "shap_interpretation":
            return _semantic(ctx, "shap_summary")
        if section == "model_quality":
            return _semantic(ctx, "model_quality")
        if section == "age_group_season_overview":
            return _semantic(ctx, "age_group_season")
        if section == "model_description":
            model_info = _as_mapping(ctx.get("model_info"))
            return {
                "family_code": _as_mapping(model_info.get("semantic")).get("family_code"),
                "strategy_code": _as_mapping(model_info.get("semantic")).get("strategy_code"),
                "calibration_start_year": model_info.get("calibration_start_year"),
            }
        return {}

    @staticmethod
    def _adjudicate_claims(
        canonical_claims: Sequence[CanonicalClaim],
        extracted_claims: Sequence[ExtractedClaim],
    ) -> list[ClaimVerdict]:
        verdicts: list[ClaimVerdict] = []
        for claim in canonical_claims:
            candidates = [
                ext
                for ext in extracted_claims
                if ext.section == claim.section
                and ext.predicate == claim.predicate
                and (claim.horizon_weeks is None or ext.horizon_weeks in (claim.horizon_weeks, None))
            ]
            if not candidates:
                verdicts.append(
                    ClaimVerdict(
                        claim_id=claim.claim_id,
                        status="missing",
                        message=f"Утверждение не отражено: {claim.predicate}",
                        detail=f"Ожидалось значение {claim.value!r} в разделе {claim.section!r}.",
                    )
                )
                continue
            matched = next((ext for ext in candidates if _compare_claim_values(claim.value, ext.value)), None)
            if matched is not None:
                verdicts.append(
                    ClaimVerdict(
                        claim_id=claim.claim_id,
                        status="supported",
                        message=f"Утверждение подтверждено: {claim.predicate}",
                        detail=f"Фрагмент: «{matched.evidence[:200]}»",
                    )
                )
                continue
            first = candidates[0]
            verdicts.append(
                ClaimVerdict(
                    claim_id=claim.claim_id,
                    status="contradicted",
                    message=f"Утверждение противоречит ctx: {claim.predicate}",
                    detail=f"Ожидалось {claim.value!r}, извлечено {first.value!r}; фрагмент: «{first.evidence[:200]}»",
                )
            )
        return verdicts


class LogicEvaluator(BaseEvaluator):
    """Проверка логической согласованности."""

    name = "logic"
    ABSOLUTE_CERTAINTY_WORDS = frozenset(
        [
            "гарантированно",
            "несомненно",
            "безусловно",
            "однозначно",
            "точно произойдет",
            "обязательно произойдет",
            "определенно приведет",
        ]
    )

    SHAP_DIRECTION_SCHEMA: JsonObject = {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fragment": {"type": "string"},
                        "horizon": {"type": "string"},
                        "factor": {"type": "string"},
                        "claim_kind": {"type": "string"},
                        "direction": {"type": "string"},
                        "modality": {"type": "string"},
                    },
                    "required": ["fragment", "horizon", "factor", "claim_kind", "direction", "modality"],
                },
            }
        },
        "required": ["claims"],
    }

    SHAP_SYSTEM_PROMPT = """
Ты извлекаешь ТОЛЬКО структурированные высказывания из раздела SHAP-интерпретации.

Вход содержит inventory — допустимые канонические факторы по горизонтам — и section_text.
Для каждого явного высказывания о факторе верни:
- fragment: дословный фрагмент текста;
- horizon: один из горизонтов из inventory;
- factor: точное каноническое название фактора из inventory;
- claim_kind: directional, importance или other;
- direction: positive, negative, mixed или unknown;
- modality: hard, soft или conditional.

Не придумывай факторов вне inventory. Не ставь интегральных оценок. Верни только JSON.
""".strip()

    def evaluate(self, ctx: Mapping[str, Any], bulletin: Mapping[str, Any], config: EvalConfig) -> EvalScore:
        issues: list[Issue] = []
        weighted_total = 0.0
        weighted_passed = 0.0
        checks_total = 0
        checks_passed = 0

        for name, section, description, weight, passed, explanation in self._deterministic_results(ctx, bulletin):
            weighted_total += weight
            checks_total += 1
            if passed:
                weighted_passed += weight
                checks_passed += 1
            else:
                issues.append(
                    Issue(
                        evaluator=self.name,
                        severity="error",
                        section=section,
                        message=f"[{name}] {description}",
                        detail=explanation,
                    )
                )

        shap_passed, shap_issues, shap_weight, shap_checks = self._evaluate_shap_direction_contract(ctx, bulletin, config)
        issues.extend(shap_issues)
        weighted_total += shap_weight
        checks_total += shap_checks
        if shap_passed is True:
            weighted_passed += shap_weight
            checks_passed += shap_checks

        score = weighted_passed / weighted_total if weighted_total > 0 else 1.0
        return EvalScore(
            name=self.name,
            score=round(score, 4),
            evaluated=True,
            issues=issues,
            checks_total=checks_total,
            checks_passed=checks_passed,
        )

    def _deterministic_results(
        self,
        ctx: Mapping[str, Any],
        bulletin: Mapping[str, Any],
    ) -> Iterable[tuple[str, str, str, float, bool, str]]:
        yield (
            "no_overconfidence",
            "forecast_risks",
            "Нет слов абсолютной уверенности при высокой неопределённости",
            2.0,
            *self._rule_no_overconfidence_under_uncertainty(ctx, bulletin),
        )
        yield (
            "correct_interval_term",
            "forecast_risks",
            "Корректная терминология интервалов",
            1.5,
            *self._rule_correct_interval_terminology(ctx, bulletin),
        )
        yield (
            "adverse_is_high",
            "forecast_risks",
            "Неблагоприятный сценарий = высокая заболеваемость",
            2.0,
            *self._rule_adverse_scenario_is_high_incidence(ctx, bulletin),
        )
        yield (
            "trend_matches_numbers",
            "forecast_risks",
            "Текстовый тренд согласован с числами прогноза",
            3.0,
            *self._rule_forecast_trend_matches_numbers(ctx, bulletin),
        )

    def _rule_no_overconfidence_under_uncertainty(self, ctx: Mapping[str, Any], bulletin: Mapping[str, Any]) -> tuple[bool, str]:
        forecast = _as_mapping(ctx.get("forecast"))
        unc_pct = _to_float_or_none(forecast.get("relative_uncertainty_pct")) or 0.0
        if unc_pct <= 80:
            return True, "Неопределённость не высокая — проверка не применяется."
        text = _normalize_text(_section_text(bulletin, "forecast_risks"))
        found = [word for word in self.ABSOLUTE_CERTAINTY_WORDS if word in text]
        if found:
            return False, f"Неопределённость {unc_pct:g}%, но текст содержит слова абсолютной уверенности: {found}."
        return True, "Нет слов абсолютной уверенности."

    @staticmethod
    def _rule_correct_interval_terminology(ctx: Mapping[str, Any], bulletin: Mapping[str, Any]) -> tuple[bool, str]:
        text = _normalize_text(_section_text(bulletin, "forecast_risks"))
        has_confidence = "доверительн" in text and "интервал" in text
        has_prediction = "прогнозн" in text and "интервал" in text
        if has_confidence and not has_prediction:
            return False, "Использован термин 'доверительный интервал'. Корректный термин — 'прогнозный интервал'."
        return True, "Терминология интервалов корректна."

    @staticmethod
    def _rule_adverse_scenario_is_high_incidence(ctx: Mapping[str, Any], bulletin: Mapping[str, Any]) -> tuple[bool, str]:
        text = _normalize_text(_section_text(bulletin, "forecast_risks"))
        adverse_words = ["неблагоприятн", "негативн", "пессимистичн", "худш"]
        low_words = ["нулев", "нижн", "минимальн"]
        high_words = ["верхн", "высок", "максимальн"]
        if not any(word in text for word in adverse_words):
            return True, "Термины неблагоприятного сценария не используются."
        near_low = any(word in text for word in low_words)
        near_high = any(word in text for word in high_words)
        if near_low and not near_high:
            return False, "Неблагоприятный сценарий ассоциирован с нижней границей интервала; в эпидемиологии неблагоприятный = высокая заболеваемость."
        return True, "Риск-фрейминг корректен."

    @staticmethod
    def _rule_forecast_trend_matches_numbers(ctx: Mapping[str, Any], bulletin: Mapping[str, Any]) -> tuple[bool, str]:
        horizons = [_as_mapping(item) for item in _as_sequence(_as_mapping(ctx.get("forecast")).get("horizons"))]
        values = [_to_float_or_none(item.get("point_forecast")) for item in horizons]
        pf_values = [value for value in values if value is not None]
        if len(pf_values) < 2:
            return True, "Недостаточно горизонтов для проверки."
        diffs = [pf_values[i + 1] - pf_values[i] for i in range(len(pf_values) - 1)]
        all_increasing = all(diff > 0 for diff in diffs)
        all_decreasing = all(diff < 0 for diff in diffs)
        text = _normalize_text(_section_text(bulletin, "forecast_risks"))
        qualifier_words = ["затем", "однако", "после", "сменил", "колебан", "временн", "впоследств"]
        has_qualifier = any(word in text for word in qualifier_words)
        if all_increasing:
            decline_words = ["снижени", "уменьшени", "падени", "сокращени"]
            if any(word in text for word in decline_words) and not has_qualifier:
                return False, f"Все прогнозные значения монотонно растут ({' → '.join(f'{v:.2f}' for v in pf_values)}), но текст утверждает снижение без оговорок."
        if all_decreasing:
            growth_words = ["рост", "повышени", "увеличени", "возрастани"]
            if any(word in text for word in growth_words) and not has_qualifier:
                return False, f"Все прогнозные значения монотонно снижаются ({' → '.join(f'{v:.2f}' for v in pf_values)}), но текст утверждает рост без оговорок."
        return True, "Текстовый тренд не противоречит числам."

    def _evaluate_shap_direction_contract(
        self,
        ctx: Mapping[str, Any],
        bulletin: Mapping[str, Any],
        config: EvalConfig,
    ) -> tuple[bool | None, list[Issue], float, int]:
        text = _section_text(bulletin, "shap_interpretation")
        inventory = _shap_inventory(ctx)
        if not text.strip() or not inventory:
            return True, [], 0.0, 0
        if config.llm is None:
            return None, [Issue(evaluator=self.name, severity="info", section="shap_interpretation", message="LLM не предоставлен — SHAP direction judge пропущен")], 0.0, 0

        parsed = _call_json_llm(
            config.llm,
            system=self.SHAP_SYSTEM_PROMPT,
            payload={"inventory": inventory, "section_text": text},
            schema=self.SHAP_DIRECTION_SCHEMA,
            config=config,
            max_tokens=1600,
            evaluator_name=self.name,
        )
        inventory_map = {
            (item["horizon"], item["factor"]): item["direction_reliable"]
            for item in inventory
            if isinstance(item, Mapping)
        }
        issues: list[Issue] = []
        checks = 0
        passed = True
        for raw_claim in parsed.get("claims", []) or []:
            if not isinstance(raw_claim, Mapping):
                continue
            if raw_claim.get("claim_kind") != "directional":
                continue
            checks += 1
            key = (str(raw_claim.get("horizon") or ""), str(raw_claim.get("factor") or ""))
            reliable = inventory_map.get(key, True)
            modality = str(raw_claim.get("modality") or "").lower()
            if reliable is False and modality == "hard":
                passed = False
                issues.append(
                    Issue(
                        evaluator=self.name,
                        severity="warn",
                        section="shap_interpretation",
                        message="Ненадёжное направление SHAP описано как устойчивый факт",
                        detail=str(raw_claim.get("fragment") or ""),
                    )
                )
        if checks == 0:
            return True, issues, 0.0, 0
        return passed, issues, 1.0, checks


class WaterEvaluator(BaseEvaluator):
    """LLM structural adjudication информационной плотности."""

    name = "water"
    RESPONSE_SCHEMA: JsonObject = {
        "type": "object",
        "properties": {
            "document_claims": {"type": "array", "items": {"type": "string"}},
            "span_to_claim_links": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fragment": {"type": "string"},
                        "section": {"type": "string"},
                        "linked_claims": {"type": "array", "items": {"type": "string"}},
                        "role": {"type": "string"},
                    },
                    "required": ["fragment", "section", "linked_claims", "role"],
                },
            },
            "orphan_spans": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fragment": {"type": "string"},
                        "section": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["fragment", "section", "reason"],
                },
            },
        },
        "required": ["document_claims", "span_to_claim_links", "orphan_spans"],
    }
    SYSTEM_PROMPT = """
Ты — строгий редактор научно-аналитического бюллетеня.

ОБЛАСТЬ ОЦЕНКИ: только narrative-разделы. Таблицы и подписи рисунков сюда не входят.

ЗАДАЧА:
1) восстановить множество содержательных claims документа;
2) построить отображение fragment -> linked_claims;
3) вернуть orphan_spans — только те фрагменты, которые не реализуют ни одного claim
   и не вносят модальность, ограничение, риск, интерпретацию или причинную связь.

Не считай пустыми аналитические выводы, риск/неопределённость, ограничения и причинные связи.
Верни только JSON заданной структуры.
""".strip()

    def evaluate(self, ctx: Mapping[str, Any], bulletin: Mapping[str, Any], config: EvalConfig) -> EvalScore:
        return _evaluate_fragment_judge(
            evaluator_name=self.name,
            llm=config.llm,
            bulletin=bulletin,
            config=config,
            section_names=_narrative_section_names(bulletin),
            system_prompt=self.SYSTEM_PROMPT,
            schema=self.RESPONSE_SCHEMA,
            response_array_key="orphan_spans",
            fragment_key="fragment",
            issue_type_key="reason",
            default_issue_type="не вносит собственного семантического вклада",
            empty_message="LLM не предоставлен — оценка воды пропущена",
            max_tokens=1400,
            message_prefix="Пустой фрагмент",
        )


class TautologyEvaluator(BaseEvaluator):
    """LLM structural adjudication повторов мыслей.

    Реализация намеренно разделена на два компактных LLM-контракта:
    1) извлечение коротких claim-candidates по каждому разделу;
    2) поиск настоящих повторов в compact inventory.

    Это сохраняет смысл notebook evaluator-а, но не заставляет модель
    возвращать длинный вложенный JSON с дословными фрагментами всего раздела.
    """

    name = "tautology"
    RESPONSE_SCHEMA: JsonObject = {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {"type": "string"},
                        "evidence": {"type": "string"},
                    },
                    "required": ["claim", "evidence"],
                },
            }
        },
        "required": ["claims"],
    }
    REPETITION_SCHEMA: JsonObject = {
        "type": "object",
        "properties": {
            "repetitions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "repetition_id": {"type": "string"},
                        "primary_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["repetition_id", "primary_id", "reason"],
                },
            }
        },
        "required": ["repetitions"],
    }
    SECTION_SYSTEM_PROMPT = (
        "Ты — редактор научно-аналитического бюллетеня.\n\n"
        "Для ОДНОГО раздела извлеки компактный список уникальных содержательных утверждений.\n"
        "Не возвращай длинные фрагменты текста. Поле evidence должно быть короткой опорной цитатой или "
        "сжатым фрагментом, достаточным для проверки, желательно не длиннее одного предложения.\n"
        "Не объединяй разные мысли: новое число, причина, ограничение, временная перспектива или "
        "аналитический аспект означают отдельный claim. Верни только JSON указанной структуры."
    )
    REPETITION_SYSTEM_PROMPT = (
        "Ты — редактор научно-аналитического бюллетеня.\n\n"
        "На входе compact inventory утверждений: claim_id, section, claim, evidence. "
        "Нужно найти только настоящие повторы одной и той же мысли.\n"
        "Не считай повтором refinement: если утверждение добавляет новое число, условие, причину, "
        "ограничение, временную перспективу или аналитический аспект, его НЕ надо возвращать.\n"
        "Верни только JSON с массивом repetitions. Каждый элемент — это claim_id повторной реализации "
        "и claim_id первичной реализации. Не возвращай неповторяющиеся claims."
    )

    def evaluate(self, ctx: Mapping[str, Any], bulletin: Mapping[str, Any], config: EvalConfig) -> EvalScore:
        if config.llm is None:
            return EvalScore(
                name=self.name,
                score=0.0,
                evaluated=False,
                issues=[Issue(evaluator=self.name, severity="info", section="global", message="LLM не предоставлен — оценка тавтологии пропущена")],
            )
        section_texts = _section_text_map(bulletin, _narrative_section_names(bulletin))
        if not section_texts:
            return EvalScore(name=self.name, score=1.0, evaluated=True)

        candidates: list[JsonObject] = []
        candidate_by_id: dict[str, JsonObject] = {}
        technical_issues: list[Issue] = []

        for section_name, section_text in section_texts.items():
            try:
                parsed = _call_json_llm(
                    config.llm,
                    system=self.SECTION_SYSTEM_PROMPT,
                    payload={"section": section_name, "section_text": section_text},
                    schema=self.RESPONSE_SCHEMA,
                    config=config,
                    max_tokens=1000,
                    evaluator_name=self.name,
                )
            except Exception as exc:
                technical_issues.append(
                    Issue(
                        evaluator=self.name,
                        severity="error",
                        section=section_name,
                        message="Ошибка извлечения claims для проверки тавтологии",
                        detail=str(exc),
                    )
                )
                continue

            raw_claims = parsed.get("claims", []) or []
            if not isinstance(raw_claims, Sequence) or isinstance(raw_claims, (str, bytes, bytearray)):
                technical_issues.append(
                    Issue(
                        evaluator=self.name,
                        severity="error",
                        section=section_name,
                        message="Ответ LLM нарушает top-level contract",
                        detail="Ключ 'claims' должен содержать массив.",
                    )
                )
                continue

            for claim_index, item in enumerate(raw_claims):
                if not isinstance(item, Mapping):
                    continue
                claim = str(item.get("claim") or "").strip()
                evidence = str(item.get("evidence") or "").strip()
                if not claim:
                    continue
                claim_id = f"{section_name}:{claim_index}"
                candidate = {
                    "claim_id": claim_id,
                    "section": section_name,
                    "claim": claim,
                    "evidence": evidence,
                }
                candidates.append(candidate)
                candidate_by_id[claim_id] = candidate

        if technical_issues:
            return EvalScore(
                name=self.name,
                score=0.0,
                evaluated=False,
                issues=technical_issues,
                checks_total=len(candidates),
                checks_passed=0,
            )
        if not candidates:
            return EvalScore(name=self.name, score=1.0, evaluated=True)

        try:
            parsed_repetitions = _call_json_llm(
                config.llm,
                system=self.REPETITION_SYSTEM_PROMPT,
                payload={"claims": candidates},
                schema=self.REPETITION_SCHEMA,
                config=config,
                max_tokens=1400,
                evaluator_name=self.name,
            )
        except Exception as exc:
            return EvalScore(
                name=self.name,
                score=0.0,
                evaluated=False,
                issues=[
                    Issue(
                        evaluator=self.name,
                        severity="error",
                        section="global",
                        message="Ошибка поиска повторов claims для проверки тавтологии",
                        detail=str(exc),
                    )
                ],
                checks_total=len(candidates),
                checks_passed=0,
            )

        raw_repetitions = parsed_repetitions.get("repetitions", []) or []
        if not isinstance(raw_repetitions, Sequence) or isinstance(raw_repetitions, (str, bytes, bytearray)):
            return EvalScore(
                name=self.name,
                score=0.0,
                evaluated=False,
                issues=[
                    Issue(
                        evaluator=self.name,
                        severity="error",
                        section="global",
                        message="Ответ LLM нарушает top-level contract",
                        detail="Ключ 'repetitions' должен содержать массив.",
                    )
                ],
                checks_total=len(candidates),
                checks_passed=0,
            )

        issues: list[Issue] = []
        repetition_ids: set[str] = set()
        for item in raw_repetitions:
            if not isinstance(item, Mapping):
                continue
            repetition_id = str(item.get("repetition_id") or "").strip()
            primary_id = str(item.get("primary_id") or "").strip()
            if not repetition_id or repetition_id not in candidate_by_id:
                continue
            if primary_id and primary_id not in candidate_by_id:
                continue
            if repetition_id == primary_id:
                continue
            if repetition_id in repetition_ids:
                continue
            repetition = candidate_by_id[repetition_id]
            primary = candidate_by_id.get(primary_id, {})
            repetition_ids.add(repetition_id)
            reason = str(item.get("reason") or "").strip()
            detail_parts = [f"Повтор: {repetition.get('evidence') or repetition.get('claim')}"]
            if primary:
                detail_parts.append(f"Первичная реализация: {primary.get('evidence') or primary.get('claim')}")
            if reason:
                detail_parts.append(f"Обоснование: {reason}")
            issues.append(
                Issue(
                    evaluator=self.name,
                    severity="warn",
                    section=str(repetition.get("section") or "global"),
                    message=f"Повтор мысли: {repetition.get('claim')}",
                    detail="; ".join(detail_parts),
                )
            )

        total_claims = len(candidates)
        repetition_count = len(repetition_ids)
        score = max(0.0, min(1.0, 1.0 - repetition_count / total_claims))
        return EvalScore(
            name=self.name,
            score=round(score, 4),
            evaluated=True,
            issues=issues,
            checks_total=total_claims,
            checks_passed=max(0, total_claims - repetition_count),
        )

def _language_issue_schema(array_key: str, issue_type_key: str) -> JsonObject:
    return {
        "type": "object",
        "properties": {
            array_key: {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fragment": {"type": "string"},
                        "section": {"type": "string"},
                        issue_type_key: {"type": "string"},
                        "suggestion": {"type": "string"},
                        "severity": {"type": "string"},
                    },
                    "required": ["fragment", "section", issue_type_key, "suggestion", "severity"],
                },
            }
        },
        "required": [array_key],
    }


class GrammarEvaluator(BaseEvaluator):
    """LLM structural adjudication собственно языковой нормы."""

    name = "grammar"
    RESPONSE_SCHEMA: JsonObject = _language_issue_schema("errors", "error_type")
    SYSTEM_PROMPT = """
Ты — строгий корректор русского языка.

ОБЛАСТЬ ОЦЕНКИ: только собственно языковая норма.
Считать ошибкой: орфографию слов, согласование, управление, реальные синтаксические нарушения.
Не считать ошибкой: десятичные дроби с точкой или запятой, оформление недель, диапазонов, процентов,
сокращений, специальные обозначения MAE, RMSE, R², SHAP, а также чисто стилевые замечания.
Верни только подтверждённые ошибки в JSON.
""".strip()

    def evaluate(self, ctx: Mapping[str, Any], bulletin: Mapping[str, Any], config: EvalConfig) -> EvalScore:
        return _evaluate_fragment_judge(
            evaluator_name=self.name,
            llm=config.llm,
            bulletin=bulletin,
            config=config,
            section_names=_prose_section_names(bulletin),
            system_prompt=self.SYSTEM_PROMPT,
            schema=self.RESPONSE_SCHEMA,
            response_array_key="errors",
            fragment_key="fragment",
            issue_type_key="error_type",
            default_issue_type="language",
            empty_message="LLM не предоставлен — оценка грамматики пропущена",
            max_tokens=2200,
        )


class OrthotypographyEvaluator(BaseEvaluator):
    """LLM structural adjudication орфо-типографической нормы."""

    name = "orthotypography"
    RESPONSE_SCHEMA: JsonObject = _language_issue_schema("issues", "issue_type")
    SYSTEM_PROMPT = """
Ты — редактор научного русского текста.

ОБЛАСТЬ ОЦЕНКИ: орфо-типографическая норма, а не грамматика и не стиль.
Считать нарушениями: оформление порядковых номеров недель в беглом тексте, десятичные дроби в русском
связном тексте, тире/дефис, проценты, сокращения и сходные орфо-типографические случаи.
Не считать нарушениями: содержательные числовые значения сами по себе, интервалы, табличные форматы,
грамматические и стилистические дефекты. Верни только подтверждённые нарушения в JSON.
""".strip()

    def evaluate(self, ctx: Mapping[str, Any], bulletin: Mapping[str, Any], config: EvalConfig) -> EvalScore:
        return _evaluate_fragment_judge(
            evaluator_name=self.name,
            llm=config.llm,
            bulletin=bulletin,
            config=config,
            section_names=_prose_section_names(bulletin),
            system_prompt=self.SYSTEM_PROMPT,
            schema=self.RESPONSE_SCHEMA,
            response_array_key="issues",
            fragment_key="fragment",
            issue_type_key="issue_type",
            default_issue_type="orthotypography",
            empty_message="LLM не предоставлен — орфо-типографическая оценка пропущена",
            max_tokens=1300,
        )


class StyleEvaluator(BaseEvaluator):
    """LLM structural adjudication редакторской ясности."""

    name = "style"
    RESPONSE_SCHEMA: JsonObject = _language_issue_schema("issues", "issue_type")
    SYSTEM_PROMPT = """
Ты — редактор научно-аналитического текста.

ОБЛАСТЬ ОЦЕНКИ: стиль и редакторская ясность, а не грамматика, не орфо-типографика и не фактология.
Считать проблемами: тяжёлые или расплывчатые конструкции, редакторски неудачные фрагменты, которые можно
сделать яснее без изменения смысла, лишний метатекст. Не считать проблемами аналитические выводы сами по
себе, повторы с новым уточнением, грамматические и орфо-типографические нарушения.
Верни только подтверждённые стилевые проблемы в JSON.
""".strip()

    def evaluate(self, ctx: Mapping[str, Any], bulletin: Mapping[str, Any], config: EvalConfig) -> EvalScore:
        return _evaluate_fragment_judge(
            evaluator_name=self.name,
            llm=config.llm,
            bulletin=bulletin,
            config=config,
            section_names=_narrative_section_names(bulletin),
            system_prompt=self.SYSTEM_PROMPT,
            schema=self.RESPONSE_SCHEMA,
            response_array_key="issues",
            fragment_key="fragment",
            issue_type_key="issue_type",
            default_issue_type="style",
            empty_message="LLM не предоставлен — стилевая оценка пропущена",
            max_tokens=2200,
        )


ALL_EVALUATOR_FACTORIES: "OrderedDict[str, type[BaseEvaluator]]" = OrderedDict(
    [
        ("numeric", NumericEvaluator),
        ("factual", FactualEvaluator),
        ("logic", LogicEvaluator),
        ("water", WaterEvaluator),
        ("tautology", TautologyEvaluator),
        ("grammar", GrammarEvaluator),
        ("orthotypography", OrthotypographyEvaluator),
        ("style", StyleEvaluator),
    ]
)


def evaluate_bulletin(
    ctx: GlobalContext | Mapping[str, Any],
    bulletin: Bulletin | Mapping[str, Any],
    config: EvalConfig | None = None,
) -> EvalReport:
    """Выполнить полную оценку бюллетеня по настроенному набору evaluators."""

    eval_config = config or EvalConfig()
    ctx_dict = _context_dict(ctx)
    bulletin_dict = _bulletin_dict(bulletin)
    scores: "OrderedDict[str, EvalScore]" = OrderedDict()
    all_issues: list[JsonObject] = []

    for name in eval_config.enabled_evaluators:
        evaluator = ALL_EVALUATOR_FACTORIES[name]()
        try:
            result = evaluator.evaluate(ctx_dict, bulletin_dict, eval_config)
        except Exception as exc:
            result = EvalScore(
                name=name,
                score=0.0,
                evaluated=False,
                issues=[
                    Issue(
                        evaluator=name,
                        severity="warn",
                        section="global",
                        message="Evaluator не был выполнен из-за технической ошибки",
                        detail=str(exc),
                    )
                ],
            )
        scores[name] = result
        for issue in result.issues:
            item = issue.model_dump(mode="json")
            item["criterion"] = name
            all_issues.append(item)

    evaluated_weights = {
        name: eval_config.weights[name]
        for name, score in scores.items()
        if score.evaluated and name in eval_config.weights
    }
    weight_sum = sum(evaluated_weights.values())
    aggregate = (
        sum(scores[name].score * weight / weight_sum for name, weight in evaluated_weights.items())
        if weight_sum > 0
        else 0.0
    )
    aggregate = round(aggregate, 4)
    semantic_findings = _findings_for_criteria(all_issues, SEMANTIC_EVALUATORS)
    editorial_findings = _findings_for_criteria(all_issues, EDITORIAL_EVALUATORS)
    technical_warnings = [
        item
        for item in all_issues
        if item.get("severity") == "warn" and not scores.get(str(item.get("criterion") or item.get("evaluator") or ""), EvalScore(name="unknown", score=0.0, evaluated=True)).evaluated
    ]
    semantic_integrity = _diagnostic_level(scores, semantic_findings, SEMANTIC_EVALUATORS)
    editorial_quality = _diagnostic_level(scores, editorial_findings, EDITORIAL_EVALUATORS)
    status = "completed_with_findings" if all_issues else "completed"
    summary = _build_summary(
        scores,
        all_issues,
        eval_config,
        aggregate,
        semantic_integrity=semantic_integrity,
        editorial_quality=editorial_quality,
    )
    return EvalReport(
        scores=dict(scores),
        aggregate=aggregate,
        aggregate_pre_gate=aggregate,
        status=status,
        semantic_integrity=semantic_integrity,
        editorial_quality=editorial_quality,
        semantic_findings=semantic_findings,
        editorial_findings=editorial_findings,
        technical_warnings=technical_warnings,
        all_issues=all_issues,
        summary=summary,
    )


def evaluate_bulletin_from_files(
    *,
    context_path: str | Path,
    bulletin_path: str | Path,
    config: EvalConfig | None = None,
) -> EvalReport:
    """Загрузить GlobalContext и Bulletin из JSON-файлов и выполнить evaluation."""

    return evaluate_bulletin(load_global_context(context_path), load_bulletin(bulletin_path), config=config)


def save_eval_report(report: EvalReport, path: str | Path, *, indent: int = 2) -> None:
    """Сохранить EvalReport как UTF-8 JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(report.to_public_dict(), stream, ensure_ascii=False, indent=indent)


def load_eval_report(path: str | Path) -> EvalReport:
    """Загрузить EvalReport из JSON."""

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as stream:
        return EvalReport.model_validate(json.load(stream))


def print_eval_report(report: EvalReport) -> None:
    """Напечатать человекочитаемую сводку."""

    print(report.summary)


def _evaluate_fragment_judge(
    *,
    evaluator_name: str,
    llm: ChatBackend | None,
    bulletin: Mapping[str, Any],
    config: EvalConfig,
    section_names: Sequence[str],
    system_prompt: str,
    schema: Mapping[str, Any],
    response_array_key: str,
    fragment_key: str,
    issue_type_key: str,
    default_issue_type: str,
    empty_message: str,
    max_tokens: int,
    message_prefix: str | None = None,
) -> EvalScore:
    if llm is None:
        return EvalScore(
            name=evaluator_name,
            score=0.0,
            evaluated=False,
            issues=[Issue(evaluator=evaluator_name, severity="info", section="global", message=empty_message)],
        )
    section_texts = _section_text_map(bulletin, section_names)
    if not section_texts:
        return EvalScore(name=evaluator_name, score=1.0, evaluated=True)

    issues: list[Issue] = []
    section_fragments: list[tuple[str, str]] = []
    failed_sections: list[str] = []

    for section_name, section_text in section_texts.items():
        try:
            parsed = _call_json_llm(
                llm,
                system=system_prompt,
                payload=f"=== РАЗДЕЛ: {section_name} ===\n{section_text}",
                schema=schema,
                config=config,
                max_tokens=max_tokens,
                evaluator_name=evaluator_name,
            )
        except Exception as exc:
            failed_sections.append(section_name)
            issues.append(
                Issue(
                    evaluator=evaluator_name,
                    severity="error",
                    section=section_name,
                    message="Ошибка парсинга ответа LLM",
                    detail=str(exc),
                )
            )
            continue

        raw_items = parsed.get(response_array_key, []) or []
        if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes, bytearray)):
            failed_sections.append(section_name)
            issues.append(
                Issue(
                    evaluator=evaluator_name,
                    severity="error",
                    section=section_name,
                    message="Ответ LLM нарушает top-level contract",
                    detail=f"Ключ {response_array_key!r} должен содержать массив.",
                )
            )
            continue

        for item in raw_items:
            if not isinstance(item, Mapping):
                continue
            fragment = str(item.get(fragment_key) or "").strip()
            section = section_name
            issue_type = str(item.get(issue_type_key) or default_issue_type).strip() or default_issue_type
            suggestion = str(item.get("suggestion") or "").strip()
            severity = str(item.get("severity") or "warn").strip().lower()
            if severity not in {"warn", "error", "info"}:
                severity = "warn"
            if not fragment:
                continue
            section_fragments.append((section, fragment))
            if message_prefix:
                message = f"{message_prefix}: «{fragment[:120]}» ({issue_type})"
            else:
                message = f"[{issue_type}] «{fragment[:120]}»"
            if suggestion:
                message += f" → {suggestion}"
            issues.append(
                Issue(
                    evaluator=evaluator_name,
                    severity=severity,  # type: ignore[arg-type]
                    section=section,
                    message=message,
                )
            )

    score = _score_from_section_fragment_coverage(section_texts, section_fragments)
    evaluated = not failed_sections
    return EvalScore(name=evaluator_name, score=round(score, 4), issues=issues, evaluated=evaluated)


def _schema_top_level_contract(schema: Mapping[str, Any]) -> dict[str, str]:
    """Извлечь обязательные top-level ключи и их ожидаемые типы из schema."""

    if not isinstance(schema, Mapping):
        raise EvaluationError("Evaluation schema must be a mapping.")
    properties = schema.get("properties", {})
    required = schema.get("required", [])
    if not isinstance(properties, Mapping):
        raise EvaluationError("Evaluation schema.properties must be a mapping.")
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes, bytearray)):
        raise EvaluationError("Evaluation schema.required must be a sequence of keys.")

    contract: dict[str, str] = {}
    for key in required:
        if not isinstance(key, str):
            raise EvaluationError("Evaluation schema.required must contain only string keys.")
        prop = properties.get(key)
        if not isinstance(prop, Mapping):
            raise EvaluationError(f"Evaluation schema has no property for required key: {key!r}.")
        expected_type = prop.get("type")
        if isinstance(expected_type, str):
            contract[key] = expected_type
        else:
            contract[key] = "any"
    return contract


def _strip_llm_wrappers(text: str) -> str:
    source = (text or "").strip()
    source = re.sub(r"<think>[\s\S]*?</think>", "", source, flags=re.IGNORECASE).strip()
    fence_match = re.fullmatch(r"```(?:json)?\s*([\s\S]*?)\s*```", source, flags=re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip()
    return source


def _extract_top_level_eval_json(
    raw_content: str,
    *,
    contract: Mapping[str, str],
    evaluator_name: str,
    raw_preview_chars: int,
) -> JsonObject:
    """Извлечь внешний JSON-объект LLM-ответа и проверить top-level contract.

    Функция намеренно не сканирует все вложенные ``{...}``. Если внешний JSON
    повреждён или усечён, это техническая ошибка evaluator-а, а не пустой список
    замечаний. Такой контракт предотвращает ложные 100% при разборе внутреннего
    объекта массива.
    """

    source = _strip_llm_wrappers(raw_content)
    first_brace = source.find("{")
    if first_brace < 0:
        raise EvaluationError(
            f"{evaluator_name}: LLM не вернула JSON-объект. "
            f"Raw response preview:\n{source[:raw_preview_chars]}"
        )

    candidate = source[first_brace:]
    decoder = json.JSONDecoder()
    try:
        obj, end = decoder.raw_decode(candidate)
    except json.JSONDecodeError as exc:
        raise EvaluationError(
            f"{evaluator_name}: внешний JSON-объект не разобран: {exc}. "
            f"Raw response preview:\n{source[:raw_preview_chars]}"
        ) from exc

    trailing = candidate[end:].strip()
    if trailing and not trailing.startswith("```"):
        # Допускаем служебный текст после валидного JSON, но не используем его.
        pass

    if not isinstance(obj, Mapping):
        raise EvaluationError(
            f"{evaluator_name}: LLM вернула JSON типа {type(obj).__name__}, ожидался object."
        )

    result = dict(obj)
    for key, expected_type in contract.items():
        if key not in result:
            raise EvaluationError(
                f"{evaluator_name}: во внешнем JSON отсутствует обязательный ключ {key!r}. "
                f"Доступные ключи: {list(result.keys())!r}. "
                f"Raw response preview:\n{source[:raw_preview_chars]}"
            )
        value = result[key]
        if expected_type == "array" and not isinstance(value, list):
            raise EvaluationError(
                f"{evaluator_name}: ключ {key!r} должен содержать array, получено {type(value).__name__}."
            )
        if expected_type == "object" and not isinstance(value, Mapping):
            raise EvaluationError(
                f"{evaluator_name}: ключ {key!r} должен содержать object, получено {type(value).__name__}."
            )
    return result


def _call_json_llm(
    llm: ChatBackend,
    *,
    system: str,
    payload: str | Mapping[str, Any],
    schema: Mapping[str, Any],
    config: EvalConfig,
    max_tokens: int,
    evaluator_name: str,
) -> JsonObject:
    user_content = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False, indent=2)
    response = llm.chat(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_content},
        ],
        temperature=config.judge_temperature,
        max_tokens=max_tokens,
        format=dict(schema),
        think=False,
        timeout=config.request_timeout_sec,
    )
    raw_content = _response_content(response)
    contract = _schema_top_level_contract(schema)
    return _extract_top_level_eval_json(
        raw_content,
        contract=contract,
        evaluator_name=evaluator_name,
        raw_preview_chars=config.raw_preview_chars,
    )

def _response_content(response: Mapping[str, Any] | str) -> str:
    if isinstance(response, str):
        return response
    if "content" in response:
        return str(response.get("content") or "")
    message = response.get("message")
    if isinstance(message, Mapping):
        return str(message.get("content") or "")
    return str(response)


def _findings_for_criteria(
    all_issues: Sequence[Mapping[str, Any]],
    criteria: frozenset[str],
) -> list[JsonObject]:
    findings: list[JsonObject] = []
    for item in all_issues:
        criterion = str(item.get("criterion") or item.get("evaluator") or "").strip()
        if criterion in criteria:
            findings.append(dict(item))
    return findings


def _diagnostic_level(
    scores: Mapping[str, EvalScore],
    findings: Sequence[Mapping[str, Any]],
    criteria: frozenset[str],
) -> Literal["ok", "warnings", "issues", "not_evaluated"]:
    has_evaluated = any(score.evaluated for name, score in scores.items() if name in criteria)
    if not has_evaluated:
        return "not_evaluated"
    if any(item.get("severity") == "error" for item in findings):
        return "issues"
    if any(item.get("severity") == "warn" for item in findings):
        return "warnings"
    return "ok"


def _build_summary(
    scores: Mapping[str, EvalScore],
    all_issues: Sequence[Mapping[str, Any]],
    config: EvalConfig,
    aggregate: float,
    *,
    semantic_integrity: str = "not_evaluated",
    editorial_quality: str = "not_evaluated",
) -> str:
    disabled = [name for name in ALL_EVALUATOR_FACTORIES if name not in config.enabled_evaluators]
    error_count = sum(1 for item in all_issues if item.get("severity") == "error")
    warn_count = sum(1 for item in all_issues if item.get("severity") == "warn")
    lines = [
        f"{'═' * 50}",
        f"  ИТОГОВАЯ ОЦЕНКА: {aggregate:.1%}",
        f"{'═' * 50}",
        f"  Используемые evaluators: {', '.join(config.enabled_evaluators)}",
        f"  Judge temperature: {config.judge_temperature}",
        "  Semantic gate: отключён; findings являются report-only",
        f"  Semantic integrity: {semantic_integrity}",
        f"  Editorial quality: {editorial_quality}",
    ]
    if disabled:
        lines.append(f"  Отключены evaluators: {', '.join(disabled)}")
    for name, score in scores.items():
        status = "✓" if score.evaluated else "—"
        filled = int(score.score * 20)
        bar = "█" * filled + "░" * (20 - filled)
        lines.append(f"  {status} {name:.<25s} {bar} {score.score:.1%}")
    lines.append(f"\n  Ошибок: {error_count}, предупреждений: {warn_count}")
    errors = [item for item in all_issues if item.get("severity") == "error"]
    if errors:
        lines.append("\n  Критические проблемы:")
        for item in errors[:10]:
            lines.append(f"    [{item.get('criterion')}] {item.get('section')}: {item.get('message')}")
    not_evaluated = [name for name, score in scores.items() if not score.evaluated]
    if not_evaluated:
        lines.append(f"\n  Не оценены: {', '.join(not_evaluated)}")
    return "\n".join(lines)


def _context_dict(ctx: GlobalContext | Mapping[str, Any]) -> JsonObject:
    if isinstance(ctx, GlobalContext):
        return ctx.to_public_dict()
    return dict(ctx)


def _bulletin_dict(bulletin: Bulletin | Mapping[str, Any]) -> JsonObject:
    if isinstance(bulletin, Bulletin):
        return bulletin.to_public_dict()
    return dict(bulletin)


def _sections_dict(bulletin: Mapping[str, Any]) -> Mapping[str, Any]:
    sections = bulletin.get("sections", {})
    return sections if isinstance(sections, Mapping) else {}


def _section_text(bulletin: Mapping[str, Any], section_name: str) -> str:
    section = _sections_dict(bulletin).get(section_name)
    if section is None:
        return ""
    if section_name == "age_group_season_table" and isinstance(section, Sequence) and not isinstance(section, (str, bytes, bytearray)):
        lines: list[str] = []
        for row in section:
            if not isinstance(row, Mapping):
                continue
            lines.append(
                f"{row.get('age_group_label', '')}: случаи за сезон {row.get('season_cases', '')}; "
                f"накопленная сезонная заболеваемость {row.get('cumulative_incidence_pct', '')}%; "
                f"неделя пика {row.get('peak_week', '')}; "
                f"пик на 10 тыс. {row.get('peak_inc_per_10k', '')}; "
                f"средняя недельная заболеваемость на 10 тыс. {row.get('mean_weekly_inc_per_10k', '')}; "
                f"ширина главного пика {row.get('peak_width_weeks', '')}; "
                f"доля сезонных случаев {row.get('share_of_total_cases_pct', '')}%"
            )
        return "\n".join(lines)
    return flatten_text(section)


def _narrative_section_names(bulletin: Mapping[str, Any]) -> list[str]:
    defaults = [
        "current_situation",
        "epidemic_wave_comparison",
        "age_group_season_overview",
        "forecast_risks",
        "shap_interpretation",
        "model_quality",
        "model_description",
    ]
    existing = _sections_dict(bulletin)
    return [name for name in defaults if name in existing] or defaults


def _prose_section_names(bulletin: Mapping[str, Any]) -> list[str]:
    extras = ["figure_caption", "forecast_figure_caption", "wave_figure_caption"]
    names = [*_narrative_section_names(bulletin), *extras]
    existing = _sections_dict(bulletin)
    return [name for name in names if name in existing] or names


def _section_text_map(bulletin: Mapping[str, Any], section_names: Sequence[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in section_names:
        text = _section_text(bulletin, name)
        if text.strip():
            result[name] = text
    return result


def _labeled_sections_text_from_map(section_texts: Mapping[str, str]) -> str:
    return "\n\n".join(f"=== РАЗДЕЛ: {name} ===\n{text}" for name, text in section_texts.items())


def _score_from_section_fragment_coverage(
    section_texts: Mapping[str, str],
    section_fragments: Sequence[tuple[str, str]],
) -> float:
    total = 0
    bad = 0
    grouped: dict[str, list[str]] = {}
    for section, fragment in section_fragments:
        section = (section or "").strip()
        fragment = (fragment or "").strip()
        if section and fragment:
            grouped.setdefault(section, []).append(fragment)
    for section, text in section_texts.items():
        clean_text = (text or "").strip()
        if not clean_text:
            continue
        total += len(clean_text)
        bad += _merged_coverage_length(_find_span_ranges(clean_text, grouped.get(section, [])))
    return 1.0 if total <= 0 else max(0.0, min(1.0, 1.0 - bad / total))


def _find_span_ranges(text: str, spans: Sequence[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for span in spans:
        needle = (span or "").strip()
        if not needle:
            continue
        start = text.find(needle)
        if start != -1:
            ranges.append((start, start + len(needle)))
    return ranges


def _merged_coverage_length(ranges: Sequence[tuple[int, int]]) -> int:
    if not ranges:
        return 0
    sorted_ranges = sorted(ranges)
    merged: list[tuple[int, int]] = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)


def _extract_numbers(text: str) -> list[float]:
    results: list[float] = []
    for match in re.finditer(r"(?<![а-яА-ЯёЁa-zA-Z])(\d+[.,]\d+|\d+)(?![а-яА-ЯёЁa-zA-Z])", text):
        try:
            results.append(float(match.group(1).replace(",", ".")))
        except ValueError:
            continue
    return results


def _number_present(expected: float, numbers: Sequence[float], tolerance_pct: float) -> bool:
    tolerance = max(0.005, abs(expected) * tolerance_pct / 100)
    return any(abs(number - expected) <= tolerance for number in numbers)


def _append_number(
    expectations: list[ExpectedNumber],
    label: str,
    value: Any,
    section: str,
    *,
    required: bool = True,
) -> None:
    number = _to_float_or_none(value)
    if number is None:
        return
    expectations.append(ExpectedNumber(label=label, value=number, section=section, required=required))


def _to_float_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result != result:
        return None
    return result


def _semantic(ctx: Mapping[str, Any], block_name: str) -> JsonObject:
    block = _as_mapping(ctx.get(block_name))
    semantic = block.get("semantic")
    return dict(semantic) if isinstance(semantic, Mapping) else {}


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_sequence(value: Any) -> Sequence[Any]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return []


def _normalize_claim_value(value: Any) -> Any:
    if isinstance(value, str):
        return value.strip().lower().replace("ё", "е")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        normalized = [_normalize_claim_value(item) for item in value]
        return tuple(sorted(str(item) for item in normalized))
    return value


def _compare_claim_values(expected: Any, actual: Any) -> bool:
    return _normalize_claim_value(expected) == _normalize_claim_value(actual)


def _normalize_text(text: str) -> str:
    return (text or "").lower().replace("ё", "е")


def _claim_extraction_schema(predicates: Sequence[str]) -> JsonObject:
    return {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "predicate": {"type": "string", "enum": list(predicates)},
                        "value": {
                            "anyOf": [
                                {"type": "string"},
                                {"type": "integer"},
                                {"type": "number"},
                                {"type": "boolean"},
                                {"type": "array", "items": {"type": "string"}},
                                {"type": "null"},
                            ]
                        },
                        "horizon_weeks": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                        "modality": {"type": "string", "enum": ["asserted", "hedged", "unclear"]},
                        "evidence": {"type": "string"},
                    },
                    "required": ["predicate", "value", "horizon_weeks", "modality", "evidence"],
                },
            }
        },
        "required": ["claims"],
    }


def _language_issue_schema(array_key: str, issue_type_key: str) -> JsonObject:
    return {
        "type": "object",
        "properties": {
            array_key: {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "fragment": {"type": "string"},
                        "section": {"type": "string"},
                        issue_type_key: {"type": "string"},
                        "suggestion": {"type": "string"},
                        "severity": {"type": "string"},
                    },
                    "required": ["fragment", "section", issue_type_key, "suggestion", "severity"],
                },
            }
        },
        "required": [array_key],
    }


def _shap_inventory(ctx: Mapping[str, Any]) -> list[JsonObject]:
    shap = _as_mapping(ctx.get("shap_summary"))
    by_horizon = _as_mapping(shap.get("by_horizon"))
    inventory: list[JsonObject] = []
    for horizon, factors in by_horizon.items():
        for factor in _as_sequence(factors):
            factor_map = _as_mapping(factor)
            name = factor_map.get("название") or factor_map.get("name")
            if not name:
                continue
            reliable = factor_map.get("направление_надёжное", factor_map.get("direction_reliable", True))
            inventory.append(
                {
                    "horizon": str(horizon),
                    "factor": str(name),
                    "direction_reliable": bool(reliable),
                    "direction": factor_map.get("направление") or factor_map.get("direction"),
                }
            )
    return inventory


__all__ = [
    "ALL_EVALUATOR_FACTORIES",
    "BaseEvaluator",
    "CanonicalClaim",
    "ClaimVerdict",
    "DEFAULT_ENABLED_EVALUATORS",
    "DEFAULT_EVAL_WEIGHTS",
    "DEFAULT_JUDGE_TEMPERATURE",
    "EvalConfig",
    "EvalReport",
    "EvalScore",
    "EvaluationError",
    "ExpectedNumber",
    "ExtractedClaim",
    "FactualEvaluator",
    "GrammarEvaluator",
    "Issue",
    "LogicEvaluator",
    "NumericEvaluator",
    "OrthotypographyEvaluator",
    "StyleEvaluator",
    "TautologyEvaluator",
    "WaterEvaluator",
    "evaluate_bulletin",
    "evaluate_bulletin_from_files",
    "load_eval_report",
    "print_eval_report",
    "save_eval_report",
]
