"""
Детерминированные runtime-проверки для ai4epi.

Модуль проверяет уже сгенерированные секционные тексты без обращения к LLM:
числовые якоря, техническое качество русского текста, терминологию прогнозных
интервалов и базовые логические противоречия, которые выводятся напрямую из
GlobalContext. Runtime-checks не редактируют текст и не добавляют факты; они
возвращают структурированный отчёт, пригодный для repair-loop, evaluator-слоя
или CI-проверок репозитория.
"""

from __future__ import annotations

import math
import re
from enum import Enum
from typing import Any, Iterable, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator

from ai4epi.core.context import GlobalContext
from ai4epi.core.sections import SectionRegistry


JsonObject = dict[str, Any]


class RuntimeCheckError(ValueError):
    """Ошибка входных данных runtime-check слоя."""


class IssueSeverity(str, Enum):
    """Степень серьёзности найденного нарушения."""

    ERROR = "error"
    WARN = "warn"
    INFO = "info"


class IssueCategory(str, Enum):
    """Класс runtime-проверки."""

    STRUCTURE = "structure"
    NUMERIC = "numeric"
    TEXT = "text"
    TERMINOLOGY = "terminology"
    LOGIC = "logic"


class StrictModel(BaseModel):
    """Базовая pydantic-модель с закрытым контрактом."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        arbitrary_types_allowed=False,
    )


class RuntimeIssue(StrictModel):
    """Одно нарушение, найденное runtime-check слоем."""

    category: IssueCategory
    severity: IssueSeverity
    section_id: str
    check_name: str
    message: str
    detail: JsonObject = Field(default_factory=dict)


class ExpectedNumber(StrictModel):
    """Один числовой якорь, который должен присутствовать в тексте секции."""

    label: str = Field(min_length=1)
    value: float
    section_id: str = Field(min_length=1)
    required: bool = True
    tolerance_pct: float | None = Field(default=None, ge=0.0)

    @field_validator("value")
    @classmethod
    def validate_finite_value(cls, value: float) -> float:
        if not math.isfinite(float(value)):
            raise ValueError("Expected number must be finite.")
        return float(value)


class RuntimeCheckConfig(StrictModel):
    """Настройки детерминированного runtime-check слоя."""

    numeric_tolerance_pct: float = Field(default=1.5, ge=0.0)
    min_section_words: int = Field(default=15, ge=0)
    high_uncertainty_pct_threshold: float = Field(default=80.0, ge=0.0)
    check_markdown_artifacts: bool = True
    check_text_quality: bool = True
    check_numeric_anchors: bool = True
    check_logic_rules: bool = True
    allowed_latin_terms: tuple[str, ...] = (
        "AIC",
        "API",
        "CI",
        "CSV",
        "FWHM",
        "GBDT",
        "ISO",
        "JSON",
        "LightGBM",
        "MAE",
        "MAPE",
        "PDF",
        "Pydantic",
        "RMSE",
        "R2",
        "R²",
        "SHAP",
        "XGBoost",
        "h",
        "q",
    )
    absolute_certainty_phrases: tuple[str, ...] = (
        "гарантированно",
        "несомненно",
        "безусловно",
        "однозначно",
        "точно произойдет",
        "точно произойдёт",
        "обязательно произойдет",
        "обязательно произойдёт",
        "определенно приведет",
        "определённо приведёт",
    )
    adverse_scenario_markers: tuple[str, ...] = (
        "неблагоприятн",
        "негативн",
        "пессимистичн",
        "худш",
    )
    low_interval_markers: tuple[str, ...] = (
        "нулев",
        "нижн",
        "минимальн",
    )
    high_interval_markers: tuple[str, ...] = (
        "верхн",
        "высок",
        "максимальн",
    )
    decline_markers: tuple[str, ...] = (
        "снижени",
        "уменьшени",
        "падени",
        "сокращени",
    )
    growth_markers: tuple[str, ...] = (
        "рост",
        "повышени",
        "увеличени",
        "возрастани",
    )
    trend_qualifier_markers: tuple[str, ...] = (
        "затем",
        "однако",
        "после",
        "сменил",
        "сменяет",
        "колебан",
        "временн",
        "впоследств",
    )
    prose_section_ids: tuple[str, ...] = (
        "current_situation",
        "epidemic_wave_comparison",
        "age_group_season_overview",
        "forecast_risks",
        "shap_interpretation",
        "model_quality",
        "model_description",
        "figure_caption",
        "forecast_figure_caption",
        "wave_figure_caption",
    )

    @property
    def allowed_latin_terms_normalized(self) -> frozenset[str]:
        """Вернуть whitelist латинских токенов в нормализованном виде."""

        return frozenset(term.casefold() for term in self.allowed_latin_terms)


class RuntimeCheckReport(StrictModel):
    """Итог runtime-проверки одного бюллетеня."""

    ok: bool
    issues: list[RuntimeIssue] = Field(default_factory=list)
    checks_total: int = 0
    checks_failed: int = 0

    @property
    def errors(self) -> list[RuntimeIssue]:
        """Вернуть только критические нарушения."""

        return [issue for issue in self.issues if issue.severity == IssueSeverity.ERROR]

    @property
    def warnings(self) -> list[RuntimeIssue]:
        """Вернуть предупреждения."""

        return [issue for issue in self.issues if issue.severity == IssueSeverity.WARN]

    def raise_for_errors(self) -> None:
        """Выбросить исключение, если отчёт содержит критические нарушения."""

        if self.errors:
            messages = "; ".join(issue.message for issue in self.errors[:5])
            raise RuntimeCheckError(f"Runtime checks failed: {messages}")

    def to_feedback_text(self, *, include_warnings: bool = True) -> str:
        """
        Сформировать компактный feedback для repair-loop narrator/editor.

        Текст содержит только найденные нарушения и не добавляет новых фактов.
        """

        selected = [
            issue
            for issue in self.issues
            if issue.severity == IssueSeverity.ERROR or include_warnings
        ]
        if not selected:
            return "Нарушений runtime-checks не найдено."

        lines = []
        for issue in selected:
            prefix = "ОШИБКА" if issue.severity == IssueSeverity.ERROR else "ПРЕДУПРЕЖДЕНИЕ"
            lines.append(f"[{prefix}] {issue.section_id}/{issue.check_name}: {issue.message}")
        return "\n".join(lines)


class RuntimeChecker:
    """Исполнитель детерминированных runtime-проверок."""

    def __init__(self, config: RuntimeCheckConfig | None = None) -> None:
        self.config = config or RuntimeCheckConfig()

    def check_bulletin(
        self,
        *,
        context: GlobalContext | Mapping[str, Any],
        bulletin: Mapping[str, Any],
        registry: SectionRegistry | None = None,
        section_ids: Sequence[str] | None = None,
        extra_expected_numbers: Sequence[ExpectedNumber | Mapping[str, Any]] | None = None,
    ) -> RuntimeCheckReport:
        """
        Проверить сгенерированный бюллетень или словарь секционных payload.

        Parameters
        ----------
        context:
            Валидированный GlobalContext или совместимый словарь.
        bulletin:
            Либо полный объект вида ``{"sections": {...}}``, либо сам словарь
            ``{section_id: section_payload}``.
        registry:
            Необязательный реестр секций. Если задан, используется для проверки
            обязательных секций и для извлечения дополнительных числовых якорей
            из ``SectionConfig.metadata["expected_numbers"]``.
        section_ids:
            Необязательный явный список секций для текстовых проверок.
        extra_expected_numbers:
            Дополнительные числовые якоря для пользовательских секций.
        """

        ctx = _context_to_dict(context)
        sections = _coerce_sections_mapping(bulletin)
        effective_section_ids = self._resolve_section_ids(sections, registry, section_ids)

        issues: list[RuntimeIssue] = []
        checks_total = 0

        structure_issues, structure_checks = self._check_required_sections(sections, registry)
        issues.extend(structure_issues)
        checks_total += structure_checks

        if self.config.check_text_quality:
            text_issues, text_checks = self._check_text_quality(sections, effective_section_ids)
            issues.extend(text_issues)
            checks_total += text_checks

        if self.config.check_numeric_anchors:
            expected_numbers = self._build_expected_numbers(ctx, registry, extra_expected_numbers)
            numeric_issues, numeric_checks = self._check_expected_numbers(sections, expected_numbers)
            issues.extend(numeric_issues)
            checks_total += numeric_checks

        if self.config.check_logic_rules:
            logic_issues, logic_checks = self._check_logic_rules(ctx, sections)
            issues.extend(logic_issues)
            checks_total += logic_checks

        checks_failed = sum(1 for issue in issues if issue.severity == IssueSeverity.ERROR)
        return RuntimeCheckReport(
            ok=checks_failed == 0,
            issues=issues,
            checks_total=checks_total,
            checks_failed=checks_failed,
        )

    def _resolve_section_ids(
        self,
        sections: Mapping[str, Any],
        registry: SectionRegistry | None,
        section_ids: Sequence[str] | None,
    ) -> list[str]:
        if section_ids is not None:
            return list(dict.fromkeys(str(item) for item in section_ids))
        if registry is not None:
            return registry.section_ids()
        default_ids = [section_id for section_id in self.config.prose_section_ids if section_id in sections]
        extra_ids = [section_id for section_id in sections if section_id not in default_ids]
        return default_ids + extra_ids

    def _check_required_sections(
        self,
        sections: Mapping[str, Any],
        registry: SectionRegistry | None,
    ) -> tuple[list[RuntimeIssue], int]:
        if registry is None:
            return [], 0

        issues: list[RuntimeIssue] = []
        checks = 0
        for section in registry.ordered_sections():
            if not section.required:
                continue
            checks += 1
            if section.section_id not in sections:
                issues.append(RuntimeIssue(
                    category=IssueCategory.STRUCTURE,
                    severity=IssueSeverity.ERROR,
                    section_id=section.section_id,
                    check_name="required_section_present",
                    message=f"Обязательная секция {section.section_id!r} отсутствует в бюллетене.",
                ))
        return issues, checks

    def _check_text_quality(
        self,
        sections: Mapping[str, Any],
        section_ids: Sequence[str],
    ) -> tuple[list[RuntimeIssue], int]:
        issues: list[RuntimeIssue] = []
        checks = 0
        for section_id in section_ids:
            text = section_text(sections, section_id)
            if not text.strip():
                checks += 1
                issues.append(RuntimeIssue(
                    category=IssueCategory.TEXT,
                    severity=IssueSeverity.ERROR,
                    section_id=section_id,
                    check_name="non_empty_text",
                    message="Текст секции пустой.",
                ))
                continue

            section_issues = validate_text_quality(
                text,
                section_id=section_id,
                config=self.config,
            )
            checks += max(1, len(section_issues))
            issues.extend(section_issues)
        return issues, checks

    def _build_expected_numbers(
        self,
        ctx: Mapping[str, Any],
        registry: SectionRegistry | None,
        extra_expected_numbers: Sequence[ExpectedNumber | Mapping[str, Any]] | None,
    ) -> list[ExpectedNumber]:
        expected = build_default_expected_numbers(ctx)
        if registry is not None:
            expected.extend(build_registry_expected_numbers(ctx, registry))
        if extra_expected_numbers:
            expected.extend(
                item if isinstance(item, ExpectedNumber) else ExpectedNumber.model_validate(item)
                for item in extra_expected_numbers
            )
        return expected

    def _check_expected_numbers(
        self,
        sections: Mapping[str, Any],
        expected_numbers: Sequence[ExpectedNumber],
    ) -> tuple[list[RuntimeIssue], int]:
        issues: list[RuntimeIssue] = []
        checks = 0
        for expected in expected_numbers:
            checks += 1
            text = section_text(sections, expected.section_id)
            numbers = extract_numbers(text)
            tolerance_pct = expected.tolerance_pct
            if tolerance_pct is None:
                tolerance_pct = self.config.numeric_tolerance_pct

            if number_present(expected.value, numbers, tolerance_pct=tolerance_pct):
                continue

            issues.append(RuntimeIssue(
                category=IssueCategory.NUMERIC,
                severity=IssueSeverity.ERROR if expected.required else IssueSeverity.WARN,
                section_id=expected.section_id,
                check_name="expected_number_present",
                message=f"Число '{expected.label}' = {expected.value:g} не найдено в тексте или искажено.",
                detail={
                    "expected_value": expected.value,
                    "tolerance_pct": tolerance_pct,
                    "found_numbers_preview": sorted(set(numbers))[:20],
                },
            ))
        return issues, checks

    def _check_logic_rules(
        self,
        ctx: Mapping[str, Any],
        sections: Mapping[str, Any],
    ) -> tuple[list[RuntimeIssue], int]:
        rules = (
            self._rule_no_overconfidence_under_uncertainty,
            self._rule_correct_interval_terminology,
            self._rule_adverse_scenario_is_high_incidence,
            self._rule_forecast_trend_matches_numbers,
        )
        issues: list[RuntimeIssue] = []
        checks = 0
        for rule in rules:
            checks += 1
            issue = rule(ctx, sections)
            if issue is not None:
                issues.append(issue)
        return issues, checks

    def _rule_no_overconfidence_under_uncertainty(
        self,
        ctx: Mapping[str, Any],
        sections: Mapping[str, Any],
    ) -> RuntimeIssue | None:
        uncertainty_pct = _get_path(ctx, "forecast.relative_uncertainty_pct", default=0.0)
        try:
            uncertainty_pct = float(uncertainty_pct or 0.0)
        except (TypeError, ValueError):
            uncertainty_pct = 0.0

        if uncertainty_pct <= self.config.high_uncertainty_pct_threshold:
            return None

        text = normalize_ru(section_text(sections, "forecast_risks"))
        found = [phrase for phrase in self.config.absolute_certainty_phrases if normalize_ru(phrase) in text]
        if not found:
            return None

        return RuntimeIssue(
            category=IssueCategory.LOGIC,
            severity=IssueSeverity.ERROR,
            section_id="forecast_risks",
            check_name="no_overconfidence_under_uncertainty",
            message="При высокой неопределённости прогноз описан словами абсолютной уверенности.",
            detail={
                "relative_uncertainty_pct": uncertainty_pct,
                "threshold_pct": self.config.high_uncertainty_pct_threshold,
                "found_phrases": found,
            },
        )

    def _rule_correct_interval_terminology(
        self,
        ctx: Mapping[str, Any],
        sections: Mapping[str, Any],
    ) -> RuntimeIssue | None:
        del ctx
        text = normalize_ru(section_text(sections, "forecast_risks"))
        has_confidence_interval = "доверительн" in text and "интервал" in text
        has_prediction_interval = "прогнозн" in text and "интервал" in text
        if has_confidence_interval and not has_prediction_interval:
            return RuntimeIssue(
                category=IssueCategory.TERMINOLOGY,
                severity=IssueSeverity.ERROR,
                section_id="forecast_risks",
                check_name="correct_interval_terminology",
                message="Использован термин 'доверительный интервал'; для прогноза нужен термин 'прогнозный интервал'.",
            )
        return None

    def _rule_adverse_scenario_is_high_incidence(
        self,
        ctx: Mapping[str, Any],
        sections: Mapping[str, Any],
    ) -> RuntimeIssue | None:
        del ctx
        text = normalize_ru(section_text(sections, "forecast_risks"))
        has_adverse = any(marker in text for marker in self.config.adverse_scenario_markers)
        if not has_adverse:
            return None

        near_low = any(marker in text for marker in self.config.low_interval_markers)
        near_high = any(marker in text for marker in self.config.high_interval_markers)
        if near_low and not near_high:
            return RuntimeIssue(
                category=IssueCategory.LOGIC,
                severity=IssueSeverity.ERROR,
                section_id="forecast_risks",
                check_name="adverse_scenario_is_high_incidence",
                message="Неблагоприятный сценарий связан с нижней границей интервала; в эпидемиологии неблагоприятный сценарий соответствует высокой заболеваемости.",
            )
        return None

    def _rule_forecast_trend_matches_numbers(
        self,
        ctx: Mapping[str, Any],
        sections: Mapping[str, Any],
    ) -> RuntimeIssue | None:
        horizons = _get_path(ctx, "forecast.horizons", default=[])
        if not isinstance(horizons, Sequence) or isinstance(horizons, (str, bytes, bytearray)):
            return None

        values = []
        for item in horizons:
            if isinstance(item, Mapping):
                value = item.get("point_forecast", item.get("y_pred"))
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    continue
        if len(values) < 2:
            return None

        diffs = [values[i + 1] - values[i] for i in range(len(values) - 1)]
        all_increasing = all(diff > 0 for diff in diffs)
        all_decreasing = all(diff < 0 for diff in diffs)
        if not all_increasing and not all_decreasing:
            return None

        text = normalize_ru(section_text(sections, "forecast_risks"))
        has_qualifier = any(marker in text for marker in self.config.trend_qualifier_markers)

        if all_increasing:
            says_decline = any(marker in text for marker in self.config.decline_markers)
            if says_decline and not has_qualifier:
                return RuntimeIssue(
                    category=IssueCategory.LOGIC,
                    severity=IssueSeverity.ERROR,
                    section_id="forecast_risks",
                    check_name="forecast_trend_matches_numbers",
                    message="Все точечные прогнозы монотонно растут, но текст утверждает снижение без оговорки.",
                    detail={"point_forecasts": values},
                )

        if all_decreasing:
            says_growth = any(marker in text for marker in self.config.growth_markers)
            if says_growth and not has_qualifier:
                return RuntimeIssue(
                    category=IssueCategory.LOGIC,
                    severity=IssueSeverity.ERROR,
                    section_id="forecast_risks",
                    check_name="forecast_trend_matches_numbers",
                    message="Все точечные прогнозы монотонно снижаются, но текст утверждает рост без оговорки.",
                    detail={"point_forecasts": values},
                )

        return None


def section_text(bulletin_or_sections: Mapping[str, Any], section_id: str) -> str:
    """
    Извлечь текст секции из полного бюллетеня или словаря секций.

    Если payload секции является словарём, строковые leaf-поля объединяются в
    порядке обхода. Это соответствует текущим JSON-ответам narrator-агентов.
    """

    sections = _coerce_sections_mapping(bulletin_or_sections)
    payload = sections.get(section_id)
    if payload is None:
        return ""
    return _flatten_text(payload).strip()


def validate_text_quality(
    text: str,
    *,
    section_id: str,
    config: RuntimeCheckConfig | None = None,
) -> list[RuntimeIssue]:
    """Проверить техническое качество текста одной секции."""

    cfg = config or RuntimeCheckConfig()
    issues: list[RuntimeIssue] = []

    if not text or not text.strip():
        return [RuntimeIssue(
            category=IssueCategory.TEXT,
            severity=IssueSeverity.ERROR,
            section_id=section_id,
            check_name="non_empty_text",
            message="Текст секции пустой.",
        )]

    latin_words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]{1,}\b", text)
    bad_latin = sorted({
        word
        for word in latin_words
        if word.casefold() not in cfg.allowed_latin_terms_normalized
    })
    if bad_latin:
        issues.append(RuntimeIssue(
            category=IssueCategory.TEXT,
            severity=IssueSeverity.WARN,
            section_id=section_id,
            check_name="latin_tokens_whitelist",
            message="В русском тексте найдены латинские или кодовые токены вне whitelist.",
            detail={"tokens": bad_latin},
        ))

    bad_chars = find_unsupported_pdf_characters(text)
    if bad_chars:
        issues.append(RuntimeIssue(
            category=IssueCategory.TEXT,
            severity=IssueSeverity.ERROR,
            section_id=section_id,
            check_name="pdf_supported_characters",
            message="Найдены символы, потенциально несовместимые с PDF-рендерингом.",
            detail={"characters": [f"U+{ord(char):04X}" for char in bad_chars]},
        ))

    punctuation_issues = []
    if re.search(r"[.,]{2,}", text):
        punctuation_issues.append("двойная пунктуация")
    if re.search(r"\s{3,}", text):
        punctuation_issues.append("три и более пробела подряд")
    if re.search(r"[.!?],", text):
        punctuation_issues.append("точка, вопросительный или восклицательный знак перед запятой")
    if punctuation_issues:
        issues.append(RuntimeIssue(
            category=IssueCategory.TEXT,
            severity=IssueSeverity.WARN,
            section_id=section_id,
            check_name="punctuation_artifacts",
            message="Найдены пунктуационные артефакты.",
            detail={"artifacts": punctuation_issues},
        ))

    if cfg.check_markdown_artifacts:
        markdown_issues = find_markdown_artifacts(text)
        if markdown_issues:
            issues.append(RuntimeIssue(
                category=IssueCategory.TEXT,
                severity=IssueSeverity.WARN,
                section_id=section_id,
                check_name="markdown_artifacts",
                message="В прозе найдены markdown-артефакты.",
                detail={"artifacts": markdown_issues},
            ))

    word_count = len(re.findall(r"\b[а-яА-ЯёЁa-zA-Z0-9]+\b", text))
    if word_count < cfg.min_section_words:
        issues.append(RuntimeIssue(
            category=IssueCategory.TEXT,
            severity=IssueSeverity.WARN,
            section_id=section_id,
            check_name="minimum_word_count",
            message=f"Текст подозрительно короткий: {word_count} слов.",
            detail={"word_count": word_count, "minimum": cfg.min_section_words},
        ))

    return issues


def build_default_expected_numbers(ctx: Mapping[str, Any]) -> list[ExpectedNumber]:
    """
    Построить стандартные числовые якоря из GlobalContext-совместимого словаря.

    Значения не хардкодируются: все якоря извлекаются из контекста текущего
    запуска. Если необязательный блок отсутствует, соответствующие якоря не
    создаются.
    """

    expected: list[ExpectedNumber] = []

    _append_expected_number(
        expected,
        ctx,
        path="origin.iso_week",
        label="номер эпидемиологической недели",
        section_id="current_situation",
        required=True,
    )
    _append_expected_number(
        expected,
        ctx,
        path="current_situation.current_value",
        label="заболеваемость текущей недели",
        section_id="current_situation",
        required=True,
    )
    _append_expected_number(
        expected,
        ctx,
        path="current_situation.previous_value",
        label="заболеваемость предыдущей недели",
        section_id="current_situation",
        required=False,
    )

    for horizon in _iter_mappings(_get_path(ctx, "forecast.horizons", default=[])):
        horizon_weeks = horizon.get("horizon_weeks", "?")
        value = horizon.get("point_forecast", horizon.get("y_pred"))
        _append_literal_expected_number(
            expected,
            value=value,
            label=f"точечный прогноз, горизонт {horizon_weeks} нед.",
            section_id="forecast_risks",
            required=True,
        )

    metrics = list(_iter_mappings(_get_path(ctx, "model_quality.metrics", default=[])))
    for metric_index, metric in enumerate(metrics):
        horizon_weeks = metric.get("horizon_weeks", "?")
        for metric_name in ("mae", "rmse", "r2"):
            _append_literal_expected_number(
                expected,
                value=metric.get(metric_name),
                label=f"{metric_name.upper()}, h={horizon_weeks}",
                section_id="model_quality",
                required=(metric_index == 0),
            )

    _append_expected_number(
        expected,
        ctx,
        path="model_quality.error_stats_h1.median_error_h1",
        label="медиана ошибки, h=1",
        section_id="model_quality",
        required=False,
    )
    _append_expected_number(
        expected,
        ctx,
        path="model_quality.error_stats_h1.max_error_h1",
        label="максимальная ошибка, h=1",
        section_id="model_quality",
        required=False,
    )
    _append_expected_number(
        expected,
        ctx,
        path="model_info.calibration_start_year",
        label="год начала калибровки",
        section_id="model_description",
        required=True,
    )

    return expected


def build_registry_expected_numbers(ctx: Mapping[str, Any], registry: SectionRegistry) -> list[ExpectedNumber]:
    """
    Построить числовые якоря из metadata секций.

    Поддерживаемый формат ``SectionConfig.metadata``::

        {
          "expected_numbers": [
            {
              "label": "описание числа",
              "path": "custom_block.metric",
              "section_id": "custom_section",   # необязательно
              "required": true,                  # необязательно
              "tolerance_pct": 1.5               # необязательно
            }
          ]
        }
    """

    expected: list[ExpectedNumber] = []
    for section in registry.ordered_sections(include_disabled=False):
        raw_items = section.metadata.get("expected_numbers", [])
        if not isinstance(raw_items, Sequence) or isinstance(raw_items, (str, bytes, bytearray)):
            continue
        for raw in raw_items:
            if not isinstance(raw, Mapping):
                continue
            path = raw.get("path")
            label = raw.get("label")
            if not isinstance(path, str) or not isinstance(label, str):
                continue
            value = _get_path(ctx, path, default=None)
            _append_literal_expected_number(
                expected,
                value=value,
                label=label,
                section_id=str(raw.get("section_id") or section.section_id),
                required=bool(raw.get("required", True)),
                tolerance_pct=_optional_float(raw.get("tolerance_pct")),
            )
    return expected


def extract_numbers(text: str) -> list[float]:
    """Извлечь числа из текста, не захватывая части слов и кодов."""

    numbers: list[float] = []
    pattern = r"(?<![а-яА-ЯёЁa-zA-Z0-9_])([-+]?\d+[.,]\d+|[-+]?\d+)(?![а-яА-ЯёЁa-zA-Z0-9_])"
    for match in re.finditer(pattern, text or ""):
        raw = match.group(1).replace(",", ".")
        try:
            value = float(raw)
        except ValueError:
            continue
        if math.isfinite(value):
            numbers.append(value)
    return numbers


def number_present(expected: float, numbers: Sequence[float], *, tolerance_pct: float) -> bool:
    """Проверить наличие ожидаемого числа с явным процентным допуском."""

    expected_value = float(expected)
    tolerance = max(0.005, abs(expected_value) * tolerance_pct / 100.0)
    return any(abs(float(number) - expected_value) <= tolerance for number in numbers)


def normalize_ru(text: str) -> str:
    """Нормализовать русский текст для простых детерминированных проверок."""

    return (text or "").casefold().replace("ё", "е")


def find_unsupported_pdf_characters(text: str) -> list[str]:
    """Вернуть уникальные символы вне набора, поддерживаемого текущим PDF-слоем."""

    allowed_pattern = re.compile(
        r"^[\u0020-\u007E\u00A0-\u00FF\u0400-\u04FF\u2010-\u2027\u2030-\u205E\u2070-\u209F\u2116\n\r\t]+$"
    )
    bad = []
    for char in text or "":
        if not allowed_pattern.match(char):
            bad.append(char)
    return sorted(set(bad), key=ord)


def find_markdown_artifacts(text: str) -> list[str]:
    """Найти markdown-артефакты в прозе секции."""

    artifacts = []
    if "```" in text:
        artifacts.append("code fence")
    if re.search(r"(?m)^\s{0,3}#{1,6}\s+", text):
        artifacts.append("markdown heading")
    if re.search(r"(?m)^\s*[-*+]\s+", text):
        artifacts.append("bullet list")
    if re.search(r"(?m)^\s*\d+[.)]\s+", text):
        artifacts.append("numbered list")
    return artifacts


def _context_to_dict(context: GlobalContext | Mapping[str, Any]) -> JsonObject:
    if isinstance(context, GlobalContext):
        return context.to_public_dict()
    if isinstance(context, Mapping):
        return dict(context)
    raise RuntimeCheckError("context must be GlobalContext or Mapping[str, Any].")


def _coerce_sections_mapping(bulletin_or_sections: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(bulletin_or_sections, Mapping):
        raise RuntimeCheckError("bulletin must be a mapping.")
    sections = bulletin_or_sections.get("sections")
    if isinstance(sections, Mapping):
        return sections
    return bulletin_or_sections


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        parts = [_flatten_text(item) for item in value.values()]
        return "\n".join(part for part in parts if part.strip())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts = [_flatten_text(item) for item in value]
        return "\n".join(part for part in parts if part.strip())
    return "" if value is None else str(value)


def _get_path(mapping: Mapping[str, Any], path: str, *, default: Any = None) -> Any:
    current: Any = mapping
    for raw_segment in path.split("."):
        if isinstance(current, Mapping):
            if raw_segment not in current:
                return default
            current = current[raw_segment]
            continue
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            if not raw_segment.isdecimal():
                return default
            index = int(raw_segment)
            if index >= len(current):
                return default
            current = current[index]
            continue
        return default
    return current


def _iter_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            if isinstance(item, Mapping):
                yield item


def _append_expected_number(
    target: list[ExpectedNumber],
    ctx: Mapping[str, Any],
    *,
    path: str,
    label: str,
    section_id: str,
    required: bool,
    tolerance_pct: float | None = None,
) -> None:
    _append_literal_expected_number(
        target,
        value=_get_path(ctx, path, default=None),
        label=label,
        section_id=section_id,
        required=required,
        tolerance_pct=tolerance_pct,
    )


def _append_literal_expected_number(
    target: list[ExpectedNumber],
    *,
    value: Any,
    label: str,
    section_id: str,
    required: bool,
    tolerance_pct: float | None = None,
) -> None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return
    if not math.isfinite(number):
        return
    target.append(ExpectedNumber(
        label=label,
        value=number,
        section_id=section_id,
        required=required,
        tolerance_pct=tolerance_pct,
    ))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


__all__ = [
    "ExpectedNumber",
    "IssueCategory",
    "IssueSeverity",
    "RuntimeCheckConfig",
    "RuntimeCheckError",
    "RuntimeCheckReport",
    "RuntimeChecker",
    "RuntimeIssue",
    "build_default_expected_numbers",
    "build_registry_expected_numbers",
    "extract_numbers",
    "find_markdown_artifacts",
    "find_unsupported_pdf_characters",
    "normalize_ru",
    "number_present",
    "section_text",
    "validate_text_quality",
]

