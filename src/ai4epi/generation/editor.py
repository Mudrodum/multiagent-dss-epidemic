"""
Редакторский слой ai4epi.

Модуль реализует независимый editor-pass поверх уже собранного бюллетеня.
Editor улучшает читаемость текстовых leaf-полей секций, но не меняет JSON-
структуру, числовые якоря и фактический контракт секции. Он не зависит от
LLM-evaluator-слоя: evaluation является downstream diagnostic/comparison layer.
"""

from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path
from collections.abc import MutableMapping
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ai4epi.generation.bulletin import Bulletin, build_bulletin_metadata, load_bulletin, save_bulletin
from ai4epi.core.context import GlobalContext, load_global_context
from ai4epi.generation.narrator import (
    ChatBackend,
    LLMResponseValidationError,
    strict_parse_and_validate_json_response,
    validate_llm_payload,
)
from ai4epi.quality.runtime_checks import RuntimeChecker
from ai4epi.core.sections import SectionConfig, SectionRegistry, default_section_registry


JsonObject = dict[str, Any]
TextPathSegment = str | int
TextPath = tuple[TextPathSegment, ...]
EditorStatus = Literal["ok", "partial", "failed"]
FieldStatus = Literal["changed", "no_op", "failed", "skipped"]


class EditorError(RuntimeError):
    """Базовая ошибка редакторского слоя."""


class EditorContractError(EditorError):
    """Редакторская правка нарушила контракт сохранения фактов или схемы."""

    def __init__(self, message: str, *, trace: "EditorFieldTrace | None" = None) -> None:
        super().__init__(message)
        self.trace = trace


class StrictModel(BaseModel):
    """Базовая pydantic-модель с закрытым контрактом."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        arbitrary_types_allowed=False,
    )


class EditorSettings(StrictModel):
    """Настройки редакторского прохода."""

    temperature: float = Field(default=0.0, ge=0.0)
    max_tokens: int = Field(default=700, ge=1)
    timeout: int | float | None = Field(default=None, gt=0)
    validate_section_schema: bool = True
    run_runtime_checks_after_edit: bool = False
    include_full_section_evidence: bool = True
    fail_on_field_errors: bool = False


class EditorFieldTrace(StrictModel):
    """Трассировка редактирования одного текстового поля."""

    section_id: str = Field(min_length=1)
    field_path: str = Field(min_length=1)
    status: FieldStatus
    changed: bool = False
    source_length: int = Field(default=0, ge=0)
    revised_length: int = Field(default=0, ge=0)
    latency_sec: float = Field(default=0.0, ge=0.0)
    raw_content_preview: str = ""
    schema_error: str | None = None
    validation_error: str | None = None
    original_preview: str = ""
    revised_preview: str = ""


class EditorRunResult(StrictModel):
    """Результат редакторского прохода."""

    status: EditorStatus
    bulletin: Bulletin
    traces: list[EditorFieldTrace] = Field(default_factory=list)
    changed_fields: dict[str, list[str]] = Field(default_factory=dict)
    no_op_fields: dict[str, list[str]] = Field(default_factory=dict)
    skipped_fields: dict[str, list[str]] = Field(default_factory=dict)
    errors: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def infer_status_consistency(self) -> "EditorRunResult":
        if self.status == "ok" and self.errors:
            object.__setattr__(self, "status", "partial")
        if self.status == "failed" and not self.errors:
            raise ValueError("failed editor result must contain at least one error.")
        return self

    def to_public_dict(self) -> JsonObject:
        """Вернуть JSON-сериализуемый отчёт редакторского прохода."""

        return self.model_dump(mode="json", exclude_none=False)

    def raise_for_failure(self) -> None:
        """Выбросить исключение, если редакторский проход завершился неуспешно."""

        if self.status == "failed":
            raise EditorError(json.dumps(self.errors, ensure_ascii=False, indent=2))


class PlainLanguageEditorAgent:
    """LLM-редактор одного текстового поля narrator draft."""

    def __init__(self, llm: ChatBackend, settings: EditorSettings | None = None) -> None:
        self.llm = llm
        self.settings = settings or EditorSettings()

    def edit_field(
        self,
        *,
        section: SectionConfig,
        field_path: TextPath,
        original_text: str,
        field_evidence_packet: Mapping[str, Any],
    ) -> tuple[str, EditorFieldTrace]:
        """
        Отредактировать одно текстовое поле.

        При нарушении JSON-контракта, protected-span контракта или числовых
        якорей вызывающий слой должен оставить исходный текст.
        """

        field_path_str = format_text_path(field_path)
        if not original_text.strip():
            trace = EditorFieldTrace(
                section_id=section.section_id,
                field_path=field_path_str,
                status="skipped",
                changed=False,
                source_length=len(original_text),
                revised_length=len(original_text),
                original_preview=original_text[:160],
                revised_preview=original_text[:160],
                validation_error="Пустое текстовое поле не отправлено в editor.",
            )
            return original_text, trace

        schema = make_field_editor_schema()
        masked_source_text, numeric_token_map = protect_numeric_spans(original_text)
        payload = make_minimal_editor_payload(
            section=section,
            field_path=field_path,
            original_text=original_text,
            masked_source_text=masked_source_text,
            numeric_token_map=numeric_token_map,
            field_evidence_packet=field_evidence_packet,
        )

        started_at = time.perf_counter()
        raw_content = ""
        try:
            response = self.llm.chat(
                [
                    {"role": "system", "content": build_editor_system_prompt()},
                    {"role": "user", "content": make_editor_user_message(payload)},
                ],
                temperature=self.settings.temperature,
                max_tokens=self.settings.max_tokens,
                format=schema,
                think=False,
                timeout=self.settings.timeout,
            )
            raw_content = response_content(response)
            parsed = strict_parse_and_validate_json_response(
                agent_name=f"PlainLanguageEditorAgent[{section.section_id}.{field_path_str}]",
                raw_content=raw_content,
                schema=schema,
            )
            revised_masked = extract_required_revised_text(parsed)
            token_error = validate_protected_tokens(masked_source_text, revised_masked, numeric_token_map)
            if token_error is not None:
                raise EditorContractError(token_error)

            revised_text = restore_numeric_spans(revised_masked, numeric_token_map)
            validation_error = validate_revised_text(original_text, revised_text)
            if validation_error is not None:
                raise EditorContractError(validation_error)

            changed = normalize_text_for_change_detection(original_text) != normalize_text_for_change_detection(revised_text)
            trace = EditorFieldTrace(
                section_id=section.section_id,
                field_path=field_path_str,
                status="changed" if changed else "no_op",
                changed=changed,
                source_length=len(original_text),
                revised_length=len(revised_text),
                latency_sec=time.perf_counter() - started_at,
                raw_content_preview=raw_content[:500],
                original_preview=original_text[:160],
                revised_preview=revised_text[:160],
            )
            return revised_text, trace
        except LLMResponseValidationError as exc:
            trace = EditorFieldTrace(
                section_id=section.section_id,
                field_path=field_path_str,
                status="failed",
                changed=False,
                source_length=len(original_text),
                revised_length=len(original_text),
                latency_sec=time.perf_counter() - started_at,
                raw_content_preview=raw_content[:500],
                schema_error=str(exc),
                original_preview=original_text[:160],
                revised_preview=original_text[:160],
            )
            raise EditorContractError(str(exc), trace=trace) from exc
        except Exception as exc:
            trace = EditorFieldTrace(
                section_id=section.section_id,
                field_path=field_path_str,
                status="failed",
                changed=False,
                source_length=len(original_text),
                revised_length=len(original_text),
                latency_sec=time.perf_counter() - started_at,
                raw_content_preview=raw_content[:500],
                validation_error=str(exc),
                original_preview=original_text[:160],
                revised_preview=original_text[:160],
            )
            raise EditorContractError(str(exc), trace=trace) from exc


class BulletinEditor:
    """Редакторский проход поверх готового Bulletin."""

    def __init__(
        self,
        llm: ChatBackend,
        *,
        registry: SectionRegistry | None = None,
        settings: EditorSettings | None = None,
        runtime_checker: RuntimeChecker | None = None,
    ) -> None:
        self.registry = registry or default_section_registry()
        self.settings = settings or EditorSettings()
        self.runtime_checker = runtime_checker or RuntimeChecker()
        self.agent = PlainLanguageEditorAgent(llm, self.settings)

    def edit_bulletin(
        self,
        *,
        context: GlobalContext | Mapping[str, Any],
        bulletin: Bulletin,
    ) -> EditorRunResult:
        """Отредактировать все доступные текстовые поля бюллетеня."""

        ctx = context if isinstance(context, GlobalContext) else GlobalContext.model_validate(context)
        edited_sections: dict[str, JsonObject] = copy.deepcopy(bulletin.sections)
        traces: list[EditorFieldTrace] = []
        errors: dict[str, str] = {}
        changed_fields: dict[str, list[str]] = {}
        no_op_fields: dict[str, list[str]] = {}
        skipped_fields: dict[str, list[str]] = {}

        for section in self.registry.available_sections(ctx):
            if section.section_id not in edited_sections:
                continue

            section_original = copy.deepcopy(edited_sections[section.section_id])
            section_current = copy.deepcopy(section_original)

            try:
                section_evidence = build_editor_evidence_packet(
                    context=ctx,
                    section=section,
                    include_full_section_evidence=self.settings.include_full_section_evidence,
                )
            except Exception as exc:
                key = section.section_id
                errors[key] = f"Не удалось собрать editor evidence: {exc}"
                continue

            for field_path, original_text in iter_text_leaf_fields(section_current):
                field_key = f"{section.section_id}.{format_text_path(field_path)}"
                field_evidence_packet = build_field_evidence_packet(
                    section=section,
                    field_path=field_path,
                    original_text=original_text,
                    section_evidence_packet=section_evidence,
                )
                try:
                    revised_text, trace = self.agent.edit_field(
                        section=section,
                        field_path=field_path,
                        original_text=original_text,
                        field_evidence_packet=field_evidence_packet,
                    )
                    traces.append(trace)
                except EditorContractError as exc:
                    errors[field_key] = str(exc)
                    trace = exc.trace or EditorFieldTrace(
                        section_id=section.section_id,
                        field_path=format_text_path(field_path),
                        status="failed",
                        changed=False,
                        source_length=len(original_text),
                        revised_length=len(original_text),
                        validation_error=str(exc),
                        original_preview=original_text[:160],
                        revised_preview=original_text[:160],
                    )
                    traces.append(trace)
                    revised_text = original_text

                if trace.status == "changed":
                    set_text_path(section_current, field_path, revised_text)
                    changed_fields.setdefault(section.section_id, []).append(format_text_path(field_path))
                elif trace.status == "no_op":
                    no_op_fields.setdefault(section.section_id, []).append(format_text_path(field_path))
                elif trace.status == "skipped":
                    skipped_fields.setdefault(section.section_id, []).append(format_text_path(field_path))

            if self.settings.validate_section_schema:
                try:
                    validate_llm_payload(section_current, section.output_schema)
                except Exception as exc:
                    errors[section.section_id] = (
                        "После редакторского прохода секция не соответствует output_schema; "
                        f"секция возвращена к исходному состоянию: {exc}"
                    )
                    edited_sections[section.section_id] = section_original
                    changed_fields.pop(section.section_id, None)
                    no_op_fields.pop(section.section_id, None)
                    skipped_fields.pop(section.section_id, None)
                    continue

            edited_sections[section.section_id] = section_current

        if self.settings.fail_on_field_errors and errors:
            status: EditorStatus = "failed"
        elif errors:
            status = "partial"
        else:
            status = "ok"

        runtime_report = None
        if self.settings.run_runtime_checks_after_edit:
            runtime_report = self.runtime_checker.check_bulletin(
                context=ctx,
                bulletin={"sections": edited_sections},
                registry=self.registry,
            )

        edited_bulletin = Bulletin(
            metadata=build_bulletin_metadata(
                ctx,
                extra={
                    **dict(bulletin.metadata.extra or {}),
                    "editor": {
                        "applied": True,
                        "status": status,
                        "changed_fields": changed_fields,
                        "errors": errors,
                    },
                },
            ),
            status="checked" if runtime_report is not None else "draft",
            sections=edited_sections,
            section_order=list(bulletin.section_order),
            section_titles=dict(bulletin.section_titles),
            section_orders=dict(bulletin.section_orders),
            traces_by_section=append_editor_traces_to_bulletin_traces(
                bulletin.traces_by_section,
                traces,
            ),
            failures_by_section=dict(bulletin.failures_by_section),
            runtime_check_report=runtime_report,
            assets=copy.deepcopy(bulletin.assets),
            editorial_notes=[
                *list(bulletin.editorial_notes or []),
                "Plain-language editor applied without evaluator feedback.",
            ],
        )

        return EditorRunResult(
            status=status,
            bulletin=edited_bulletin,
            traces=traces,
            changed_fields=changed_fields,
            no_op_fields=no_op_fields,
            skipped_fields=skipped_fields,
            errors=errors,
        )


def make_field_editor_schema() -> JsonObject:
    """JSON Schema ответа редактора одного поля."""

    return {
        "type": "object",
        "properties": {
            "revised_text": {"type": "string"},
        },
        "required": ["revised_text"],
        "additionalProperties": False,
    }


def editor_output_contract_text() -> str:
    """Текстовый контракт JSON-ответа редактора."""

    return (
        'Верни ровно один JSON-объект и ничего больше. '
        'Допустим только формат {"revised_text": "..."} без дополнительных ключей. '
        'Первый символ ответа должен быть "{", последний символ ответа должен быть "}". '
        'Используй только двойные кавычки JSON. '
        'Запрещены markdown, пояснения, комментарии, префиксы, суффиксы, списки и любой текст вне JSON. '
        'Если правка не нужна, верни {"revised_text": "<исходный текст без изменений>"}.'
    )


def editor_few_shots_text() -> str:
    """Нормативные примеры поведения editor-агента."""

    return (
        "Пример 1 — допустимо делать фразу короче и прямее\n"
        "Исходный текст: \"Прогноз на четыре недели показывает временное снижение с последующим ростом.\"\n"
        "Хорошая правка: \"Прогноз на четыре недели указывает на временное снижение с последующим ростом.\"\n"
        "Норма: если формулировку можно сделать прямее без потери смысла, это следует сделать.\n\n"
        "Пример 2 — допустимо разбивать тяжёлую конструкцию\n"
        "Исходный текст: \"Влияние температуры за шестидневное скользящее окно характеризуется нестабильностью направления, поэтому его эффект зависит от текущей эпидемической ситуации и может варьироваться как в сторону повышения, так и понижения прогноза.\"\n"
        "Хорошая правка: \"Влияние температуры за шестидневное скользящее окно нестабильно. Поэтому его эффект зависит от текущей эпидемической ситуации и может как повышать, так и понижать прогноз.\"\n"
        "Норма: разрешено упростить длинную фразу и разделить её на два предложения, если фактические отношения сохранены.\n\n"
        "Пример 3 — допустимо убирать канцелярит\n"
        "Исходный текст: \"Ширина прогнозного интервала для лица, принимающего решения, указывает на значительный разброс возможных значений.\"\n"
        "Хорошая правка: \"Ширина прогнозного интервала указывает на значительный разброс возможных значений.\"\n"
        "Норма: убирай тяжёлые служебные обороты, если они не добавляют отдельного факта.\n\n"
        "Пример 4 — narrator draft первичен\n"
        "Исходный текст: \"Вероятность резкого роста активности вируса не исключается.\"\n"
        "Хорошая правка: \"Вероятность резкого роста активности вируса не исключается.\"\n"
        "Норма: нельзя усиливать или ослаблять модальность, даже если evidence позволяет более сильную формулировку.\n\n"
        "Пример 5 — protected span переносится дословно\n"
        "Исходный текст: \"Ширина главного пика составляет [[NUM_1]] недели.\"\n"
        "Хорошая правка: \"Ширина главного пика составляет [[NUM_1]] недели.\"\n"
        "Норма: protected span и связанная с ним фактическая конструкция не изменяются.\n\n"
        "Пример 6 — нельзя импортировать факт из evidence\n"
        "Исходный текст: \"Суммарная сезонная нагрузка уступает предыдущему сезону.\"\n"
        "Хорошая правка: \"Суммарная сезонная нагрузка уступает предыдущему сезону.\"\n"
        "Норма: запрещено добавлять числа, сравнения или акценты из evidence, если их нет в narrator draft.\n\n"
        "Пример 7 — для численно плотного абзаца допустим no-op\n"
        "Исходный текст: \"Ширина волны по FWHM для сезона 2025–2026 составляет [[NUM_1]] недели, что лишь незначительно отличается от ширины волны 2023–2024 ([[NUM_2]] недели).\"\n"
        "Хорошая правка: \"Ширина волны по FWHM для сезона 2025–2026 составляет [[NUM_1]] недели, что лишь незначительно отличается от ширины волны 2023–2024 ([[NUM_2]] недели).\"\n"
        "Норма: если абзац насыщен числами и сравнениями, правь его только при очевидном и безопасном выигрыше.\n\n"
        "Пример 8 — формат ответа\n"
        'Допустимый ответ: {"revised_text": "Исходный текст без изменений."}\n'
        "Норма: ответ должен быть одним валидным JSON-объектом с одним ключом revised_text."
    )


def build_editor_system_prompt() -> str:
    """System prompt редакторского агента."""

    return (
        "Ты редактор аналитического текста. "
        "Твоя задача — улучшить язык narrator draft: сделать формулировку яснее, проще, компактнее и естественнее, "
        "сохранив фактическое содержание, числа, смысловые отношения и степень уверенности.\n\n"
        "Иерархия источников:\n"
        "1) narrator draft — основной и канонический текст для редактирования;\n"
        "2) evidence — только вспомогательная проверка того, что при перефразировании не искажены факты;\n"
        "3) запрещено добавлять в итоговый текст факты, числа, акценты или интерпретации из evidence, если их нет в narrator draft.\n\n"
        "Целевая стилистика:\n"
        "Пиши на ясном профессиональном русском языке. Предпочитай прямой синтаксис, более короткие фразы и естественные формулировки. "
        "Устраняй канцелярит, тяжеловесные обороты, перегруженные перечисления и ненужные служебные конструкции. "
        "Не упрощай текст до разговорного регистра и не теряй научную точность.\n\n"
        "Разрешено:\n"
        "- свободно переписывать предложение и порядок его частей;\n"
        "- разбивать длинную фразу на две более ясные;\n"
        "- заменять тяжёлую конструкцию на более прямую;\n"
        "- сокращать текст, если смысл и фактические связи полностью сохранены;\n"
        "- убирать канцелярские и избыточные обороты.\n\n"
        "Запрещено:\n"
        "- менять любые protected spans;\n"
        "- менять числа, даты, проценты, интервалы, возрастные группы, сезоны, горизонты прогноза, единицы измерения и метрики;\n"
        "- менять сравнения, причинно-следственные связи, временные отношения и модальность;\n"
        "- усиливать или ослаблять риск, уверенность, неопределённость или интерпретацию;\n"
        "- добавлять сведения из evidence;\n"
        "- делать выводы, которых не было в narrator draft.\n\n"
        "Правило protected spans:\n"
        "Каждый protected span должен быть перенесён в revised_text дословно, без изменения символов, порядка, формы и ближайшей фактической связи. "
        "Если удачное упрощение требует изменить protected span или связанный с ним факт, верни исходный текст без изменений.\n\n"
        "Правило решения:\n"
        "- для prose-heavy полей предпочитай заметное улучшение читаемости;\n"
        "- для fact-dense полей будь консервативен и правь только при явном и безопасном выигрыше;\n"
        "- narrator draft важнее surface-closeness: можно переписывать смелее, если фактические якоря сохранены.\n\n"
        f"Нормативные примеры поведения редактора:\n\n{editor_few_shots_text()}\n\n"
        f"{editor_output_contract_text()}"
    )


def make_editor_user_message(payload: Mapping[str, Any]) -> str:
    """Сформировать user-payload редакторского агента."""

    packet = {
        "task": "plain_language_controlled_rewrite_from_narrator",
        "goal": (
            "Улучшить язык draft_text: сделать формулировку проще, яснее и естественнее, "
            "сохранив фактическое содержание narrator draft."
        ),
        "source_priority": {
            "primary_source_for_writing": "draft_text",
            "secondary_source_for_fact_check": "field_evidence",
            "do_not_import_new_facts_from_evidence": True,
        },
        "editing_policy": {
            "mode": "plain_language_controlled_rewrite_from_narrator",
            "allow_sentence_rewrite": True,
            "allow_sentence_split": True,
            "allow_syntax_simplification": True,
            "allow_clarity_rephrase": True,
            "allow_plain_language_shift": True,
            "allow_remove_heavy_bureaucratic_phrases": True,
            "prefer_readability_gain_over_surface_closeness": True,
            "preserve_all_protected_spans_verbatim": True,
            "preserve_modality": True,
            "preserve_causal_temporal_and_comparative_relations": True,
            "fallback_to_no_op_if_risky": True,
        },
        "field_profile": payload.get("field_profile", {}),
        "context": {
            "section_id": payload.get("section_id", ""),
            "section_title": payload.get("section_title", ""),
            "field_path": payload.get("field_path", ""),
            "field_role": payload.get("field_role", ""),
            "section_role": payload.get("section_role", ""),
        },
        "input": {
            "draft_text": payload.get("draft_text", ""),
            "field_evidence": payload.get("field_evidence", {}),
            "protected_spans": payload.get("protected_spans", []),
        },
        "output_contract": {
            "only_json_object": True,
            "json_must_be_valid": True,
            "first_char_must_be": "{",
            "last_char_must_be": "}",
            "single_allowed_key": "revised_text",
            "use_only_double_quotes": True,
            "valid_example": {"revised_text": "Исходный текст без изменений."},
        },
    }
    return json.dumps(packet, ensure_ascii=False)


def editor_field_profile(section_id: str, field_path: TextPath) -> JsonObject:
    """Вернуть профиль допустимой свободы редактирования для поля."""

    field_name = leaf_field_name(field_path)
    prose_heavy = {
        ("current_situation", "paragraph_forecast_brief"),
        ("forecast_risks", "point_forecast_text"),
        ("forecast_risks", "uncertainty_text"),
        ("forecast_risks", "risk_assessment"),
        ("shap_interpretation", "short_term_factors"),
        ("shap_interpretation", "long_term_factors"),
        ("shap_interpretation", "overall_insight"),
        ("model_quality", "quality_summary"),
        ("model_description", "description_text"),
    }
    fact_dense = {
        ("epidemic_wave_comparison", "comparison_text"),
        ("age_group_season_overview", "overview_text"),
        ("model_quality", "limitations_text"),
    }

    if (section_id, field_name) in prose_heavy:
        return {
            "rewrite_freedom": "high",
            "style_target": "plain_language_scientific",
            "instruction": (
                "Это prose-heavy поле. Предпочтительны заметные улучшения читаемости: "
                "прямой синтаксис, меньше канцелярита, более короткие фразы и ясные причинные связи."
            ),
        }

    if (section_id, field_name) in fact_dense:
        return {
            "rewrite_freedom": "conservative",
            "style_target": "fact_dense_conservative",
            "instruction": (
                "Это фактологически плотное поле с числами и сравнениями. "
                "Правь только при очевидном выигрыше в ясности и нулевом риске для фактов."
            ),
        }

    return {
        "rewrite_freedom": "medium",
        "style_target": "plain_language_balanced",
        "instruction": (
            "Предпочтительно умеренное упрощение: сделать текст яснее и компактнее, "
            "не переходя в разговорный регистр."
        ),
    }


def describe_field_role(section: SectionConfig, field_path: TextPath) -> str:
    """Описать роль текстового поля для editor-prompt."""

    path_key = format_text_path(field_path)
    metadata_roles = section.metadata.get("editor_field_roles", {}) if isinstance(section.metadata, Mapping) else {}
    if isinstance(metadata_roles, Mapping) and path_key in metadata_roles:
        return str(metadata_roles[path_key])

    field_name = leaf_field_name(field_path)
    roles = {
        ("current_situation", "paragraph_situation"): "абзац о текущей эпидемиологической ситуации",
        ("current_situation", "paragraph_forecast_brief"): "краткий абзац о ближайшем прогнозе",
        ("epidemic_wave_comparison", "comparison_text"): "абзац со сравнением последних эпидемических волн",
        ("age_group_season_overview", "overview_text"): "абзац с возрастным обзором текущего сезона",
        ("forecast_risks", "point_forecast_text"): "абзац о точечном прогнозе",
        ("forecast_risks", "uncertainty_text"): "абзац о неопределённости прогноза",
        ("forecast_risks", "risk_assessment"): "абзац об оценке рисков",
        ("shap_interpretation", "short_term_factors"): "абзац о краткосрочных факторах модели",
        ("shap_interpretation", "long_term_factors"): "абзац о долгосрочных факторах модели",
        ("shap_interpretation", "overall_insight"): "итоговый абзац об интерпретации факторов",
        ("model_quality", "quality_summary"): "абзац о качестве модели",
        ("model_quality", "limitations_text"): "абзац об ограничениях модели",
        ("model_description", "description_text"): "абзац с описанием модели",
    }
    return roles.get((section.section_id, field_name), f"текстовое поле {section.section_id}.{path_key}")


def build_editor_evidence_packet(
    *,
    context: GlobalContext,
    section: SectionConfig,
    include_full_section_evidence: bool = True,
) -> JsonObject:
    """Собрать evidence-packet для редактора секции."""

    section_role = ""
    if isinstance(section.metadata, Mapping):
        section_role = str(section.metadata.get("editor_section_role") or "")
    if not section_role:
        section_role = section.title

    return {
        "section_id": section.section_id,
        "section_title": section.title,
        "section_role": section_role,
        "section_specific_evidence": section.build_evidence(context) if include_full_section_evidence else {},
        "required_numeric_facts": copy.deepcopy(section.metadata.get("expected_numbers", {})),
    }


def build_field_evidence_packet(
    *,
    section: SectionConfig,
    field_path: TextPath,
    original_text: str,
    section_evidence_packet: Mapping[str, Any],
) -> JsonObject:
    """Собрать field-level evidence для editor-agent."""

    masked_source_text, numeric_token_map = protect_numeric_spans(original_text)
    return {
        "section_id": section.section_id,
        "section_title": section.title,
        "field_path": format_text_path(field_path),
        "field_role": describe_field_role(section, field_path),
        "section_role": section_evidence_packet.get("section_role", ""),
        "source_text": original_text,
        "masked_source_text": masked_source_text,
        "numeric_token_map": numeric_token_map,
        "clean_field_evidence": copy.deepcopy(section_evidence_packet.get("section_specific_evidence", {})),
    }


def make_minimal_editor_payload(
    *,
    section: SectionConfig,
    field_path: TextPath,
    original_text: str,
    masked_source_text: str,
    numeric_token_map: Mapping[str, str],
    field_evidence_packet: Mapping[str, Any],
) -> JsonObject:
    """Сформировать минимальный payload для редактирования одного поля."""

    return {
        "editing_mode": "plain_language_controlled_rewrite_from_narrator",
        "section_id": section.section_id,
        "section_title": section.title,
        "field_path": format_text_path(field_path),
        "field_role": field_evidence_packet.get("field_role", ""),
        "section_role": field_evidence_packet.get("section_role", ""),
        "field_profile": editor_field_profile(section.section_id, field_path),
        "draft_text": masked_source_text,
        "field_evidence": field_evidence_packet.get("clean_field_evidence", {}),
        "protected_spans": list(numeric_token_map.keys()),
        "source_text_unmasked": original_text,
    }


def extract_required_revised_text(payload: Mapping[str, Any]) -> str:
    """Извлечь revised_text из ответа редактора."""

    if not isinstance(payload, Mapping):
        raise ValueError("payload должен быть mapping")
    if "revised_text" not in payload:
        raise ValueError("ответ модели нарушает schema contract: отсутствует обязательный ключ revised_text")
    revised_text = payload.get("revised_text")
    if not isinstance(revised_text, str):
        raise ValueError("revised_text должен быть строкой")
    return revised_text.strip()


def normalize_text_for_change_detection(text: str | None) -> str:
    """Нормализовать текст только для определения факта изменения."""

    return " ".join((text or "").split())


def extract_numeric_anchors(text: str) -> list[str]:
    """Извлечь числовые якоря в порядке появления."""

    return re.findall(r"\d+(?:[.,]\d+)?", text or "")


def protect_numeric_spans(text: str) -> tuple[str, dict[str, str]]:
    """Заменить числовые фрагменты на protected tokens."""

    if not isinstance(text, str) or not text:
        return "", {}

    mapping: dict[str, str] = {}
    counter = 0

    def repl(match: re.Match[str]) -> str:
        nonlocal counter
        counter += 1
        token = f"[[NUM_{counter}]]"
        mapping[token] = match.group(0)
        return token

    masked = re.sub(r"\d+(?:[.,]\d+)?", repl, text)
    return masked, mapping


def restore_numeric_spans(text: str, mapping: Mapping[str, str]) -> str:
    """Восстановить protected numeric spans."""

    out = text or ""
    for token, raw_value in mapping.items():
        out = out.replace(token, raw_value)
    return out


def validate_protected_tokens(
    original_masked_text: str,
    revised_masked_text: str,
    mapping: Mapping[str, str],
) -> str | None:
    """Проверить, что editor сохранил protected tokens в исходном порядке."""

    if not mapping:
        return None
    token_re = r"\[\[NUM_\d+\]\]"
    original_tokens = re.findall(token_re, original_masked_text or "")
    revised_tokens = re.findall(token_re, revised_masked_text or "")
    if original_tokens != revised_tokens:
        return (
            "нарушено сохранение protected numeric spans: "
            f"source={original_tokens}, revised={revised_tokens}"
        )
    return None


def validate_revised_text(original_text: str, revised_text: str) -> str | None:
    """Проверить field-level контракт после восстановления protected spans."""

    if not isinstance(revised_text, str):
        return "revised_text не является строкой"
    if not revised_text.strip():
        return "revised_text пуст"

    source_anchors = extract_numeric_anchors(original_text)
    revised_anchors = extract_numeric_anchors(revised_text)
    if source_anchors != revised_anchors:
        return (
            "нарушено сохранение numeric anchors: "
            f"source={source_anchors}, revised={revised_anchors}"
        )

    return None


def iter_text_leaf_fields(value: Any, prefix: TextPath = ()) -> list[tuple[TextPath, str]]:
    """Вернуть все строковые leaf-поля JSON-подобной структуры."""

    if isinstance(value, str):
        return [(prefix, value)]
    if isinstance(value, Mapping):
        fields: list[tuple[TextPath, str]] = []
        for key, item in value.items():
            fields.extend(iter_text_leaf_fields(item, (*prefix, str(key))))
        return fields
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        fields = []
        for index, item in enumerate(value):
            fields.extend(iter_text_leaf_fields(item, (*prefix, index)))
        return fields
    return []


def set_text_path(target: Any, path: TextPath, value: str) -> None:
    """Установить строковое значение по пути внутри JSON-подобной структуры."""

    if not path:
        raise ValueError("Cannot set value at empty path.")
    current = target
    for segment in path[:-1]:
        if isinstance(current, list) and isinstance(segment, int):
            current = current[segment]
        elif isinstance(current, MutableMapping) and isinstance(segment, str):
            current = current[segment]
        else:
            raise TypeError(f"Cannot traverse segment {segment!r} in path {path!r}.")
    leaf = path[-1]
    if isinstance(current, list) and isinstance(leaf, int):
        current[leaf] = value
    elif isinstance(current, MutableMapping) and isinstance(leaf, str):
        current[leaf] = value
    else:
        raise TypeError(f"Cannot set leaf {leaf!r} in path {path!r}.")


def format_text_path(path: TextPath) -> str:
    """Преобразовать путь поля в стабильную строку."""

    if not path:
        return "$"
    return ".".join(str(segment) for segment in path)


def leaf_field_name(path: TextPath) -> str:
    """Вернуть последнее строковое имя поля из пути."""

    for segment in reversed(path):
        if isinstance(segment, str):
            return segment
    return format_text_path(path)


def response_content(response: Mapping[str, Any] | str) -> str:
    """Извлечь текст ответа из backend-результата."""

    if isinstance(response, str):
        return response
    content = response.get("content", "")
    return "" if content is None else str(content)


def append_editor_traces_to_bulletin_traces(
    existing: Mapping[str, Sequence[Mapping[str, Any]]],
    traces: Sequence[EditorFieldTrace],
) -> dict[str, list[JsonObject]]:
    """Добавить editor trace к секционным traces бюллетеня."""

    out: dict[str, list[JsonObject]] = {
        str(section_id): [dict(item) for item in trace_items]
        for section_id, trace_items in existing.items()
    }
    grouped: dict[str, list[JsonObject]] = {}
    for trace in traces:
        grouped.setdefault(trace.section_id, []).append(trace.model_dump(mode="json", exclude_none=False))

    for section_id, field_traces in grouped.items():
        out.setdefault(section_id, []).append(
            {
                "stage": "plain_language_editor",
                "field_level": True,
                "field_traces": field_traces,
                "changed_fields": [item["field_path"] for item in field_traces if item.get("changed")],
                "attempted_fields": [item["field_path"] for item in field_traces],
            }
        )
    return out


def edit_bulletin(
    *,
    context: GlobalContext | Mapping[str, Any],
    bulletin: Bulletin,
    llm: ChatBackend,
    registry: SectionRegistry | None = None,
    settings: EditorSettings | None = None,
    runtime_checker: RuntimeChecker | None = None,
) -> EditorRunResult:
    """Функциональная обёртка над BulletinEditor."""

    editor = BulletinEditor(
        llm,
        registry=registry,
        settings=settings,
        runtime_checker=runtime_checker,
    )
    return editor.edit_bulletin(context=context, bulletin=bulletin)


def edit_bulletin_from_files(
    *,
    context_path: str | Path,
    bulletin_path: str | Path,
    llm: ChatBackend,
    output_bulletin_path: str | Path | None = None,
    output_report_path: str | Path | None = None,
    registry: SectionRegistry | None = None,
    settings: EditorSettings | None = None,
    runtime_checker: RuntimeChecker | None = None,
) -> EditorRunResult:
    """Загрузить context и bulletin из файлов, выполнить editor-pass и сохранить результат."""

    context = load_global_context(context_path)
    bulletin = load_bulletin(bulletin_path)
    result = edit_bulletin(
        context=context,
        bulletin=bulletin,
        llm=llm,
        registry=registry,
        settings=settings,
        runtime_checker=runtime_checker,
    )

    if output_bulletin_path is not None:
        save_bulletin(result.bulletin, output_bulletin_path)
    if output_report_path is not None:
        output_path = Path(output_report_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(result.to_public_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return result


__all__ = [
    "BulletinEditor",
    "EditorContractError",
    "EditorError",
    "EditorFieldTrace",
    "EditorRunResult",
    "EditorSettings",
    "EditorStatus",
    "FieldStatus",
    "PlainLanguageEditorAgent",
    "append_editor_traces_to_bulletin_traces",
    "build_editor_evidence_packet",
    "build_editor_system_prompt",
    "build_field_evidence_packet",
    "describe_field_role",
    "edit_bulletin",
    "edit_bulletin_from_files",
    "editor_field_profile",
    "extract_numeric_anchors",
    "format_text_path",
    "iter_text_leaf_fields",
    "make_field_editor_schema",
    "normalize_text_for_change_detection",
    "protect_numeric_spans",
    "restore_numeric_spans",
    "validate_revised_text",
]

