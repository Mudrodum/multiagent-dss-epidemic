"""
Сборка и сериализация итогового бюллетеня ai4epi.

Модуль соединяет результаты секционных narrator-агентов, порядок секций,
метаданные запуска, трассировку генерации, assets и отчёт runtime-checks в
единый сериализуемый объект. Он не вызывает LLM и не редактирует текст: входом
являются уже полученные структурированные ответы секций.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4epi.core.context import GlobalContext
from ai4epi.generation.narrator import LLMResponseValidationError, SectionNarrationResult, validate_llm_payload
from ai4epi.quality.runtime_checks import RuntimeCheckReport, RuntimeChecker
from ai4epi.core.sections import SectionConfig, SectionRegistry, default_section_registry


JsonObject = dict[str, Any]
BulletinStatus = Literal["draft", "checked", "failed"]


class BulletinBuildError(RuntimeError):
    """Ошибка сборки бюллетеня из секционных результатов."""


class StrictModel(BaseModel):
    """Базовая pydantic-модель с закрытым контрактом."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        arbitrary_types_allowed=False,
    )


class BulletinMetadata(StrictModel):
    """Метаданные одного выпуска бюллетеня."""

    bulletin_id: str = Field(min_length=1)
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    schema_version: str = Field(default="1.0", min_length=1)
    package_name: str = Field(default="ai4epi", min_length=1)
    origin_date: str = Field(min_length=1)
    iso_year: int
    iso_week: int = Field(ge=1, le=53)
    unit: str = Field(min_length=1)
    per_population: int | float = Field(gt=0)
    context_blocks: list[str] = Field(default_factory=list)
    extra: JsonObject = Field(default_factory=dict)

    @field_validator("context_blocks")
    @classmethod
    def validate_context_blocks(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("context_blocks must not contain duplicates.")
        return value

    @model_validator(mode="after")
    def normalize_generated_at_timezone(self) -> "BulletinMetadata":
        if self.generated_at.tzinfo is None:
            object.__setattr__(self, "generated_at", self.generated_at.replace(tzinfo=timezone.utc))
        return self


class BulletinSection(StrictModel):
    """Одна упорядоченная секция бюллетеня."""

    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str = Field(min_length=1)
    order: int = Field(ge=0)
    content: JsonObject
    plain_text: str = ""

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: JsonObject) -> JsonObject:
        if not value:
            raise ValueError("section content must not be empty.")
        return value

    @model_validator(mode="after")
    def populate_plain_text(self) -> "BulletinSection":
        if not self.plain_text.strip():
            self.plain_text = flatten_text(self.content)
        return self

    @classmethod
    def from_config(cls, section: SectionConfig, content: Mapping[str, Any]) -> "BulletinSection":
        """Создать объект секции по конфигурации и структурированному payload."""

        return cls(
            section_id=section.section_id,
            title=section.title,
            order=section.order,
            content=dict(content),
            plain_text=flatten_text(content),
        )


class Bulletin(StrictModel):
    """Итоговый структурированный бюллетень."""

    metadata: BulletinMetadata
    status: BulletinStatus = "draft"
    sections: dict[str, JsonObject]
    section_order: list[str]
    section_titles: dict[str, str]
    section_orders: dict[str, int]
    traces_by_section: dict[str, list[JsonObject]] = Field(default_factory=dict)
    failures_by_section: dict[str, str] = Field(default_factory=dict)
    runtime_check_report: RuntimeCheckReport | None = None
    assets: JsonObject = Field(default_factory=dict)
    editorial_notes: list[str] = Field(default_factory=list)

    @field_validator("sections")
    @classmethod
    def validate_sections_not_empty(cls, value: dict[str, JsonObject]) -> dict[str, JsonObject]:
        if not value:
            raise ValueError("bulletin must contain at least one generated section.")
        for section_id, content in value.items():
            if not section_id:
                raise ValueError("section_id must not be empty.")
            if not isinstance(content, Mapping) or not content:
                raise ValueError(f"section {section_id!r} must contain a non-empty object.")
        return value

    @field_validator("section_order")
    @classmethod
    def validate_section_order_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("section_order must not contain duplicates.")
        return value

    @model_validator(mode="after")
    def validate_section_index_consistency(self) -> "Bulletin":
        section_ids = set(self.sections)
        order_ids = set(self.section_order)
        title_ids = set(self.section_titles)
        numeric_order_ids = set(self.section_orders)

        missing_in_order = section_ids - order_ids
        missing_in_titles = section_ids - title_ids
        missing_in_numeric_orders = section_ids - numeric_order_ids
        unknown_order_ids = order_ids - section_ids

        if missing_in_order:
            raise ValueError(f"section_order misses generated sections: {sorted(missing_in_order)!r}.")
        if missing_in_titles:
            raise ValueError(f"section_titles misses generated sections: {sorted(missing_in_titles)!r}.")
        if missing_in_numeric_orders:
            raise ValueError(f"section_orders misses generated sections: {sorted(missing_in_numeric_orders)!r}.")
        if unknown_order_ids:
            raise ValueError(f"section_order contains unknown sections: {sorted(unknown_order_ids)!r}.")

        if self.failures_by_section:
            object.__setattr__(self, "status", "failed")
        elif self.runtime_check_report is not None:
            object.__setattr__(self, "status", "checked")

        return self

    @property
    def ordered_section_ids(self) -> list[str]:
        """Вернуть section_id в порядке отображения."""

        return [section_id for section_id in self.section_order if section_id in self.sections]

    @property
    def ordered_sections(self) -> list[BulletinSection]:
        """Вернуть секции как список объектов с title, order и plain_text."""

        return [
            BulletinSection(
                section_id=section_id,
                title=self.section_titles[section_id],
                order=self.section_orders[section_id],
                content=self.sections[section_id],
            )
            for index, section_id in enumerate(self.ordered_section_ids)
        ]

    def section_text(self, section_id: str) -> str:
        """Вернуть плоский текст одной секции."""

        if section_id not in self.sections:
            raise KeyError(f"Unknown section_id: {section_id!r}.")
        return flatten_text(self.sections[section_id])

    def to_public_dict(self) -> JsonObject:
        """Вернуть JSON-сериализуемый вид бюллетеня."""

        return self.model_dump(mode="json", exclude_none=False)

    def to_markdown(self, *, include_metadata: bool = False) -> str:
        """Сформировать человекочитаемый markdown без изменения содержимого секций."""

        lines: list[str] = []
        if include_metadata:
            lines.append(f"# Бюллетень {self.metadata.iso_year}-W{self.metadata.iso_week:02d}")
            lines.append("")
            lines.append(f"- Идентификатор: `{self.metadata.bulletin_id}`")
            lines.append(f"- Дата прогноза: {self.metadata.origin_date}")
            lines.append(f"- Статус: {self.status}")
            lines.append("")

        for section in self.ordered_sections:
            lines.append(f"## {section.title}")
            lines.append("")
            text = section.plain_text.strip()
            if text:
                lines.append(text)
                lines.append("")

        return "\n".join(lines).strip() + "\n"


class BulletinBuilder:
    """Сборщик Bulletin из результатов narrator-слоя."""

    def __init__(
        self,
        registry: SectionRegistry | None = None,
        *,
        runtime_checker: RuntimeChecker | None = None,
    ) -> None:
        self.registry = registry or default_section_registry()
        self.runtime_checker = runtime_checker or RuntimeChecker()

    def build_from_narration_results(
        self,
        *,
        context: GlobalContext | Mapping[str, Any],
        results: Mapping[str, SectionNarrationResult | Mapping[str, Any]],
        run_runtime_checks: bool = True,
        require_complete: bool = True,
        validate_section_payloads: bool = True,
        assets: Mapping[str, Any] | None = None,
        metadata_extra: Mapping[str, Any] | None = None,
        editorial_notes: Sequence[str] | None = None,
    ) -> Bulletin:
        """
        Собрать Bulletin из результатов секционных narrator-агентов.

        Parameters
        ----------
        context:
            Глобальный контекст выпуска.
        results:
            Mapping ``section_id -> SectionNarrationResult``. Для тестов и
            интеграций допускается также mapping с полями ``draft``, ``trace``,
            ``status`` и ``failure_reason``.
        run_runtime_checks:
            Выполнить deterministic runtime-checks после сборки секций.
        require_complete:
            Если ``True``, отсутствие доступной обязательной секции или её
            неуспешный статус прерывает сборку.
        validate_section_payloads:
            Повторно проверить payload каждой секции по её JSON Schema.
        """

        ctx = context if isinstance(context, GlobalContext) else GlobalContext.model_validate(context)
        available_sections = self.registry.available_sections(ctx)
        sections: dict[str, JsonObject] = {}
        section_order: list[str] = []
        section_titles: dict[str, str] = {}
        section_orders: dict[str, int] = {}
        traces: dict[str, list[JsonObject]] = {}
        failures: dict[str, str] = {}

        for section in available_sections:
            raw_result = results.get(section.section_id)
            if raw_result is None:
                message = f"Result for section {section.section_id!r} is absent."
                if require_complete and section.required:
                    raise BulletinBuildError(message)
                failures[section.section_id] = message
                continue

            draft, trace, failure_reason, ok = _coerce_narration_result(raw_result)
            if trace:
                traces[section.section_id] = trace

            if not ok or draft is None:
                message = failure_reason or f"Section {section.section_id!r} failed without a reason."
                if require_complete and section.required:
                    raise BulletinBuildError(message)
                failures[section.section_id] = message
                continue

            if validate_section_payloads:
                _validate_section_payload(section, draft)

            sections[section.section_id] = _to_jsonable_object(draft)
            section_order.append(section.section_id)
            section_titles[section.section_id] = section.title
            section_orders[section.section_id] = section.order

        if not sections:
            raise BulletinBuildError("No valid sections were generated; bulletin cannot be built.")

        metadata = build_bulletin_metadata(ctx, extra=metadata_extra)
        runtime_report = None
        if run_runtime_checks:
            runtime_report = self.runtime_checker.check_bulletin(
                context=ctx,
                bulletin={"sections": sections},
                registry=self.registry,
            )

        return Bulletin(
            metadata=metadata,
            sections=sections,
            section_order=section_order,
            section_titles=section_titles,
            section_orders=section_orders,
            traces_by_section=traces,
            failures_by_section=failures,
            runtime_check_report=runtime_report,
            assets=dict(assets or {}),
            editorial_notes=list(editorial_notes or []),
        )

    def build_from_sections(
        self,
        *,
        context: GlobalContext | Mapping[str, Any],
        sections: Mapping[str, Mapping[str, Any]],
        run_runtime_checks: bool = True,
        validate_section_payloads: bool = True,
        assets: Mapping[str, Any] | None = None,
        metadata_extra: Mapping[str, Any] | None = None,
    ) -> Bulletin:
        """Собрать Bulletin из уже готового словаря секционных payload."""

        normalized_results = {
            section_id: {
                "status": "ok",
                "draft": dict(payload),
                "trace": [],
                "failure_reason": None,
            }
            for section_id, payload in sections.items()
        }
        return self.build_from_narration_results(
            context=context,
            results=normalized_results,
            run_runtime_checks=run_runtime_checks,
            require_complete=False,
            validate_section_payloads=validate_section_payloads,
            assets=assets,
            metadata_extra=metadata_extra,
        )


def build_bulletin_metadata(
    context: GlobalContext | Mapping[str, Any],
    *,
    extra: Mapping[str, Any] | None = None,
) -> BulletinMetadata:
    """Сформировать метаданные выпуска из GlobalContext."""

    ctx = context if isinstance(context, GlobalContext) else GlobalContext.model_validate(context)
    origin_date = ctx.origin.origin_date.isoformat()
    bulletin_id = f"ai4epi-{ctx.origin.iso_year}-W{ctx.origin.iso_week:02d}-{origin_date}"
    return BulletinMetadata(
        bulletin_id=bulletin_id,
        origin_date=origin_date,
        iso_year=ctx.origin.iso_year,
        iso_week=ctx.origin.iso_week,
        unit=ctx.unit,
        per_population=ctx.per_population,
        context_blocks=ctx.block_names(),
        extra=dict(extra or {}),
    )


def build_bulletin_from_results(
    *,
    context: GlobalContext | Mapping[str, Any],
    results: Mapping[str, SectionNarrationResult | Mapping[str, Any]],
    registry: SectionRegistry | None = None,
    runtime_checker: RuntimeChecker | None = None,
    run_runtime_checks: bool = True,
    require_complete: bool = True,
    validate_section_payloads: bool = True,
    assets: Mapping[str, Any] | None = None,
    metadata_extra: Mapping[str, Any] | None = None,
    editorial_notes: Sequence[str] | None = None,
) -> Bulletin:
    """Функциональная обёртка над BulletinBuilder."""

    builder = BulletinBuilder(registry=registry, runtime_checker=runtime_checker)
    return builder.build_from_narration_results(
        context=context,
        results=results,
        run_runtime_checks=run_runtime_checks,
        require_complete=require_complete,
        validate_section_payloads=validate_section_payloads,
        assets=assets,
        metadata_extra=metadata_extra,
        editorial_notes=editorial_notes,
    )


def flatten_text(value: Any) -> str:
    """Рекурсивно извлечь текстовые leaf-поля из JSON-подобного объекта."""

    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        parts = [flatten_text(item) for item in value.values()]
        return "\n\n".join(part for part in parts if part.strip())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        parts = [flatten_text(item) for item in value]
        return "\n\n".join(part for part in parts if part.strip())
    return "" if value is None else str(value)


def save_bulletin(bulletin: Bulletin, path: str | Path, *, indent: int = 2) -> None:
    """Сохранить Bulletin как UTF-8 JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(bulletin.to_public_dict(), stream, ensure_ascii=False, indent=indent)


def load_bulletin(path: str | Path) -> Bulletin:
    """Загрузить и валидировать Bulletin из JSON-файла."""

    input_path = Path(path)
    with input_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    return Bulletin.model_validate(data)


def save_bulletin_markdown(
    bulletin: Bulletin,
    path: str | Path,
    *,
    include_metadata: bool = False,
) -> None:
    """Сохранить человекочитаемое markdown-представление бюллетеня."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(bulletin.to_markdown(include_metadata=include_metadata), encoding="utf-8")


def _coerce_narration_result(
    result: SectionNarrationResult | Mapping[str, Any],
) -> tuple[JsonObject | None, list[JsonObject], str | None, bool]:
    if isinstance(result, SectionNarrationResult):
        return result.draft, list(result.trace), result.failure_reason, result.status == "ok"

    if not isinstance(result, Mapping):
        raise BulletinBuildError(f"Unsupported narration result type: {type(result).__name__}.")

    status = str(result.get("status", "ok"))
    draft = result.get("draft")
    trace_raw = result.get("trace", [])
    failure_reason = result.get("failure_reason")

    if draft is not None and not isinstance(draft, Mapping):
        raise BulletinBuildError("Narration result field 'draft' must be an object or None.")
    if not isinstance(trace_raw, Sequence) or isinstance(trace_raw, (str, bytes, bytearray)):
        raise BulletinBuildError("Narration result field 'trace' must be a sequence.")

    trace = [dict(item) for item in trace_raw if isinstance(item, Mapping)]
    ok = status == "ok" and draft is not None
    return (dict(draft) if draft is not None else None), trace, None if failure_reason is None else str(failure_reason), ok


def _validate_section_payload(section: SectionConfig, payload: Mapping[str, Any]) -> None:
    try:
        validate_llm_payload(dict(payload), section.output_schema)
    except LLMResponseValidationError as exc:
        raise BulletinBuildError(
            f"Payload of section {section.section_id!r} does not satisfy output_schema: {exc}"
        ) from exc


def _to_jsonable_object(value: Mapping[str, Any]) -> JsonObject:
    converted = _to_jsonable(value)
    if not isinstance(converted, dict):
        raise BulletinBuildError("Section payload must be converted to a JSON object.")
    return converted


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True, exclude_none=False)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_jsonable(item) for item in value]
    return value


__all__ = [
    "Bulletin",
    "BulletinBuildError",
    "BulletinBuilder",
    "BulletinMetadata",
    "BulletinSection",
    "BulletinStatus",
    "JsonObject",
    "build_bulletin_from_results",
    "build_bulletin_metadata",
    "flatten_text",
    "load_bulletin",
    "save_bulletin",
    "save_bulletin_markdown",
]

