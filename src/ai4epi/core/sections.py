"""
Конфигурация секций для пайплайна ai4epi.

Модуль фиксирует контракт между глобальным контекстом и секционными
narrator-агентами. Он не выполняет LLM-запросы: его задача — описать,
какие секции существуют, какие поля JSON они должны вернуть, какие
переменные из GlobalContext им доступны и как из этих переменных собрать
строго структурированный пользовательский payload.

Добавление новой секции в типовом случае сводится к регистрации нового
SectionConfig: указать section_id, prompt, output_schema, evidence_key и
context_mapping. Если нужные данные уже находятся в GlobalContext, менять
ядро пайплайна не требуется.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4epi.core.context import ContextPathError, GlobalContext


ContextPath = str | Sequence[str | int]
JsonObject = dict[str, Any]


class SectionConfigError(ValueError):
    """Ошибка конфигурации секции."""


class SectionRegistryError(ValueError):
    """Ошибка реестра секций."""


class StrictModel(BaseModel):
    """Базовая модель с закрытым контрактом."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        arbitrary_types_allowed=False,
    )


class SectionConfig(StrictModel):
    """
    Декларативное описание одной секции бюллетеня.

    Parameters
    ----------
    section_id:
        Машинное имя секции. Используется в trace, runtime-checks,
        evaluator/editor и итоговом bulletin.sections.
    title:
        Человекочитаемый заголовок секции.
    prompt:
        System prompt секционного narrator-агента.
    output_schema:
        JSON Schema ответа narrator-агента. На этом уровне проверяется
        форма схемы; полная валидация ответа выполняется в narrator-слое.
    context_mapping:
        Отображение ``имя_поля_evidence -> путь_в_GlobalContext``.
        Путь может быть строкой с dot-notation, например
        ``forecast.horizons.0.point_forecast``, или последовательностью
        сегментов, например ``["forecast", "horizons", 0, "point_forecast"]``.
    evidence_key:
        Имя ключа в пользовательском payload, под которым будет передан
        evidence-packet секции.
    order:
        Порядок секции в пайплайне.
    max_tokens:
        Верхняя граница длины ответа narrator-агента.
    activation_path:
        Необязательный путь в GlobalContext. Если путь отсутствует или даёт
        ``None``, секция считается недоступной для данного запуска.
    include_prompt_mode:
        Добавлять ли поле ``mode`` в пользовательский payload.
    feedback_key:
        Имя поля, через которое narrator-слой может передать обратную связь
        для исправления ответа.
    """

    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str = Field(min_length=1)
    prompt: str = Field(min_length=1)
    output_schema: JsonObject
    context_mapping: dict[str, ContextPath] = Field(default_factory=dict)
    evidence_key: str = Field(default="structured_evidence", pattern=r"^[a-z][a-z0-9_]*$")
    order: int = Field(ge=0)
    max_tokens: int = Field(default=700, ge=1)
    activation_path: ContextPath | None = None
    include_prompt_mode: bool = True
    feedback_key: str = Field(default="feedback_to_fix", pattern=r"^[a-z][a-z0-9_]*$")
    enabled: bool = True
    required: bool = True
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("prompt")
    @classmethod
    def normalize_prompt(cls, value: str) -> str:
        text = " ".join(value.split())
        if not text:
            raise ValueError("prompt must not be blank.")
        return text

    @field_validator("context_mapping")
    @classmethod
    def validate_context_mapping(cls, value: dict[str, ContextPath]) -> dict[str, ContextPath]:
        for output_key, path in value.items():
            if not output_key or not isinstance(output_key, str):
                raise ValueError("context_mapping keys must be non-empty strings.")
            _validate_context_path(path)
        return value

    @field_validator("activation_path")
    @classmethod
    def validate_activation_path(cls, value: ContextPath | None) -> ContextPath | None:
        if value is not None:
            _validate_context_path(value)
        return value

    @field_validator("output_schema")
    @classmethod
    def validate_output_schema(cls, value: JsonObject) -> JsonObject:
        _validate_json_object_schema(value)
        return value

    def is_available(self, context: GlobalContext) -> bool:
        """
        Проверить, может ли секция быть запущена на данном GlobalContext.

        Если activation_path не задан, секция считается доступной. Если путь
        задан, секция доступна только при успешном разрешении пути и значении,
        отличном от None.
        """

        if not self.enabled:
            return False
        if self.activation_path is None:
            return True
        try:
            return context.resolve_path(self.activation_path) is not None
        except ContextPathError:
            return False

    def build_evidence(self, context: GlobalContext) -> JsonObject:
        """
        Собрать evidence-packet секции из GlobalContext.

        Все пути из context_mapping являются строгими: если хотя бы один путь
        не разрешается, выбрасывается ContextPathError. Это предотвращает
        тихое выпадение фактов из промпта.
        """

        return _to_jsonable(context.project(self.context_mapping))

    def build_user_payload(
        self,
        context: GlobalContext,
        *,
        prompt_mode: str | None = None,
        feedback: str | None = None,
    ) -> JsonObject:
        """
        Сформировать пользовательский payload для narrator-агента.

        Структура payload совпадает с идеей notebook: поле mode, затем
        секционный evidence-packet под именованным ключом, затем при
        необходимости feedback_to_fix.
        """

        if not self.is_available(context):
            raise SectionConfigError(f"Section {self.section_id!r} is not available for this context.")

        payload: JsonObject = {}
        if self.include_prompt_mode and prompt_mode is not None:
            payload["mode"] = prompt_mode
        payload[self.evidence_key] = self.build_evidence(context)
        if feedback:
            payload[self.feedback_key] = feedback
        return payload

    def validate_against_context(self, context: GlobalContext) -> None:
        """
        Проверить, что секция может быть собрана из GlobalContext.

        Метод не вызывает LLM и не проверяет качество текста. Он гарантирует,
        что все заявленные пути в контекст существуют и разрешаются.
        """

        if not self.is_available(context):
            if self.required:
                raise SectionConfigError(f"Required section {self.section_id!r} is not available.")
            return
        self.build_evidence(context)

    @property
    def output_fields(self) -> tuple[str, ...]:
        """Вернуть поля, которые narrator обязан сгенерировать."""

        required = self.output_schema.get("required", [])
        return tuple(str(item) for item in required)


class SectionRegistry(StrictModel):
    """Упорядоченный реестр секций пайплайна."""

    sections: list[SectionConfig]

    @model_validator(mode="after")
    def validate_unique_sections(self) -> "SectionRegistry":
        ids = [section.section_id for section in self.sections]
        duplicated_ids = sorted({section_id for section_id in ids if ids.count(section_id) > 1})
        if duplicated_ids:
            raise SectionRegistryError(f"Duplicated section_id values: {duplicated_ids!r}.")

        orders = [section.order for section in self.sections if section.enabled]
        duplicated_orders = sorted({order for order in orders if orders.count(order) > 1})
        if duplicated_orders:
            raise SectionRegistryError(f"Duplicated enabled section order values: {duplicated_orders!r}.")

        return self

    def ordered_sections(self, *, include_disabled: bool = False) -> list[SectionConfig]:
        """Вернуть секции в порядке исполнения."""

        sections = self.sections if include_disabled else [section for section in self.sections if section.enabled]
        return sorted(sections, key=lambda section: (section.order, section.section_id))

    def section_ids(self, *, include_disabled: bool = False) -> list[str]:
        """Вернуть machine-readable имена секций в порядке исполнения."""

        return [section.section_id for section in self.ordered_sections(include_disabled=include_disabled)]

    def get(self, section_id: str) -> SectionConfig:
        """Вернуть секцию по section_id."""

        for section in self.sections:
            if section.section_id == section_id:
                return section
        raise SectionRegistryError(f"Unknown section_id: {section_id!r}.")

    def register(self, section: SectionConfig, *, replace: bool = False) -> "SectionRegistry":
        """
        Вернуть новый реестр с добавленной секцией.

        Реестр иммутабельно пересоздаётся через model_validate, поэтому все
        проверки уникальности section_id и order выполняются автоматически.
        """

        existing = [item for item in self.sections if item.section_id == section.section_id]
        if existing and not replace:
            raise SectionRegistryError(f"Section {section.section_id!r} already exists.")

        if replace:
            new_sections = [item for item in self.sections if item.section_id != section.section_id]
        else:
            new_sections = list(self.sections)
        new_sections.append(section)
        return SectionRegistry(sections=new_sections)

    def available_sections(self, context: GlobalContext) -> list[SectionConfig]:
        """Вернуть включённые секции, доступные для данного GlobalContext."""

        return [section for section in self.ordered_sections() if section.is_available(context)]

    def validate_against_context(self, context: GlobalContext, *, include_disabled: bool = False) -> None:
        """Проверить все секции реестра относительно GlobalContext."""

        for section in self.ordered_sections(include_disabled=include_disabled):
            section.validate_against_context(context)

    def build_user_payloads(
        self,
        context: GlobalContext,
        *,
        prompt_mode: str | None = None,
        feedback_by_section: Mapping[str, str] | None = None,
    ) -> dict[str, JsonObject]:
        """Собрать пользовательские payload для всех доступных секций."""

        feedback_by_section = feedback_by_section or {}
        return {
            section.section_id: section.build_user_payload(
                context,
                prompt_mode=prompt_mode,
                feedback=feedback_by_section.get(section.section_id),
            )
            for section in self.available_sections(context)
        }

    def to_public_dict(self) -> JsonObject:
        """Вернуть JSON-сериализуемый вид реестра."""

        return self.model_dump(mode="json", exclude_none=False)


def _to_jsonable(value: Any) -> Any:
    """
    Рекурсивно привести pydantic-модели и контейнеры к JSON-совместимому виду.
    """

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True, exclude_none=False)
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_jsonable(item) for item in value]
    return value


def _validate_context_path(path: ContextPath) -> None:
    if isinstance(path, str):
        if not path.strip():
            raise ValueError("Context path must not be blank.")
        if ".." in path:
            raise ValueError(f"Context path contains an empty segment: {path!r}.")
        return

    if isinstance(path, Sequence) and not isinstance(path, (str, bytes, bytearray)):
        if not path:
            raise ValueError("Context path sequence must not be empty.")
        for segment in path:
            if not isinstance(segment, (str, int)):
                raise ValueError(f"Context path segment must be str or int; got {type(segment).__name__}.")
        return

    raise ValueError("Context path must be a dot-separated string or a sequence of str/int segments.")


def _validate_json_object_schema(schema: Mapping[str, Any]) -> None:
    if schema.get("type") != "object":
        raise ValueError("Section output_schema must be a JSON object schema.")

    properties = schema.get("properties")
    if not isinstance(properties, Mapping) or not properties:
        raise ValueError("Section output_schema must define non-empty properties.")

    required = schema.get("required")
    if not isinstance(required, list) or not required:
        raise ValueError("Section output_schema must define non-empty required list.")

    missing = [field for field in required if field not in properties]
    if missing:
        raise ValueError(f"Required fields are absent from output_schema.properties: {missing!r}.")

    if schema.get("additionalProperties") is not False:
        raise ValueError("Section output_schema must set additionalProperties=false.")


def make_text_object_schema(*fields: str) -> JsonObject:
    """
    Создать строгую JSON Schema для объекта, состоящего из строковых полей.

    Функция используется и в дефолтных секциях, и при добавлении новых
    простых narrative-секций.
    """

    if not fields:
        raise SectionConfigError("At least one output field is required.")
    properties = {field: {"type": "string"} for field in fields}
    return {
        "type": "object",
        "properties": properties,
        "required": list(fields),
        "additionalProperties": False,
    }


CURRENT_SITUATION_PROMPT = (
    "Ты Narrator — автор еженедельного бюллетеня по гриппу и ОРВИ. "
    "Сгенерируй два абзаца для раздела 'Текущая ситуация'. "
    "Верни JSON с полями paragraph_situation и paragraph_forecast_brief. "
    "АБЗАЦ 1 (paragraph_situation): по значениям current_value и previous_value "
    "определи изменение относительно предыдущей недели; обязательно укажи текущее "
    "наблюдаемое значение current_value в явном виде. Затем кратко охарактеризуй "
    "динамику последних четырёх недель по ряду trend_4w_values и при необходимости "
    "по полю trend_4w_label. "
    "АБЗАЦ 2 (paragraph_forecast_brief): начни СТРОГО фразой "
    "'Прогноз на четыре недели показывает ...'. Далее по weekly_point_forecast и, "
    "при необходимости, по human-readable полю forecast_dynamics_description кратко "
    "опиши форму 4-недельной траектории. Если внутри траектории есть смена направления, "
    "обязательно отрази её явно. "
    "ПРАВИЛА: используй только данные из JSON-контекста; не придумывай дополнительные "
    "факты; используй только недельную шкалу времени — не пиши про дни, сутки, вчера, "
    "сегодня или месяцы; стиль официальный, нейтральный, без списков."
)

EPIDEMIC_WAVE_COMPARISON_PROMPT = (
    "Ты Narrator — автор аналитического подпункта к рисунку о сравнении трёх последних "
    "эпидемических волн. Сгенерируй ОДИН плотный абзац и верни JSON {comparison_text}. "
    "Сравни последнюю волну с двумя предыдущими сезонами только по structured evidence bundle. "
    "Обязательно отрази: 1) сравнительную высоту пиков, 2) положение пиков по неделям, "
    "3) ширину волны по FWHM или, если полная ширина последней волны ещё не наблюдалась, "
    "явное указание на это. Допустимо кратко описать асимметрию волны и суммарную сезонную "
    "нагрузку, если это поддержано данными. Запрещено упоминать MEM, базовые линии, "
    "эпидемические пороги, формальное начало или окончание эпидемии, а также причинные "
    "объяснения. Текст должен быть в 4–6 предложениях, без списков, официальным и нейтральным."
)

AGE_GROUP_SEASON_OVERVIEW_PROMPT = (
    "Ты Narrator — автор одного аналитического подпункта бюллетеня. "
    "Верни строго один JSON-объект с единственным ключом overview_text. "
    "Значение overview_text — один плотный абзац на русском языке в официальном нейтральном стиле. "
    "Используй только structured evidence bundle. Обязательно отрази: "
    "1) у какой возрастной группы максимальный пик на 10 тыс. населения; "
    "2) у какой группы максимальная накопленная сезонная заболеваемость; "
    "3) у какой возрастной группы максимальна ширина главного пика на уровне 50% prominence, "
    "если это поддержано данными; "
    "4) как распределяется вклад возрастных групп в общий объём зарегистрированных случаев ОРВИ. "
    "Если latest_peak_group отсутствует или равен null, не сравнивай группы по времени пика. "
    "Не называй строку «Все население» возрастной группой. Не смешивай эту секцию с прогнозом, "
    "лабораторной реконструкцией гриппа и другими разделами бюллетеня. Не выдумывай числа, "
    "причины и медицинские интерпретации."
)

FORECAST_RISKS_PROMPT = (
    "Ты Narrator — автор аналитического раздела бюллетеня по гриппу. "
    "Сгенерируй три абзаца для раздела 'Прогноз и оценка рисков'. "
    "Верни JSON: {point_forecast_text, uncertainty_text, risk_assessment}. "
    "АБЗАЦ 1 (point_forecast_text): по weekly_forecast и, при необходимости, "
    "по human-readable полю dynamics_description опиши точечный прогноз на 4 недели. "
    "Если внутри траектории есть промежуточный рост или промежуточное снижение, обязательно "
    "отрази это явно; не своди немонотонную траекторию к одному слову. "
    "АБЗАЦ 2 (uncertainty_text): объясни неопределённость прогноза по границам интервалов, "
    "их ширине, максимуму max_interval_width, относительной неопределённости "
    "relative_uncertainty_pct и, при необходимости, по полю uncertainty_label. Используй термин "
    "'прогнозный интервал'. Не используй термины 'доверительный интервал' или "
    "'интервал достоверности'. Объясни, что верхняя граница прогноза соответствует "
    "неблагоприятному сценарию с более высокой заболеваемостью, а нижняя граница, равная нулю, "
    "не является неблагоприятным исходом. Кратко поясни значение ширины интервалов для лица, "
    "принимающего решения. "
    "АБЗАЦ 3 (risk_assessment): дай сдержанную оценку риска для недельного мониторинга "
    "на ближайшие 4 недели. Не выходи за рамки данных и не делай причинных выводов. "
    "ПРАВИЛА: используй только данные из JSON-контекста; не придумывай дополнительные факты; "
    "используй только недельную шкалу времени — не пиши про дни, сутки, вчера, сегодня или месяцы; "
    "стиль официальный и нейтральный."
)

SHAP_INTERPRETATION_PROMPT = (
    "Ты Narrator — автор раздела 'Интерпретация модели' в эпиднадзорном бюллетене. "
    "Объясни, какие факторы влияют на прогноз, используя SHAP-данные. "
    "Верни JSON: {short_term_factors, long_term_factors, overall_insight}. "
    "АБЗАЦ 1 (short_term_factors): опиши top-факторы для краткосрочного прогноза h=1, "
    "опираясь на short_term_features. Для факторов с признаком направление_надёжное=false "
    "не утверждай фиксированное повышение или понижение прогноза; пиши, что влияние зависит "
    "от текущей эпидемической ситуации. "
    "АБЗАЦ 2 (long_term_factors): аналогично опиши top-факторы для горизонта h=4, "
    "опираясь на long_term_features. "
    "АБЗАЦ 3 (overall_insight): сравни горизонты h=1 и h=4 и сформулируй общий вывод "
    "о смене структуры факторов, используя key_insight только как опорное human-readable "
    "резюме, а не как готовую фразу. Допустимы только три типа утверждений: "
    "1) относительная важность факторов, 2) различия между горизонтами, "
    "3) устойчивость или контекстная зависимость направления влияния. "
    "Запрещены причинные объяснения, домыслы и выводы вне данных. "
    "ПРАВИЛА: h=1 и h=4 означают 1 и 4 недели вперёд; не пиши про дни или месяцы. "
    "Не используй английские термины. Вместо 'SHAP-значение' пиши 'вклад фактора' "
    "или 'степень влияния'. Сохраняй официальный стиль."
)

MODEL_QUALITY_PROMPT = (
    "Ты Narrator — автор раздела 'Качество модели и ограничения'. "
    "Верни JSON: {quality_summary, limitations_text}. "
    "АБЗАЦ 1 (quality_summary): кратко охарактеризуй качество модели по метрикам "
    "на разных горизонтах и по статистике ошибок для h=1. Используй только факты. "
    "Не используй оценочные слова вроде 'хороший', 'сильный' или 'отличный'. "
    "АБЗАЦ 2 (limitations_text): сформулируй ограничения модели по полю "
    "bias_description_ru, по месяцам peak_error_months и по статистике ошибок. "
    "Объясни, что это означает для практического использования прогноза. "
    "Не добавляй недоказанные причины ошибок. "
    "СТИЛЬ: нейтральный, для специалиста. Аббревиатуры MAE, RMSE и R² можно "
    "использовать как общепринятые обозначения."
)

MODEL_DESCRIPTION_PROMPT = (
    "Ты Narrator — автор раздела 'Описание модели'. На основе структурированной "
    "карточки модели напиши компактное описание для бюллетеня. "
    "Верни JSON: {description_text}. "
    "Требования к тексту: 1) 2–3 предложения, официальный нейтральный стиль. "
    "2) Обязательно отрази тип модели, стратегию многошагового прогноза, диапазон "
    "горизонтов в неделях, основные группы признаков и стартовый год калибровки. "
    "3) Если поле отсутствует или неизвестно, опирайся только на доступные данные "
    "и не придумывай недостающие свойства. 4) Не используй англоязычные имена классов "
    "и кодовые идентификаторы. Не добавляй маркетинговые или оценочные утверждения "
    "о качестве модели. 5) Текст должен оставаться корректным и для других семейств "
    "моделей, а не только для текущей реализации."
)


def default_section_configs() -> list[SectionConfig]:
    """Вернуть дефолтные секции, перенесённые из notebook."""

    return [
        SectionConfig(
            section_id="current_situation",
            title="Текущая ситуация",
            prompt=CURRENT_SITUATION_PROMPT,
            output_schema=make_text_object_schema("paragraph_situation", "paragraph_forecast_brief"),
            evidence_key="current_situation_context",
            order=10,
            max_tokens=700,
            context_mapping={
                "iso_week": "origin.iso_week",
                "iso_year": "origin.iso_year",
                "unit": "unit",
                "current_value": "current_situation.current_value",
                "previous_value": "current_situation.previous_value",
                "change_pct": "current_situation.change_pct",
                "direction_word": "current_situation.direction_word",
                "trend_4w_values": "current_situation.trend_4w_values",
                "trend_4w_label": "current_situation.trend_4w_label",
                "weekly_point_forecast": "forecast.horizons",
                "forecast_trend_label": "forecast.trend_label",
                "forecast_dynamics_description": "forecast.dynamics_description",
            },
        ),
        SectionConfig(
            section_id="epidemic_wave_comparison",
            title="Сравнение эпидемических волн",
            prompt=EPIDEMIC_WAVE_COMPARISON_PROMPT,
            output_schema=make_text_object_schema("comparison_text"),
            evidence_key="wave_comparison_context",
            order=20,
            max_tokens=900,
            context_mapping={
                "series_name": "epidemic_wave_comparison.series_name",
                "series_label_ru": "epidemic_wave_comparison.series_label_ru",
                "season_definition": "epidemic_wave_comparison.season_definition",
                "width_definition": "epidemic_wave_comparison.width_definition",
                "smoothing": "epidemic_wave_comparison.smoothing",
                "season_labels": "epidemic_wave_comparison.season_labels",
                "waves": "epidemic_wave_comparison.waves",
                "latest_vs_previous": "epidemic_wave_comparison.latest_vs_previous",
                "peak_ranking": "epidemic_wave_comparison.peak_ranking",
                "width_ranking_complete": "epidemic_wave_comparison.width_ranking_complete",
                "latest_wave_status": "epidemic_wave_comparison.latest_wave_status",
                "allowed_claims": "epidemic_wave_comparison.allowed_claims",
                "forbidden_claims": "epidemic_wave_comparison.forbidden_claims",
            },
        ),
        SectionConfig(
            section_id="age_group_season_overview",
            title="Возрастная структура зарегистрированной заболеваемости ОРВИ",
            prompt=AGE_GROUP_SEASON_OVERVIEW_PROMPT,
            output_schema=make_text_object_schema("overview_text"),
            evidence_key="age_group_season_context",
            order=30,
            max_tokens=500,
            activation_path="age_group_season",
            required=False,
            context_mapping={
                "series_label_ru": "age_group_season.metric_label_ru",
                "season_label": "age_group_season.season_label",
                "width_definition": "age_group_season.width_definition",
                "rows": "age_group_season.rows",
                "derived_findings": "age_group_season.derived_findings",
                "width_undefined_note": "age_group_season.peak_width_undefined_note",
                "comparison_scope_note": "age_group_season.comparison_scope_note",
                "semantic_codes": "age_group_season.semantic",
            },
        ),
        SectionConfig(
            section_id="forecast_risks",
            title="Прогноз и оценка рисков",
            prompt=FORECAST_RISKS_PROMPT,
            output_schema=make_text_object_schema("point_forecast_text", "uncertainty_text", "risk_assessment"),
            evidence_key="forecast_context",
            order=40,
            max_tokens=900,
            context_mapping={
                "weekly_forecast": "forecast.horizons",
                "trend_label": "forecast.trend_label",
                "dynamics_description": "forecast.dynamics_description",
                "uncertainty_label": "forecast.uncertainty_label",
                "relative_uncertainty_pct": "forecast.relative_uncertainty_pct",
                "max_interval_width": "forecast.max_interval_width",
                "semantic": "forecast.semantic",
            },
        ),
        SectionConfig(
            section_id="shap_interpretation",
            title="Интерпретация модели",
            prompt=SHAP_INTERPRETATION_PROMPT,
            output_schema=make_text_object_schema("short_term_factors", "long_term_factors", "overall_insight"),
            evidence_key="shap_context",
            order=50,
            max_tokens=1000,
            context_mapping={
                "short_term_features": "shap_summary.by_horizon.1",
                "long_term_features": "shap_summary.by_horizon.4",
                "key_insight": "shap_summary.key_insight",
                "semantic": "shap_summary.semantic",
            },
        ),
        SectionConfig(
            section_id="model_quality",
            title="Качество модели и ограничения",
            prompt=MODEL_QUALITY_PROMPT,
            output_schema=make_text_object_schema("quality_summary", "limitations_text"),
            evidence_key="model_quality_context",
            order=60,
            max_tokens=800,
            context_mapping={
                "metrics": "model_quality.metrics",
                "error_stats_h1": "model_quality.error_stats_h1",
                "bias_description_ru": "model_quality.worst_case_pattern",
                "semantic": "model_quality.semantic",
            },
        ),
        SectionConfig(
            section_id="model_description",
            title="Описание модели",
            prompt=MODEL_DESCRIPTION_PROMPT,
            output_schema=make_text_object_schema("description_text"),
            evidence_key="model_card",
            order=70,
            max_tokens=700,
            context_mapping={
                "family_ru": "model_info.family_ru",
                "strategy_ru": "model_info.strategy_ru",
                "task_type_ru": "model_info.task_type_ru",
                "forecast_target_ru": "model_info.forecast_target_ru",
                "horizons": "model_info.horizons",
                "feature_groups": "model_info.feature_groups",
                "feature_groups_ru": "model_info.feature_groups_ru",
                "calibration_start_year": "model_info.calibration_start_year",
                "calibration_data_description": "model_info.calibration_data_description",
                "semantic": "model_info.semantic",
            },
        ),
    ]


def default_section_registry() -> SectionRegistry:
    """Вернуть дефолтный реестр секций."""

    return SectionRegistry(sections=deepcopy(default_section_configs()))


def load_section_registry(path: str | Path) -> SectionRegistry:
    """Загрузить реестр секций из JSON-файла."""

    registry_path = Path(path)
    with registry_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    return SectionRegistry.model_validate(data)


def save_section_registry(registry: SectionRegistry, path: str | Path, *, indent: int = 2) -> None:
    """Сохранить реестр секций в UTF-8 JSON."""

    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("w", encoding="utf-8") as stream:
        json.dump(registry.to_public_dict(), stream, ensure_ascii=False, indent=indent)


__all__ = [
    "AGE_GROUP_SEASON_OVERVIEW_PROMPT",
    "CURRENT_SITUATION_PROMPT",
    "ContextPath",
    "EPIDEMIC_WAVE_COMPARISON_PROMPT",
    "FORECAST_RISKS_PROMPT",
    "JsonObject",
    "MODEL_DESCRIPTION_PROMPT",
    "MODEL_QUALITY_PROMPT",
    "SHAP_INTERPRETATION_PROMPT",
    "SectionConfig",
    "SectionConfigError",
    "SectionRegistry",
    "SectionRegistryError",
    "default_section_configs",
    "default_section_registry",
    "load_section_registry",
    "make_text_object_schema",
    "save_section_registry",
]

