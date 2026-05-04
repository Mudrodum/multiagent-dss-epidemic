"""
Секционный narrator-слой пайплайна ai4epi.

Модуль отвечает только за исполнение секционных narrator-агентов:
- формирование сообщений для LLM по SectionConfig;
- извлечение JSON из ответа модели;
- строгую проверку ответа по JSON Schema секции;
- локальный repair-проход при нарушении JSON-контракта;
- трассировку попыток генерации.

Модуль не собирает итоговый бюллетень, не выполняет deterministic/text checks
и не редактирует готовый текст. Эти обязанности принадлежат последующим слоям
пайплайна.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Mapping, Protocol, Sequence

from ai4epi.core.context import GlobalContext
from ai4epi.core.sections import SectionConfig, SectionRegistry


JsonObject = dict[str, Any]
ChatMessage = Mapping[str, str]
PromptMode = Literal["research", "assisted"]
SectionStatus = Literal["ok", "failed"]


class JsonExtractionError(ValueError):
    """Ответ LLM не содержит корректного JSON-объекта."""


class LLMResponseValidationError(ValueError):
    """Ответ LLM не соответствует JSON Schema секции."""


class SectionGenerationError(RuntimeError):
    """Секция не была сгенерирована в валидном структурированном виде."""

    def __init__(self, message: str, *, trace: Sequence[Mapping[str, Any]] | None = None) -> None:
        super().__init__(message)
        self.trace = list(trace or [])


class ChatBackend(Protocol):
    """
    Минимальный контракт LLM-бэкенда.

    Метод совместим с локальным Ollama API и с backend-классом из исходного
    notebook: schema-aware режим передаётся через аргумент ``format``.
    """

    model: str

    def chat(
        self,
        messages: Sequence[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 2000,
        format: JsonObject | str | None = None,
        think: bool | None = None,
        timeout: int | float | None = None,
    ) -> Mapping[str, Any] | str:
        """Вернуть ответ модели. Ожидается поле content или строка."""


@dataclass(frozen=True)
class OllamaChatBackend:
    """
    Реальный backend для локального Ollama API.

    Parameters
    ----------
    model:
        Имя модели в Ollama, например ``qwen3.5:9b``.
    base_url:
        Базовый URL Ollama server API.
    default_timeout:
        Таймаут одного HTTP-запроса в секундах.
    """

    model: str
    base_url: str = "http://localhost:11434"
    default_timeout: int = 180

    def chat(
        self,
        messages: Sequence[ChatMessage],
        temperature: float = 0.0,
        max_tokens: int = 2000,
        format: JsonObject | str | None = None,
        think: bool | None = None,
        timeout: int | float | None = None,
    ) -> Mapping[str, Any]:
        try:
            import requests
        except ImportError as exc:  # pragma: no cover - зависит от окружения установки
            raise RuntimeError("Для OllamaChatBackend требуется пакет requests.") from exc

        payload: JsonObject = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "options": {
                "temperature": float(temperature),
                "num_predict": int(max_tokens),
            },
            "stream": False,
        }
        if format is not None:
            payload["format"] = format
        if think is not None:
            payload["think"] = bool(think)

        response = requests.post(
            f"{self.base_url.rstrip('/')}/api/chat",
            json=payload,
            timeout=timeout if timeout is not None else self.default_timeout,
        )
        response.raise_for_status()
        data = response.json()
        message = data.get("message") or {}
        content = message.get("content") or ""
        if not str(content).strip() and isinstance(message.get("thinking"), str):
            content = message["thinking"]
        return {"content": content, "raw": data}


@dataclass(frozen=True)
class NarratorAttempt:
    """
    Описание одной попытки получить валидный JSON от LLM.

    ``use_schema_format=True`` означает, что JSON Schema секции будет передана
    в backend как structured-output формат, если backend это поддерживает.
    """

    label: str
    temperature: float = 0.0
    use_schema_format: bool = True


@dataclass(frozen=True)
class NarratorSettings:
    """Настройки секционного narrator-слоя."""

    prompt_mode: PromptMode = "research"
    think: bool | None = False
    request_timeout_sec: int | None = None
    raw_preview_chars: int = 800
    attempts: tuple[NarratorAttempt, ...] = (
        NarratorAttempt("structured_output", temperature=0.0, use_schema_format=True),
        NarratorAttempt("repair", temperature=0.0, use_schema_format=True),
        NarratorAttempt("no_format", temperature=0.0, use_schema_format=False),
    )

    def __post_init__(self) -> None:
        if self.prompt_mode not in {"research", "assisted"}:
            raise ValueError("prompt_mode must be 'research' or 'assisted'.")
        if self.raw_preview_chars <= 0:
            raise ValueError("raw_preview_chars must be positive.")
        if not self.attempts:
            raise ValueError("At least one narrator attempt is required.")


@dataclass
class AttemptTrace:
    """Трассировка одной LLM-попытки."""

    label: str
    temperature: float
    schema_format_requested: bool
    success: bool
    latency_sec: float
    raw_content_preview: str = ""
    validation_error: str | None = None

    def to_public_dict(self) -> JsonObject:
        return asdict(self)


@dataclass
class SectionNarrationResult:
    """Результат генерации одной секции."""

    section_id: str
    status: SectionStatus
    draft: JsonObject | None
    trace: list[JsonObject] = field(default_factory=list)
    failure_reason: str | None = None

    def require_ok(self) -> JsonObject:
        """Вернуть draft или выбросить SectionGenerationError."""

        if self.status == "ok" and self.draft is not None:
            return self.draft
        raise SectionGenerationError(
            self.failure_reason or f"Section {self.section_id!r} was not generated.",
            trace=self.trace,
        )


def make_chat_backend(
    backend: str,
    *,
    model: str,
    base_url: str = "http://localhost:11434",
    default_timeout: int = 180,
) -> ChatBackend:
    """Создать LLM backend по имени."""

    if backend == "ollama":
        return OllamaChatBackend(model=model, base_url=base_url, default_timeout=default_timeout)
    raise ValueError(f"Unknown LLM backend: {backend!r}.")


def is_very_small_model_name(model_name: str) -> bool:
    """
    Проверить, относится ли имя модели к очень малым моделям.

    Для таких моделей repair-инструкция сохраняет уже сгенерированный текст и
    исправляет только JSON-формат, потому что повторная полная генерация чаще
    приводит к деградации структуры.
    """

    if not model_name:
        return False
    match = re.search(r":\s*(\d+(?:\.\d+)?)b\s*$", str(model_name).strip().lower())
    if not match:
        return False
    return float(match.group(1)) <= 1.0


def extract_json(text: str) -> JsonObject:
    """
    Извлечь JSON-объект из ответа LLM.

    Порядок разбора соответствует исходному notebook:
    1. прямой ``json.loads``;
    2. JSON внутри markdown fence;
    3. raw_decode с любой позиции ``{``;
    4. внешний сбалансированный объект;
    5. опциональный YAML fallback, если PyYAML установлен.
    """

    source = (text or "").strip()
    if not source:
        raise JsonExtractionError("Пустой ответ LLM.")

    source = re.sub(r"<think>[\s\S]*?</think>", "", source, flags=re.IGNORECASE).strip()

    candidates: list[str] = []
    fence_matches = re.findall(r"```(?:json)?\s*([\s\S]*?)```", source, flags=re.IGNORECASE)
    candidates.extend(candidate.strip() for candidate in fence_matches if candidate.strip())

    stripped = _strip_outer_code_fence(source)
    if stripped and stripped not in candidates:
        candidates.append(stripped)

    if source not in candidates:
        candidates.append(source)

    for candidate in candidates:
        try:
            obj = json.loads(candidate)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj

    for candidate in candidates:
        for obj in _iter_raw_json_objects(candidate):
            return obj

    for candidate in candidates:
        outer = _extract_outermost_json_candidate(candidate)
        if not outer:
            continue
        try:
            obj = json.loads(outer)
        except Exception:
            continue
        if isinstance(obj, dict):
            return obj

    try:
        import yaml  # type: ignore[import-untyped]
    except Exception:
        yaml = None

    if yaml is not None:
        for candidate in candidates:
            variants = [candidate]
            outer = _extract_outermost_json_candidate(candidate)
            if outer and outer != candidate:
                variants.append(outer)
            for variant in variants:
                try:
                    obj = yaml.safe_load(variant)
                except Exception:
                    continue
                if isinstance(obj, dict):
                    return obj

    raise JsonExtractionError("JSON не найден в ответе LLM.")


def validate_llm_payload(obj: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    """
    Проверить объект LLM по ограниченному подмножеству JSON Schema.

    Поддерживаются конструкции, используемые в секционных контрактах:
    ``type``, ``properties``, ``required``, ``additionalProperties``,
    ``items``, ``enum``, ``const``, числовые границы и минимальные длины строк
    или массивов.
    """

    if not isinstance(schema, Mapping):
        return

    if "anyOf" in schema:
        errors = []
        for variant in schema["anyOf"]:
            try:
                validate_llm_payload(obj, variant, path=path)
                return
            except Exception as exc:
                errors.append(str(exc))
        raise LLMResponseValidationError(f"{path}: ни один вариант anyOf не подошёл: {errors!r}.")

    if "oneOf" in schema:
        success_count = 0
        last_error: str | None = None
        for variant in schema["oneOf"]:
            try:
                validate_llm_payload(obj, variant, path=path)
                success_count += 1
            except Exception as exc:
                last_error = str(exc)
        if success_count == 1:
            return
        raise LLMResponseValidationError(
            f"{path}: ожидался ровно один подходящий вариант oneOf, получено {success_count}; "
            f"последняя ошибка: {last_error}"
        )

    if "enum" in schema and obj not in schema["enum"]:
        raise LLMResponseValidationError(f"{path}: значение {obj!r} отсутствует в enum {schema['enum']!r}.")

    if "const" in schema and obj != schema["const"]:
        raise LLMResponseValidationError(f"{path}: ожидалось const={schema['const']!r}, получено {obj!r}.")

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        errors = []
        for one_type in schema_type:
            variant_schema = dict(schema)
            variant_schema["type"] = one_type
            try:
                validate_llm_payload(obj, variant_schema, path=path)
                return
            except Exception as exc:
                errors.append(str(exc))
        raise LLMResponseValidationError(f"{path}: тип не соответствует ни одному из {schema_type!r}: {errors!r}.")

    if schema_type == "null":
        if obj is not None:
            raise LLMResponseValidationError(f"{path}: ожидался null, получено {_type_name(obj)}.")
        return

    if schema_type == "object" or "properties" in schema or "required" in schema:
        _validate_object_payload(obj, schema, path)
        return

    if schema_type == "array":
        _validate_array_payload(obj, schema, path)
        return

    if schema_type == "string":
        if not isinstance(obj, str):
            raise LLMResponseValidationError(f"{path}: ожидалась строка, получено {_type_name(obj)}.")
        min_length = schema.get("minLength")
        if min_length is not None and len(obj) < int(min_length):
            raise LLMResponseValidationError(f"{path}: длина строки меньше minLength={min_length}.")
        if schema.get("nonEmpty", False) and not obj.strip():
            raise LLMResponseValidationError(f"{path}: пустая строка.")
        if "pattern" in schema and re.search(str(schema["pattern"]), obj) is None:
            raise LLMResponseValidationError(f"{path}: строка не соответствует pattern={schema['pattern']!r}.")
        return

    if schema_type == "boolean":
        if not isinstance(obj, bool):
            raise LLMResponseValidationError(f"{path}: ожидался boolean, получено {_type_name(obj)}.")
        return

    if schema_type == "integer":
        if not isinstance(obj, int) or isinstance(obj, bool):
            raise LLMResponseValidationError(f"{path}: ожидался integer, получено {_type_name(obj)}.")
        _validate_numeric_bounds(obj, schema, path)
        return

    if schema_type == "number":
        if not isinstance(obj, (int, float)) or isinstance(obj, bool):
            raise LLMResponseValidationError(f"{path}: ожидался number, получено {_type_name(obj)}.")
        _validate_numeric_bounds(float(obj), schema, path)
        return


def strict_parse_and_validate_json_response(
    *,
    agent_name: str,
    raw_content: str,
    schema: Mapping[str, Any],
) -> JsonObject:
    """Извлечь JSON из raw-ответа LLM и проверить его по схеме."""

    parsed = extract_json(raw_content)
    try:
        validate_llm_payload(parsed, schema)
    except LLMResponseValidationError:
        raise
    except Exception as exc:
        raise LLMResponseValidationError(f"{agent_name}: ошибка валидации JSON: {exc}") from exc
    return parsed


class Narrator:
    """Исполнитель секционных narrator-агентов."""

    def __init__(self, llm: ChatBackend, settings: NarratorSettings | None = None) -> None:
        self.llm = llm
        self.settings = settings or NarratorSettings()

    def generate_section(
        self,
        section: SectionConfig,
        context: GlobalContext,
        *,
        feedback: str | None = None,
    ) -> SectionNarrationResult:
        """
        Сгенерировать одну секцию.

        Метод возвращает объект результата и не выбрасывает исключение при
        ошибке LLM. Для fail-fast поведения используйте ``generate_section_or_raise``.
        """

        user_payload = section.build_user_payload(
            context,
            prompt_mode=self.settings.prompt_mode,
            feedback=feedback,
        )
        try:
            draft, trace = self._call_llm(
                system_prompt=section.prompt,
                user_payload=user_payload,
                schema=section.output_schema,
                max_tokens=section.max_tokens,
                section_id=section.section_id,
            )
            return SectionNarrationResult(
                section_id=section.section_id,
                status="ok",
                draft=draft,
                trace=trace,
            )
        except SectionGenerationError as exc:
            return SectionNarrationResult(
                section_id=section.section_id,
                status="failed",
                draft=None,
                trace=[dict(item) for item in exc.trace],
                failure_reason=str(exc),
            )
        except Exception as exc:
            return SectionNarrationResult(
                section_id=section.section_id,
                status="failed",
                draft=None,
                trace=[],
                failure_reason=str(exc),
            )

    def generate_section_or_raise(
        self,
        section: SectionConfig,
        context: GlobalContext,
        *,
        feedback: str | None = None,
    ) -> JsonObject:
        """Сгенерировать одну секцию и выбросить исключение при ошибке."""

        result = self.generate_section(section, context, feedback=feedback)
        return result.require_ok()

    def generate_all(
        self,
        registry: SectionRegistry,
        context: GlobalContext,
        *,
        feedback_by_section: Mapping[str, str] | None = None,
        fail_fast: bool = False,
    ) -> dict[str, SectionNarrationResult]:
        """
        Сгенерировать все доступные секции из реестра.

        ``fail_fast=False`` позволяет получить частичные результаты по всем
        секциям. ``fail_fast=True`` прерывает выполнение на первой ошибке.
        """

        feedback_by_section = feedback_by_section or {}
        results: dict[str, SectionNarrationResult] = {}

        for section in registry.available_sections(context):
            result = self.generate_section(
                section,
                context,
                feedback=feedback_by_section.get(section.section_id),
            )
            results[section.section_id] = result
            if fail_fast and result.status != "ok":
                raise SectionGenerationError(
                    result.failure_reason or f"Section {section.section_id!r} failed.",
                    trace=result.trace,
                )

        return results

    def _call_llm(
        self,
        *,
        system_prompt: str,
        user_payload: JsonObject,
        schema: Mapping[str, Any],
        max_tokens: int,
        section_id: str,
    ) -> tuple[JsonObject, list[JsonObject]]:
        base_messages: list[ChatMessage] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ]

        trace: list[JsonObject] = []
        errors: list[str] = []
        last_content = ""
        last_error: str | None = None

        for attempt in self.settings.attempts:
            messages = self._messages_for_attempt(
                attempt=attempt,
                base_messages=base_messages,
                system_prompt=system_prompt,
                user_payload=user_payload,
                schema=schema,
                last_content=last_content,
                last_error=last_error,
            )
            started = time.perf_counter()
            schema_format = schema if attempt.use_schema_format else None

            try:
                response = self.llm.chat(
                    messages,
                    temperature=attempt.temperature,
                    max_tokens=max_tokens,
                    format=schema_format,
                    think=self.settings.think,
                    timeout=self.settings.request_timeout_sec,
                )
                content = _extract_content_from_backend_response(response)
                last_content = content
                payload = strict_parse_and_validate_json_response(
                    agent_name=f"narrator:{section_id}:{attempt.label}",
                    raw_content=content,
                    schema=schema,
                )
                trace.append(
                    AttemptTrace(
                        label=attempt.label,
                        temperature=attempt.temperature,
                        schema_format_requested=attempt.use_schema_format,
                        success=True,
                        latency_sec=round(time.perf_counter() - started, 3),
                        raw_content_preview=content[: self.settings.raw_preview_chars],
                        validation_error=None,
                    ).to_public_dict()
                )
                return payload, trace

            except Exception as exc:
                last_error = str(exc)
                errors.append(f"{attempt.label}: {last_error}")
                trace.append(
                    AttemptTrace(
                        label=attempt.label,
                        temperature=attempt.temperature,
                        schema_format_requested=attempt.use_schema_format,
                        success=False,
                        latency_sec=round(time.perf_counter() - started, 3),
                        raw_content_preview=last_content[: self.settings.raw_preview_chars],
                        validation_error=last_error,
                    ).to_public_dict()
                )

        raise SectionGenerationError(
            "LLM не вернула валидный JSON после всех попыток. "
            f"Ошибки: {' | '.join(errors)}. "
            f"Последний ответ: {last_content[: self.settings.raw_preview_chars]}",
            trace=trace,
        )

    def _messages_for_attempt(
        self,
        *,
        attempt: NarratorAttempt,
        base_messages: Sequence[ChatMessage],
        system_prompt: str,
        user_payload: JsonObject,
        schema: Mapping[str, Any],
        last_content: str,
        last_error: str | None,
    ) -> list[ChatMessage]:
        if attempt.label == "structured_output":
            return [dict(message) for message in base_messages]

        if attempt.label == "repair":
            repair_payload = self._repair_payload(
                user_payload=user_payload,
                schema=schema,
                previous_response=last_content,
                schema_error=last_error,
            )
            return [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(repair_payload, ensure_ascii=False)},
            ]

        no_format_payload: JsonObject = {
            "instruction": (
                "Верни строго один JSON-объект указанной структуры. "
                "Никаких markdown-блоков, пояснений, комментариев или текста вне JSON."
            ),
            "schema_error": last_error or "unknown schema error",
            "output_schema": dict(schema),
            "original_data": user_payload,
        }
        if last_content:
            no_format_payload["previous_response"] = last_content

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(no_format_payload, ensure_ascii=False)},
        ]

    def _repair_payload(
        self,
        *,
        user_payload: JsonObject,
        schema: Mapping[str, Any],
        previous_response: str,
        schema_error: str | None,
    ) -> JsonObject:
        model_name = getattr(self.llm, "model", "")
        if is_very_small_model_name(model_name):
            instruction = (
                "Ниже дан предыдущий ответ и ошибка валидации. "
                "Не генерируй раздел заново. Исправь только JSON-структуру, кавычки, "
                "скобки, названия ключей, недостающие поля и лишний текст вне JSON. "
                "Сохрани уже имеющееся содержание максимально близко. "
                "Не добавляй новые факты, которых нет в previous_response или original_data. "
                "Верни строго один JSON-объект указанной схемы и ничего кроме JSON."
            )
        else:
            instruction = (
                "Ниже дан предыдущий неудачный ответ и ошибка валидации. "
                "Сгенерируй ответ заново по исходным данным, но верни строго один "
                "JSON-объект указанной схемы и ничего кроме JSON."
            )

        return {
            "instruction": instruction,
            "schema_error": schema_error or "unknown schema error",
            "output_schema": dict(schema),
            "previous_response": previous_response,
            "original_data": user_payload,
        }


def _strip_outer_code_fence(text: str) -> str:
    value = (text or "").strip()
    value = re.sub(r"^```(?:json)?\s*", "", value, flags=re.IGNORECASE)
    value = re.sub(r"\s*```\s*$", "", value)
    return value.strip()


def _iter_raw_json_objects(text: str):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text or ""):
        try:
            obj, _ = decoder.raw_decode(text[match.start() :])
        except Exception:
            continue
        if isinstance(obj, dict):
            yield obj


def _extract_outermost_json_candidate(text: str) -> str | None:
    value = (text or "").strip()
    if not value:
        return None

    start = value.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for index in range(start, len(value)):
        char = value[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return value[start : index + 1]

    return None


def _validate_object_payload(obj: Any, schema: Mapping[str, Any], path: str) -> None:
    if not isinstance(obj, dict):
        raise LLMResponseValidationError(f"{path}: ожидался object, получено {_type_name(obj)}.")

    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise LLMResponseValidationError(f"{path}: schema.properties должен быть объектом.")

    required = schema.get("required", [])
    if not isinstance(required, Sequence) or isinstance(required, (str, bytes, bytearray)):
        raise LLMResponseValidationError(f"{path}: schema.required должен быть массивом.")

    for key in required:
        if key not in obj:
            raise LLMResponseValidationError(f"{path}: отсутствует обязательный ключ: {key}.")
        validate_llm_payload(obj[key], properties.get(key, {}), path=f"{path}.{key}")

    if schema.get("additionalProperties") is False:
        extra_keys = [key for key in obj if key not in properties]
        if extra_keys:
            raise LLMResponseValidationError(f"{path}: запрещены дополнительные ключи: {extra_keys!r}.")

    additional_schema = schema.get("additionalProperties")
    for key, value in obj.items():
        if key in properties:
            validate_llm_payload(value, properties[key], path=f"{path}.{key}")
        elif isinstance(additional_schema, Mapping):
            validate_llm_payload(value, additional_schema, path=f"{path}.{key}")


def _validate_array_payload(obj: Any, schema: Mapping[str, Any], path: str) -> None:
    if not isinstance(obj, list):
        raise LLMResponseValidationError(f"{path}: ожидался array, получено {_type_name(obj)}.")

    min_items = schema.get("minItems")
    if min_items is not None and len(obj) < int(min_items):
        raise LLMResponseValidationError(f"{path}: длина массива меньше minItems={min_items}.")

    max_items = schema.get("maxItems")
    if max_items is not None and len(obj) > int(max_items):
        raise LLMResponseValidationError(f"{path}: длина массива больше maxItems={max_items}.")

    item_schema = schema.get("items", {})
    if isinstance(item_schema, Mapping):
        for index, item in enumerate(obj):
            validate_llm_payload(item, item_schema, path=f"{path}[{index}]")


def _validate_numeric_bounds(value: float | int, schema: Mapping[str, Any], path: str) -> None:
    if "minimum" in schema and value < schema["minimum"]:
        raise LLMResponseValidationError(f"{path}: значение меньше minimum={schema['minimum']}.")
    if "maximum" in schema and value > schema["maximum"]:
        raise LLMResponseValidationError(f"{path}: значение больше maximum={schema['maximum']}.")
    if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
        raise LLMResponseValidationError(f"{path}: значение не больше exclusiveMinimum={schema['exclusiveMinimum']}.")
    if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
        raise LLMResponseValidationError(f"{path}: значение не меньше exclusiveMaximum={schema['exclusiveMaximum']}.")


def _extract_content_from_backend_response(response: Mapping[str, Any] | str) -> str:
    if isinstance(response, str):
        return response
    content = response.get("content")
    if isinstance(content, str):
        return content
    message = response.get("message")
    if isinstance(message, Mapping):
        content = message.get("content")
        if isinstance(content, str):
            return content
    raise TypeError("LLM backend response must be a string or a mapping with string field 'content'.")


def _type_name(value: Any) -> str:
    if value is None:
        return "null"
    return type(value).__name__


__all__ = [
    "AttemptTrace",
    "ChatBackend",
    "JsonExtractionError",
    "LLMResponseValidationError",
    "Narrator",
    "NarratorAttempt",
    "NarratorSettings",
    "OllamaChatBackend",
    "PromptMode",
    "SectionGenerationError",
    "SectionNarrationResult",
    "extract_json",
    "is_very_small_model_name",
    "make_chat_backend",
    "strict_parse_and_validate_json_response",
    "validate_llm_payload",
]

