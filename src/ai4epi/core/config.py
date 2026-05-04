"""
Декларативная конфигурация ai4epi.

Модуль задаёт пользовательский вход в пакетную систему: пути к данным,
LLM-backend-и, настройки narrator/editor/runtime/evaluation слоёв и правила
сохранения артефактов. Он не содержит доменных эвристик и не добавляет новой
логики генерации; его задача — строго валидировать конфигурацию и собрать уже
существующие компоненты пайплайна.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4epi.generation.editor import EditorSettings
from ai4epi.quality.evaluation import DEFAULT_ENABLED_EVALUATORS, DEFAULT_EVAL_WEIGHTS, EvalConfig
from ai4epi.generation.narrator import ChatBackend, NarratorSettings, make_chat_backend
from ai4epi.generation.pipeline import (
    Ai4EpiPipeline,
    PipelineOutputConfig,
    PipelineRunResult,
    PipelineSettings,
    create_pipeline,
    load_pipeline_registry,
)
from ai4epi.quality.runtime_checks import RuntimeCheckConfig, RuntimeChecker
from ai4epi.core.sections import SectionRegistry


JsonObject = dict[str, Any]
ConfigFormat = Literal["json", "yaml", "yml"]
EvaluationBulletinKind = Literal["final", "base", "edited"]


class ConfigError(ValueError):
    """Ошибка пользовательской конфигурации ai4epi."""


class StrictModel(BaseModel):
    """Базовая модель с закрытым контрактом."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


class BackendConfig(StrictModel):
    """
    Конфигурация одного LLM-backend-а.

    Сейчас пакетный backend-фабричный слой поддерживает ``ollama``. Поле
    ``backend`` сохранено явным, чтобы позднее добавить другие реализации без
    изменения структуры пользовательского конфигурационного файла.
    """

    backend: str = Field(default="ollama", min_length=1)
    model: str = Field(min_length=1)
    base_url: str = Field(default="http://localhost:11434", min_length=1)
    timeout_sec: int = Field(default=180, gt=0)

    def make_backend(self) -> ChatBackend:
        """Создать ChatBackend по текущей конфигурации."""

        return make_chat_backend(
            self.backend,
            model=self.model,
            base_url=self.base_url,
            default_timeout=self.timeout_sec,
        )


class LLMConfig(StrictModel):
    """
    Набор LLM-backend-ов для разных стадий.

    ``editor`` и ``evaluator`` необязательны. Если editor включён в
    ``PipelineSettings`` и отдельный backend не задан, pipeline переиспользует
    narrator-backend. Для evaluation действует тот же принцип на уровне
    ``make_eval_config``.
    """

    narrator: BackendConfig
    editor: BackendConfig | None = None
    evaluator: BackendConfig | None = None


class EvaluationRunConfig(StrictModel):
    """
    Декларативные настройки downstream evaluation-слоя.

    Этот блок намеренно не запускается внутри основного pipeline. Он описывает,
    как оценивать уже готовый бюллетень после генерации и editor-pass.
    """

    enabled: bool = False
    bulletin_kind: EvaluationBulletinKind = "final"
    output_filename: str = Field(default="evaluation_report.json", min_length=1)
    numeric_tolerance_pct: float = Field(default=1.5, ge=0.0)
    weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_EVAL_WEIGHTS))
    enabled_evaluators: tuple[str, ...] = Field(default_factory=lambda: tuple(DEFAULT_ENABLED_EVALUATORS))
    judge_temperature: float = Field(default=0.0, ge=0.0)
    request_timeout_sec: int | None = Field(default=None, gt=0)
    raw_preview_chars: int = Field(default=500, gt=0)

    @field_validator("output_filename")
    @classmethod
    def validate_output_filename(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or len(path.parts) != 1:
            raise ValueError("Evaluation output filename must be a simple relative filename.")
        return value

    def make_eval_config(self, *, llm: ChatBackend | None = None) -> EvalConfig:
        """Построить EvalConfig для evaluation.py."""

        return EvalConfig(
            numeric_tolerance_pct=self.numeric_tolerance_pct,
            weights=dict(self.weights),
            enabled_evaluators=tuple(self.enabled_evaluators),
            judge_temperature=self.judge_temperature,
            llm=llm,
            request_timeout_sec=self.request_timeout_sec,
            raw_preview_chars=self.raw_preview_chars,
        )


class Ai4EpiConfig(StrictModel):
    """
    Полная конфигурация одного запуска ai4epi.

    Минимально необходимы ``context_path`` и ``llm.narrator.model``. Остальные
    блоки имеют явные значения по умолчанию и могут переопределяться в JSON/YAML.
    """

    context_path: Path
    registry_path: Path | None = None
    output: PipelineOutputConfig | None = None
    llm: LLMConfig
    narrator_settings: NarratorSettings = Field(default_factory=NarratorSettings)
    pipeline_settings: PipelineSettings = Field(default_factory=PipelineSettings)
    editor_settings: EditorSettings = Field(default_factory=EditorSettings)
    runtime_checks: RuntimeCheckConfig = Field(default_factory=RuntimeCheckConfig)
    evaluation: EvaluationRunConfig = Field(default_factory=EvaluationRunConfig)
    metadata_extra: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_editor_configuration(self) -> "Ai4EpiConfig":
        if not self.pipeline_settings.run_editor and self.llm.editor is not None:
            # Это не ошибка: пользователь мог заранее описать editor backend.
            return self
        return self

    def resolve_paths(self, base_dir: str | Path) -> "Ai4EpiConfig":
        """
        Вернуть копию конфигурации с путями, разрешёнными относительно base_dir.

        Относительные пути внутри конфигурационного файла должны интерпретироваться
        относительно каталога этого файла, а не текущего рабочего каталога процесса.
        """

        base = Path(base_dir).expanduser().resolve()
        output = self.output
        if output is not None:
            output = output.model_copy(update={"output_dir": _resolve_path(output.output_dir, base)})

        return self.model_copy(
            update={
                "context_path": _resolve_path(self.context_path, base),
                "registry_path": _resolve_optional_path(self.registry_path, base),
                "output": output,
            },
            deep=True,
        )

    def load_registry(self) -> SectionRegistry:
        """Загрузить пользовательский реестр секций или вернуть дефолтный."""

        return load_pipeline_registry(self.registry_path)

    def make_runtime_checker(self) -> RuntimeChecker:
        """Создать RuntimeChecker по конфигурации runtime-check слоя."""

        return RuntimeChecker(config=self.runtime_checks)

    def make_pipeline(self) -> Ai4EpiPipeline:
        """Собрать Ai4EpiPipeline из декларативной конфигурации."""

        registry = self.load_registry()
        runtime_checker = self.make_runtime_checker()
        editor = self.llm.editor

        return create_pipeline(
            backend=self.llm.narrator.backend,
            model=self.llm.narrator.model,
            base_url=self.llm.narrator.base_url,
            backend_timeout_sec=self.llm.narrator.timeout_sec,
            editor_backend=editor.backend if editor is not None else None,
            editor_model=editor.model if editor is not None else None,
            editor_base_url=editor.base_url if editor is not None else None,
            editor_backend_timeout_sec=editor.timeout_sec if editor is not None else None,
            narrator_settings=self.narrator_settings,
            editor_settings=self.editor_settings,
            pipeline_settings=self.pipeline_settings,
            registry=registry,
            runtime_checker=runtime_checker,
        )

    def make_evaluation_llm(self, *, require_enabled: bool = True) -> ChatBackend | None:
        """Создать LLM для evaluation-слоя.

        ``require_enabled`` оставляет прежнюю семантику для автоматического workflow:
        если ``evaluation.enabled=False``, evaluation LLM не создаётся. Для явной
        команды ``evaluate-bulletin`` этот флаг должен быть ``False``: пользователь
        уже запросил evaluation, поэтому конфигурационный флаг не должен отключать
        LLM-based evaluator-ы.
        """

        if require_enabled and not self.evaluation.enabled:
            return None
        backend = self.llm.evaluator or self.llm.narrator
        return backend.make_backend()

    def make_eval_config(self, *, require_enabled: bool = True) -> EvalConfig:
        """Построить EvalConfig с корректным LLM-backend-ом."""

        return self.evaluation.make_eval_config(
            llm=self.make_evaluation_llm(require_enabled=require_enabled)
        )

    def run_pipeline(self) -> PipelineRunResult:
        """Запустить основной pipeline по текущей конфигурации."""

        pipeline = self.make_pipeline()
        return pipeline.run_from_files(
            context_path=self.context_path,
            output=self.output,
            metadata_extra=self.metadata_extra,
        )

    def to_public_dict(self) -> JsonObject:
        """Вернуть JSON-сериализуемое представление конфигурации."""

        return self.model_dump(mode="json", exclude_none=False)


class ConfigBundle(StrictModel):
    """
    Результат загрузки конфигурационного файла.

    Поле ``path`` сохраняет источник конфигурации; ``config`` уже содержит
    пути, разрешённые относительно каталога файла.
    """

    path: Path
    config: Ai4EpiConfig



def default_config(
    *,
    context_path: str | Path,
    model: str,
    output_dir: str | Path | None = "outputs",
    registry_path: str | Path | None = None,
    run_editor: bool = False,
) -> Ai4EpiConfig:
    """Создать минимальную валидную конфигурацию для типового запуска."""

    pipeline_settings = PipelineSettings(run_editor=run_editor)
    output = PipelineOutputConfig(output_dir=Path(output_dir)) if output_dir is not None else None
    return Ai4EpiConfig(
        context_path=Path(context_path),
        registry_path=Path(registry_path) if registry_path is not None else None,
        output=output,
        llm=LLMConfig(narrator=BackendConfig(model=model)),
        pipeline_settings=pipeline_settings,
    )



def load_ai4epi_config(path: str | Path, *, resolve_relative_paths: bool = True) -> Ai4EpiConfig:
    """Загрузить Ai4EpiConfig из JSON или YAML."""

    config_path = Path(path).expanduser().resolve()
    data = _read_mapping(config_path)
    config = Ai4EpiConfig.model_validate(data)
    if resolve_relative_paths:
        config = config.resolve_paths(config_path.parent)
    return config



def load_config_bundle(path: str | Path, *, resolve_relative_paths: bool = True) -> ConfigBundle:
    """Загрузить конфигурацию вместе с путём её источника."""

    config_path = Path(path).expanduser().resolve()
    config = load_ai4epi_config(config_path, resolve_relative_paths=resolve_relative_paths)
    return ConfigBundle(path=config_path, config=config)



def save_ai4epi_config(config: Ai4EpiConfig, path: str | Path, *, format: ConfigFormat | None = None) -> None:
    """Сохранить конфигурацию в JSON или YAML."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    effective_format = format or _infer_format(output_path)
    data = config.to_public_dict()

    if effective_format == "json":
        with output_path.open("w", encoding="utf-8") as stream:
            json.dump(data, stream, ensure_ascii=False, indent=2)
        return

    if effective_format in {"yaml", "yml"}:
        yaml = _import_yaml()
        with output_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(data, stream, allow_unicode=True, sort_keys=False)
        return

    raise ConfigError(f"Unsupported config format: {effective_format!r}.")



def run_pipeline_from_config(config: Ai4EpiConfig | str | Path) -> PipelineRunResult:
    """Запустить основной pipeline из объекта Ai4EpiConfig или пути к конфигу."""

    effective_config = load_ai4epi_config(config) if isinstance(config, (str, Path)) else config
    return effective_config.run_pipeline()



def _read_mapping(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise ConfigError(f"Config file does not exist: {path}")

    file_format = _infer_format(path)
    if file_format == "json":
        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    elif file_format in {"yaml", "yml"}:
        yaml = _import_yaml()
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    else:
        raise ConfigError(f"Unsupported config format: {file_format!r}.")

    if not isinstance(data, Mapping):
        raise ConfigError("Config root must be a JSON/YAML object.")
    return data



def _infer_format(path: Path) -> ConfigFormat:
    suffix = path.suffix.lower().lstrip(".")
    if suffix in {"json", "yaml", "yml"}:
        return suffix  # type: ignore[return-value]
    raise ConfigError("Config file extension must be .json, .yaml or .yml.")



def _import_yaml() -> Any:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - зависит от окружения установки
        raise ConfigError("Для YAML-конфигураций требуется пакет PyYAML.") from exc
    return yaml



def _resolve_optional_path(path: Path | None, base_dir: Path) -> Path | None:
    if path is None:
        return None
    return _resolve_path(path, base_dir)



def _resolve_path(path: str | Path, base_dir: Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    return (base_dir / value).resolve()


__all__ = [
    "Ai4EpiConfig",
    "BackendConfig",
    "ConfigBundle",
    "ConfigError",
    "ConfigFormat",
    "EvaluationBulletinKind",
    "EvaluationRunConfig",
    "JsonObject",
    "LLMConfig",
    "default_config",
    "load_ai4epi_config",
    "load_config_bundle",
    "run_pipeline_from_config",
    "save_ai4epi_config",
]

