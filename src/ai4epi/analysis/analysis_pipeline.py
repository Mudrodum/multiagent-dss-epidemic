"""
Численный orchestration-layer для ai4epi.

Модуль собирает воспроизводимый численный pipeline от подготовленных таблиц
или внешних источников до валидного ``GlobalContext``. Он намеренно отделён от
``pipeline.py``: тот отвечает за LLM/bulletin-часть, а этот файл — за данные,
признаки, прогноз, SHAP, сезонные аналитические блоки и сохранение
``context_relevant.json``.

Основная схема:

``influenza_weekly + weather_weekly``
    -> ``merged_weekly``
    -> ``SupervisedDataset``
    -> ``ForecastRunResult``
    -> ``ShapRunResult``
    -> ``EpidemicWaveRunResult``
    -> optional ``AgeGroupSeasonRunResult``
    -> ``ContextBuildResult``.

Два уровня API:

* ``run_analysis_pipeline`` принимает уже подготовленные pandas.DataFrame;
* ``run_analysis_pipeline_from_sources`` загружает influenza/weather данные
  через source-модули и затем вызывает тот же табличный pipeline.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import time
from typing import Any, Literal, Mapping

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

try:  # package import
    from ai4epi.analysis.age_group_season import (
        AgeGroupSeasonConfig,
        AgeGroupSeasonOutputConfig,
        AgeGroupSeasonRunResult,
        run_age_group_season_analysis,
    )
    from ai4epi.core.context import GlobalContext
    from ai4epi.analysis.context_builders import (
        ContextBuildConfig,
        ContextBuildResult,
        ContextOutputConfig,
        build_global_context_result,
    )
    from ai4epi.data.influenza import (
        InfluenzaDbConfig,
        InfluenzaFrameBundle,
        load_influenza_weekly_from_api,
        save_influenza_bundle,
    )
    from ai4epi.analysis.epidemic_waves import (
        EpidemicWaveConfig,
        EpidemicWaveOutputConfig,
        EpidemicWaveRunResult,
        run_epidemic_wave_analysis,
    )
    from ai4epi.analysis.explainability import (
        ExplainabilityConfig,
        ExplainabilityOutputConfig,
        ShapRunResult,
        run_shap_analysis,
    )
    from ai4epi.analysis.forecasting import (
        ForecastingConfig,
        ForecastOutputConfig,
        ForecastRunResult,
        run_forecast,
    )
    from ai4epi.core.io import TableWriteOptions, write_json, write_table
    from ai4epi.analysis.preprocessing import (
        FeatureEngineeringConfig,
        SupervisedDataset,
        WeatherAggregationConfig,
        WeeklyFrameConfig,
        aggregate_hourly_weather_to_weekly,
        build_supervised,
        merge_weekly_influenza_weather,
        normalize_weekly_frame,
        save_supervised_dataset,
    )
    from ai4epi.data.weather import (
        WeatherApiConfig,
        WeatherFrameBundle,
        WeatherOutputConfig,
        load_weather_aligned_to_influenza,
        save_weather_bundle,
    )
except ImportError:  # pragma: no cover - compatibility with current notebook-like flat layout
    from ai4epi.analysis.age_group_season import (  # type: ignore[no-redef]
        AgeGroupSeasonConfig,
        AgeGroupSeasonOutputConfig,
        AgeGroupSeasonRunResult,
        run_age_group_season_analysis,
    )
    from ai4epi.core.context import GlobalContext  # type: ignore[no-redef]
    from ai4epi.analysis.context_builders import (  # type: ignore[no-redef]
        ContextBuildConfig,
        ContextBuildResult,
        ContextOutputConfig,
        build_global_context_result,
    )
    from ai4epi.data.influenza import (  # type: ignore[no-redef]
        InfluenzaDbConfig,
        InfluenzaFrameBundle,
        load_influenza_weekly_from_api,
        save_influenza_bundle,
    )
    from ai4epi.analysis.epidemic_waves import (  # type: ignore[no-redef]
        EpidemicWaveConfig,
        EpidemicWaveOutputConfig,
        EpidemicWaveRunResult,
        run_epidemic_wave_analysis,
    )
    from ai4epi.analysis.explainability import (  # type: ignore[no-redef]
        ExplainabilityConfig,
        ExplainabilityOutputConfig,
        ShapRunResult,
        run_shap_analysis,
    )
    from ai4epi.analysis.forecasting import (  # type: ignore[no-redef]
        ForecastingConfig,
        ForecastOutputConfig,
        ForecastRunResult,
        run_forecast,
    )
    from ai4epi.core.io import TableWriteOptions, write_json, write_table  # type: ignore[no-redef]
    from ai4epi.analysis.preprocessing import (  # type: ignore[no-redef]
        FeatureEngineeringConfig,
        SupervisedDataset,
        WeatherAggregationConfig,
        WeeklyFrameConfig,
        aggregate_hourly_weather_to_weekly,
        build_supervised,
        merge_weekly_influenza_weather,
        normalize_weekly_frame,
        save_supervised_dataset,
    )
    from ai4epi.data.weather import (  # type: ignore[no-redef]
        WeatherApiConfig,
        WeatherFrameBundle,
        WeatherOutputConfig,
        load_weather_aligned_to_influenza,
        save_weather_bundle,
    )


JsonObject = dict[str, Any]
AnalysisStatus = Literal["ok", "partial", "failed"]


class AnalysisPipelineError(RuntimeError):
    """Базовая ошибка численного analysis pipeline."""


class AnalysisPipelineInputError(AnalysisPipelineError):
    """Входные таблицы или source-параметры не позволяют выполнить pipeline."""


class StrictModel(BaseModel):
    """Базовая pydantic-модель с закрытым контрактом."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


class AnalysisPipelineSettings(StrictModel):
    """Настройки выполнения численного pipeline.

    ``require_weather`` защищает от скрытого изменения feature space: если
    ``FeatureEngineeringConfig.temp_cols`` содержит погодные признаки, но
    weather-таблица не передана, pipeline завершается ошибкой вместо тихого
    удаления признаков.
    """

    require_weather: bool = True
    run_explainability: bool = True
    run_epidemic_waves: bool = True
    run_age_group_season: bool = True
    require_age_group_season: bool = False
    build_context: bool = True
    raise_on_error: bool = False
    save_source_artifacts: bool = True
    save_intermediate_tables: bool = True

    @model_validator(mode="after")
    def validate_context_dependencies(self) -> "AnalysisPipelineSettings":
        if self.build_context and not self.run_explainability:
            raise ValueError("build_context=True требует run_explainability=True.")
        if self.build_context and not self.run_epidemic_waves:
            raise ValueError("build_context=True требует run_epidemic_waves=True.")
        if self.require_age_group_season and not self.run_age_group_season:
            raise ValueError("require_age_group_season=True требует run_age_group_season=True.")
        return self


class AnalysisOutputConfig(StrictModel):
    """Настройки сохранения артефактов численного pipeline."""

    output_dir: Path = Path("analysis_outputs")
    merged_weekly_filename: str = Field(default="merged_weekly.csv", min_length=1)
    run_report_filename: str = Field(default="analysis_run.json", min_length=1)
    context_filename: str = Field(default="context_relevant.json", min_length=1)
    sources_subdir: str = Field(default="sources", min_length=1)
    preprocessing_subdir: str = Field(default="preprocessing", min_length=1)
    forecasting_subdir: str = Field(default="forecasting", min_length=1)
    explainability_subdir: str = Field(default="explainability", min_length=1)
    epidemic_waves_subdir: str = Field(default="epidemic_waves", min_length=1)
    age_group_subdir: str = Field(default="age_group_season", min_length=1)
    save_run_report: bool = True
    save_merged_weekly: bool = True
    csv_encoding: str = Field(default="utf-8-sig", min_length=1)

    @field_validator(
        "merged_weekly_filename",
        "run_report_filename",
        "context_filename",
        "sources_subdir",
        "preprocessing_subdir",
        "forecasting_subdir",
        "explainability_subdir",
        "epidemic_waves_subdir",
        "age_group_subdir",
    )
    @classmethod
    def validate_relative_name(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or len(path.parts) != 1:
            raise ValueError("Имена файлов и подкаталогов должны быть простыми относительными именами.")
        return value

    @property
    def sources_dir(self) -> Path:
        return self.output_dir / self.sources_subdir

    @property
    def preprocessing_dir(self) -> Path:
        return self.output_dir / self.preprocessing_subdir

    @property
    def forecasting_dir(self) -> Path:
        return self.output_dir / self.forecasting_subdir

    @property
    def explainability_dir(self) -> Path:
        return self.output_dir / self.explainability_subdir

    @property
    def epidemic_waves_dir(self) -> Path:
        return self.output_dir / self.epidemic_waves_subdir

    @property
    def age_group_dir(self) -> Path:
        return self.output_dir / self.age_group_subdir

    @property
    def context_path(self) -> Path:
        return self.output_dir / self.context_filename

    @property
    def run_report_path(self) -> Path:
        return self.output_dir / self.run_report_filename

    def make_forecast_output(self) -> ForecastOutputConfig:
        return ForecastOutputConfig(output_dir=self.forecasting_dir)

    def make_explainability_output(self) -> ExplainabilityOutputConfig:
        return ExplainabilityOutputConfig(output_dir=self.explainability_dir)

    def make_epidemic_wave_output(self) -> EpidemicWaveOutputConfig:
        return EpidemicWaveOutputConfig(output_dir=self.epidemic_waves_dir)

    def make_age_group_output(self) -> AgeGroupSeasonOutputConfig:
        return AgeGroupSeasonOutputConfig(output_dir=self.age_group_dir)

    def make_context_output(self) -> ContextOutputConfig:
        return ContextOutputConfig(output_path=self.context_path)


class AnalysisSourceConfig(StrictModel):
    """Параметры convenience-загрузки данных из внешних источников."""

    city: str = Field(min_length=1)
    begin_year: int = Field(default=2011, ge=1900, le=2100)
    begin_week: int = Field(default=1, ge=1, le=53)
    end_date: date | datetime | pd.Timestamp | str | None = None
    fetch_weather: bool = True
    weather_latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    weather_longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    weather_timezone: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_weather_coordinates(self) -> "AnalysisSourceConfig":
        provided = [self.weather_latitude is not None, self.weather_longitude is not None, self.weather_timezone is not None]
        if any(provided) and not all(provided):
            raise ValueError("weather_latitude, weather_longitude и weather_timezone должны задаваться совместно.")
        return self


class AnalysisRunResult(StrictModel):
    """Структурированный результат численного analysis pipeline."""

    status: AnalysisStatus
    duration_sec: float = Field(ge=0.0)
    merged_weekly: Any | None = None
    supervised: SupervisedDataset | None = None
    forecast_result: ForecastRunResult | None = None
    shap_result: ShapRunResult | None = None
    epidemic_wave_result: EpidemicWaveRunResult | None = None
    age_group_result: AgeGroupSeasonRunResult | None = None
    context_result: ContextBuildResult | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error_message: str | None = None

    @field_validator("merged_weekly")
    @classmethod
    def validate_optional_dataframe(cls, value: Any) -> pd.DataFrame | None:
        if value is None:
            return None
        if not isinstance(value, pd.DataFrame):
            raise TypeError("merged_weekly должен быть pandas.DataFrame или None.")
        return value

    @property
    def context(self) -> GlobalContext | None:
        """Вернуть собранный GlobalContext, если pipeline дошёл до этого этапа."""

        return self.context_result.context if self.context_result is not None else None

    def raise_for_failure(self) -> None:
        """Выбросить исключение, если pipeline завершился неуспешно."""

        if self.status == "failed":
            raise AnalysisPipelineError(self.error_message or "ai4epi analysis pipeline failed.")

    def to_public_dict(self, *, include_heavy_objects: bool = False) -> JsonObject:
        """Вернуть JSON-сериализуемый отчёт без тяжёлых таблиц по умолчанию."""

        data: JsonObject = {
            "status": self.status,
            "duration_sec": self.duration_sec,
            "artifacts": dict(self.artifacts),
            "warnings": list(self.warnings),
            "error_message": self.error_message,
            "has_merged_weekly": self.merged_weekly is not None,
            "has_supervised": self.supervised is not None,
            "has_forecast_result": self.forecast_result is not None,
            "has_shap_result": self.shap_result is not None,
            "has_epidemic_wave_result": self.epidemic_wave_result is not None,
            "has_age_group_result": self.age_group_result is not None,
            "has_context": self.context_result is not None,
        }
        if self.context_result is not None:
            data["context"] = self.context_result.context.to_public_dict()
        if include_heavy_objects:
            data["forecast"] = self.forecast_result.to_notebook_dict() if self.forecast_result is not None else None
            data["shap"] = self.shap_result.to_public_dict() if self.shap_result is not None else None
        return data


# ---------------------------------------------------------------------------
# Main public API
# ---------------------------------------------------------------------------


def run_analysis_pipeline(
    *,
    influenza_weekly: pd.DataFrame,
    weather_weekly: pd.DataFrame | None = None,
    hourly_weather: pd.DataFrame | None = None,
    age_group_frame: pd.DataFrame | None = None,
    settings: AnalysisPipelineSettings | None = None,
    output: AnalysisOutputConfig | None = None,
    weekly_config: WeeklyFrameConfig | None = None,
    weather_aggregation_config: WeatherAggregationConfig | None = None,
    feature_config: FeatureEngineeringConfig | None = None,
    forecasting_config: ForecastingConfig | None = None,
    explainability_config: ExplainabilityConfig | None = None,
    epidemic_wave_config: EpidemicWaveConfig | None = None,
    age_group_config: AgeGroupSeasonConfig | None = None,
    context_config: ContextBuildConfig | None = None,
    extra_context_blocks: Mapping[str, Any] | None = None,
) -> AnalysisRunResult:
    """Выполнить численный pipeline на уже подготовленных таблицах.

    Это основной масштабируемый вход: пользователь может подать собственные
    таблицы вместо использования встроенных source-клиентов. Все этапы
    возвращают типизированные result-объекты; сохранение файлов включается
    через ``output``.
    """

    started = time.perf_counter()
    cfg = settings or AnalysisPipelineSettings()
    out = output
    artifacts: dict[str, str] = {}
    warnings: list[str] = []

    merged_weekly: pd.DataFrame | None = None
    supervised: SupervisedDataset | None = None
    forecast_result: ForecastRunResult | None = None
    shap_result: ShapRunResult | None = None
    epidemic_wave_result: EpidemicWaveRunResult | None = None
    age_group_result: AgeGroupSeasonRunResult | None = None
    context_result: ContextBuildResult | None = None
    error_message: str | None = None

    try:
        fcfg = feature_config or FeatureEngineeringConfig()
        merged_weekly = prepare_analysis_weekly_frame(
            influenza_weekly=influenza_weekly,
            weather_weekly=weather_weekly,
            hourly_weather=hourly_weather,
            weekly_config=weekly_config,
            weather_aggregation_config=weather_aggregation_config,
            feature_config=fcfg,
            require_weather=cfg.require_weather,
        )

        if out is not None:
            out.output_dir.mkdir(parents=True, exist_ok=True)
            if cfg.save_intermediate_tables and out.save_merged_weekly:
                path = write_table(
                    merged_weekly,
                    out.output_dir / out.merged_weekly_filename,
                    options=TableWriteOptions(index=False, encoding=out.csv_encoding),
                )
                artifacts["merged_weekly"] = str(path)

        supervised = build_supervised(merged_weekly, config=fcfg)
        if out is not None and cfg.save_intermediate_tables:
            for name, path in save_supervised_dataset(supervised, out.preprocessing_dir).items():
                artifacts[f"preprocessing.{name}"] = str(path)

        forecast_result = run_forecast(
            supervised,
            config=forecasting_config,
            output=out.make_forecast_output() if out is not None else None,
        )
        artifacts.update(_prefix_artifacts("forecasting", forecast_result.artifacts))

        if cfg.run_explainability:
            shap_result = run_shap_analysis(
                forecast_result,
                config=explainability_config,
                output=out.make_explainability_output() if out is not None else None,
            )
            artifacts.update(_prefix_artifacts("explainability", shap_result.artifacts))

        if cfg.run_epidemic_waves:
            epidemic_wave_result = run_epidemic_wave_analysis(
                influenza_weekly,
                config=epidemic_wave_config,
                output=out.make_epidemic_wave_output() if out is not None else None,
            )
            artifacts.update(_prefix_artifacts("epidemic_waves", epidemic_wave_result.artifacts))

        if cfg.run_age_group_season:
            if age_group_frame is not None:
                age_group_result = run_age_group_season_analysis(
                    age_group_frame,
                    config=age_group_config,
                    output=out.make_age_group_output() if out is not None else None,
                )
                artifacts.update(_prefix_artifacts("age_group_season", age_group_result.artifacts))
            elif cfg.require_age_group_season:
                raise AnalysisPipelineInputError("age_group_frame не передан, но require_age_group_season=True.")
            else:
                warnings.append("age_group_frame не передан; блок age_group_season не будет включён в GlobalContext.")

        if cfg.build_context:
            if shap_result is None:
                raise AnalysisPipelineError("GlobalContext требует shap_result; SHAP-pass не выполнен.")
            if epidemic_wave_result is None:
                raise AnalysisPipelineError("GlobalContext требует epidemic_wave_result; epidemic wave analysis не выполнен.")
            context_result = build_global_context_result(
                forecast_result=forecast_result,
                shap_result=shap_result,
                epidemic_wave_comparison=epidemic_wave_result,
                age_group_season=age_group_result,
                extra_blocks=extra_context_blocks,
                config=context_config,
                output=out.make_context_output() if out is not None else None,
            )
            artifacts.update(_prefix_artifacts("context", context_result.artifacts))

    except Exception as exc:
        error_message = str(exc)
        if cfg.raise_on_error:
            raise

    status = _infer_analysis_status(error_message=error_message, warnings=warnings, context_result=context_result, settings=cfg)
    result = AnalysisRunResult(
        status=status,
        duration_sec=round(time.perf_counter() - started, 3),
        merged_weekly=merged_weekly,
        supervised=supervised,
        forecast_result=forecast_result,
        shap_result=shap_result,
        epidemic_wave_result=epidemic_wave_result,
        age_group_result=age_group_result,
        context_result=context_result,
        artifacts=artifacts,
        warnings=warnings,
        error_message=error_message,
    )

    if out is not None and out.save_run_report:
        out.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = write_json(result.to_public_dict(include_heavy_objects=False), out.run_report_path)
        result.artifacts["analysis_run"] = str(report_path)

    return result


def run_analysis_pipeline_from_sources(
    *,
    source: AnalysisSourceConfig,
    settings: AnalysisPipelineSettings | None = None,
    output: AnalysisOutputConfig | None = None,
    influenza_db_config: InfluenzaDbConfig | None = None,
    weather_api_config: WeatherApiConfig | None = None,
    weather_aggregation_config: WeatherAggregationConfig | None = None,
    feature_config: FeatureEngineeringConfig | None = None,
    forecasting_config: ForecastingConfig | None = None,
    explainability_config: ExplainabilityConfig | None = None,
    epidemic_wave_config: EpidemicWaveConfig | None = None,
    age_group_config: AgeGroupSeasonConfig | None = None,
    context_config: ContextBuildConfig | None = None,
    extra_context_blocks: Mapping[str, Any] | None = None,
    session: Any | None = None,
) -> AnalysisRunResult:
    """Загрузить данные из source-модулей и выполнить тот же численный pipeline."""

    cfg = settings or AnalysisPipelineSettings()
    out = output
    source_artifacts: dict[str, str] = {}

    influenza_bundle = load_influenza_weekly_from_api(
        source.city,
        end_date=source.end_date,
        begin_year=source.begin_year,
        begin_week=source.begin_week,
        config=influenza_db_config,
        session=session,
    )

    if out is not None and cfg.save_source_artifacts:
        for name, path in save_influenza_bundle(
            influenza_bundle,
            output_dir=out.sources_dir,
            save_raw=False,
            save_cases=True,
        ).items():
            source_artifacts[f"sources.influenza.{name}"] = str(path)

    weather_bundle: WeatherFrameBundle | None = None
    if source.fetch_weather:
        weather_bundle = load_weather_aligned_to_influenza(
            source.city,
            influenza_bundle.weekly,
            latitude=source.weather_latitude,
            longitude=source.weather_longitude,
            timezone=source.weather_timezone,
            api_config=weather_api_config,
            aggregation_config=weather_aggregation_config,
            session=session,
        )
        if out is not None and cfg.save_source_artifacts:
            for name, path in save_weather_bundle(
                weather_bundle,
                WeatherOutputConfig(output_dir=out.sources_dir),
            ).items():
                source_artifacts[f"sources.weather.{name}"] = str(path)

    result = run_analysis_pipeline(
        influenza_weekly=influenza_bundle.weekly,
        weather_weekly=weather_bundle.weekly if weather_bundle is not None else None,
        age_group_frame=influenza_bundle.cases,
        settings=cfg,
        output=out,
        weather_aggregation_config=weather_aggregation_config,
        feature_config=feature_config,
        forecasting_config=forecasting_config,
        explainability_config=explainability_config,
        epidemic_wave_config=epidemic_wave_config,
        age_group_config=age_group_config,
        context_config=context_config,
        extra_context_blocks=extra_context_blocks,
    )
    result.artifacts = {**source_artifacts, **result.artifacts}
    return result


def run_analysis_pipeline_from_source_params(
    *,
    city: str,
    begin_year: int = 2011,
    begin_week: int = 1,
    end_date: date | datetime | pd.Timestamp | str | None = None,
    output_dir: str | Path | None = "analysis_outputs",
    fetch_weather: bool = True,
    settings: AnalysisPipelineSettings | None = None,
    **kwargs: Any,
) -> AnalysisRunResult:
    """Краткая обёртка для интерактивного запуска."""

    output = AnalysisOutputConfig(output_dir=Path(output_dir)) if output_dir is not None else None
    return run_analysis_pipeline_from_sources(
        source=AnalysisSourceConfig(
            city=city,
            begin_year=begin_year,
            begin_week=begin_week,
            end_date=end_date,
            fetch_weather=fetch_weather,
        ),
        settings=settings,
        output=output,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Preparation helpers
# ---------------------------------------------------------------------------


def prepare_analysis_weekly_frame(
    *,
    influenza_weekly: pd.DataFrame,
    weather_weekly: pd.DataFrame | None = None,
    hourly_weather: pd.DataFrame | None = None,
    weekly_config: WeeklyFrameConfig | None = None,
    weather_aggregation_config: WeatherAggregationConfig | None = None,
    feature_config: FeatureEngineeringConfig | None = None,
    require_weather: bool = True,
) -> pd.DataFrame:
    """Подготовить weekly frame для ``build_supervised``.

    Если передан ``hourly_weather``, он агрегируется в weekly weather. Если
    передан ``weather_weekly``, он используется напрямую. Если погодные данные
    отсутствуют, pipeline разрешает это только при ``require_weather=False`` и
    пустом списке ``FeatureEngineeringConfig.temp_cols``.
    """

    if not isinstance(influenza_weekly, pd.DataFrame):
        raise AnalysisPipelineInputError("influenza_weekly должен быть pandas.DataFrame.")

    fcfg = feature_config or FeatureEngineeringConfig()
    weekly_cfg = weekly_config or fcfg.weekly_frame_config

    if weather_weekly is not None and hourly_weather is not None:
        raise AnalysisPipelineInputError("Передайте либо weather_weekly, либо hourly_weather, но не оба одновременно.")

    effective_weather_weekly = weather_weekly
    if effective_weather_weekly is None and hourly_weather is not None:
        effective_weather_weekly = aggregate_hourly_weather_to_weekly(
            hourly_weather,
            config=weather_aggregation_config,
        )

    if effective_weather_weekly is not None:
        return merge_weekly_influenza_weather(
            influenza_weekly,
            effective_weather_weekly,
            weekly_config=weekly_cfg,
        )

    if require_weather and fcfg.temp_cols:
        raise AnalysisPipelineInputError(
            "Погодные данные не переданы, но FeatureEngineeringConfig.temp_cols содержит погодные признаки. "
            "Передайте weather_weekly/hourly_weather либо задайте temp_cols=() и require_weather=False."
        )

    normalized = normalize_weekly_frame(influenza_weekly, config=weekly_cfg)
    missing_temp_cols = [column for column in fcfg.temp_cols if column not in normalized.columns]
    if missing_temp_cols:
        raise AnalysisPipelineInputError(
            "В weekly frame отсутствуют погодные признаки, требуемые feature_config: "
            f"{missing_temp_cols!r}."
        )
    return normalized


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _prefix_artifacts(prefix: str, artifacts: Mapping[str, Path | str]) -> dict[str, str]:
    return {f"{prefix}.{key}": str(value) for key, value in artifacts.items()}


def _infer_analysis_status(
    *,
    error_message: str | None,
    warnings: list[str],
    context_result: ContextBuildResult | None,
    settings: AnalysisPipelineSettings,
) -> AnalysisStatus:
    if error_message:
        return "failed"
    if settings.build_context and context_result is None:
        return "failed"
    if warnings:
        return "partial"
    return "ok"


__all__ = [
    "AnalysisPipelineError",
    "AnalysisPipelineInputError",
    "AnalysisPipelineSettings",
    "AnalysisOutputConfig",
    "AnalysisSourceConfig",
    "AnalysisRunResult",
    "prepare_analysis_weekly_frame",
    "run_analysis_pipeline",
    "run_analysis_pipeline_from_sources",
    "run_analysis_pipeline_from_source_params",
]

