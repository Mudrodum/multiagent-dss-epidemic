"""
Сборка GlobalContext для ai4epi.

Модуль является мостом между численной частью пайплайна и LLM/narrator-слоем.
Он принимает уже готовые результаты forecasting/explainability и готовые
аналитические payload-и для дополнительных эпидемиологических блоков, после
чего собирает валидированный ``GlobalContext``.

Важно: модуль не загружает данные из БД, не строит признаки, не обучает модели,
не считает SHAP и не вызывает LLM. Отсутствующие обязательные аналитические
блоки не заменяются заглушками: вызывающий код должен передать их явно.
"""

from __future__ import annotations

import json
import math
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4epi.core.context import GlobalContext, save_global_context
from ai4epi.analysis.forecasting import ForecastRunResult


JsonObject = dict[str, Any]

FEATURE_GROUPS_RU: dict[str, str] = {
    "target_lags": "лаги заболеваемости",
    "target_rolling_stats": "скользящие статистики заболеваемости",
    "epidemic_dynamics": "показатели эпидемической динамики",
    "fourier_seasonality": "сезонные гармоники (ряды Фурье)",
    "temperature_lags": "лаги температуры воздуха",
    "temperature_rolling_stats": "скользящие статистики температуры",
    "calendar_features": "календарные признаки (номер недели)",
    # Backward-compatible aliases used by older model registries.
    "calendar": "календарные и сезонные признаки",
    "target_rolling": "скользящие статистики заболеваемости",
    "target_dynamics": "показатели эпидемической динамики",
    "weather_lags": "лаги температуры воздуха",
    "weather_rolling": "скользящие статистики температуры",
    "other": "прочие признаки",
}

BaseTrendCode = Literal["increase", "decrease", "stable"]
CurrentTrendCode = Literal[
    "increase",
    "decrease",
    "stable",
    "up_with_fluctuations",
    "down_with_fluctuations",
    "mixed",
]
ForecastShapeCode = Literal[
    "increase",
    "decrease",
    "stable",
    "rise_then_fall",
    "fall_then_rise",
    "mixed",
]
TrendCode = BaseTrendCode | CurrentTrendCode | ForecastShapeCode
UncertaintyCode = Literal["low", "moderate", "high", "very_high"]


class ContextBuilderError(ValueError):
    """Базовая ошибка сборки GlobalContext."""


class MissingContextBlockError(ContextBuilderError):
    """Обязательный блок для GlobalContext не был передан."""


class StrictModel(BaseModel):
    """Базовая pydantic-модель с закрытым контрактом."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


class ContextBuildConfig(StrictModel):
    """Настройки детерминированной сборки GlobalContext.

    Эти параметры не являются скрытыми эвристиками: они задают системные
    правила классификации текстовых ярлыков, которые затем попадают в
    ``semantic``-поля контекста и используются секционными narrator-агентами.
    """

    unit: str = Field(default="случаев на 10 тыс. населения", min_length=1)
    per_population: int = Field(default=10_000, gt=0)
    current_trend_window_weeks: int = Field(default=4, ge=2)
    current_change_pct_threshold: float = Field(default=5.0, ge=0.0)
    forecast_flat_change_pct_threshold: float = Field(default=5.0, ge=0.0)
    uncertainty_low_pct_threshold: float = Field(default=30.0, ge=0.0)
    uncertainty_high_pct_threshold: float = Field(default=70.0, ge=0.0)
    uncertainty_very_high_pct_threshold: float = Field(default=150.0, ge=0.0)
    model_family_ru: str = Field(default="градиентный бустинг на гистограммах", min_length=1)
    model_strategy_ru: str = Field(
        default="прямая многошаговая (отдельная модель для каждого горизонта прогноза)",
        min_length=1,
    )
    task_type_ru: str = Field(default="многошаговый недельный прогноз заболеваемости", min_length=1)
    forecast_target_ru: str = Field(default="уровень заболеваемости гриппом и ОРВИ", min_length=1)
    calibration_data_description: str = Field(
        default=(
            "данные о зарегистрированных случаях заболеваний гриппом и ОРВИ "
            "и данные лабораторной диагностики гриппа (ПЦР)"
        ),
        min_length=1,
    )
    require_shap_summary: bool = True
    require_epidemic_wave_comparison: bool = True

    @model_validator(mode="after")
    def validate_uncertainty_thresholds(self) -> "ContextBuildConfig":
        if self.uncertainty_high_pct_threshold < self.uncertainty_low_pct_threshold:
            raise ValueError("uncertainty_high_pct_threshold должен быть не ниже uncertainty_low_pct_threshold.")
        if self.uncertainty_very_high_pct_threshold < self.uncertainty_high_pct_threshold:
            raise ValueError("uncertainty_very_high_pct_threshold должен быть не ниже uncertainty_high_pct_threshold.")
        return self


class ContextBuildInput(StrictModel):
    """Явные входы для сборки GlobalContext."""

    forecast_result: ForecastRunResult
    shap_result: Any
    epidemic_wave_comparison: Any | None
    age_group_season: Any | None = None
    extra_blocks: Mapping[str, Any] = Field(default_factory=dict)

    @field_validator("extra_blocks")
    @classmethod
    def validate_extra_blocks_do_not_override_standard(cls, value: Mapping[str, Any]) -> Mapping[str, Any]:
        reserved = {
            "origin",
            "unit",
            "per_population",
            "current_situation",
            "epidemic_wave_comparison",
            "forecast",
            "shap_summary",
            "model_quality",
            "model_info",
            "age_group_season",
        }
        overlap = reserved.intersection(value.keys())
        if overlap:
            raise ValueError(f"extra_blocks не должен переопределять стандартные блоки: {sorted(overlap)!r}.")
        return value


class ContextBuildResult(StrictModel):
    """Результат сборки контекста."""

    context: GlobalContext
    payload: JsonObject
    artifacts: dict[str, Path] = Field(default_factory=dict)

    def to_public_dict(self) -> JsonObject:
        """Вернуть JSON-сериализуемое представление результата."""

        return {
            "context": self.context.to_public_dict(),
            "payload": self.payload,
            "artifacts": {key: str(path) for key, path in self.artifacts.items()},
        }


class ContextOutputConfig(StrictModel):
    """Настройки сохранения GlobalContext."""

    output_path: Path = Path("context_relevant.json")
    save_context: bool = True
    indent: int = Field(default=2, ge=0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_global_context(
    *,
    forecast_result: ForecastRunResult,
    shap_result: Any,
    epidemic_wave_comparison: Any | None,
    age_group_season: Any | None = None,
    extra_blocks: Mapping[str, Any] | None = None,
    config: ContextBuildConfig | None = None,
) -> GlobalContext:
    """Собрать и валидировать ``GlobalContext``.

    ``forecast_result`` даёт численную часть: текущую ситуацию, прогноз,
    качество модели и сведения о модели. ``shap_result`` даёт блок
    ``shap_summary``. ``epidemic_wave_comparison`` передаётся готовым payload-ом,
    потому что расчёт волн является отдельным аналитическим модулем.
    """

    return build_global_context_result(
        forecast_result=forecast_result,
        shap_result=shap_result,
        epidemic_wave_comparison=epidemic_wave_comparison,
        age_group_season=age_group_season,
        extra_blocks=extra_blocks,
        config=config,
        output=None,
    ).context


def build_global_context_result(
    *,
    forecast_result: ForecastRunResult,
    shap_result: Any,
    epidemic_wave_comparison: Any | None,
    age_group_season: Any | None = None,
    extra_blocks: Mapping[str, Any] | None = None,
    config: ContextBuildConfig | None = None,
    output: ContextOutputConfig | None = None,
) -> ContextBuildResult:
    """Собрать ``GlobalContext`` и при необходимости сохранить его в JSON."""

    cfg = config or ContextBuildConfig()
    payload = build_global_context_payload(
        forecast_result=forecast_result,
        shap_result=shap_result,
        epidemic_wave_comparison=epidemic_wave_comparison,
        age_group_season=age_group_season,
        extra_blocks=extra_blocks,
        config=cfg,
    )
    context = GlobalContext.model_validate(payload)
    artifacts: dict[str, Path] = {}

    if output is not None and output.save_context:
        path = Path(output.output_path)
        save_global_context(context, path, indent=output.indent)
        artifacts["context"] = path.resolve()

    return ContextBuildResult(context=context, payload=payload, artifacts=artifacts)


def build_global_context_payload(
    *,
    forecast_result: ForecastRunResult,
    shap_result: Any,
    epidemic_wave_comparison: Any | None,
    age_group_season: Any | None = None,
    extra_blocks: Mapping[str, Any] | None = None,
    config: ContextBuildConfig | None = None,
) -> JsonObject:
    """Собрать JSON-представление ``GlobalContext`` до pydantic-валидации."""

    cfg = config or ContextBuildConfig()
    if shap_result is None and cfg.require_shap_summary:
        raise MissingContextBlockError("Для GlobalContext требуется shap_summary; передайте ShapRunResult или payload.")
    if epidemic_wave_comparison is None and cfg.require_epidemic_wave_comparison:
        raise MissingContextBlockError(
            "Для GlobalContext требуется epidemic_wave_comparison; передайте готовый payload анализа волн."
        )

    payload: JsonObject = {
        "origin": build_origin_context(forecast_result),
        "unit": cfg.unit,
        "per_population": cfg.per_population,
        "current_situation": build_current_situation_context(forecast_result, config=cfg),
        "epidemic_wave_comparison": build_epidemic_wave_comparison_context(epidemic_wave_comparison),
        "forecast": build_forecast_context(forecast_result, config=cfg),
        "shap_summary": build_shap_summary_context(shap_result),
        "model_quality": build_model_quality_context(forecast_result),
        "model_info": build_model_info_context(forecast_result, config=cfg),
        "age_group_season": build_age_group_season_context(age_group_season) if age_group_season is not None else None,
    }
    payload.update(_json_object(extra_blocks or {}))
    return _to_jsonable(payload)


# ---------------------------------------------------------------------------
# Standard context block builders
# ---------------------------------------------------------------------------


def build_origin_context(forecast_result: ForecastRunResult) -> JsonObject:
    """Построить блок ``origin``."""

    origin = _as_timestamp(forecast_result.origin_date, name="forecast_result.origin_date")
    origin_date = origin.date()
    return {
        "origin_date": origin_date.isoformat(),
        "iso_year": int(origin_date.isocalendar().year),
        "iso_week": int(origin_date.isocalendar().week),
        "week_start": origin_date.isoformat(),
        "week_end": (origin_date + timedelta(days=6)).isoformat(),
    }


def build_current_situation_context(
    forecast_result: ForecastRunResult,
    *,
    config: ContextBuildConfig | None = None,
) -> JsonObject:
    """Построить блок ``current_situation`` из последней наблюдаемой недели."""

    cfg = config or ContextBuildConfig()
    data = _history_frame(forecast_result)
    date_col = forecast_result.config.datetime_col
    target_col = forecast_result.config.target_col
    origin_date = _as_timestamp(forecast_result.origin_date, name="forecast_result.origin_date")
    history = data.loc[pd.to_datetime(data[date_col]) <= origin_date].sort_values(date_col).reset_index(drop=True)
    if len(history) < 2:
        raise ContextBuilderError("Для current_situation требуется минимум две наблюдаемые недели до origin_date.")

    current_row = history.iloc[-1]
    previous_row = history.iloc[-2]
    current_value = _finite_float(current_row[target_col], name="current_value")
    previous_value = _finite_float(previous_row[target_col], name="previous_value")
    change_pct = _safe_relative_change_pct(current_value, previous_value)
    direction_code = classify_change(change_pct, threshold_pct=cfg.current_change_pct_threshold)
    direction_word = direction_word_ru(direction_code)

    window = max(2, int(cfg.current_trend_window_weeks))
    trend_values = [float(item) for item in history[target_col].tail(window).to_numpy(dtype=float)]
    trend_code = classify_current_window_shape(trend_values, flat_threshold_pct=cfg.current_change_pct_threshold)
    trend_label = trend_label_ru(trend_code)

    return {
        "current_week_date": pd.to_datetime(current_row[date_col]).date().isoformat(),
        "current_value": round(current_value, 3),
        "previous_value": round(previous_value, 3),
        "direction_word": direction_word,
        "change_pct": round(change_pct, 1),
        "trend_4w_values": [round(value, 3) for value in trend_values],
        "trend_4w_label": trend_label,
        "semantic": {
            "direction_code": direction_code,
            "trend_4w_code": trend_code,
            "threshold_pct": cfg.current_change_pct_threshold,
            "origin_date": origin_date.date().isoformat(),
        },
    }


def build_forecast_context(
    forecast_result: ForecastRunResult,
    *,
    config: ContextBuildConfig | None = None,
) -> JsonObject:
    """Построить блок ``forecast``."""

    cfg = config or ContextBuildConfig()
    point = _finite_1d(forecast_result.y_hat, name="forecast_result.y_hat")
    lo = _finite_1d(forecast_result.y_lo, name="forecast_result.y_lo")
    hi = _finite_1d(forecast_result.y_hi, name="forecast_result.y_hi")
    future_dates = pd.to_datetime(pd.Series(forecast_result.future_dates)).reset_index(drop=True)
    if not (len(point) == len(lo) == len(hi) == len(future_dates)):
        raise ContextBuilderError("Длины y_hat/y_lo/y_hi/future_dates должны совпадать.")

    horizons: list[JsonObject] = []
    widths: list[float] = []
    for idx, value in enumerate(point):
        target_date = pd.to_datetime(future_dates.iloc[idx]).date()
        q_lo = float(min(lo[idx], hi[idx]))
        q_hi = float(max(lo[idx], hi[idx]))
        width = q_hi - q_lo
        widths.append(width)
        horizons.append(
            {
                "horizon_weeks": idx + 1,
                "target_date": target_date.isoformat(),
                "target_iso_week": int(target_date.isocalendar().week),
                "point_forecast": round(float(value), 3),
                "q_lo": round(q_lo, 3),
                "q_hi": round(q_hi, 3),
                "interval_width": round(width, 3),
                "pi_width": round(width, 3),
            }
        )

    trend_code = classify_forecast_shape(point, flat_threshold_pct=cfg.forecast_flat_change_pct_threshold)
    slope = _linear_slope(point)
    relative_uncertainty_pct = _relative_uncertainty_pct(point, widths)
    uncertainty_code = classify_uncertainty(relative_uncertainty_pct, config=cfg)
    has_rise, has_decline = _has_internal_direction_changes(point, threshold_pct=cfg.forecast_flat_change_pct_threshold)

    return {
        "horizons": horizons,
        "trend_label": trend_label_ru(trend_code),
        "dynamics_description": forecast_dynamics_description(point, trend_code=trend_code),
        "slope_per_week": round(float(slope), 4),
        "uncertainty_label": uncertainty_label_ru(uncertainty_code),
        "relative_uncertainty_pct": round(relative_uncertainty_pct, 1),
        "max_interval_width": round(max(widths) if widths else 0.0, 3),
        "semantic": {
            "shape_code": trend_code,
            "forecast_shape_code": trend_code,
            "uncertainty_level_code": uncertainty_code,
            "has_intermediate_rise": has_rise,
            "has_intermediate_decline": has_decline,
            "point_forecast_values": [round(float(value), 3) for value in point],
            "interval_method": getattr(forecast_result.config, "interval_method", "unknown"),
            "nominal_coverage": _nominal_coverage(forecast_result),
        },
    }


def build_shap_summary_context(shap_result: Any) -> JsonObject:
    """Построить блок ``shap_summary`` из результата explainability-слоя."""

    if shap_result is None:
        raise MissingContextBlockError("shap_summary не передан.")
    if hasattr(shap_result, "summary") and hasattr(shap_result.summary, "to_context_payload"):
        return _normalize_shap_summary_payload(_json_object(shap_result.summary.to_context_payload()))
    if hasattr(shap_result, "to_context_payload"):
        return _normalize_shap_summary_payload(_json_object(shap_result.to_context_payload()))
    payload = _json_object(shap_result)
    if "summary" in payload and isinstance(payload["summary"], Mapping):
        summary = _json_object(payload["summary"])
        if "by_horizon" in summary and "key_insight" in summary:
            return _normalize_shap_summary_payload(
                {
                    "by_horizon": summary["by_horizon"],
                    "key_insight": summary["key_insight"],
                    "semantic": summary.get("semantic", {}),
                }
            )
    if "by_horizon" in payload and "key_insight" in payload:
        return _normalize_shap_summary_payload(
            {
                "by_horizon": payload["by_horizon"],
                "key_insight": payload["key_insight"],
                "semantic": payload.get("semantic", {}),
            }
        )
    raise ContextBuilderError("shap_result должен содержать поля by_horizon и key_insight.")

def build_epidemic_wave_comparison_context(value: Any) -> JsonObject:
    """Построить блок ``epidemic_wave_comparison`` из run-result или payload.

    Принимает готовый mapping или объект с методом ``to_context_payload()``,
    например ``EpidemicWaveRunResult``. Полный run-result содержит
    pandas.DataFrame в ``tables`` и не должен сериализоваться целиком при
    сборке GlobalContext.
    """

    payload = _context_payload_object(value, block_name="epidemic_wave_comparison")
    required = ("waves", "season_labels", "latest_vs_previous")
    missing = [field for field in required if field not in payload]
    if missing:
        raise ContextBuilderError(
            "epidemic_wave_comparison payload не содержит обязательные поля: "
            f"{missing!r}."
        )
    return payload


def build_age_group_season_context(value: Any) -> JsonObject:
    """Построить блок ``age_group_season`` из run-result или payload.

    Принимает готовый mapping или объект с методом ``to_context_payload()``,
    например ``AgeGroupSeasonRunResult``.
    """

    payload = _context_payload_object(value, block_name="age_group_season")
    required = ("season_label", "metric_label_ru", "rows", "derived_findings")
    missing = [field for field in required if field not in payload]
    if missing:
        raise ContextBuilderError(
            "age_group_season payload не содержит обязательные поля: "
            f"{missing!r}."
        )
    return payload


def build_model_quality_context(forecast_result: ForecastRunResult) -> JsonObject:
    """Построить блок ``model_quality``."""

    per_h = forecast_result.metrics.per_horizon.copy()
    required = {"horizon_weeks", "mae", "rmse", "r2"}
    missing = required.difference(per_h.columns)
    if missing:
        raise ContextBuilderError(f"В per_horizon отсутствуют колонки: {sorted(missing)!r}.")

    n_test = int(np.asarray(forecast_result.y_test_mat).shape[0])
    metrics: list[JsonObject] = []
    for _, row in per_h.sort_values("horizon_weeks").iterrows():
        metrics.append(
            {
                "horizon_weeks": int(row["horizon_weeks"]),
                "mae": round(_finite_float(row["mae"], name="mae"), 4),
                "rmse": round(_finite_float(row["rmse"], name="rmse"), 4),
                "r2": round(_finite_float(row["r2"], name="r2"), 4),
                "n_test": n_test,
            }
        )

    y_true = np.asarray(forecast_result.y_test_mat, dtype=float)
    y_pred = np.asarray(forecast_result.y_pred_mat, dtype=float)
    if y_true.ndim != 2 or y_pred.ndim != 2 or y_true.shape != y_pred.shape:
        raise ContextBuilderError("y_test_mat и y_pred_mat должны быть двумерными матрицами одинаковой формы.")
    if y_true.shape[1] < 1:
        raise ContextBuilderError("Для error_stats_h1 требуется минимум один горизонт.")

    h1_abs = np.abs(y_true[:, 0] - y_pred[:, 0])
    dates_test = pd.to_datetime(pd.Series(forecast_result.dates_test)).reset_index(drop=True)
    worst_case_pattern, worst_case_regime, peak_error_months, peak_error_month_numbers, pct_underestimated = build_worst_case_pattern(
        h1_abs,
        dates_test,
        y_true_h1=y_true[:, 0],
        y_pred_h1=y_pred[:, 0],
    )

    return {
        "metrics": metrics,
        "error_stats_h1": {
            "median_error_h1": round(float(np.median(h1_abs)), 4),
            "p75_error_h1": round(float(np.quantile(h1_abs, 0.75)), 4),
            "max_error_h1": round(float(np.max(h1_abs)), 4),
            "mean_error_h1": round(float(np.mean(h1_abs)), 4),
        },
        "worst_case_pattern": worst_case_pattern,
        "semantic": {
            "worst_case_regime": worst_case_regime,
            "peak_error_months": peak_error_months,
            "peak_error_month_numbers": peak_error_month_numbers,
            "pct_underestimated_worst_cases": pct_underestimated,
            "test_weeks": n_test,
            "overall": {key: round(float(value), 4) for key, value in forecast_result.metrics.overall.items()},
        },
    }


def build_model_info_context(
    forecast_result: ForecastRunResult,
    *,
    config: ContextBuildConfig | None = None,
) -> JsonObject:
    """Построить блок ``model_info``."""

    cfg = config or ContextBuildConfig()
    horizons = [int(item) for item in range(1, len(forecast_result.target_cols) + 1)]
    feature_groups = list(dict.fromkeys(str(item) for item in _model_registry_feature_groups(forecast_result)))
    if not feature_groups:
        feature_groups = [
            "target_lags",
            "target_rolling_stats",
            "epidemic_dynamics",
            "fourier_seasonality",
            "temperature_lags",
            "temperature_rolling_stats",
            "calendar_features",
        ]

    calibration_start_year = int(pd.to_datetime(forecast_result.split.train_start).year)
    n_point_models = len(horizons)
    interval_method = getattr(forecast_result.config, "interval_method", "unknown")
    return {
        "family": "HistGradientBoostingRegressor",
        "family_ru": cfg.model_family_ru,
        "strategy": "direct_multi_step",
        "strategy_ru": cfg.model_strategy_ru,
        "task_type": "multi_horizon_weekly_forecast",
        "task_type_ru": cfg.task_type_ru,
        "forecast_target_ru": cfg.forecast_target_ru,
        "horizons": horizons,
        "n_point_models": n_point_models,
        "feature_groups": feature_groups,
        "feature_groups_ru": {name: feature_group_ru(name) for name in feature_groups},
        "calibration_start_year": calibration_start_year,
        "calibration_data_description": cfg.calibration_data_description,
        "semantic": {
            "family_code": "hist_gradient_boosting_regressor",
            "family_sklearn_name": "HistGradientBoostingRegressor",
            "strategy_code": "direct_multi_step",
            "task_type_code": "multi_horizon_weekly_forecast",
            "target_variable": forecast_result.config.target_variable,
            "target_transform": forecast_result.config.target_transform,
            "interval_method": interval_method,
            "n_features": len(forecast_result.feature_cols),
            "n_point_models": n_point_models,
        },
    }


# ---------------------------------------------------------------------------
# Classification and text helpers
# ---------------------------------------------------------------------------


def classify_change(change_pct: float, *, threshold_pct: float) -> TrendCode:
    """Классифицировать одно относительное изменение."""

    if change_pct >= threshold_pct:
        return "increase"
    if change_pct <= -threshold_pct:
        return "decrease"
    return "stable"


def _direction_sequence(values: Sequence[float], *, threshold_pct: float) -> list[BaseTrendCode]:
    """Вернуть последовательность значимых направлений между соседними точками."""

    arr = _finite_1d(values, name="values")
    directions: list[BaseTrendCode] = []
    for prev, curr in zip(arr[:-1], arr[1:]):
        directions.append(classify_change(_safe_relative_change_pct(float(curr), float(prev)), threshold_pct=threshold_pct))
    return directions


def _non_stable_directions(values: Sequence[float], *, threshold_pct: float) -> list[BaseTrendCode]:
    return [direction for direction in _direction_sequence(values, threshold_pct=threshold_pct) if direction != "stable"]


def classify_current_window_shape(values: Sequence[float], *, flat_threshold_pct: float) -> CurrentTrendCode:
    """Классифицировать наблюдаемую динамику за короткое окно."""

    arr = _finite_1d(values, name="values")
    if len(arr) < 2:
        return "stable"

    directions = _non_stable_directions(arr, threshold_pct=flat_threshold_pct)
    if not directions:
        return "stable"
    if all(direction == "increase" for direction in directions):
        return "increase"
    if all(direction == "decrease" for direction in directions):
        return "decrease"

    net_code = classify_change(_safe_relative_change_pct(float(arr[-1]), float(arr[0])), threshold_pct=flat_threshold_pct)
    if net_code == "increase":
        return "up_with_fluctuations"
    if net_code == "decrease":
        return "down_with_fluctuations"
    return "mixed"


def classify_forecast_shape(values: Sequence[float], *, flat_threshold_pct: float) -> ForecastShapeCode:
    """Классифицировать форму точечной прогнозной траектории."""

    arr = _finite_1d(values, name="values")
    if len(arr) < 2:
        return "stable"

    directions = _non_stable_directions(arr, threshold_pct=flat_threshold_pct)
    if not directions:
        return "stable"
    if all(direction == "increase" for direction in directions):
        return "increase"
    if all(direction == "decrease" for direction in directions):
        return "decrease"

    compressed: list[BaseTrendCode] = []
    for direction in directions:
        if not compressed or compressed[-1] != direction:
            compressed.append(direction)
    if compressed == ["increase", "decrease"]:
        return "rise_then_fall"
    if compressed == ["decrease", "increase"]:
        return "fall_then_rise"
    return "mixed"

def classify_uncertainty(relative_uncertainty_pct: float, *, config: ContextBuildConfig) -> UncertaintyCode:
    """Классифицировать относительную ширину прогнозного интервала."""

    value = float(relative_uncertainty_pct)
    if value >= config.uncertainty_very_high_pct_threshold:
        return "very_high"
    if value >= config.uncertainty_high_pct_threshold:
        return "high"
    if value <= config.uncertainty_low_pct_threshold:
        return "low"
    return "moderate"


def direction_word_ru(code: TrendCode) -> str:
    """Русский ярлык изменения текущей недели."""

    return {
        "increase": "повысился",
        "decrease": "понизился",
        "stable": "не изменился",
        "mixed": "нестабильная динамика",
        "up_with_fluctuations": "повысился",
        "down_with_fluctuations": "понизился",
        "rise_then_fall": "разнонаправленная динамика",
        "fall_then_rise": "разнонаправленная динамика",
    }[code]


def trend_label_ru(code: TrendCode) -> str:
    """Русский ярлык формы временной траектории."""

    return {
        "increase": "устойчивый рост",
        "decrease": "устойчивое снижение",
        "stable": "стабилизация",
        "up_with_fluctuations": "рост с колебаниями",
        "down_with_fluctuations": "снижение с колебаниями",
        "rise_then_fall": "временный подъём с последующим снижением",
        "fall_then_rise": "временное снижение с последующим ростом",
        "mixed": "разнонаправленная динамика",
    }[code]


def uncertainty_label_ru(code: UncertaintyCode) -> str:
    """Русский ярлык уровня неопределённости."""

    return {
        "low": "низкая неопределённость",
        "moderate": "умеренная неопределённость",
        "high": "высокая неопределённость",
        "very_high": "очень высокая неопределённость",
    }[code]


def forecast_dynamics_description(values: Sequence[float], *, trend_code: TrendCode) -> str:
    """Сформировать короткое численно обусловленное описание формы прогноза."""

    arr = _finite_1d(values, name="forecast values")
    start = float(arr[0])
    end = float(arr[-1])
    if trend_code == "increase":
        return f"центральная траектория указывает на преимущественный рост с {start:.2f} до {end:.2f}"
    if trend_code == "decrease":
        return f"центральная траектория указывает на преимущественное снижение с {start:.2f} до {end:.2f}"
    if trend_code == "stable":
        return f"центральная траектория остаётся близкой к стабильной около {float(np.mean(arr)):.2f}"
    min_idx = int(np.argmin(arr)) + 1
    max_idx = int(np.argmax(arr)) + 1
    if trend_code == "rise_then_fall":
        return f"центральная траектория: подъём до {float(np.max(arr)):.2f} на горизонте h={max_idx}, затем снижение до {end:.2f} к концу периода"
    if trend_code == "fall_then_rise":
        return f"центральная траектория: снижение до {float(np.min(arr)):.2f} на горизонте h={min_idx}, затем подъём до {end:.2f} к концу периода"
    return (
        f"центральная траектория разнонаправленная: минимум {float(np.min(arr)):.2f} на горизонте h={min_idx}, "
        f"максимум {float(np.max(arr)):.2f} на горизонте h={max_idx}"
    )


def build_worst_case_pattern(
    abs_errors_h1: np.ndarray,
    dates_test: pd.Series,
    *,
    y_true_h1: Sequence[float] | np.ndarray | None = None,
    y_pred_h1: Sequence[float] | np.ndarray | None = None,
    top_n: int = 5,
) -> tuple[str, str, list[str], list[int], float | None]:
    """Описать режим худших ошибок h=1 без обращения к LLM.

    В отличие от простого перечисления ISO-недель, этот слой готовит
    narrator-facing семантику: режим ошибок и месяцы, где они чаще всего
    встречались. Абсолютные ошибки используются для выбора worst cases; знак
    ошибки используется только если переданы ``y_true_h1`` и ``y_pred_h1``.
    """

    errors = _finite_1d(abs_errors_h1, name="abs_errors_h1")
    dates = pd.to_datetime(pd.Series(dates_test)).reset_index(drop=True)
    if len(errors) != len(dates):
        raise ContextBuilderError("Длины abs_errors_h1 и dates_test должны совпадать.")
    if len(errors) == 0:
        raise ContextBuilderError("Нельзя описать worst-case pattern: test-набор пуст.")

    order = np.argsort(-errors)[: max(1, min(top_n, len(errors)))]
    top_dates = [pd.to_datetime(dates.iloc[int(idx)]).date() for idx in order]
    top_month_numbers = [int(item.month) for item in top_dates]
    month_counts: dict[int, int] = {}
    for month in top_month_numbers:
        month_counts[month] = month_counts.get(month, 0) + 1
    peak_month_numbers = [month for month, _ in sorted(month_counts.items(), key=lambda item: (-item[1], item[0]))[:2]]
    peak_month_names = [_month_name_ru(month) for month in peak_month_numbers]

    pct_under: float | None = None
    if y_true_h1 is not None and y_pred_h1 is not None:
        true_arr = _finite_1d(y_true_h1, name="y_true_h1")
        pred_arr = _finite_1d(y_pred_h1, name="y_pred_h1")
        if len(true_arr) != len(pred_arr) or len(true_arr) != len(errors):
            raise ContextBuilderError("y_true_h1/y_pred_h1 должны совпадать по длине с abs_errors_h1.")
        pct_under = round(float(np.mean(true_arr[order] > pred_arr[order]) * 100.0), 1)

    months_text = ", ".join(peak_month_names)
    if pct_under is not None and pct_under > 60.0:
        pattern = (
            "Модель систематически недооценивает пики заболеваемости "
            f"({pct_under}% worst cases — недооценка). "
            f"Наибольшие ошибки приходятся на {months_text}."
        )
        regime = "systematic_underestimation_peaks"
    elif pct_under is not None:
        pattern = (
            "Ошибки распределены между недооценкой и переоценкой "
            f"(недооценка: {pct_under}%). "
            f"Пиковые ошибки в {months_text}."
        )
        regime = "mixed_errors"
    else:
        pattern = f"Наибольшие абсолютные ошибки однонедельного прогноза приходятся на {months_text}."
        regime = "largest_h1_absolute_errors_in_test_holdout"

    return pattern, regime, peak_month_names, peak_month_numbers, pct_under


def _month_name_ru(month: int) -> str:
    return {
        1: "январь",
        2: "февраль",
        3: "март",
        4: "апрель",
        5: "май",
        6: "июнь",
        7: "июль",
        8: "август",
        9: "сентябрь",
        10: "октябрь",
        11: "ноябрь",
        12: "декабрь",
    }.get(int(month), str(month))


def feature_group_ru(name: str) -> str:
    """Русское название группы признаков."""

    return FEATURE_GROUPS_RU.get(name, name)



def _normalize_shap_summary_payload(payload: Mapping[str, Any]) -> JsonObject:
    """Стабилизировать semantic-контракт SHAP-блока для narrator-слоя."""

    result = _json_object(payload)
    by_horizon = _json_object(result.get("by_horizon", {}))
    semantic = _json_object(result.get("semantic", {}))

    h1_names = _top_feature_display_names(by_horizon, horizon="1")
    h4_names = _top_feature_display_names(by_horizon, horizon="4")
    if h1_names:
        semantic.setdefault("h1_top_feature_names", h1_names)
    if h4_names:
        semantic.setdefault("h4_top_feature_names", h4_names)

    h1_feature_ids = [str(item) for item in semantic.get("h1_top_features", [])]
    h4_feature_ids = [str(item) for item in semantic.get("h4_top_features", [])]
    if h1_feature_ids:
        semantic.setdefault("h1_top_feature_ids", h1_feature_ids)
    if h4_feature_ids:
        semantic.setdefault("h4_top_feature_ids", h4_feature_ids)

    h1_group = _notebook_compatible_shap_group(h1_feature_ids, fallback=semantic.get("h1_dominant_group"))
    h4_group = _notebook_compatible_shap_group(h4_feature_ids, fallback=semantic.get("h4_dominant_group"))
    if h1_group:
        semantic["h1_dominant_group"] = h1_group
    if h4_group:
        semantic["h4_dominant_group"] = h4_group
    if h1_group and h4_group:
        transition = _transition_pattern(h1_group, h4_group)
        semantic["transition_pattern"] = transition
        if transition == "lags_to_temp_seasonality":
            result["key_insight"] = (
                "На коротком горизонте (1 нед.) доминируют лаговые признаки заболеваемости "
                "(инерция эпидпроцесса). На длинном горизонте (4 нед.) возрастает роль "
                "температурных и сезонных факторов."
            )

    result["semantic"] = semantic
    return result


def _top_feature_display_names(by_horizon: Mapping[str, Any], *, horizon: str, limit: int = 3) -> list[str]:
    records = by_horizon.get(str(horizon), [])
    if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
        return []
    names: list[str] = []
    for record in records[:limit]:
        if isinstance(record, Mapping):
            name = record.get("название") or record.get("name")
            if name is not None:
                names.append(str(name))
    return names


def _notebook_compatible_shap_group(feature_ids: Sequence[str], *, fallback: Any = None) -> str | None:
    """Сопоставить raw SHAP feature ids с narrator-facing группой."""

    if not feature_ids:
        if fallback is None:
            return None
        fallback_text = str(fallback)
        return {
            "target_lags": "lags",
            "target_rolling": "lags",
            "target_rolling_stats": "lags",
            "weather_rolling": "temp_seasonality",
            "temperature_rolling_stats": "temp_seasonality",
            "calendar": "temp_seasonality",
            "calendar_features": "temp_seasonality",
            "fourier_seasonality": "temp_seasonality",
        }.get(fallback_text, fallback_text)

    top = [str(item) for item in feature_ids[:3]]
    counts = {
        "lags": 0,
        "temp_seasonality": 0,
        "epidemic_dynamics": 0,
        "other": 0,
    }
    for feature in top:
        if feature.startswith("y_lag"):
            counts["lags"] += 1
        elif feature.startswith("temp_") or feature.startswith(("sin_", "cos_")) or feature == "iso_week":
            counts["temp_seasonality"] += 1
        elif feature.startswith("y_roll") or feature.startswith("y_diff") or feature == "y_accel":
            counts["epidemic_dynamics"] += 1
        else:
            counts["other"] += 1

    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    if ordered[0][1] == 0:
        return str(fallback) if fallback is not None else "other"
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        if fallback is not None:
            return _notebook_compatible_shap_group([], fallback=fallback)
    return ordered[0][0]


def _transition_pattern(h1_group: str, h4_group: str) -> str:
    if h1_group == "lags" and h4_group == "temp_seasonality":
        return "lags_to_temp_seasonality"
    if h1_group == h4_group:
        return "stable_feature_regime"
    return "mixed_transition"

# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def save_context_payload(payload: Mapping[str, Any], path: str | Path, *, indent: int = 2) -> Path:
    """Сохранить JSON payload контекста до или после pydantic-валидации."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=indent), encoding="utf-8")
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _history_frame(forecast_result: ForecastRunResult) -> pd.DataFrame:
    data = forecast_result.supervised.data.copy()
    date_col = forecast_result.config.datetime_col
    target_col = forecast_result.config.target_col
    missing = {date_col, target_col}.difference(data.columns)
    if missing:
        raise ContextBuilderError(f"В supervised.data отсутствуют обязательные колонки: {sorted(missing)!r}.")
    data[date_col] = pd.to_datetime(data[date_col])
    data[target_col] = pd.to_numeric(data[target_col], errors="coerce")
    if data[target_col].isna().any():
        raise ContextBuilderError(f"Колонка {target_col!r} содержит NaN после численного приведения.")
    return data.sort_values(date_col).reset_index(drop=True)


def _safe_relative_change_pct(current: float, previous: float) -> float:
    if not math.isfinite(current) or not math.isfinite(previous):
        raise ContextBuilderError("Нельзя посчитать относительное изменение по нечисловым значениям.")
    denom = abs(previous)
    if denom <= 1e-12:
        if abs(current) <= 1e-12:
            return 0.0
        return math.copysign(100.0, current - previous)
    return 100.0 * (current - previous) / denom


def _linear_slope(values: Sequence[float]) -> float:
    arr = _finite_1d(values, name="values")
    if len(arr) < 2:
        return 0.0
    x = np.arange(1, len(arr) + 1, dtype=float)
    return float(np.polyfit(x, arr, 1)[0])


def _relative_uncertainty_pct(points: Sequence[float], widths: Sequence[float]) -> float:
    """Вернуть относительную среднюю ширину интервалов к средней точечной оценке."""

    p = np.abs(_finite_1d(points, name="points"))
    w = _finite_1d(widths, name="widths")
    if len(p) != len(w):
        raise ContextBuilderError("points и widths должны иметь одинаковую длину.")
    denom = max(float(np.mean(p)), 1e-9)
    return 100.0 * float(np.mean(w)) / denom


def _has_internal_direction_changes(values: Sequence[float], *, threshold_pct: float) -> tuple[bool, bool]:
    arr = _finite_1d(values, name="values")
    has_rise = False
    has_decline = False
    for prev, curr in zip(arr[:-1], arr[1:]):
        code = classify_change(_safe_relative_change_pct(float(curr), float(prev)), threshold_pct=threshold_pct)
        has_rise = has_rise or code == "increase"
        has_decline = has_decline or code == "decrease"
    return has_rise, has_decline


def _nominal_coverage(forecast_result: ForecastRunResult) -> float | None:
    alpha = getattr(forecast_result.config, "interval_alpha", None)
    if alpha is None:
        quantiles = getattr(forecast_result.config, "quantiles", None)
        if quantiles and len(quantiles) >= 2:
            return round(float(max(quantiles) - min(quantiles)), 4)
        return None
    return round(1.0 - float(alpha), 4)


def _model_registry_feature_groups(forecast_result: ForecastRunResult) -> list[str]:
    registry = forecast_result.model_registry or {}
    groups = registry.get("feature_groups") if isinstance(registry, Mapping) else None
    if isinstance(groups, Sequence) and not isinstance(groups, (str, bytes, bytearray)):
        return [str(item) for item in groups]
    return []


def _as_timestamp(value: Any, *, name: str) -> pd.Timestamp:
    ts = pd.to_datetime(value)
    if pd.isna(ts):
        raise ContextBuilderError(f"{name} не является валидной датой.")
    return pd.Timestamp(ts).tz_localize(None) if getattr(pd.Timestamp(ts), "tzinfo", None) is not None else pd.Timestamp(ts)


def _finite_float(value: Any, *, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ContextBuilderError(f"{name} должен быть числом, получено {value!r}.") from exc
    if not math.isfinite(result):
        raise ContextBuilderError(f"{name} должен быть конечным числом, получено {value!r}.")
    return result


def _finite_1d(values: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ContextBuilderError(f"{name} должен быть одномерным массивом, получена форма {arr.shape}.")
    if len(arr) == 0:
        raise ContextBuilderError(f"{name} не должен быть пустым.")
    if not np.isfinite(arr).all():
        raise ContextBuilderError(f"{name} содержит NaN или бесконечные значения.")
    return arr


def _context_payload_object(value: Any, *, block_name: str) -> JsonObject:
    if value is None:
        raise MissingContextBlockError(f"Для GlobalContext требуется блок {block_name}.")
    if hasattr(value, "to_context_payload"):
        return _json_object(value.to_context_payload())
    if hasattr(value, "bundle"):
        return _json_object(getattr(value, "bundle"))
    return _json_object(value)


def _json_object(value: Any | None) -> JsonObject:
    if value is None:
        return {}
    if isinstance(value, BaseModel):
        data = value.model_dump(mode="python", exclude_none=False)
        if not isinstance(data, dict):
            raise ContextBuilderError("Ожидался JSON object.")
        return {str(key): _to_jsonable(item) for key, item in data.items()}
    if not isinstance(value, Mapping):
        raise ContextBuilderError(f"Ожидался mapping, получено {type(value).__name__}.")
    return {str(key): _to_jsonable(item) for key, item in value.items()}


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return _to_jsonable(value.model_dump(mode="python", by_alias=True, exclude_none=False))
    if isinstance(value, Mapping):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, (pd.Timestamp, date)):
        return value.isoformat()
    if isinstance(value, pd.Series):
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return [_to_jsonable(item) for item in value.to_dict(orient="records")]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_to_jsonable(item) for item in value]
    return value


__all__ = [
    "ContextBuildConfig",
    "ContextBuildInput",
    "ContextBuildResult",
    "ContextBuilderError",
    "ContextOutputConfig",
    "MissingContextBlockError",
    "build_current_situation_context",
    "build_forecast_context",
    "build_global_context",
    "build_global_context_payload",
    "build_global_context_result",
    "build_epidemic_wave_comparison_context",
    "build_age_group_season_context",
    "build_model_info_context",
    "build_model_quality_context",
    "build_origin_context",
    "build_shap_summary_context",
    "classify_change",
    "classify_current_window_shape",
    "classify_forecast_shape",
    "classify_uncertainty",
    "save_context_payload",
]

