"""
Модельный слой многошагового прогноза заболеваемости для ai4epi.

Модуль переносит forecasting-часть исследовательского notebook в
переиспользуемый контракт репозитория. Он не загружает данные и не строит
признаки самостоятельно: входом является ``SupervisedDataset`` из
``ai4epi.preprocessing``.

Базовая схема соответствует notebook:

* direct multi-step forecasting;
* одна point-модель ``HistGradientBoostingRegressor(loss='poisson')`` на каждый
  горизонт ``h=1..H``;
* основной режим интервалов — split conformal prediction по абсолютным ошибкам
  на calibration-блоке;
* legacy-режим quantile-моделей сохранён как явная опция;
* time-based holdout на последних ``test_weeks`` валидных неделях;
* для conformal-режима интервалы калибруются отдельно, а production point fit
  выполняется на всех валидных строках, как в notebook;
* прогноз от последней строки с полностью определёнными признаками;
* CSV/JSON-артефакты: ``metrics_summary.csv``, ``test_predictions.csv``,
  ``forecast_next_{H}w.csv``, ``feature_list.csv``,
  ``history_plus_forecast_40.csv``, ``conformal_radii.csv``,
  ``interval_metrics.csv``, ``model_registry.json``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4epi.core.io import TableWriteOptions, write_table
from ai4epi.analysis.preprocessing import (
    FeatureEngineeringConfig,
    HoldoutSplitConfig,
    SupervisedDataset,
    TimeHoldoutSplit,
    build_supervised,
    make_time_holdout_split,
    supervised_to_feature_list_frame,
)

try:  # pragma: no cover - отсутствие sklearn проверяется интеграционным окружением.
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
except ImportError:  # pragma: no cover
    HistGradientBoostingRegressor = None  # type: ignore[assignment]
    mean_absolute_error = None  # type: ignore[assignment]
    mean_squared_error = None  # type: ignore[assignment]
    r2_score = None  # type: ignore[assignment]


TargetTransformName = Literal["none", "log1p"]
ForecastModelFamily = Literal["hist_gradient_boosting"]
IntervalMethod = Literal["conformal", "quantile", "none"]


class ForecastingError(ValueError):
    """Базовая ошибка forecasting-слоя."""


class ForecastingConfigError(ForecastingError):
    """Некорректная конфигурация прогноза."""


class ModelTrainingError(ForecastingError):
    """Ошибка обучения модели."""


class ForecastArtifactError(ForecastingError):
    """Ошибка сохранения forecast-артефактов."""


class StrictModel(BaseModel):
    """Базовая pydantic-модель с закрытым контрактом."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )


class HistGradientBoostingPointConfig(StrictModel):
    """Параметры point-моделей из активного notebook-блока."""

    loss: Literal["poisson", "squared_error", "absolute_error", "gamma"] = "poisson"
    max_depth: int | None = None
    max_leaf_nodes: int | None = None
    learning_rate: float = Field(default=0.001, gt=0)
    max_iter: int = Field(default=8000, ge=1)
    min_samples_leaf: int = Field(default=95, ge=1)
    l2_regularization: float = Field(default=1.0, ge=0)
    early_stopping: bool = True
    validation_fraction: float | None = None

    def to_sklearn_params(self) -> dict[str, Any]:
        """Вернуть параметры конструктора HistGradientBoostingRegressor."""

        return {
            "loss": self.loss,
            "max_depth": self.max_depth,
            "max_leaf_nodes": self.max_leaf_nodes,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "early_stopping": self.early_stopping,
            "validation_fraction": self.validation_fraction,
        }


class HistGradientBoostingQuantileConfig(StrictModel):
    """Параметры quantile-моделей из notebook."""

    max_depth: int | None = None
    max_leaf_nodes: int | None = None
    learning_rate: float = Field(default=0.001, gt=0)
    max_iter: int = Field(default=8000, ge=1)
    min_samples_leaf: int = Field(default=25, ge=1)
    l2_regularization: float = Field(default=1.0, ge=0)
    early_stopping: bool = True
    validation_fraction: float | None = None

    def to_sklearn_params(self, *, quantile: float) -> dict[str, Any]:
        """Вернуть параметры конструктора quantile-регрессора."""

        return {
            "loss": "quantile",
            "quantile": float(quantile),
            "max_depth": self.max_depth,
            "max_leaf_nodes": self.max_leaf_nodes,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "min_samples_leaf": self.min_samples_leaf,
            "l2_regularization": self.l2_regularization,
            "early_stopping": self.early_stopping,
            "validation_fraction": self.validation_fraction,
        }


class ForecastingConfig(StrictModel):
    """Настройки полного forecasting-pass."""

    horizons: int = Field(default=4, ge=1, le=52)
    test_weeks: int = Field(default=52, ge=1)
    min_train_weeks: int = Field(default=104, ge=1)
    target_col: str = Field(default="inc_per_10k", min_length=1)
    datetime_col: str = Field(default="datetime", min_length=1)
    target_transform: TargetTransformName = "none"
    interval_method: IntervalMethod = "conformal"
    interval_alpha: float = Field(default=0.20, gt=0.0, lt=1.0)
    calib_weeks: int = Field(default=52, ge=1)
    conformal_refit_on_train_plus_calib: bool = True
    quantiles: tuple[float, float] = (0.10, 0.90)
    random_state: int = 42
    point_model: HistGradientBoostingPointConfig = Field(default_factory=HistGradientBoostingPointConfig)
    quantile_model: HistGradientBoostingQuantileConfig = Field(default_factory=HistGradientBoostingQuantileConfig)
    notebook_id: str = Field(default="gbdt_influenza_forecast_4w", min_length=1)
    target_variable: str = Field(default="inc_per_10k", min_length=1)
    time_unit: str = Field(default="week", min_length=1)
    history_window: int = Field(default=40, ge=1)
    clip_predictions_to_non_negative: bool = True

    @field_validator("quantiles")
    @classmethod
    def validate_quantiles(cls, value: tuple[float, float]) -> tuple[float, float]:
        if len(value) != 2:
            raise ValueError("quantiles должен содержать ровно два значения: нижний и верхний квантили.")
        lower, upper = tuple(float(v) for v in value)
        if not 0.0 < lower < upper < 1.0:
            raise ValueError("Ожидается 0 < lower_quantile < upper_quantile < 1.")
        return lower, upper

    @model_validator(mode="after")
    def validate_point_loss_with_transform(self) -> "ForecastingConfig":
        # Poisson loss в sklearn требует неотрицательную целевую переменную.
        # log1p-преобразование сохраняет неотрицательность при исходном y >= 0.
        if self.point_model.loss == "poisson" and self.target_transform not in {"none", "log1p"}:
            raise ValueError("Poisson-модель поддерживает только неотрицательные target transform.")
        if self.interval_method == "conformal" and self.calib_weeks < 1:
            raise ValueError("Для conformal-интервалов calib_weeks должен быть положительным.")
        return self

    @property
    def holdout_config(self) -> HoldoutSplitConfig:
        """Конфигурация time-based holdout-разбиения."""

        return HoldoutSplitConfig(test_weeks=self.test_weeks, min_train_weeks=self.min_train_weeks)


class ForecastOutputConfig(StrictModel):
    """Настройки сохранения forecast-артефактов."""

    output_dir: Path = Path("results_csv")
    registry_path: Path | None = Path("model_registry.json")
    save_metrics_summary: bool = True
    save_test_predictions: bool = True
    save_forecast_next: bool = True
    save_feature_list: bool = True
    save_history_plus_forecast: bool = True
    save_conformal_radii: bool = True
    save_interval_metrics: bool = True
    save_model_registry: bool = True
    csv_encoding: str = Field(default="utf-8-sig", min_length=1)


class ForecastMetrics(StrictModel):
    """Метрики качества многошагового прогноза."""

    per_horizon: Any
    overall: dict[str, float]

    @field_validator("per_horizon")
    @classmethod
    def validate_per_horizon_df(cls, value: Any) -> pd.DataFrame:
        if not isinstance(value, pd.DataFrame):
            raise TypeError("per_horizon должен быть pandas.DataFrame.")
        required = {"horizon_weeks", "r2", "rmse", "mae"}
        missing = required.difference(value.columns)
        if missing:
            raise ValueError(f"В per_horizon отсутствуют колонки: {sorted(missing)}.")
        return value


class ForecastTables(StrictModel):
    """Табличные артефакты forecast-pass."""

    metrics_summary: Any
    test_predictions: Any
    forecast_next: Any
    feature_list: Any
    history_plus_forecast: Any
    conformal_radii: Any | None = None
    interval_metrics: Any | None = None

    @field_validator(
        "metrics_summary",
        "test_predictions",
        "forecast_next",
        "feature_list",
        "history_plus_forecast",
        "conformal_radii",
        "interval_metrics",
    )
    @classmethod
    def validate_dataframe(cls, value: Any) -> pd.DataFrame | None:
        if value is None:
            return None
        if not isinstance(value, pd.DataFrame):
            raise TypeError("ForecastTables содержит только pandas.DataFrame или None.")
        return value


class ForecastRunResult(StrictModel):
    """Результат полного forecasting-pass."""

    supervised: SupervisedDataset
    split: TimeHoldoutSplit
    config: ForecastingConfig
    feature_cols: tuple[str, ...]
    target_cols: tuple[str, ...]
    models_eval: list[Any]
    models_prod: list[Any]
    q_models_prod: dict[float, list[Any]]
    conformal_radii: Any | None = None
    interval_metrics: Any | None = None
    X_test: Any
    y_test_mat: Any
    y_pred_mat: Any
    dates_test: Any
    metrics: ForecastMetrics
    y_hat: Any
    y_lo: Any
    y_hi: Any
    origin_date: pd.Timestamp
    future_dates: Any
    model_registry: dict[str, Any]
    tables: ForecastTables
    artifacts: dict[str, Path] = Field(default_factory=dict)

    @field_validator("X_test", "y_test_mat", "y_pred_mat", "y_hat", "y_lo", "y_hi", "conformal_radii")
    @classmethod
    def validate_numpy_array(cls, value: Any) -> np.ndarray | None:
        if value is None:
            return None
        if not isinstance(value, np.ndarray):
            raise TypeError("Ожидался numpy.ndarray или None.")
        return value

    @field_validator("interval_metrics")
    @classmethod
    def validate_optional_dataframe(cls, value: Any) -> pd.DataFrame | None:
        if value is None:
            return None
        if not isinstance(value, pd.DataFrame):
            raise TypeError("interval_metrics должен быть pandas.DataFrame или None.")
        return value

    @field_validator("dates_test", "future_dates")
    @classmethod
    def validate_dates(cls, value: Any) -> pd.Series | pd.DatetimeIndex:
        if not isinstance(value, (pd.Series, pd.DatetimeIndex)):
            raise TypeError("Ожидался pandas.Series или pandas.DatetimeIndex с датами.")
        return value

    @property
    def per_h(self) -> pd.DataFrame:
        """Совместимое с notebook имя таблицы метрик по горизонтам."""

        return self.metrics.per_horizon

    @property
    def overall(self) -> dict[str, float]:
        """Совместимое с notebook имя агрегированных метрик."""

        return self.metrics.overall

    def to_notebook_dict(self) -> dict[str, Any]:
        """Вернуть словарь с ключами, совместимыми с текущим notebook."""

        return {
            "feature_cols": list(self.feature_cols),
            "data": self.supervised.data,
            "per_h": self.metrics.per_horizon,
            "overall": self.metrics.overall,
            "models_eval": self.models_eval,
            "models_prod": self.models_prod,
            "q_models_prod": self.q_models_prod,
            "conformal_radii": self.conformal_radii,
            "interval_metrics": self.interval_metrics,
            "y_hat": self.y_hat,
            "y_lo": self.y_lo,
            "y_hi": self.y_hi,
            "origin_date": self.origin_date,
            "future_dates": self.future_dates,
            "model_registry": self.model_registry,
            "fig": None,
            "X_test": self.X_test,
            "y_test_mat": self.y_test_mat,
            "y_pred_mat": self.y_pred_mat,
            "y_inverse": get_target_transform(self.config.target_transform).inverse,
            "dates_test": self.dates_test,
        }


class ConformalCalibrationSplit(StrictModel):
    """Time-based TRAIN/CALIB/TEST-разбиение для split conformal prediction."""

    data_valid: Any
    feature_cols: tuple[str, ...]
    target_cols: tuple[str, ...]
    train_mask: Any
    calib_mask: Any
    test_mask: Any
    X_train: Any
    X_calib: Any
    X_test: Any
    y_train: Any
    y_calib: Any
    y_test: Any
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    calib_start: pd.Timestamp
    calib_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @field_validator("data_valid")
    @classmethod
    def validate_data_valid(cls, value: Any) -> pd.DataFrame:
        if not isinstance(value, pd.DataFrame):
            raise TypeError("data_valid должен быть pandas.DataFrame.")
        return value

    @field_validator("train_mask", "calib_mask", "test_mask")
    @classmethod
    def validate_mask(cls, value: Any) -> pd.Series:
        if not isinstance(value, pd.Series):
            raise TypeError("mask должен быть pandas.Series.")
        return value

    @field_validator("X_train", "X_calib", "X_test", "y_train", "y_calib", "y_test")
    @classmethod
    def validate_matrix(cls, value: Any) -> np.ndarray:
        if not isinstance(value, np.ndarray):
            raise TypeError("Ожидался numpy.ndarray.")
        return value


@dataclass(frozen=True)
class TargetTransform:
    """Пара forward/inverse-преобразований целевой переменной."""

    name: TargetTransformName
    forward: Callable[[np.ndarray], np.ndarray]
    inverse: Callable[[np.ndarray], np.ndarray]


def get_target_transform(name: TargetTransformName) -> TargetTransform:
    """Вернуть forward/inverse-преобразование target."""

    if name == "none":
        return TargetTransform(
            name="none",
            forward=lambda y: np.asarray(y, dtype=float),
            inverse=lambda z: np.asarray(z, dtype=float),
        )
    if name == "log1p":
        return TargetTransform(
            name="log1p",
            forward=lambda y: np.log1p(np.asarray(y, dtype=float)),
            inverse=lambda z: np.expm1(np.asarray(z, dtype=float)),
        )
    raise ForecastingConfigError(f"Неизвестное target-преобразование: {name!r}.")


def fit_models_hist_gbdt(
    X_train: np.ndarray,
    y_train_list: Sequence[np.ndarray],
    *,
    config: HistGradientBoostingPointConfig | None = None,
    random_state: int = 42,
) -> list[Any]:
    """Обучить point-модели HistGradientBoostingRegressor по горизонтам."""

    _require_sklearn()
    cfg = config or HistGradientBoostingPointConfig()
    X = _validate_2d_array(X_train, name="X_train")
    models: list[Any] = []
    for horizon, y_train in enumerate(y_train_list, start=1):
        y = _validate_1d_array(y_train, name=f"y_train_h{horizon}")
        _validate_train_shapes(X, y, horizon=horizon)
        if cfg.loss == "poisson" and np.any(y < 0):
            raise ModelTrainingError(
                f"Poisson loss требует неотрицательные target-значения; нарушен горизонт h={horizon}."
            )
        params = cfg.to_sklearn_params()
        params["random_state"] = int(random_state) + horizon
        model = HistGradientBoostingRegressor(**params)  # type: ignore[operator]
        model.fit(X, y)
        models.append(model)
    return models


def fit_models_hist_gbdt_quantiles(
    X_train: np.ndarray,
    y_train_list: Sequence[np.ndarray],
    *,
    quantiles: Sequence[float] = (0.05, 0.95),
    config: HistGradientBoostingQuantileConfig | None = None,
    random_state: int = 42,
) -> dict[float, list[Any]]:
    """Обучить quantile-модели HistGradientBoostingRegressor по горизонтам."""

    _require_sklearn()
    qs = _validate_quantile_sequence(quantiles)
    cfg = config or HistGradientBoostingQuantileConfig()
    X = _validate_2d_array(X_train, name="X_train")
    q_models: dict[float, list[Any]] = {q: [] for q in qs}
    for horizon, y_train in enumerate(y_train_list, start=1):
        y = _validate_1d_array(y_train, name=f"y_train_h{horizon}")
        _validate_train_shapes(X, y, horizon=horizon)
        for q in qs:
            params = cfg.to_sklearn_params(quantile=q)
            params["random_state"] = int(random_state) + 100 * horizon + int(q * 100)
            model = HistGradientBoostingRegressor(**params)  # type: ignore[operator]
            model.fit(X, y)
            q_models[q].append(model)
    return q_models


def evaluate_multistep(y_true_mat: np.ndarray, y_pred_mat: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    """Посчитать R2/RMSE/MAE по каждому горизонту и в целом."""

    _require_sklearn()
    y_true = _validate_2d_array(y_true_mat, name="y_true_mat")
    y_pred = _validate_2d_array(y_pred_mat, name="y_pred_mat")
    if y_true.shape != y_pred.shape:
        raise ForecastingError(f"Формы y_true_mat и y_pred_mat должны совпадать: {y_true.shape} != {y_pred.shape}.")

    rows: list[dict[str, float | int]] = []
    for idx in range(y_true.shape[1]):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        rows.append(
            {
                "horizon_weeks": idx + 1,
                "r2": float(r2_score(yt, yp)),  # type: ignore[misc]
                "rmse": float(np.sqrt(mean_squared_error(yt, yp))),  # type: ignore[misc]
                "mae": float(mean_absolute_error(yt, yp)),  # type: ignore[misc]
            }
        )
    overall = {
        "r2_overall": float(r2_score(y_true.reshape(-1), y_pred.reshape(-1))),  # type: ignore[misc]
        "rmse_overall": float(np.sqrt(mean_squared_error(y_true.reshape(-1), y_pred.reshape(-1)))),  # type: ignore[misc]
        "mae_overall": float(mean_absolute_error(y_true.reshape(-1), y_pred.reshape(-1))),  # type: ignore[misc]
    }
    return pd.DataFrame(rows), overall


def predict_matrix(models: Sequence[Any], X: np.ndarray, *, inverse: Callable[[np.ndarray], np.ndarray] | None = None, clip_non_negative: bool = True) -> np.ndarray:
    """Получить матрицу прогнозов shape=(n_samples, H)."""

    if not models:
        raise ForecastingError("Список моделей пуст.")
    X_arr = _validate_2d_array(X, name="X")
    z_pred = np.column_stack([model.predict(X_arr) for model in models])
    y_pred = inverse(z_pred) if inverse is not None else z_pred
    return np.clip(y_pred, 0.0, None) if clip_non_negative else np.asarray(y_pred, dtype=float)


def make_conformal_calibration_split(
    supervised: SupervisedDataset,
    *,
    config: ForecastingConfig | None = None,
) -> ConformalCalibrationSplit:
    """Построить TRAIN/CALIB/TEST-разбиение для conformal-интервалов.

    Последние ``test_weeks`` валидных строк образуют TEST. Предшествующие
    ``calib_weeks`` строк образуют CALIB. Все более ранние строки образуют
    TRAIN. Такое разбиение соответствует второму notebook с conformal fixed PI.
    """

    cfg = config or ForecastingConfig(horizons=len(supervised.target_cols))
    data_valid = supervised.data_valid.reset_index(drop=True)
    n_valid = len(data_valid)
    train_end = n_valid - (cfg.calib_weeks + cfg.test_weeks)
    calib_end = n_valid - cfg.test_weeks
    if train_end < cfg.min_train_weeks:
        raise ForecastingConfigError(
            "Недостаточно valid-строк для TRAIN/CALIB/TEST-разбиения: "
            f"n_valid={n_valid}, train={train_end}, calib={cfg.calib_weeks}, "
            f"test={cfg.test_weeks}, min_train_weeks={cfg.min_train_weeks}."
        )
    if cfg.calib_weeks <= 0 or cfg.test_weeks <= 0:
        raise ForecastingConfigError("calib_weeks и test_weeks должны быть положительными.")

    train_array = np.zeros(n_valid, dtype=bool)
    calib_array = np.zeros(n_valid, dtype=bool)
    test_array = np.zeros(n_valid, dtype=bool)
    train_array[:train_end] = True
    calib_array[train_end:calib_end] = True
    test_array[calib_end:] = True

    datetime_col = cfg.datetime_col
    X_all = data_valid.loc[:, list(supervised.feature_cols)].to_numpy(dtype=float)
    y_all = data_valid.loc[:, list(supervised.target_cols)].to_numpy(dtype=float)

    return ConformalCalibrationSplit(
        data_valid=data_valid,
        feature_cols=supervised.feature_cols,
        target_cols=supervised.target_cols,
        train_mask=pd.Series(train_array),
        calib_mask=pd.Series(calib_array),
        test_mask=pd.Series(test_array),
        X_train=X_all[train_array],
        X_calib=X_all[calib_array],
        X_test=X_all[test_array],
        y_train=y_all[train_array],
        y_calib=y_all[calib_array],
        y_test=y_all[test_array],
        train_start=pd.to_datetime(data_valid.loc[train_array, datetime_col].min()),
        train_end=pd.to_datetime(data_valid.loc[train_array, datetime_col].max()),
        calib_start=pd.to_datetime(data_valid.loc[calib_array, datetime_col].min()),
        calib_end=pd.to_datetime(data_valid.loc[calib_array, datetime_col].max()),
        test_start=pd.to_datetime(data_valid.loc[test_array, datetime_col].min()),
        test_end=pd.to_datetime(data_valid.loc[test_array, datetime_col].max()),
    )


def conformal_radius_from_abs_errors(abs_errors: np.ndarray, *, alpha: float = 0.20) -> float:
    """Посчитать split-conformal radius по абсолютным ошибкам.

    Используется finite-sample correction из notebook:
    ``ceil((n + 1) * (1 - alpha)) / n`` и ``method='higher'``.
    """

    if not 0.0 < float(alpha) < 1.0:
        raise ForecastingConfigError("alpha должен лежать в интервале (0, 1).")
    errors = np.asarray(abs_errors, dtype=float)
    errors = errors[np.isfinite(errors)]
    n = len(errors)
    if n == 0:
        raise ForecastingError("Нельзя посчитать conformal radius: calibration-errors пусты.")
    q_level = np.ceil((n + 1) * (1.0 - float(alpha))) / n
    q_level = min(max(float(q_level), 0.0), 1.0)
    return float(np.quantile(errors, q_level, method="higher"))


def compute_conformal_radii(
    y_true_calib: np.ndarray,
    y_pred_calib: np.ndarray,
    *,
    alpha: float = 0.20,
) -> np.ndarray:
    """Посчитать conformal radii по каждому горизонту."""

    y_true = _validate_2d_array(y_true_calib, name="y_true_calib")
    y_pred = _validate_2d_array(y_pred_calib, name="y_pred_calib")
    if y_true.shape != y_pred.shape:
        raise ForecastingError(f"Формы calibration y_true/y_pred должны совпадать: {y_true.shape} != {y_pred.shape}.")
    abs_errors = np.abs(y_true - y_pred)
    return np.array(
        [conformal_radius_from_abs_errors(abs_errors[:, h], alpha=alpha) for h in range(abs_errors.shape[1])],
        dtype=float,
    )


def apply_symmetric_intervals(
    y_pred_mat: np.ndarray,
    radii: np.ndarray,
    *,
    clip_non_negative: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Построить симметричные интервалы ``y_pred ± radius_h``."""

    y_pred = _validate_2d_array(y_pred_mat, name="y_pred_mat")
    r = _validate_1d_array(radii, name="radii")
    if y_pred.shape[1] != len(r):
        raise ForecastingError(f"Число радиусов ({len(r)}) не совпадает с числом горизонтов ({y_pred.shape[1]}).")
    lo = y_pred - r.reshape(1, -1)
    hi = y_pred + r.reshape(1, -1)
    if clip_non_negative:
        lo = np.clip(lo, 0.0, None)
        hi = np.clip(hi, 0.0, None)
    return np.minimum(lo, hi), np.maximum(lo, hi)


def evaluate_intervals(y_true_mat: np.ndarray, lo_mat: np.ndarray, hi_mat: np.ndarray) -> tuple[pd.DataFrame, dict[str, float]]:
    """Посчитать empirical coverage и среднюю ширину интервалов."""

    y_true = _validate_2d_array(y_true_mat, name="y_true_mat")
    lo = _validate_2d_array(lo_mat, name="lo_mat")
    hi = _validate_2d_array(hi_mat, name="hi_mat")
    if not (y_true.shape == lo.shape == hi.shape):
        raise ForecastingError(f"Формы y_true/lo/hi должны совпадать: {y_true.shape}, {lo.shape}, {hi.shape}.")

    rows: list[dict[str, float | int]] = []
    for idx in range(y_true.shape[1]):
        yt = y_true[:, idx]
        lower = lo[:, idx]
        upper = hi[:, idx]
        rows.append(
            {
                "horizon_weeks": idx + 1,
                "coverage": float(((yt >= lower) & (yt <= upper)).mean()),
                "avg_width": float(np.mean(upper - lower)),
            }
        )
    overall = {
        "coverage_mean": float(np.mean([row["coverage"] for row in rows])),
        "avg_width_mean": float(np.mean([row["avg_width"] for row in rows])),
    }
    return pd.DataFrame(rows), overall




def run_forecast(
    supervised: SupervisedDataset,
    *,
    config: ForecastingConfig | None = None,
    output: ForecastOutputConfig | None = None,
) -> ForecastRunResult:
    """Выполнить полный forecasting-pass от ``SupervisedDataset`` до артефактов.

    В conformal-режиме намеренно разделены три сущности:

    * ``models_eval``: notebook-parity evaluation models, обученные на
      time-based TRAIN и проверяемые на последних ``test_weeks`` строках;
    * conformal calibration models: служебные модели, обученные только на
      TRAIN-блоке до CALIB и используемые исключительно для оценки calibration
      residuals и conformal radii;
    * ``models_prod``: production point-модели, обученные на всех валидных
      строках с известными target-горизонтами, как в notebook.

    Поэтому ``interval_method='conformal'`` влияет на интервалы, но не сужает
    обучающий набор финальной point-модели.
    """

    cfg = config or ForecastingConfig(horizons=len(supervised.target_cols))
    _validate_supervised_against_config(supervised, cfg)
    transform = get_target_transform(cfg.target_transform)

    q_models_prod: dict[float, list[Any]] = {}
    conformal_radii: np.ndarray | None = None
    interval_metrics: pd.DataFrame | None = None
    test_lo_mat: np.ndarray | None = None
    test_hi_mat: np.ndarray | None = None

    # Notebook-parity evaluation split: последние test_weeks валидных строк — TEST,
    # все более ранние валидные строки — TRAIN.
    split = make_time_holdout_split(supervised, config=cfg.holdout_config)
    y_eval_train_list = [transform.forward(split.y_train[:, h]) for h in range(split.y_train.shape[1])]
    models_eval = fit_models_hist_gbdt(
        split.X_train,
        y_eval_train_list,
        config=cfg.point_model,
        random_state=cfg.random_state,
    )
    y_pred_mat = predict_matrix(
        models_eval,
        split.X_test,
        inverse=transform.inverse,
        clip_non_negative=cfg.clip_predictions_to_non_negative,
    )
    per_h, overall = evaluate_multistep(split.y_test, y_pred_mat)

    # Production point fit: все валидные строки с известными y_h1..y_hH.
    # Это должно совпадать с notebook-логикой финального прогноза.
    prod_data = supervised.data_valid
    X_prod = prod_data.loc[:, list(supervised.feature_cols)].to_numpy(dtype=float)
    y_prod = prod_data.loc[:, list(supervised.target_cols)].to_numpy(dtype=float)
    y_prod_list = [transform.forward(y_prod[:, h]) for h in range(y_prod.shape[1])]
    models_prod = fit_models_hist_gbdt(
        X_prod,
        y_prod_list,
        config=cfg.point_model,
        random_state=cfg.random_state,
    )

    if cfg.interval_method == "conformal":
        # Notebook-parity for the point forecast and SHAP/eval models:
        # eval metrics are computed on the standard time holdout split
        # (TRAIN = all valid rows before the last test_weeks rows;
        # TEST = last test_weeks rows), independently of conformal calibration.
        split = make_time_holdout_split(supervised, config=cfg.holdout_config)
        y_train_list = [transform.forward(split.y_train[:, h]) for h in range(split.y_train.shape[1])]

        models_eval = fit_models_hist_gbdt(
            split.X_train,
            y_train_list,
            config=cfg.point_model,
            random_state=cfg.random_state,
        )
        y_pred_mat = predict_matrix(
            models_eval,
            split.X_test,
            inverse=transform.inverse,
            clip_non_negative=cfg.clip_predictions_to_non_negative,
        )
        per_h, overall = evaluate_multistep(split.y_test, y_pred_mat)

        # Split-conformal calibration is a separate interval layer. It must not
        # change either the eval models above or the final production point model
        # below; otherwise point forecasts diverge from notebook logic.
        conformal_split = make_conformal_calibration_split(supervised, config=cfg)
        y_conformal_train_list = [
            transform.forward(conformal_split.y_train[:, h])
            for h in range(conformal_split.y_train.shape[1])
        ]
        models_conformal = fit_models_hist_gbdt(
            conformal_split.X_train,
            y_conformal_train_list,
            config=cfg.point_model,
            random_state=cfg.random_state,
        )
        y_calib_pred = predict_matrix(
            models_conformal,
            conformal_split.X_calib,
            inverse=transform.inverse,
            clip_non_negative=cfg.clip_predictions_to_non_negative,
        )
        conformal_radii = compute_conformal_radii(
            conformal_split.y_calib,
            y_calib_pred,
            alpha=cfg.interval_alpha,
        )

        prod_data = supervised.data_valid
        X_prod = prod_data.loc[:, list(supervised.feature_cols)].to_numpy(dtype=float)
        y_prod = prod_data.loc[:, list(supervised.target_cols)].to_numpy(dtype=float)
        y_prod_list = [transform.forward(y_prod[:, h]) for h in range(y_prod.shape[1])]

        models_prod = fit_models_hist_gbdt(
            X_prod,
            y_prod_list,
            config=cfg.point_model,
            random_state=cfg.random_state,
        )

        test_lo_mat, test_hi_mat = apply_symmetric_intervals(
            y_pred_mat,
            conformal_radii,
            clip_non_negative=cfg.clip_predictions_to_non_negative,
        )
        per_h_pi, overall_pi = evaluate_intervals(split.y_test, test_lo_mat, test_hi_mat)
        interval_metrics = build_interval_metrics_table(
            per_h=per_h_pi,
            overall=overall_pi,
            alpha=cfg.interval_alpha,
            method="split_conformal_absolute_residual",
            stage="test_eval_model",
        )
    else:
        split = make_time_holdout_split(supervised, config=cfg.holdout_config)
        y_train_list = [transform.forward(split.y_train[:, h]) for h in range(split.y_train.shape[1])]

        models_eval = fit_models_hist_gbdt(
            split.X_train,
            y_train_list,
            config=cfg.point_model,
            random_state=cfg.random_state,
        )
        y_pred_mat = predict_matrix(
            models_eval,
            split.X_test,
            inverse=transform.inverse,
            clip_non_negative=cfg.clip_predictions_to_non_negative,
        )
        per_h, overall = evaluate_multistep(split.y_test, y_pred_mat)

        prod_data = supervised.data_valid
        X_prod = prod_data.loc[:, list(supervised.feature_cols)].to_numpy(dtype=float)
        y_prod = prod_data.loc[:, list(supervised.target_cols)].to_numpy(dtype=float)
        y_prod_list = [transform.forward(y_prod[:, h]) for h in range(y_prod.shape[1])]

        models_prod = fit_models_hist_gbdt(
            X_prod,
            y_prod_list,
            config=cfg.point_model,
            random_state=cfg.random_state,
        )
        if cfg.interval_method == "quantile":
            q_models_prod = fit_models_hist_gbdt_quantiles(
                X_prod,
                y_prod_list,
                quantiles=cfg.quantiles,
                config=cfg.quantile_model,
                random_state=cfg.random_state,
            )

    x0 = supervised.latest_feature_row.to_numpy(dtype=float)
    z_hat = np.array([model.predict(x0)[0] for model in models_prod], dtype=float)
    y_hat = transform.inverse(z_hat)
    if cfg.clip_predictions_to_non_negative:
        y_hat = np.clip(y_hat, 0.0, None)

    if cfg.interval_method == "conformal":
        if conformal_radii is None:
            raise ForecastingError("conformal_radii не рассчитаны.")
        y_lo = y_hat - conformal_radii
        y_hi = y_hat + conformal_radii
        if cfg.clip_predictions_to_non_negative:
            y_lo = np.clip(y_lo, 0.0, None)
            y_hi = np.clip(y_hi, 0.0, None)
        y_lo, y_hi = np.minimum(y_lo, y_hi), np.maximum(y_lo, y_hi)
    elif cfg.interval_method == "quantile":
        q_keys = sorted(q_models_prod.keys())
        if len(q_keys) < 2:
            raise ForecastingError("Для quantile-интервала требуется минимум два набора quantile-моделей.")
        lower_q, upper_q = q_keys[0], q_keys[-1]
        z_lo = np.array([q_models_prod[lower_q][h].predict(x0)[0] for h in range(cfg.horizons)], dtype=float)
        z_hi = np.array([q_models_prod[upper_q][h].predict(x0)[0] for h in range(cfg.horizons)], dtype=float)
        y_lo = transform.inverse(z_lo)
        y_hi = transform.inverse(z_hi)
        if cfg.clip_predictions_to_non_negative:
            y_lo = np.clip(y_lo, 0.0, None)
            y_hi = np.clip(y_hi, 0.0, None)
        y_lo, y_hi = np.minimum(y_lo, y_hi), np.maximum(y_lo, y_hi)
    else:
        y_lo = np.full_like(y_hat, np.nan, dtype=float)
        y_hi = np.full_like(y_hat, np.nan, dtype=float)

    origin_date = pd.to_datetime(supervised.origin_date)
    if pd.isna(origin_date):
        raise ForecastingError("origin_date не определён: невозможно построить production forecast.")
    future_dates = infer_future_dates(supervised.data, origin_date, cfg.horizons, datetime_col=cfg.datetime_col)

    model_registry = build_model_registry(
        point_models=models_prod,
        quantile_models=q_models_prod,
        horizons=cfg.horizons,
        feature_cols=supervised.feature_cols,
        test_weeks=cfg.test_weeks,
        notebook_id=cfg.notebook_id,
        target_variable=cfg.target_variable,
        time_unit=cfg.time_unit,
        interval_method=cfg.interval_method,
        interval_alpha=cfg.interval_alpha,
        calib_weeks=cfg.calib_weeks,
        conformal_radii=conformal_radii,
        conformal_refit_on_train_plus_calib=cfg.conformal_refit_on_train_plus_calib,
    )

    tables = build_forecast_tables(
        supervised=supervised,
        split=split,
        config=cfg,
        per_h=per_h,
        overall=overall,
        y_pred_mat=y_pred_mat,
        y_hat=y_hat,
        y_lo=y_lo,
        y_hi=y_hi,
        origin_date=origin_date,
        future_dates=future_dates,
        conformal_radii=conformal_radii,
        interval_metrics=interval_metrics,
        test_lo_mat=test_lo_mat,
        test_hi_mat=test_hi_mat,
    )

    result = ForecastRunResult(
        supervised=supervised,
        split=split,
        config=cfg,
        feature_cols=supervised.feature_cols,
        target_cols=supervised.target_cols,
        models_eval=models_eval,
        models_prod=models_prod,
        q_models_prod=q_models_prod,
        conformal_radii=conformal_radii,
        interval_metrics=interval_metrics,
        X_test=split.X_test,
        y_test_mat=split.y_test,
        y_pred_mat=y_pred_mat,
        dates_test=split.data_valid.loc[split.test_mask, cfg.datetime_col].reset_index(drop=True),
        metrics=ForecastMetrics(per_horizon=per_h, overall=overall),
        y_hat=np.asarray(y_hat, dtype=float),
        y_lo=np.asarray(y_lo, dtype=float),
        y_hi=np.asarray(y_hi, dtype=float),
        origin_date=origin_date,
        future_dates=future_dates,
        model_registry=model_registry,
        tables=tables,
        artifacts={},
    )

    if output is not None:
        result.artifacts = save_forecast_artifacts(result, output)
    return result

def run_forecast_from_frame(
    frame: pd.DataFrame,
    *,
    feature_config: FeatureEngineeringConfig | None = None,
    forecasting_config: ForecastingConfig | None = None,
    output: ForecastOutputConfig | None = None,
) -> ForecastRunResult:
    """Построить supervised-матрицу из DataFrame и выполнить forecasting-pass."""

    fcfg = feature_config or FeatureEngineeringConfig()
    supervised = build_supervised(frame, config=fcfg)
    cfg = forecasting_config or ForecastingConfig(horizons=fcfg.horizons)
    return run_forecast(supervised, config=cfg, output=output)


def build_forecast_tables(
    *,
    supervised: SupervisedDataset,
    split: TimeHoldoutSplit,
    config: ForecastingConfig,
    per_h: pd.DataFrame,
    overall: Mapping[str, float],
    y_pred_mat: np.ndarray,
    y_hat: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
    origin_date: pd.Timestamp,
    future_dates: pd.DatetimeIndex,
    conformal_radii: np.ndarray | None = None,
    interval_metrics: pd.DataFrame | None = None,
    test_lo_mat: np.ndarray | None = None,
    test_hi_mat: np.ndarray | None = None,
) -> ForecastTables:
    """Собрать таблицы, совместимые с CSV-артефактами notebook."""

    metrics_summary = build_metrics_summary_table(
        per_h=per_h,
        overall=overall,
        split=split,
        n_features=len(supervised.feature_cols),
        target_transform=config.target_transform,
        interval_method=config.interval_method,
        interval_alpha=config.interval_alpha,
        calib_weeks=config.calib_weeks if config.interval_method == "conformal" else None,
    )
    test_predictions = build_test_predictions_table(
        split=split,
        y_pred_mat=y_pred_mat,
        datetime_col=config.datetime_col,
        y_lo_mat=test_lo_mat,
        y_hi_mat=test_hi_mat,
    )
    forecast_next = build_forecast_next_table(
        origin_date=origin_date,
        future_dates=future_dates,
        y_hat=y_hat,
        y_lo=y_lo,
        y_hi=y_hi,
    )
    feature_list = supervised_to_feature_list_frame(supervised)
    history_plus_forecast = build_history_plus_forecast_table(
        supervised=supervised,
        origin_date=origin_date,
        future_dates=future_dates,
        y_hat=y_hat,
        y_lo=y_lo,
        y_hi=y_hi,
        history_window=config.history_window,
        datetime_col=config.datetime_col,
        target_col=config.target_col,
    )
    conformal_radii_table = (
        build_conformal_radii_table(conformal_radii, alpha=config.interval_alpha)
        if conformal_radii is not None
        else None
    )
    return ForecastTables(
        metrics_summary=metrics_summary,
        test_predictions=test_predictions,
        forecast_next=forecast_next,
        feature_list=feature_list,
        history_plus_forecast=history_plus_forecast,
        conformal_radii=conformal_radii_table,
        interval_metrics=interval_metrics,
    )


def build_conformal_radii_table(radii: np.ndarray, *, alpha: float = 0.20) -> pd.DataFrame:
    """Собрать таблицу conformal-радиусов по горизонтам."""

    r = _validate_1d_array(radii, name="conformal_radii")
    return pd.DataFrame(
        {
            "horizon_weeks": np.arange(1, len(r) + 1),
            "radius": r,
            "alpha": float(alpha),
            "nominal_coverage": 1.0 - float(alpha),
            "method": "split_conformal_absolute_residual",
        }
    )


def build_interval_metrics_table(
    *,
    per_h: pd.DataFrame,
    overall: Mapping[str, float],
    alpha: float,
    method: str,
    stage: str,
) -> pd.DataFrame:
    """Собрать таблицу качества прогнозных интервалов."""

    metrics_df = per_h.copy()
    metrics_df["section"] = "per_h"
    metrics_df["alpha"] = float(alpha)
    metrics_df["nominal_coverage"] = 1.0 - float(alpha)
    metrics_df["method"] = method
    metrics_df["stage"] = stage
    overall_df = pd.DataFrame(
        [
            {
                "section": "overall",
                "horizon_weeks": "all",
                "coverage": float(overall["coverage_mean"]),
                "avg_width": float(overall["avg_width_mean"]),
                "alpha": float(alpha),
                "nominal_coverage": 1.0 - float(alpha),
                "method": method,
                "stage": stage,
            }
        ]
    )
    all_cols = sorted(set(metrics_df.columns) | set(overall_df.columns))
    return pd.concat(
        [metrics_df.reindex(columns=all_cols), overall_df.reindex(columns=all_cols)],
        ignore_index=True,
    )

def build_metrics_summary_table(
    *,
    per_h: pd.DataFrame,
    overall: Mapping[str, float],
    split: TimeHoldoutSplit,
    n_features: int,
    target_transform: TargetTransformName,
    interval_method: IntervalMethod = "conformal",
    interval_alpha: float | None = None,
    calib_weeks: int | None = None,
) -> pd.DataFrame:
    """Собрать ``metrics_summary.csv``."""

    metrics_df = per_h.copy()
    metrics_df["section"] = "per_h"
    metrics_df["train_start"] = split.train_start.date()
    metrics_df["train_end"] = split.train_end.date()
    metrics_df["test_start"] = split.test_start.date()
    metrics_df["test_end"] = split.test_end.date()
    metrics_df["n_train"] = int(split.train_mask.sum())
    metrics_df["n_test"] = int(split.test_mask.sum())
    metrics_df["n_features"] = int(n_features)
    metrics_df["target_transform"] = target_transform
    metrics_df["interval_method"] = interval_method
    metrics_df["interval_alpha"] = interval_alpha
    metrics_df["calib_weeks"] = calib_weeks

    overall_df = pd.DataFrame(
        [
            {
                "section": "overall",
                "horizon_weeks": "all",
                "r2": float(overall["r2_overall"]),
                "rmse": float(overall["rmse_overall"]),
                "mae": float(overall["mae_overall"]),
                "train_start": split.train_start.date(),
                "train_end": split.train_end.date(),
                "test_start": split.test_start.date(),
                "test_end": split.test_end.date(),
                "n_train": int(split.train_mask.sum()),
                "n_test": int(split.test_mask.sum()),
                "n_features": int(n_features),
                "target_transform": target_transform,
                "interval_method": interval_method,
                "interval_alpha": interval_alpha,
                "calib_weeks": calib_weeks,
            }
        ]
    )
    all_cols = sorted(set(metrics_df.columns) | set(overall_df.columns))
    return pd.concat(
        [metrics_df.reindex(columns=all_cols), overall_df.reindex(columns=all_cols)],
        ignore_index=True,
    )


def build_test_predictions_table(
    *,
    split: TimeHoldoutSplit,
    y_pred_mat: np.ndarray,
    datetime_col: str = "datetime",
    y_lo_mat: np.ndarray | None = None,
    y_hi_mat: np.ndarray | None = None,
) -> pd.DataFrame:
    """Собрать ``test_predictions.csv``."""

    y_pred = _validate_2d_array(y_pred_mat, name="y_pred_mat")
    if split.y_test.shape != y_pred.shape:
        raise ForecastingError(f"Формы y_test и y_pred_mat не совпадают: {split.y_test.shape} != {y_pred.shape}.")

    lo = _validate_2d_array(y_lo_mat, name="y_lo_mat") if y_lo_mat is not None else None
    hi = _validate_2d_array(y_hi_mat, name="y_hi_mat") if y_hi_mat is not None else None
    if (lo is None) != (hi is None):
        raise ForecastingError("y_lo_mat и y_hi_mat должны задаваться одновременно.")
    if lo is not None and (lo.shape != y_pred.shape or hi.shape != y_pred.shape):
        raise ForecastingError("Формы y_lo_mat/y_hi_mat должны совпадать с y_pred_mat.")

    test_dates = pd.to_datetime(split.data_valid.loc[split.test_mask, datetime_col]).reset_index(drop=True)
    table = pd.DataFrame({"origin_date": test_dates})
    horizons = y_pred.shape[1]
    for horizon in range(1, horizons + 1):
        idx = horizon - 1
        table[f"target_date_h{horizon}"] = test_dates + pd.to_timedelta(7 * horizon, unit="D")
        table[f"y_true_h{horizon}"] = split.y_test[:, idx]
        table[f"y_pred_h{horizon}"] = y_pred[:, idx]
        table[f"abs_error_h{horizon}"] = np.abs(split.y_test[:, idx] - y_pred[:, idx])
        if lo is not None and hi is not None:
            table[f"y_lo_h{horizon}"] = lo[:, idx]
            table[f"y_hi_h{horizon}"] = hi[:, idx]
            table[f"covered_h{horizon}"] = (split.y_test[:, idx] >= lo[:, idx]) & (split.y_test[:, idx] <= hi[:, idx])
    return table

def build_forecast_next_table(
    *,
    origin_date: pd.Timestamp,
    future_dates: pd.DatetimeIndex,
    y_hat: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
) -> pd.DataFrame:
    """Собрать ``forecast_next_{H}w.csv``."""

    yh = _validate_1d_array(y_hat, name="y_hat")
    lo = _as_1d_float_array(y_lo, name="y_lo", allow_nan=True)
    hi = _as_1d_float_array(y_hi, name="y_hi", allow_nan=True)
    if not (len(yh) == len(lo) == len(hi) == len(future_dates)):
        raise ForecastingError("Длины y_hat/y_lo/y_hi/future_dates должны совпадать.")

    table = pd.DataFrame(
        {
            "origin_date": [pd.to_datetime(origin_date).date()] * len(yh),
            "target_date": pd.to_datetime(future_dates).date,
            "horizon_weeks": np.arange(1, len(yh) + 1),
            "point_forecast": yh,
            "q_lo": lo,
            "q_hi": hi,
        }
    )
    table["pi_width"] = table["q_hi"] - table["q_lo"]
    table["interval_ok"] = table["q_hi"] >= table["q_lo"]
    return table


def build_history_plus_forecast_table(
    *,
    supervised: SupervisedDataset,
    origin_date: pd.Timestamp,
    future_dates: pd.DatetimeIndex,
    y_hat: np.ndarray,
    y_lo: np.ndarray,
    y_hi: np.ndarray,
    history_window: int = 40,
    datetime_col: str = "datetime",
    target_col: str = "inc_per_10k",
) -> pd.DataFrame:
    """Собрать ``history_plus_forecast_40.csv``."""

    data = supervised.data.copy()
    required = {datetime_col, target_col}
    missing = required.difference(data.columns)
    if missing:
        raise ForecastingError(f"В supervised.data отсутствуют колонки для истории: {sorted(missing)}.")
    history = (
        data.loc[pd.to_datetime(data[datetime_col]) <= pd.to_datetime(origin_date), [datetime_col, target_col]]
        .sort_values(datetime_col)
        .tail(history_window)
        .reset_index(drop=True)
    )
    hist_out = pd.DataFrame(
        {
            "date": pd.to_datetime(history[datetime_col]),
            "value": history[target_col].astype(float),
            "row_type": "history",
            "origin_date": pd.NaT,
            "horizon_weeks": pd.NA,
            "q_lo": np.nan,
            "q_hi": np.nan,
        }
    )
    forecast_out = pd.DataFrame(
        {
            "date": pd.to_datetime(future_dates),
            "value": _validate_1d_array(y_hat, name="y_hat"),
            "row_type": "forecast",
            "origin_date": [pd.to_datetime(origin_date)] * len(future_dates),
            "horizon_weeks": np.arange(1, len(future_dates) + 1),
            "q_lo": _as_1d_float_array(y_lo, name="y_lo", allow_nan=True),
            "q_hi": _as_1d_float_array(y_hi, name="y_hi", allow_nan=True),
        }
    )
    return pd.concat([hist_out, forecast_out], ignore_index=True).sort_values("date").reset_index(drop=True)


def infer_future_dates(
    data: pd.DataFrame,
    origin_date: pd.Timestamp,
    horizons: int,
    *,
    datetime_col: str = "datetime",
) -> pd.DatetimeIndex:
    """Вывести даты будущих горизонтов по медианному шагу временного ряда."""

    if datetime_col not in data.columns:
        raise ForecastingError(f"В data отсутствует datetime_col={datetime_col!r}.")
    dates = pd.to_datetime(data[datetime_col]).sort_values().reset_index(drop=True)
    diffs = dates.diff().dropna()
    step = diffs.median() if not diffs.empty else pd.Timedelta(days=7)
    if pd.isna(step) or step <= pd.Timedelta(0):
        step = pd.Timedelta(days=7)
    return pd.date_range(pd.to_datetime(origin_date) + step, periods=horizons, freq=step)


def save_forecast_artifacts(result: ForecastRunResult, output: ForecastOutputConfig) -> dict[str, Path]:
    """Сохранить CSV/JSON-артефакты forecast-pass."""

    output_dir = Path(output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_options = TableWriteOptions(index=False, encoding=output.csv_encoding)
    artifacts: dict[str, Path] = {}

    if output.save_metrics_summary:
        artifacts["metrics_summary"] = write_table(
            result.tables.metrics_summary,
            output_dir / "metrics_summary.csv",
            options=write_options,
        )
    if output.save_test_predictions:
        artifacts["test_predictions"] = write_table(
            result.tables.test_predictions,
            output_dir / "test_predictions.csv",
            options=write_options,
        )
    if output.save_forecast_next:
        artifacts["forecast_next"] = write_table(
            result.tables.forecast_next,
            output_dir / f"forecast_next_{result.config.horizons}w.csv",
            options=write_options,
        )
    if output.save_feature_list:
        artifacts["feature_list"] = write_table(
            result.tables.feature_list,
            output_dir / "feature_list.csv",
            options=write_options,
        )
    if output.save_history_plus_forecast:
        artifacts["history_plus_forecast"] = write_table(
            result.tables.history_plus_forecast,
            output_dir / "history_plus_forecast_40.csv",
            options=write_options,
        )
    if output.save_conformal_radii and result.tables.conformal_radii is not None:
        artifacts["conformal_radii"] = write_table(
            result.tables.conformal_radii,
            output_dir / "conformal_radii.csv",
            options=write_options,
        )
    if output.save_interval_metrics and result.tables.interval_metrics is not None:
        artifacts["interval_metrics"] = write_table(
            result.tables.interval_metrics,
            output_dir / "interval_metrics.csv",
            options=write_options,
        )
    if output.save_model_registry:
        registry_path = Path(output.registry_path) if output.registry_path is not None else output_dir / "model_registry.json"
        if not registry_path.is_absolute():
            registry_path = output_dir / registry_path
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(json.dumps(result.model_registry, ensure_ascii=False, indent=2), encoding="utf-8")
        artifacts["model_registry"] = registry_path

    return artifacts


def _extract_hgbr_params(model: Any) -> dict[str, Any]:
    """Извлечь ключевые гиперпараметры HistGradientBoostingRegressor."""

    params = model.get_params()
    return {
        "learning_rate": params.get("learning_rate"),
        "max_iter": params.get("max_iter"),
        "min_samples_leaf": params.get("min_samples_leaf"),
        "l2_regularization": params.get("l2_regularization"),
        "early_stopping": params.get("early_stopping"),
    }


def _build_point_entry(model: Any, horizon: int, target_variable: str, time_unit: str) -> dict[str, Any]:
    """Собрать запись point-модели для model_registry."""

    return {
        "id": f"point_forecast_h{horizon}",
        "family": "HistGradientBoostingRegressor",
        "forecast_type": "point",
        "output_type": "single_value",
        "target_variable": target_variable,
        "target_time_offset": horizon,
        "target_time_unit": time_unit,
        "predicts": f"{target_variable} at t+{horizon}",
        "training_strategy": "direct",
        "loss": model.get_params().get("loss", "poisson"),
        "purpose": "point_forecast",
        "hyperparameters": _extract_hgbr_params(model),
    }


def _build_quantile_entry(
    model: Any,
    horizon: int,
    quantile: float,
    interval_side: Literal["lower", "upper"],
    target_variable: str,
    time_unit: str,
) -> dict[str, Any]:
    """Собрать запись quantile-модели для model_registry."""

    purpose = f"{interval_side}_prediction_interval"
    q_str = str(quantile).replace(".", "")
    return {
        "id": f"{purpose}_q{q_str}_h{horizon}",
        "family": "HistGradientBoostingRegressor",
        "forecast_type": "quantile",
        "output_type": "single_value",
        "target_variable": target_variable,
        "target_time_offset": horizon,
        "target_time_unit": time_unit,
        "predicts": f"{target_variable} at t+{horizon}",
        "training_strategy": "direct",
        "loss": "quantile",
        "quantile": float(quantile),
        "interval_side": interval_side,
        "purpose": purpose,
        "hyperparameters": _extract_hgbr_params(model),
    }


def _build_conformal_interval_entry(
    horizon: int,
    radius: float,
    *,
    alpha: float,
    target_variable: str,
    time_unit: str,
) -> dict[str, Any]:
    """Собрать запись conformal-интервала для model_registry."""

    return {
        "id": f"prediction_interval_conformal_h{horizon}",
        "family": "SplitConformalPrediction",
        "forecast_type": "prediction_interval",
        "output_type": "lower_upper_bounds",
        "target_variable": target_variable,
        "target_time_offset": horizon,
        "target_time_unit": time_unit,
        "predicts": f"{target_variable} interval at t+{horizon}",
        "training_strategy": "direct_point_model_plus_calibration_residuals",
        "interval_method": "split_conformal_absolute_residual",
        "alpha": float(alpha),
        "nominal_coverage": 1.0 - float(alpha),
        "radius": float(radius),
        "interval_formula": "point_forecast ± radius_h",
        "purpose": "prediction_interval",
    }


def build_model_registry(
    *,
    point_models: Sequence[Any],
    quantile_models: Mapping[float, Sequence[Any]] | None = None,
    horizons: int,
    feature_cols: Sequence[str],
    test_weeks: int,
    notebook_id: str = "gbdt_influenza_forecast_4w",
    target_variable: str = "inc_per_10k",
    time_unit: str = "week",
    interval_method: IntervalMethod = "conformal",
    interval_alpha: float = 0.20,
    calib_weeks: int | None = 52,
    conformal_radii: np.ndarray | None = None,
    conformal_refit_on_train_plus_calib: bool = True,
) -> dict[str, Any]:
    """Построить model_registry, совместимый с notebook и conformal-режимом."""

    if len(point_models) != horizons:
        raise ForecastingError(f"Ожидалось {horizons} point-моделей, получено {len(point_models)}.")

    model_entries: list[dict[str, Any]] = []
    for horizon, model in enumerate(point_models, start=1):
        model_entries.append(_build_point_entry(model, horizon, target_variable, time_unit))

    interval_design: dict[str, Any]
    if interval_method == "quantile":
        q_models = quantile_models or {}
        quantile_keys = sorted(float(q) for q in q_models.keys())
        if len(quantile_keys) < 2:
            raise ForecastingError("quantile_models должен содержать минимум два квантиля.")
        for q in quantile_keys:
            if len(q_models[q]) != horizons:  # type: ignore[index]
                raise ForecastingError(
                    f"Для квантиля {q} ожидалось {horizons} моделей, получено {len(q_models[q])}."  # type: ignore[index]
                )
        lower_q = quantile_keys[0]
        upper_q = quantile_keys[-1]
        for horizon, model in enumerate(q_models[lower_q], start=1):  # type: ignore[index]
            model_entries.append(_build_quantile_entry(model, horizon, lower_q, "lower", target_variable, time_unit))
        for horizon, model in enumerate(q_models[upper_q], start=1):  # type: ignore[index]
            model_entries.append(_build_quantile_entry(model, horizon, upper_q, "upper", target_variable, time_unit))
        interval_design = {
            "method": "hist_gradient_boosting_quantile",
            "quantiles": [lower_q, upper_q],
            "nominal_coverage": float(upper_q - lower_q),
        }
    elif interval_method == "conformal":
        if conformal_radii is None:
            raise ForecastingError("Для conformal model_registry требуется conformal_radii.")
        radii = _validate_1d_array(conformal_radii, name="conformal_radii")
        if len(radii) != horizons:
            raise ForecastingError(f"Ожидалось {horizons} conformal radii, получено {len(radii)}.")
        for horizon, radius in enumerate(radii, start=1):
            model_entries.append(
                _build_conformal_interval_entry(
                    horizon,
                    float(radius),
                    alpha=interval_alpha,
                    target_variable=target_variable,
                    time_unit=time_unit,
                )
            )
        interval_design = {
            "method": "split_conformal_absolute_residual",
            "alpha": float(interval_alpha),
            "nominal_coverage": 1.0 - float(interval_alpha),
            "calib_weeks": int(calib_weeks) if calib_weeks is not None else None,
            "radius_by_horizon": [float(v) for v in radii],
            "refit_after_calibration": bool(conformal_refit_on_train_plus_calib),
            "formal_guarantee_note": (
                "Radii are calibrated on the calibration block. "
                "Production point models are fit on all valid rows for notebook-parity; "
                "therefore strict split-conformal finite-sample coverage is not claimed for the final refit point model."
            ),
        }
    else:
        interval_design = {"method": "none"}

    return {
        "notebook_id": notebook_id,
        "task": f"forecast weekly {target_variable} for 1-{horizons} weeks ahead",
        "forecast_design": {
            "strategy": "direct_multi_step",
            "per_model_output": "single_value",
            "semantic_rule": "each model predicts exactly one target value for one specific future offset t+h",
            "intervals": interval_design,
        },
        "training_scheme": {
            "strategy": "direct_multi_step",
            "test_split": "time_based_holdout",
            "test_weeks": int(test_weeks),
            "calibration_split": "time_based_calibration" if interval_method == "conformal" else None,
            "calib_weeks": int(calib_weeks) if interval_method == "conformal" and calib_weeks is not None else None,
            "production_fit": "all_valid_rows",
        },
        "feature_groups": [
            "target_lags",
            "target_rolling_stats",
            "epidemic_dynamics",
            "fourier_seasonality",
            "temperature_lags",
            "temperature_rolling_stats",
            "calendar_features",
        ],
        "features": list(feature_cols),
        "models": model_entries,
    }

def _validate_supervised_against_config(supervised: SupervisedDataset, config: ForecastingConfig) -> None:
    """Проверить совместимость SupervisedDataset и ForecastingConfig."""

    if len(supervised.target_cols) != config.horizons:
        raise ForecastingConfigError(
            f"config.horizons={config.horizons}, но supervised содержит {len(supervised.target_cols)} target-колонок."
        )
    if supervised.config.target_col != config.target_col:
        raise ForecastingConfigError(
            f"target_col не согласован: supervised={supervised.config.target_col!r}, config={config.target_col!r}."
        )
    if supervised.config.datetime_col != config.datetime_col:
        raise ForecastingConfigError(
            f"datetime_col не согласован: supervised={supervised.config.datetime_col!r}, config={config.datetime_col!r}."
        )
    if supervised.valid_row_count <= config.test_weeks:
        raise ForecastingConfigError(
            f"Недостаточно валидных строк ({supervised.valid_row_count}) для test_weeks={config.test_weeks}."
        )
    if config.interval_method == "conformal":
        n_train = supervised.valid_row_count - config.test_weeks - config.calib_weeks
        if n_train < config.min_train_weeks:
            raise ForecastingConfigError(
                "Недостаточно валидных строк для conformal TRAIN/CALIB/TEST-разбиения: "
                f"n_valid={supervised.valid_row_count}, train={n_train}, "
                f"calib={config.calib_weeks}, test={config.test_weeks}, "
                f"min_train_weeks={config.min_train_weeks}."
            )


def _validate_quantile_sequence(quantiles: Sequence[float]) -> tuple[float, ...]:
    qs = tuple(sorted(float(q) for q in quantiles))
    if len(qs) < 2:
        raise ForecastingConfigError("Нужно минимум два квантиля для прогнозного интервала.")
    if len(set(qs)) != len(qs):
        raise ForecastingConfigError(f"Квантили не должны повторяться: {qs}.")
    if any(not 0.0 < q < 1.0 for q in qs):
        raise ForecastingConfigError(f"Все квантили должны лежать в интервале (0, 1): {qs}.")
    return qs


def _validate_1d_array(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ForecastingError(f"{name} должен быть одномерным массивом, получена форма {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ForecastingError(f"{name} содержит NaN или бесконечные значения.")
    return arr


def _as_1d_float_array(values: Any, *, name: str, allow_nan: bool = False) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 1:
        raise ForecastingError(f"{name} должен быть одномерным массивом, получена форма {arr.shape}.")
    finite_mask = np.isfinite(arr) | (np.isnan(arr) if allow_nan else False)
    if not bool(finite_mask.all()):
        raise ForecastingError(f"{name} содержит недопустимые значения.")
    return arr


def _validate_2d_array(values: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ForecastingError(f"{name} должен быть двумерным массивом, получена форма {arr.shape}.")
    if not np.isfinite(arr).all():
        raise ForecastingError(f"{name} содержит NaN или бесконечные значения.")
    return arr


def _validate_train_shapes(X: np.ndarray, y: np.ndarray, *, horizon: int) -> None:
    if X.shape[0] != y.shape[0]:
        raise ModelTrainingError(
            f"Размерности X_train и y_train не согласованы для h={horizon}: {X.shape[0]} != {y.shape[0]}."
        )
    if X.shape[0] == 0:
        raise ModelTrainingError(f"Пустой train-набор для h={horizon}.")


def _require_sklearn() -> None:
    if HistGradientBoostingRegressor is None:
        raise ImportError(
            "Для forecasting.py требуется scikit-learn. Установите зависимость `scikit-learn`."
        )


__all__ = [
    "TargetTransformName",
    "ForecastModelFamily",
    "IntervalMethod",
    "ForecastingError",
    "ForecastingConfigError",
    "ModelTrainingError",
    "ForecastArtifactError",
    "HistGradientBoostingPointConfig",
    "HistGradientBoostingQuantileConfig",
    "ForecastingConfig",
    "ForecastOutputConfig",
    "ForecastMetrics",
    "ForecastTables",
    "ForecastRunResult",
    "ConformalCalibrationSplit",
    "TargetTransform",
    "get_target_transform",
    "fit_models_hist_gbdt",
    "fit_models_hist_gbdt_quantiles",
    "evaluate_multistep",
    "evaluate_intervals",
    "conformal_radius_from_abs_errors",
    "compute_conformal_radii",
    "make_conformal_calibration_split",
    "predict_matrix",
    "run_forecast",
    "run_forecast_from_frame",
    "build_forecast_tables",
    "build_metrics_summary_table",
    "build_conformal_radii_table",
    "build_interval_metrics_table",
    "build_test_predictions_table",
    "build_forecast_next_table",
    "build_history_plus_forecast_table",
    "infer_future_dates",
    "save_forecast_artifacts",
    "build_model_registry",
]

