"""
Предобработка недельных эпидемиологических рядов для ai4epi.

Модуль переносит feature engineering из исследовательского notebook в
проверяемый пакетный контракт. Он не загружает данные из БД и не обучает
модели: входом является уже нормализованный недельный ряд, например результат
``ai4epi.data_sources.weekly_incidence_from_cases`` или объединённая таблица
``inc_per_10k + weather``.

Основная модель данных:

``weekly dataframe``
    datetime, iso_year, iso_week, inc_per_10k, опциональные погодные колонки.

``supervised dataframe``
    исходные недельные данные + календарные признаки + лаги + rolling-признаки
    + целевые колонки ``y_h1 ... y_hH`` для многошагового прогноза.

Все численные настройки по умолчанию соответствуют активному forecasting-блоку
notebook: горизонт 4 недели, лаги цели 0..5, лаги температуры 0..4,
rolling-окна цели 4..9, rolling-окна температуры 4..7, Fourier K=2.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4epi.core.io import TableSchema, TableWriteOptions, write_table


CalendarValidationMode = Literal["strict", "repair", "ignore"]


class PreprocessingError(ValueError):
    """Базовая ошибка контракта preprocessing-слоя."""


class WeeklyFrameError(PreprocessingError):
    """Недельный ряд не соответствует ожидаемому контракту."""


class FeatureEngineeringError(PreprocessingError):
    """Ошибка построения признаков или supervised-матрицы."""


class PreprocessingConfigError(PreprocessingError):
    """Некорректная конфигурация preprocessing-слоя."""


class StrictModel(BaseModel):
    """Базовая модель конфигурации с закрытым контрактом."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )


class WeeklyFrameConfig(StrictModel):
    """Контракт нормализации и проверки недельного временного ряда."""

    datetime_col: str = Field(default="datetime", min_length=1)
    iso_year_col: str = Field(default="iso_year", min_length=1)
    iso_week_col: str = Field(default="iso_week", min_length=1)
    target_col: str = Field(default="inc_per_10k", min_length=1)
    calendar_validation: CalendarValidationMode = "strict"
    require_unique_dates: bool = True
    require_regular_weekly_index: bool = True
    require_monday_week_start: bool = True
    sort_by_datetime: bool = True


class WeatherAggregationConfig(StrictModel):
    """Настройки агрегации часовой погоды в недельные признаки."""

    time_col: str = Field(default="time", min_length=1)
    temperature_col: str = Field(default="temp", min_length=1)
    humidity_col: str = Field(default="rh", min_length=1)
    week_start_col: str = Field(default="week_start", min_length=1)
    min_hours_per_week: int | None = Field(default=150, ge=1)
    drop_incomplete_weeks: bool = False


class FeatureEngineeringConfig(StrictModel):
    """Настройки построения supervised-признаков.

    Значения по умолчанию соответствуют фактическому вызову
    ``run_influenza_forecast_pipeline`` в notebook, а не более широким
    дефолтам helper-функции ``build_supervised``.
    """

    target_col: str = Field(default="inc_per_10k", min_length=1)
    datetime_col: str = Field(default="datetime", min_length=1)
    iso_year_col: str = Field(default="iso_year", min_length=1)
    iso_week_col: str = Field(default="iso_week", min_length=1)
    horizons: int = Field(default=4, ge=1, le=52)
    temp_cols: tuple[str, ...] = ("temp_mean",)
    y_lags: tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    temp_lags: tuple[int, ...] = (0, 1, 2, 3, 4)
    y_roll_windows: tuple[int, ...] = (4, 5, 6, 7, 8, 9)
    temp_roll_windows: tuple[int, ...] = (4, 5, 6, 7)
    fourier_k: int = Field(default=2, ge=0, le=12)
    fourier_period: int = Field(default=52, ge=1)
    growth_lags: tuple[int, ...] = (1, 2, 4)
    epidemic_dynamics_eps: float = Field(default=1e-6, gt=0)
    raw_columns_to_drop: tuple[str, ...] = (
        "total_population",
        "total_cases_formula",
        "rh_min",
        "rh_max",
    )
    require_numeric_features: bool = True
    validate_weekly_frame: bool = True
    weekly_frame_config: WeeklyFrameConfig = Field(default_factory=WeeklyFrameConfig)

    @field_validator("temp_cols")
    @classmethod
    def _temp_cols_not_empty(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("temp_cols не должен быть пустым.")
        return _validate_unique_non_negative_names(value, field_name="temp_cols")

    @field_validator("y_lags", "temp_lags", "y_roll_windows", "temp_roll_windows", "growth_lags")
    @classmethod
    def _validate_positive_integer_sequences(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if any(int(v) != v for v in value):
            raise ValueError("Последовательность должна содержать только целые числа.")
        if any(v < 0 for v in value):
            raise ValueError("Лаги и окна не могут быть отрицательными.")
        return tuple(int(v) for v in value)

    @model_validator(mode="after")
    def _validate_windows(self) -> "FeatureEngineeringConfig":
        if any(w <= 0 for w in self.y_roll_windows):
            raise ValueError("Окна rolling-статистик цели должны быть положительными.")
        if any(w <= 0 for w in self.temp_roll_windows):
            raise ValueError("Окна rolling-статистик температуры должны быть положительными.")
        if any(lag <= 0 for lag in self.growth_lags):
            raise ValueError("growth_lags должны быть положительными: лаг 0 не задаёт динамику.")
        return self

    @property
    def target_columns(self) -> tuple[str, ...]:
        """Имена целевых колонок для всех горизонтов."""

        return tuple(f"y_h{h}" for h in range(1, self.horizons + 1))


class HoldoutSplitConfig(StrictModel):
    """Настройки time-based holdout-разбиения."""

    test_weeks: int = Field(default=52, ge=1)
    min_train_weeks: int = Field(default=104, ge=1)


class SupervisedDataset(StrictModel):
    """Результат построения supervised-матрицы."""

    data: Any
    feature_cols: tuple[str, ...]
    target_cols: tuple[str, ...]
    config: FeatureEngineeringConfig
    valid_mask: Any
    valid_row_count: int = Field(ge=0)
    origin_index: int | None = Field(default=None, ge=0)
    origin_date: pd.Timestamp | None = None

    @field_validator("data")
    @classmethod
    def _dataframe_required(cls, value: Any) -> pd.DataFrame:
        if not isinstance(value, pd.DataFrame):
            raise TypeError("data должен быть pandas.DataFrame.")
        return value

    @field_validator("valid_mask")
    @classmethod
    def _series_required(cls, value: Any) -> pd.Series:
        if not isinstance(value, pd.Series):
            raise TypeError("valid_mask должен быть pandas.Series.")
        if value.dtype != bool:
            raise TypeError("valid_mask должен быть булевым pandas.Series.")
        return value

    @property
    def data_valid(self) -> pd.DataFrame:
        """Строки, пригодные для обучения и holdout-оценки."""

        return self.data.loc[self.valid_mask].copy().reset_index(drop=True)

    @property
    def latest_feature_row(self) -> pd.DataFrame:
        """Последняя строка с полностью определёнными признаками."""

        if self.origin_index is None:
            raise FeatureEngineeringError("В наборе нет строки с полностью определёнными признаками.")
        return self.data.loc[[self.origin_index], list(self.feature_cols)].copy()


class TimeHoldoutSplit(StrictModel):
    """Индексы и матрицы для time-based holdout-разбиения."""

    data_valid: Any
    feature_cols: tuple[str, ...]
    target_cols: tuple[str, ...]
    train_mask: Any
    test_mask: Any
    X_train: Any
    X_test: Any
    y_train: Any
    y_test: Any
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp

    @field_validator("data_valid")
    @classmethod
    def _dataframe_required(cls, value: Any) -> pd.DataFrame:
        if not isinstance(value, pd.DataFrame):
            raise TypeError("data_valid должен быть pandas.DataFrame.")
        return value


@dataclass(frozen=True)
class FeatureColumnGroups:
    """Группы признаков, удобные для отчётности и SHAP-агрегаций."""

    calendar: tuple[str, ...]
    target_lags: tuple[str, ...]
    target_rolling: tuple[str, ...]
    target_dynamics: tuple[str, ...]
    weather_lags: tuple[str, ...]
    weather_rolling: tuple[str, ...]
    other: tuple[str, ...]

    def as_dict(self) -> dict[str, tuple[str, ...]]:
        return {
            "calendar": self.calendar,
            "target_lags": self.target_lags,
            "target_rolling": self.target_rolling,
            "target_dynamics": self.target_dynamics,
            "weather_lags": self.weather_lags,
            "weather_rolling": self.weather_rolling,
            "other": self.other,
        }


WEEKLY_MODEL_INPUT_SCHEMA = TableSchema(
    required_columns=["datetime", "iso_year", "iso_week", "inc_per_10k"],
    column_types={
        "datetime": "datetime",
        "iso_year": "integer",
        "iso_week": "integer",
        "inc_per_10k": "numeric",
    },
    non_null_columns=["datetime", "iso_year", "iso_week", "inc_per_10k"],
    unique_key=["datetime"],
)

WEATHER_WEEKLY_SCHEMA = TableSchema(
    required_columns=[
        "week_start",
        "temp_mean",
        "temp_max",
        "temp_min",
        "rh_mean",
        "rh_max",
        "rh_min",
        "n_hours",
    ],
    column_types={
        "week_start": "datetime",
        "temp_mean": "numeric",
        "temp_max": "numeric",
        "temp_min": "numeric",
        "rh_mean": "numeric",
        "rh_max": "numeric",
        "rh_min": "numeric",
        "n_hours": "integer",
    },
    non_null_columns=["week_start", "temp_mean", "temp_max", "temp_min", "n_hours"],
    unique_key=["week_start"],
)


def normalize_weekly_frame(
    frame: pd.DataFrame,
    *,
    config: WeeklyFrameConfig | None = None,
) -> pd.DataFrame:
    """Нормализовать недельный ряд и проверить календарный контракт.

    При ``calendar_validation='strict'`` существующие ``iso_year``/``iso_week``
    должны совпадать с датой. При ``'repair'`` они пересчитываются из
    ``datetime``. При ``'ignore'`` календарная согласованность не проверяется,
    но отсутствующие ISO-колонки всё равно создаются.
    """

    cfg = config or WeeklyFrameConfig()
    _require_columns(frame, [cfg.datetime_col, cfg.target_col], frame_name="weekly frame")

    out = frame.copy()
    out[cfg.datetime_col] = _as_naive_datetime(out[cfg.datetime_col])

    if cfg.sort_by_datetime:
        out = out.sort_values(cfg.datetime_col).reset_index(drop=True)

    iso = out[cfg.datetime_col].dt.isocalendar()
    computed_year = iso.year.astype(int)
    computed_week = iso.week.astype(int)

    has_iso_year = cfg.iso_year_col in out.columns
    has_iso_week = cfg.iso_week_col in out.columns

    if cfg.calendar_validation == "strict" and has_iso_year and has_iso_week:
        year_mismatch = out[cfg.iso_year_col].astype(int).reset_index(drop=True) != computed_year.reset_index(drop=True)
        week_mismatch = out[cfg.iso_week_col].astype(int).reset_index(drop=True) != computed_week.reset_index(drop=True)
        if bool((year_mismatch | week_mismatch).any()):
            bad = out.loc[(year_mismatch | week_mismatch).to_numpy(), [cfg.datetime_col, cfg.iso_year_col, cfg.iso_week_col]].head(10)
            raise WeeklyFrameError(
                "ISO-календарь в таблице не согласован с datetime. "
                f"Первые проблемные строки:\n{bad}"
            )

    if cfg.calendar_validation in {"repair", "strict", "ignore"}:
        if cfg.calendar_validation == "repair" or not has_iso_year:
            out[cfg.iso_year_col] = computed_year
        if cfg.calendar_validation == "repair" or not has_iso_week:
            out[cfg.iso_week_col] = computed_week

    if cfg.require_unique_dates and out[cfg.datetime_col].duplicated().any():
        duplicates = out.loc[out[cfg.datetime_col].duplicated(keep=False), cfg.datetime_col].head(10).tolist()
        raise WeeklyFrameError(f"В недельном ряду есть повторяющиеся даты: {duplicates!r}.")

    if cfg.require_monday_week_start:
        non_monday = out.loc[out[cfg.datetime_col].dt.dayofweek != 0, cfg.datetime_col].head(10).tolist()
        if non_monday:
            raise WeeklyFrameError(f"Ожидались даты-понедельники начала ISO-недели. Примеры: {non_monday!r}.")

    if cfg.require_regular_weekly_index and len(out) > 1:
        diffs = out[cfg.datetime_col].diff().dropna().dt.days
        bad_diffs = diffs.loc[diffs != 7]
        if not bad_diffs.empty:
            idx = int(bad_diffs.index[0])
            prev_date = out.loc[idx - 1, cfg.datetime_col]
            this_date = out.loc[idx, cfg.datetime_col]
            raise WeeklyFrameError(
                "Недельный ряд должен иметь регулярный шаг 7 дней. "
                f"Нарушение: {prev_date.date()} → {this_date.date()} ({int(bad_diffs.iloc[0])} дней)."
            )

    out[cfg.target_col] = pd.to_numeric(out[cfg.target_col], errors="coerce")
    if out[cfg.target_col].isna().any():
        bad = out.loc[out[cfg.target_col].isna(), [cfg.datetime_col, cfg.target_col]].head(10)
        raise WeeklyFrameError(f"В целевой колонке есть NaN после численного приведения:\n{bad}")

    WEEKLY_MODEL_INPUT_SCHEMA.validate_dataframe(
        out.rename(
            columns={
                cfg.datetime_col: "datetime",
                cfg.iso_year_col: "iso_year",
                cfg.iso_week_col: "iso_week",
                cfg.target_col: "inc_per_10k",
            }
        ),
        name="weekly_model_input",
    )
    return out


def aggregate_hourly_weather_to_weekly(
    hourly_weather: pd.DataFrame,
    *,
    config: WeatherAggregationConfig | None = None,
) -> pd.DataFrame:
    """Агрегировать часовую погоду в недельные признаки.

    Контракт соответствует notebook: неделя задаётся понедельником
    ``W-SUN.start_time``, признаки — средняя/минимальная/максимальная
    температура и влажность, а также число часовых наблюдений.
    """

    cfg = config or WeatherAggregationConfig()
    _require_columns(
        hourly_weather,
        [cfg.time_col, cfg.temperature_col, cfg.humidity_col],
        frame_name="hourly weather",
    )

    out = hourly_weather.copy()
    out[cfg.time_col] = _as_naive_datetime(out[cfg.time_col])
    out[cfg.temperature_col] = pd.to_numeric(out[cfg.temperature_col], errors="coerce")
    out[cfg.humidity_col] = pd.to_numeric(out[cfg.humidity_col], errors="coerce")
    if out[[cfg.temperature_col, cfg.humidity_col]].isna().any().any():
        bad = out.loc[out[[cfg.temperature_col, cfg.humidity_col]].isna().any(axis=1)].head(10)
        raise WeeklyFrameError(f"В часовой погоде есть NaN после численного приведения:\n{bad}")

    out[cfg.week_start_col] = out[cfg.time_col].dt.to_period("W-SUN").dt.start_time
    weekly = (
        out.groupby(cfg.week_start_col, as_index=False)
        .agg(
            temp_mean=(cfg.temperature_col, "mean"),
            temp_max=(cfg.temperature_col, "max"),
            temp_min=(cfg.temperature_col, "min"),
            rh_mean=(cfg.humidity_col, "mean"),
            rh_max=(cfg.humidity_col, "max"),
            rh_min=(cfg.humidity_col, "min"),
            n_hours=(cfg.temperature_col, "count"),
        )
        .sort_values(cfg.week_start_col)
        .reset_index(drop=True)
    )
    weekly["n_hours"] = weekly["n_hours"].astype(int)

    if cfg.min_hours_per_week is not None:
        incomplete = weekly.loc[weekly["n_hours"] < cfg.min_hours_per_week, [cfg.week_start_col, "n_hours"]]
        if not incomplete.empty and not cfg.drop_incomplete_weeks:
            raise WeeklyFrameError(
                "В погодных данных есть неполные недели. "
                f"Минимум: {cfg.min_hours_per_week} часов. Примеры:\n{incomplete.head(10)}"
            )
        if not incomplete.empty and cfg.drop_incomplete_weeks:
            weekly = weekly.loc[weekly["n_hours"] >= cfg.min_hours_per_week].reset_index(drop=True)

    WEATHER_WEEKLY_SCHEMA.validate_dataframe(weekly, name="weather_weekly")
    return weekly


def merge_weekly_influenza_weather(
    influenza_weekly: pd.DataFrame,
    weather_weekly: pd.DataFrame,
    *,
    weekly_config: WeeklyFrameConfig | None = None,
    weather_week_start_col: str = "week_start",
    drop_weather_hours: bool = True,
) -> pd.DataFrame:
    """Объединить недельную заболеваемость и недельную погоду one-to-one."""

    cfg = weekly_config or WeeklyFrameConfig()
    infl = normalize_weekly_frame(influenza_weekly, config=cfg)
    weather = weather_weekly.copy()
    _require_columns(weather, [weather_week_start_col], frame_name="weather_weekly")
    weather[weather_week_start_col] = _as_naive_datetime(weather[weather_week_start_col])
    if weather[weather_week_start_col].duplicated().any():
        duplicates = weather.loc[weather[weather_week_start_col].duplicated(keep=False), weather_week_start_col].head(10).tolist()
        raise WeeklyFrameError(f"В погодных недельных данных есть повторяющиеся даты: {duplicates!r}.")

    missing_weather = pd.Index(infl[cfg.datetime_col]).difference(pd.Index(weather[weather_week_start_col]))
    if len(missing_weather) > 0:
        raise WeeklyFrameError(
            "Нет погодных данных для части недель. "
            f"Количество: {len(missing_weather)}. Примеры: {missing_weather[:5].tolist()}"
        )

    weather_to_merge = weather.copy()
    if drop_weather_hours and "n_hours" in weather_to_merge.columns:
        weather_to_merge = weather_to_merge.drop(columns=["n_hours"])

    merged = infl.merge(
        weather_to_merge,
        left_on=cfg.datetime_col,
        right_on=weather_week_start_col,
        how="left",
        validate="one_to_one",
    )
    if weather_week_start_col != cfg.datetime_col and weather_week_start_col in merged.columns:
        merged = merged.drop(columns=[weather_week_start_col])

    weather_feature_cols = [c for c in ["temp_mean", "temp_max", "temp_min", "rh_mean", "rh_max", "rh_min"] if c in merged.columns]
    if weather_feature_cols and merged[weather_feature_cols].isna().any().any():
        bad = merged.loc[merged[weather_feature_cols].isna().any(axis=1), [cfg.datetime_col, *weather_feature_cols]].head(10)
        raise WeeklyFrameError(f"После объединения есть NaN в погодных признаках:\n{bad}")
    return merged.sort_values(cfg.datetime_col).reset_index(drop=True)


def add_fourier_week_features(
    frame: pd.DataFrame,
    *,
    week_col: str = "iso_week",
    period: int = 52,
    k: int = 2,
) -> pd.DataFrame:
    """Добавить Fourier-признаки сезонности по ISO-неделе."""

    if k < 0:
        raise PreprocessingConfigError("k не может быть отрицательным.")
    if period <= 0:
        raise PreprocessingConfigError("period должен быть положительным.")
    _require_columns(frame, [week_col], frame_name="frame")
    out = frame.copy()
    week_values = pd.to_numeric(out[week_col], errors="raise").astype(float).to_numpy()
    for harmonic in range(1, k + 1):
        out[f"sin_w{harmonic}"] = np.sin(2 * np.pi * harmonic * week_values / period)
        out[f"cos_w{harmonic}"] = np.cos(2 * np.pi * harmonic * week_values / period)
    return out


def make_lag_features(
    frame: pd.DataFrame,
    col: str,
    lags: Sequence[int],
    *,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Добавить лаговые признаки ``prefix_lagL`` для одной колонки."""

    _require_columns(frame, [col], frame_name="frame")
    lag_values = _validate_non_negative_int_sequence(lags, name="lags")
    out = frame.copy()
    feature_prefix = prefix or col
    for lag in lag_values:
        out[f"{feature_prefix}_lag{lag}"] = out[col].shift(lag)
    return out


def make_rolling_features(
    frame: pd.DataFrame,
    col: str,
    windows: Sequence[int],
    *,
    prefix: str | None = None,
) -> pd.DataFrame:
    """Добавить rolling mean/std признаки ``prefix_rollmeanW`` и ``prefix_rollstdW``."""

    _require_columns(frame, [col], frame_name="frame")
    window_values = _validate_positive_int_sequence(windows, name="windows")
    out = frame.copy()
    feature_prefix = prefix or col
    for window in window_values:
        out[f"{feature_prefix}_rollmean{window}"] = out[col].rolling(window).mean()
        out[f"{feature_prefix}_rollstd{window}"] = out[col].rolling(window).std()
    return out


def add_epidemic_dynamics_features(
    frame: pd.DataFrame,
    *,
    target_col: str = "inc_per_10k",
    growth_lags: Sequence[int] = (1, 2, 4),
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Добавить признаки краткосрочной динамики цели.

    Реализованы активные признаки notebook: абсолютные разности ``y_diffL`` и
    ускорение ``y_accel``. Закомментированные в notebook growth/log-growth и
    сезонные нормы здесь не включены, чтобы сохранить фактический контракт.
    """

    del eps  # параметр оставлен в контракте для совместимости с notebook-сигнатурой.
    _require_columns(frame, [target_col], frame_name="frame")
    lag_values = _validate_positive_int_sequence(growth_lags, name="growth_lags")
    out = frame.copy()
    y = pd.to_numeric(out[target_col], errors="raise").astype(float)
    for lag in lag_values:
        out[f"y_diff{lag}"] = y - y.shift(lag)
    out["y_accel"] = (y - y.shift(1)) - (y.shift(1) - y.shift(2))
    return out


def build_supervised(
    df: pd.DataFrame,
    *,
    config: FeatureEngineeringConfig | None = None,
) -> SupervisedDataset:
    """Построить supervised-таблицу для многошагового прогноза.

    Функция соответствует notebook-функции ``build_supervised`` с активными
    настройками из ``run_influenza_forecast_pipeline``. Возвращается полный
    DataFrame, список признаков и маска строк, пригодных для обучения.
    """

    cfg = config or FeatureEngineeringConfig()
    if cfg.validate_weekly_frame:
        weekly_cfg = cfg.weekly_frame_config.model_copy(
            update={
                "datetime_col": cfg.datetime_col,
                "iso_year_col": cfg.iso_year_col,
                "iso_week_col": cfg.iso_week_col,
                "target_col": cfg.target_col,
            }
        )
        data = normalize_weekly_frame(df, config=weekly_cfg)
    else:
        data = df.copy()
        data[cfg.datetime_col] = _as_naive_datetime(data[cfg.datetime_col])
        data = data.sort_values(cfg.datetime_col).reset_index(drop=True)

    _require_columns(
        data,
        [cfg.datetime_col, cfg.iso_year_col, cfg.iso_week_col, cfg.target_col, *cfg.temp_cols],
        frame_name="model input frame",
    )
    _coerce_numeric_columns(data, [cfg.target_col, cfg.iso_year_col, cfg.iso_week_col, *cfg.temp_cols])

    data = add_fourier_week_features(
        data,
        week_col=cfg.iso_week_col,
        period=cfg.fourier_period,
        k=cfg.fourier_k,
    )
    data = make_lag_features(data, cfg.target_col, cfg.y_lags, prefix="y")
    data = make_rolling_features(data, cfg.target_col, cfg.y_roll_windows, prefix="y")
    data = add_epidemic_dynamics_features(
        data,
        target_col=cfg.target_col,
        growth_lags=cfg.growth_lags,
        eps=cfg.epidemic_dynamics_eps,
    )

    for temp_col in cfg.temp_cols:
        data = make_lag_features(data, temp_col, cfg.temp_lags, prefix=temp_col)
        data = make_rolling_features(data, temp_col, cfg.temp_roll_windows, prefix=temp_col)

    for horizon in range(1, cfg.horizons + 1):
        data[f"y_h{horizon}"] = data[cfg.target_col].shift(-horizon)

    target_cols = cfg.target_columns
    drop_cols = {
        cfg.datetime_col,
        cfg.target_col,
        *cfg.raw_columns_to_drop,
        *target_cols,
    }
    feature_cols = tuple(c for c in data.columns if c not in drop_cols)

    if cfg.require_numeric_features:
        non_numeric = [col for col in feature_cols if not is_numeric_dtype(data[col])]
        if non_numeric:
            raise FeatureEngineeringError(
                "Все модельные признаки должны быть численными. "
                f"Нечисленные признаки: {non_numeric}."
            )

    valid_mask = pd.Series(True, index=data.index)
    for target_col in target_cols:
        valid_mask &= data[target_col].notna()
    valid_mask &= data.loc[:, list(feature_cols)].notna().all(axis=1)

    feature_ready_mask = data.loc[:, list(feature_cols)].notna().all(axis=1)
    origin_indices = np.where(feature_ready_mask.to_numpy())[0]
    origin_index = int(origin_indices[-1]) if len(origin_indices) else None
    origin_date = pd.to_datetime(data.loc[origin_index, cfg.datetime_col]) if origin_index is not None else None

    return SupervisedDataset(
        data=data,
        feature_cols=feature_cols,
        target_cols=target_cols,
        config=cfg,
        valid_mask=valid_mask.astype(bool),
        valid_row_count=int(valid_mask.sum()),
        origin_index=origin_index,
        origin_date=origin_date,
    )


def make_time_holdout_split(
    supervised: SupervisedDataset,
    *,
    config: HoldoutSplitConfig | None = None,
) -> TimeHoldoutSplit:
    """Построить time-based holdout-разбиение, как в notebook."""

    cfg = config or HoldoutSplitConfig()
    data_valid = supervised.data_valid
    if len(data_valid) <= cfg.test_weeks:
        raise FeatureEngineeringError(
            f"Недостаточно валидных строк ({len(data_valid)}) для test_weeks={cfg.test_weeks}."
        )
    n_train = len(data_valid) - cfg.test_weeks
    if n_train < cfg.min_train_weeks:
        raise FeatureEngineeringError(
            f"Недостаточно обучающих недель: {n_train}; требуется минимум {cfg.min_train_weeks}."
        )

    datetime_col = supervised.config.datetime_col
    test_start_date = pd.to_datetime(data_valid[datetime_col].iloc[-cfg.test_weeks])
    train_mask = data_valid[datetime_col] < test_start_date
    test_mask = data_valid[datetime_col] >= test_start_date

    X_all = data_valid.loc[:, list(supervised.feature_cols)].to_numpy(dtype=float)
    y_all = data_valid.loc[:, list(supervised.target_cols)].to_numpy(dtype=float)
    train_array = train_mask.to_numpy(dtype=bool)
    test_array = test_mask.to_numpy(dtype=bool)

    return TimeHoldoutSplit(
        data_valid=data_valid,
        feature_cols=supervised.feature_cols,
        target_cols=supervised.target_cols,
        train_mask=train_mask.reset_index(drop=True),
        test_mask=test_mask.reset_index(drop=True),
        X_train=X_all[train_array],
        X_test=X_all[test_array],
        y_train=y_all[train_array],
        y_test=y_all[test_array],
        train_start=pd.to_datetime(data_valid.loc[train_mask, datetime_col].min()),
        train_end=pd.to_datetime(data_valid.loc[train_mask, datetime_col].max()),
        test_start=pd.to_datetime(data_valid.loc[test_mask, datetime_col].min()),
        test_end=pd.to_datetime(data_valid.loc[test_mask, datetime_col].max()),
    )


def infer_feature_column_groups(
    feature_cols: Sequence[str],
    *,
    temp_cols: Sequence[str] = ("temp_mean",),
) -> FeatureColumnGroups:
    """Разложить признаки по группам для отчётности и explainability."""

    features = tuple(feature_cols)
    calendar = tuple(c for c in features if c in {"iso_year", "iso_week"} or c.startswith("sin_w") or c.startswith("cos_w"))
    target_lags = tuple(c for c in features if c.startswith("y_lag"))
    target_rolling = tuple(c for c in features if c.startswith("y_roll"))
    target_dynamics = tuple(c for c in features if c.startswith("y_diff") or c == "y_accel")
    weather_lags = tuple(
        c
        for c in features
        if any(c.startswith(f"{temp_col}_lag") for temp_col in temp_cols)
    )
    weather_rolling = tuple(
        c
        for c in features
        if any(c.startswith(f"{temp_col}_roll") for temp_col in temp_cols)
    )
    assigned = set(calendar) | set(target_lags) | set(target_rolling) | set(target_dynamics) | set(weather_lags) | set(weather_rolling)
    other = tuple(c for c in features if c not in assigned)
    return FeatureColumnGroups(
        calendar=calendar,
        target_lags=target_lags,
        target_rolling=target_rolling,
        target_dynamics=target_dynamics,
        weather_lags=weather_lags,
        weather_rolling=weather_rolling,
        other=other,
    )


def supervised_to_feature_list_frame(supervised: SupervisedDataset) -> pd.DataFrame:
    """Вернуть таблицу feature_name/feature_index, совместимую с notebook."""

    return pd.DataFrame(
        {
            "feature_name": list(supervised.feature_cols),
            "feature_index": np.arange(len(supervised.feature_cols), dtype=int),
        }
    )


def save_supervised_dataset(
    supervised: SupervisedDataset,
    output_dir: str | Path,
    *,
    save_full_data: bool = True,
    save_feature_list: bool = True,
) -> dict[str, Path]:
    """Сохранить supervised-таблицу и список признаков."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    if save_full_data:
        paths["supervised_data"] = write_table(
            supervised.data,
            output / "supervised_data.csv",
            options=TableWriteOptions(index=False),
        )
    if save_feature_list:
        paths["feature_list"] = write_table(
            supervised_to_feature_list_frame(supervised),
            output / "feature_list.csv",
            options=TableWriteOptions(index=False),
        )
    return paths


def _as_naive_datetime(values: Any) -> pd.Series:
    series = pd.to_datetime(values)
    if isinstance(series, pd.DatetimeIndex):
        series = pd.Series(series)
    if getattr(series.dt, "tz", None) is not None:
        series = series.dt.tz_localize(None)
    return series


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, frame_name: str) -> None:
    missing = [col for col in columns if col not in frame.columns]
    if missing:
        raise WeeklyFrameError(f"В {frame_name} отсутствуют обязательные колонки: {missing}.")


def _coerce_numeric_columns(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if frame.loc[:, list(columns)].isna().any().any():
        bad = frame.loc[frame.loc[:, list(columns)].isna().any(axis=1), list(columns)].head(10)
        raise FeatureEngineeringError(f"После численного приведения появились NaN:\n{bad}")


def _validate_non_negative_int_sequence(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    out = tuple(int(v) for v in values)
    if any(v < 0 for v in out):
        raise PreprocessingConfigError(f"{name} не может содержать отрицательные значения.")
    return out


def _validate_positive_int_sequence(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    out = tuple(int(v) for v in values)
    if any(v <= 0 for v in out):
        raise PreprocessingConfigError(f"{name} должен содержать только положительные значения.")
    return out


def _validate_unique_non_negative_names(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    normalized = tuple(str(v).strip() for v in values)
    if any(not v for v in normalized):
        raise ValueError(f"{field_name} содержит пустое имя колонки.")
    duplicates = sorted({v for v in normalized if normalized.count(v) > 1})
    if duplicates:
        raise ValueError(f"{field_name} содержит повторяющиеся имена: {duplicates}.")
    return normalized


__all__ = [
    "CalendarValidationMode",
    "PreprocessingError",
    "WeeklyFrameError",
    "FeatureEngineeringError",
    "PreprocessingConfigError",
    "WeeklyFrameConfig",
    "WeatherAggregationConfig",
    "FeatureEngineeringConfig",
    "HoldoutSplitConfig",
    "SupervisedDataset",
    "TimeHoldoutSplit",
    "FeatureColumnGroups",
    "WEEKLY_MODEL_INPUT_SCHEMA",
    "WEATHER_WEEKLY_SCHEMA",
    "normalize_weekly_frame",
    "aggregate_hourly_weather_to_weekly",
    "merge_weekly_influenza_weather",
    "add_fourier_week_features",
    "make_lag_features",
    "make_rolling_features",
    "add_epidemic_dynamics_features",
    "build_supervised",
    "make_time_holdout_split",
    "infer_feature_column_groups",
    "supervised_to_feature_list_frame",
    "save_supervised_dataset",
]

