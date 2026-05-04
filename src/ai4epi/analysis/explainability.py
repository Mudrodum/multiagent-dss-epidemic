"""
SHAP-интерпретация многошагового прогноза для ai4epi.

Модуль переносит explainability-слой исследовательского notebook в пакетный
контракт. Он работает поверх уже обученного ``ForecastRunResult`` и не
загружает данные, не обучает модели, не собирает ``GlobalContext`` и не
обращается к LLM.

Основные артефакты:

* ``shap_global_importance.csv`` — глобальная важность признаков по горизонту;
* ``shap_local_values.csv`` — локальные SHAP-значения в long-format;
* ``shap_worst_cases.csv`` — худшие прогнозные недели по абсолютной ошибке;
* ``shap_summary.json`` — компактная сводка для последующего
  ``context_builders.py``.

SHAP-значения считаются для eval-моделей на test-выборке. Для
``HistGradientBoostingRegressor`` сначала используется ``shap.TreeExplainer``.
Если он недоступен для конкретной версии окружения, выполняется явный fallback
на ``shap.Explainer(..., algorithm='permutation')``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4epi.analysis.forecasting import ForecastRunResult, get_target_transform
from ai4epi.analysis.preprocessing import FeatureColumnGroups, infer_feature_column_groups


JsonObject = dict[str, Any]
ExplainerKind = Literal["tree", "permutation"]
PermutationOutputScale = Literal["model_raw", "target_scale"]


class ExplainabilityError(ValueError):
    """Базовая ошибка explainability-слоя."""


class ShapDependencyError(ExplainabilityError):
    """Пакет shap отсутствует или недоступен."""


class ShapComputationError(ExplainabilityError):
    """Ошибка вычисления SHAP-значений."""


class StrictModel(BaseModel):
    """Базовая pydantic-модель с закрытым контрактом."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )


class ExplainabilityConfig(StrictModel):
    """Настройки SHAP-pass.

    Parameters
    ----------
    shap_horizons:
        Горизонты прогноза для объяснения. ``None`` означает все горизонты из
        ``ForecastRunResult``.
    shap_background:
        Размер фоновой выборки для permutation fallback.
    shap_worst_n:
        Число худших прогнозных недель, сохраняемых для каждого горизонта.
    max_test_samples:
        Если задано, SHAP считается на воспроизводимой подвыборке test-строк.
        Worst-cases при этом всё равно считаются по полной test-выборке.
    top_features_per_horizon:
        Число признаков в компактной сводке ``shap_summary``.
    direction_confidence_threshold:
        Порог устойчивости направления: ``abs(mean_shap) / mean_abs_shap``.
        Если отношение ниже порога, направление считается неустойчивым.
    permutation_output_scale:
        Масштаб fallback-объяснений. ``model_raw`` сохраняет тот же масштаб,
        что и TreeExplainer для модели; ``target_scale`` объясняет уже
        инвертированный прогноз и используется только как явная опция.
    include_plots:
        Если ``True``, возвращает SHAP-объекты для построения графиков
        вызывающим кодом. Сам модуль файловые изображения не сохраняет.
    """

    shap_horizons: tuple[int, ...] | None = None
    shap_background: int = Field(default=100, ge=1)
    shap_worst_n: int = Field(default=5, ge=1)
    max_test_samples: int | None = Field(default=None, ge=1)
    random_state: int = 42
    top_features_per_horizon: int = Field(default=5, ge=1)
    direction_confidence_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    permutation_output_scale: PermutationOutputScale = "model_raw"
    include_plots: bool = False

    @field_validator("shap_horizons")
    @classmethod
    def validate_horizons(cls, value: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if value is None:
            return None
        normalized = tuple(int(item) for item in value)
        if not normalized:
            raise ValueError("shap_horizons не должен быть пустым.")
        if any(item <= 0 for item in normalized):
            raise ValueError("Все горизонты SHAP должны быть положительными.")
        if len(normalized) != len(set(normalized)):
            raise ValueError("shap_horizons не должен содержать дубликаты.")
        return normalized


class ExplainabilityOutputConfig(StrictModel):
    """Настройки сохранения SHAP-артефактов."""

    output_dir: Path = Path("results_csv")
    save_global_importance: bool = True
    save_local_values: bool = True
    save_worst_cases: bool = True
    save_summary_json: bool = True
    csv_encoding: str = Field(default="utf-8-sig", min_length=1)
    global_importance_filename: str = Field(default="shap_global_importance.csv", min_length=1)
    local_values_filename: str = Field(default="shap_local_values.csv", min_length=1)
    worst_cases_filename: str = Field(default="shap_worst_cases.csv", min_length=1)
    summary_filename: str = Field(default="shap_summary.json", min_length=1)

    @field_validator(
        "global_importance_filename",
        "local_values_filename",
        "worst_cases_filename",
        "summary_filename",
    )
    @classmethod
    def validate_relative_filename(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or len(path.parts) != 1:
            raise ValueError("Имена output-файлов должны быть простыми относительными именами.")
        return value

    @model_validator(mode="after")
    def validate_at_least_one_output(self) -> "ExplainabilityOutputConfig":
        if not any(
            (
                self.save_global_importance,
                self.save_local_values,
                self.save_worst_cases,
                self.save_summary_json,
            )
        ):
            raise ValueError("Должен быть включён хотя бы один SHAP-артефакт.")
        return self


class ShapTables(StrictModel):
    """Табличные результаты SHAP-pass."""

    global_importance: Any
    local_values: Any
    worst_cases: Any

    @field_validator("global_importance", "local_values", "worst_cases")
    @classmethod
    def validate_dataframe(cls, value: Any) -> pd.DataFrame:
        if not isinstance(value, pd.DataFrame):
            raise TypeError("SHAP-таблицы должны быть pandas.DataFrame.")
        return value


class ShapFeatureSummary(StrictModel):
    """Один признак в компактной SHAP-сводке для GlobalContext."""

    internal_name: str = Field(min_length=1)
    name_ru: str = Field(min_length=1)
    influence_strength: float = Field(ge=0.0)
    mean_shap: float
    direction: str = Field(min_length=1)
    direction_reliable: bool
    feature_group: str = Field(min_length=1)
    rank: int = Field(ge=1)


class ShapHorizonSummary(StrictModel):
    """Компактная SHAP-сводка для одного горизонта."""

    horizon_weeks: int = Field(ge=1)
    features: list[ShapFeatureSummary] = Field(default_factory=list)
    dominant_group: str = Field(min_length=1)
    unreliable_direction_feature_names: list[str] = Field(default_factory=list)


class ShapSummary(StrictModel):
    """Сводка explainability-слоя для последующей сборки GlobalContext."""

    by_horizon: dict[str, list[JsonObject]]
    key_insight: str = Field(min_length=1)
    horizon_summaries: list[ShapHorizonSummary]
    semantic: JsonObject = Field(default_factory=dict)

    def to_context_payload(self) -> JsonObject:
        """Вернуть payload, совместимый с ``GlobalContext.shap_summary``."""

        return {
            "by_horizon": self.by_horizon,
            "key_insight": self.key_insight,
            "semantic": self.semantic,
        }


class ShapRunResult(StrictModel):
    """Результат полного SHAP-pass."""

    tables: ShapTables
    summary: ShapSummary
    config: ExplainabilityConfig
    feature_name_map: dict[str, str]
    feature_groups: dict[str, list[str]]
    explainers_used: dict[str, ExplainerKind]
    artifacts: dict[str, Path] = Field(default_factory=dict)
    shap_objects: dict[str, Any] = Field(default_factory=dict)

    @property
    def shap_global(self) -> pd.DataFrame:
        """Совместимое с notebook имя таблицы глобальной важности."""

        return self.tables.global_importance

    @property
    def shap_local(self) -> pd.DataFrame:
        """Совместимое с notebook имя таблицы локальных SHAP-значений."""

        return self.tables.local_values

    @property
    def shap_worst(self) -> pd.DataFrame:
        """Совместимое с notebook имя таблицы худших случаев."""

        return self.tables.worst_cases

    def to_notebook_dict(self) -> dict[str, Any]:
        """Вернуть словарь с ключами, совместимыми с исходным notebook."""

        return {
            "shap_global": self.tables.global_importance,
            "shap_local": self.tables.local_values,
            "shap_worst": self.tables.worst_cases,
            "shap_summary": self.summary.to_context_payload(),
        }

    def to_public_dict(self) -> JsonObject:
        """Вернуть JSON-сериализуемую метаинформацию без тяжёлых таблиц."""

        return {
            "summary": self.summary.model_dump(mode="json", exclude_none=False),
            "config": self.config.model_dump(mode="json", exclude_none=False),
            "feature_name_map": dict(self.feature_name_map),
            "feature_groups": {key: list(value) for key, value in self.feature_groups.items()},
            "explainers_used": dict(self.explainers_used),
            "artifacts": {key: str(value) for key, value in self.artifacts.items()},
        }


@dataclass(frozen=True)
class _ModelCallableWrapper:
    """Callable wrapper для permutation fallback."""

    model: Any
    inverse_func: Callable[[np.ndarray], np.ndarray] | None = None
    clip_non_negative: bool = True

    def predict(self, X: Any) -> np.ndarray:
        raw = np.asarray(self.model.predict(X), dtype=float)
        if self.inverse_func is not None:
            raw = np.asarray(self.inverse_func(raw), dtype=float)
        if self.clip_non_negative:
            raw = np.clip(raw, 0.0, None)
        return raw


FEATURE_NAMES_RU: dict[str, str] = {
    "y_lag0": "заболеваемость предыдущей недели",
    "y_lag1": "заболеваемость 2 недели назад",
    "y_lag2": "заболеваемость 3 недели назад",
    "y_lag3": "заболеваемость 4 недели назад",
    "y_lag4": "заболеваемость 5 недель назад",
    "y_lag5": "заболеваемость 6 недель назад",
    "y_rollmean4": "скользящее среднее заболеваемости (4 нед.)",
    "y_rollmean5": "скользящее среднее заболеваемости (5 нед.)",
    "y_rollmean6": "скользящее среднее заболеваемости (6 нед.)",
    "y_rollmean7": "скользящее среднее заболеваемости (7 нед.)",
    "y_rollmean8": "скользящее среднее заболеваемости (8 нед.)",
    "y_rollmean9": "скользящее среднее заболеваемости (9 нед.)",
    "y_rollstd4": "волатильность заболеваемости (4 нед.)",
    "y_rollstd5": "волатильность заболеваемости (5 нед.)",
    "y_rollstd6": "волатильность заболеваемости (6 нед.)",
    "y_rollstd7": "волатильность заболеваемости (7 нед.)",
    "y_rollstd8": "волатильность заболеваемости (8 нед.)",
    "y_rollstd9": "волатильность заболеваемости (9 нед.)",
    "y_diff1": "изменение заболеваемости за 1 неделю",
    "y_diff2": "изменение заболеваемости за 2 недели",
    "y_diff4": "изменение заболеваемости за 4 недели",
    "y_accel": "ускорение изменения заболеваемости",
    "iso_year": "год наблюдения",
    "iso_week": "номер эпидемиологической недели",
    "sin_w1": "сезонный фактор (sin, годовой цикл)",
    "cos_w1": "сезонный фактор (cos, годовой цикл)",
    "sin_w2": "сезонный фактор (sin, полугодовой цикл)",
    "cos_w2": "сезонный фактор (cos, полугодовой цикл)",
    "temp_mean": "средняя температура воздуха",
    "temp_max": "максимальная температура воздуха",
    "temp_min": "минимальная температура воздуха",
    "temp_mean_lag0": "средняя температура текущей недели",
    "temp_mean_lag1": "температура неделю назад",
    "temp_mean_lag2": "температура 2 недели назад",
    "temp_mean_lag3": "температура 3 недели назад",
    "temp_mean_lag4": "температура 4 недели назад",
    "temp_mean_rollmean4": "скользящая средняя температура (4 нед.)",
    "temp_mean_rollmean5": "скользящая средняя температура (5 нед.)",
    "temp_mean_rollmean6": "скользящая средняя температура (6 нед.)",
    "temp_mean_rollmean7": "скользящая средняя температура (7 нед.)",
    "temp_mean_rollstd4": "волатильность температуры (4 нед.)",
    "temp_mean_rollstd5": "волатильность температуры (5 нед.)",
    "temp_mean_rollstd6": "волатильность температуры (6 нед.)",
    "temp_mean_rollstd7": "волатильность температуры (7 нед.)",
}

FEATURE_GROUPS_RU: dict[str, str] = {
    "calendar": "календарные и сезонные признаки",
    "target_lags": "лаговые признаки заболеваемости",
    "target_rolling": "скользящие статистики заболеваемости",
    "target_dynamics": "признаки динамики заболеваемости",
    "weather_lags": "лаговые погодные признаки",
    "weather_rolling": "скользящие погодные признаки",
    "other": "прочие признаки",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_shap_analysis(
    forecast_result: ForecastRunResult,
    *,
    config: ExplainabilityConfig | None = None,
    output: ExplainabilityOutputConfig | None = None,
) -> ShapRunResult:
    """Выполнить SHAP-анализ поверх ``ForecastRunResult``.

    Функция не модифицирует ``ForecastRunResult``. Все таблицы возвращаются в
    памяти; сохранение CSV/JSON включается через ``output``.
    """

    cfg = config or ExplainabilityConfig()
    shap_module = _import_shap()
    _validate_forecast_result_for_shap(forecast_result)

    feature_cols = tuple(forecast_result.feature_cols)
    horizons = _resolve_horizons(cfg.shap_horizons, n_horizons=len(forecast_result.target_cols))
    X_test_df = pd.DataFrame(forecast_result.X_test, columns=list(feature_cols))
    dates_test = pd.to_datetime(pd.Series(forecast_result.dates_test)).reset_index(drop=True)
    y_test_mat = np.asarray(forecast_result.y_test_mat, dtype=float)
    y_pred_mat = np.asarray(forecast_result.y_pred_mat, dtype=float)

    sample = _make_explain_sample(
        X_test_df=X_test_df,
        y_test_mat=y_test_mat,
        y_pred_mat=y_pred_mat,
        dates_test=dates_test,
        max_test_samples=cfg.max_test_samples,
        random_state=cfg.random_state,
    )

    global_tables: list[pd.DataFrame] = []
    local_tables: list[pd.DataFrame] = []
    worst_tables: list[pd.DataFrame] = []
    explainers_used: dict[str, ExplainerKind] = {}
    shap_objects: dict[str, Any] = {}

    transform = get_target_transform(forecast_result.config.target_transform)

    for horizon in horizons:
        model = forecast_result.models_eval[horizon - 1]
        sv_values, base_values, explainer_kind, shap_object = compute_shap_values_for_horizon(
            model=model,
            X_explain=sample.X_explain,
            feature_cols=feature_cols,
            shap_module=shap_module,
            config=cfg,
            inverse_func=transform.inverse,
            clip_non_negative=forecast_result.config.clip_predictions_to_non_negative,
        )
        explainers_used[str(horizon)] = explainer_kind
        if cfg.include_plots:
            shap_objects[str(horizon)] = shap_object

        global_tables.append(
            compute_global_shap_importance(
                shap_values=sv_values,
                X_explain=sample.X_explain,
                feature_cols=feature_cols,
                horizon_weeks=horizon,
                explainer_kind=explainer_kind,
            )
        )
        worst_tables.append(
            compute_worst_case_table(
                horizon_weeks=horizon,
                y_test_mat=y_test_mat,
                y_pred_mat=y_pred_mat,
                dates_test=dates_test,
                worst_n=cfg.shap_worst_n,
            )
        )
        local_tables.append(
            compute_local_shap_values(
                shap_values=sv_values,
                base_values=base_values,
                X_explain=sample.X_explain,
                feature_cols=feature_cols,
                horizon_weeks=horizon,
                y_test_explain=sample.y_test_explain,
                y_pred_explain=sample.y_pred_explain,
                dates_explain=sample.dates_explain,
            )
        )

    global_importance = pd.concat(global_tables, ignore_index=True) if global_tables else _empty_global_table()
    local_values = pd.concat(local_tables, ignore_index=True) if local_tables else _empty_local_table()
    worst_cases = pd.concat(worst_tables, ignore_index=True) if worst_tables else _empty_worst_table()

    feature_name_map = build_feature_name_map(feature_cols)
    groups = infer_feature_column_groups(feature_cols, temp_cols=forecast_result.supervised.config.temp_cols)
    feature_group_map = build_feature_group_map(groups)
    summary = summarize_shap_for_context(
        global_importance,
        feature_name_map=feature_name_map,
        feature_group_map=feature_group_map,
        config=cfg,
    )

    result = ShapRunResult(
        tables=ShapTables(
            global_importance=global_importance,
            local_values=local_values,
            worst_cases=worst_cases,
        ),
        summary=summary,
        config=cfg,
        feature_name_map=feature_name_map,
        feature_groups={key: list(value) for key, value in groups.as_dict().items()},
        explainers_used=explainers_used,
        artifacts={},
        shap_objects=shap_objects,
    )

    if output is not None:
        result.artifacts = save_shap_artifacts(result, output)
    return result


def compute_global_shap_importance(
    *,
    shap_values: np.ndarray,
    X_explain: pd.DataFrame,
    feature_cols: Sequence[str],
    horizon_weeks: int,
    explainer_kind: ExplainerKind,
) -> pd.DataFrame:
    """Построить таблицу глобальной важности признаков для одного горизонта."""

    values = _validate_shap_matrix(shap_values, expected_features=len(feature_cols))
    if len(X_explain) != values.shape[0]:
        raise ShapComputationError(
            f"Число строк X_explain ({len(X_explain)}) не совпадает с SHAP ({values.shape[0]})."
        )
    frame = pd.DataFrame(
        {
            "horizon_weeks": int(horizon_weeks),
            "feature": list(feature_cols),
            "mean_abs_shap": np.mean(np.abs(values), axis=0),
            "mean_shap": np.mean(values, axis=0),
            "mean_feature_value": X_explain.loc[:, list(feature_cols)].mean(axis=0).to_numpy(dtype=float),
            "explainer": explainer_kind,
        }
    )
    frame = frame.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1, dtype=int)
    return frame


def compute_local_shap_values(
    *,
    shap_values: np.ndarray,
    base_values: np.ndarray,
    X_explain: pd.DataFrame,
    feature_cols: Sequence[str],
    horizon_weeks: int,
    y_test_explain: np.ndarray,
    y_pred_explain: np.ndarray,
    dates_explain: pd.Series,
) -> pd.DataFrame:
    """Построить long-format таблицу локальных SHAP-значений."""

    values = _validate_shap_matrix(shap_values, expected_features=len(feature_cols))
    n_rows = values.shape[0]
    base = _normalize_base_values(base_values, n_rows=n_rows)

    y_true = np.asarray(y_test_explain, dtype=float)
    y_pred = np.asarray(y_pred_explain, dtype=float)
    if y_true.ndim != 2 or y_pred.ndim != 2:
        raise ShapComputationError("y_test_explain и y_pred_explain должны быть двумерными матрицами.")
    if y_true.shape[0] < n_rows or y_pred.shape[0] < n_rows:
        raise ShapComputationError("Матрицы y_test/y_pred короче SHAP-выборки.")
    h_index = int(horizon_weeks) - 1
    if h_index < 0 or h_index >= y_true.shape[1] or h_index >= y_pred.shape[1]:
        raise ShapComputationError(f"Недоступен горизонт h={horizon_weeks} в y_test/y_pred.")

    shap_long = (
        pd.DataFrame(values, columns=list(feature_cols))
        .assign(row_id=np.arange(n_rows, dtype=int))
        .melt(id_vars="row_id", var_name="feature", value_name="shap_value")
    )
    feat_long = (
        X_explain.reset_index(drop=True)
        .assign(row_id=np.arange(n_rows, dtype=int))
        .melt(id_vars="row_id", var_name="feature", value_name="feature_value")
    )
    abs_err = np.abs(y_true[:n_rows, h_index] - y_pred[:n_rows, h_index])
    row_meta = pd.DataFrame(
        {
            "row_id": np.arange(n_rows, dtype=int),
            "horizon_weeks": int(horizon_weeks),
            "date": pd.to_datetime(dates_explain.iloc[:n_rows]).to_numpy(),
            "y_true": y_true[:n_rows, h_index],
            "y_pred": y_pred[:n_rows, h_index],
            "abs_err": abs_err,
            "base_value": base,
            "base_plus_shap_sum": base + values.sum(axis=1),
        }
    )
    local = shap_long.merge(feat_long, on=["row_id", "feature"], how="left").merge(row_meta, on="row_id", how="left")
    local["abs_shap"] = local["shap_value"].abs()
    return local


def compute_worst_case_table(
    *,
    horizon_weeks: int,
    y_test_mat: np.ndarray,
    y_pred_mat: np.ndarray,
    dates_test: pd.Series,
    worst_n: int = 5,
) -> pd.DataFrame:
    """Построить таблицу худших прогнозных недель по абсолютной ошибке."""

    y_true = _validate_2d_array(y_test_mat, name="y_test_mat")
    y_pred = _validate_2d_array(y_pred_mat, name="y_pred_mat")
    if y_true.shape != y_pred.shape:
        raise ShapComputationError(f"Формы y_test_mat и y_pred_mat должны совпадать: {y_true.shape} != {y_pred.shape}.")
    h_index = int(horizon_weeks) - 1
    if h_index < 0 or h_index >= y_true.shape[1]:
        raise ShapComputationError(f"Недоступен горизонт h={horizon_weeks}.")
    abs_err = np.abs(y_true[:, h_index] - y_pred[:, h_index])
    worst_idx = np.argsort(-abs_err)[: min(int(worst_n), len(abs_err))]
    frame = pd.DataFrame(
        {
            "horizon_weeks": int(horizon_weeks),
            "row_id": worst_idx.astype(int),
            "date": pd.to_datetime(dates_test.iloc[worst_idx]).to_numpy(),
            "y_true": y_true[worst_idx, h_index],
            "y_pred": y_pred[worst_idx, h_index],
            "abs_err": abs_err[worst_idx],
        }
    )
    return frame.sort_values("abs_err", ascending=False).reset_index(drop=True)


def compute_shap_values_for_horizon(
    *,
    model: Any,
    X_explain: pd.DataFrame,
    feature_cols: Sequence[str],
    shap_module: Any,
    config: ExplainabilityConfig,
    inverse_func: Callable[[np.ndarray], np.ndarray] | None = None,
    clip_non_negative: bool = True,
) -> tuple[np.ndarray, np.ndarray, ExplainerKind, Any]:
    """Вычислить SHAP-значения для одной horizon-specific модели."""

    try:
        explainer = shap_module.TreeExplainer(model, feature_names=list(feature_cols))
        explanation = explainer(X_explain)
        values, base_values = _extract_values_and_base(explanation)
        return values, base_values, "tree", explanation
    except Exception as tree_exc:
        try:
            background = X_explain.sample(
                min(int(config.shap_background), len(X_explain)),
                random_state=config.random_state,
            )
            wrapper = _ModelCallableWrapper(
                model=model,
                inverse_func=inverse_func if config.permutation_output_scale == "target_scale" else None,
                clip_non_negative=clip_non_negative,
            )
            explainer = shap_module.Explainer(
                wrapper.predict,
                background,
                algorithm="permutation",
                feature_names=list(feature_cols),
            )
            explanation = explainer(X_explain)
            values, base_values = _extract_values_and_base(explanation)
            return values, base_values, "permutation", explanation
        except Exception as permutation_exc:
            raise ShapComputationError(
                "Не удалось вычислить SHAP ни через TreeExplainer, ни через permutation fallback. "
                f"TreeExplainer: {tree_exc}; PermutationExplainer: {permutation_exc}"
            ) from permutation_exc


def summarize_shap_for_context(
    shap_global: pd.DataFrame,
    *,
    feature_name_map: Mapping[str, str] | None = None,
    feature_group_map: Mapping[str, str] | None = None,
    config: ExplainabilityConfig | None = None,
) -> ShapSummary:
    """Построить компактную SHAP-сводку для ``GlobalContext.shap_summary``."""

    cfg = config or ExplainabilityConfig()
    _require_columns(
        shap_global,
        ["horizon_weeks", "feature", "mean_abs_shap", "mean_shap", "rank"],
        frame_name="shap_global",
    )
    name_map = dict(feature_name_map or {})
    group_map = dict(feature_group_map or {})

    by_horizon: dict[str, list[JsonObject]] = {}
    horizon_summaries: list[ShapHorizonSummary] = []

    for horizon in sorted(pd.to_numeric(shap_global["horizon_weeks"], errors="raise").astype(int).unique()):
        part = shap_global.loc[shap_global["horizon_weeks"].astype(int) == horizon].sort_values("rank")
        top = part.head(cfg.top_features_per_horizon)
        features: list[ShapFeatureSummary] = []
        for _, row in top.iterrows():
            internal_name = str(row["feature"])
            mean_shap = float(row["mean_shap"])
            mean_abs_shap = float(row["mean_abs_shap"])
            ratio = abs(mean_shap) / mean_abs_shap if mean_abs_shap > 0 else 0.0
            direction_reliable = ratio >= cfg.direction_confidence_threshold
            direction = _direction_text(mean_shap, direction_reliable=direction_reliable)
            features.append(
                ShapFeatureSummary(
                    internal_name=internal_name,
                    name_ru=name_map.get(internal_name, internal_name),
                    influence_strength=round(mean_abs_shap, 3),
                    mean_shap=round(mean_shap, 6),
                    direction=direction,
                    direction_reliable=direction_reliable,
                    feature_group=group_map.get(internal_name, "other"),
                    rank=int(row["rank"]),
                )
            )

        dominant_group = dominant_feature_group(features)
        unreliable = [item.name_ru for item in features if not item.direction_reliable]
        horizon_summaries.append(
            ShapHorizonSummary(
                horizon_weeks=int(horizon),
                features=features,
                dominant_group=dominant_group,
                unreliable_direction_feature_names=unreliable,
            )
        )
        by_horizon[str(int(horizon))] = [
            {
                "название": item.name_ru,
                "сила_влияния": item.influence_strength,
                "направление": item.direction,
                "направление_надёжное": item.direction_reliable,
            }
            for item in features
        ]

    semantic = build_shap_semantic_summary(horizon_summaries)
    key_insight = build_shap_key_insight(horizon_summaries)
    return ShapSummary(
        by_horizon=by_horizon,
        key_insight=key_insight,
        horizon_summaries=horizon_summaries,
        semantic=semantic,
    )


def save_shap_artifacts(result: ShapRunResult, output: ExplainabilityOutputConfig) -> dict[str, Path]:
    """Сохранить SHAP CSV/JSON-артефакты."""

    output_dir = Path(output.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    if output.save_global_importance:
        path = output_dir / output.global_importance_filename
        result.tables.global_importance.to_csv(path, index=False, encoding=output.csv_encoding)
        artifacts["shap_global_importance"] = path.resolve()
    if output.save_local_values:
        path = output_dir / output.local_values_filename
        result.tables.local_values.to_csv(path, index=False, encoding=output.csv_encoding)
        artifacts["shap_local_values"] = path.resolve()
    if output.save_worst_cases:
        path = output_dir / output.worst_cases_filename
        result.tables.worst_cases.to_csv(path, index=False, encoding=output.csv_encoding)
        artifacts["shap_worst_cases"] = path.resolve()
    if output.save_summary_json:
        path = output_dir / output.summary_filename
        path.write_text(
            json.dumps(result.summary.model_dump(mode="json", exclude_none=False), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        artifacts["shap_summary"] = path.resolve()
    return artifacts


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------


def build_feature_name_map(feature_cols: Sequence[str]) -> dict[str, str]:
    """Построить человекочитаемые русские имена признаков."""

    return {str(feature): FEATURE_NAMES_RU.get(str(feature), str(feature)) for feature in feature_cols}


def build_feature_group_map(groups: FeatureColumnGroups | Mapping[str, Sequence[str]]) -> dict[str, str]:
    """Построить mapping ``feature -> group``."""

    group_dict = groups.as_dict() if isinstance(groups, FeatureColumnGroups) else dict(groups)
    out: dict[str, str] = {}
    for group_name, features in group_dict.items():
        for feature in features:
            out[str(feature)] = str(group_name)
    return out


def dominant_feature_group(features: Sequence[ShapFeatureSummary]) -> str:
    """Определить доминирующую группу среди top-признаков горизонта."""

    top = list(features)[:3]
    if not top:
        return "other"
    counts: dict[str, int] = {}
    weights: dict[str, float] = {}
    for item in top:
        counts[item.feature_group] = counts.get(item.feature_group, 0) + 1
        weights[item.feature_group] = weights.get(item.feature_group, 0.0) + item.influence_strength
    ordered = sorted(counts, key=lambda group: (counts[group], weights[group], group), reverse=True)
    if not ordered:
        return "other"
    first = ordered[0]
    if len(ordered) > 1 and counts[first] == counts[ordered[1]] and abs(weights[first] - weights[ordered[1]]) < 1e-12:
        return "mixed"
    return first


def build_shap_semantic_summary(horizon_summaries: Sequence[ShapHorizonSummary]) -> JsonObject:
    """Собрать машинно-читаемую semantic-сводку для factual evaluator."""

    by_h = {item.horizon_weeks: item for item in horizon_summaries}
    h1 = by_h.get(1)
    h4 = by_h.get(4) or by_h.get(max(by_h) if by_h else 0)

    h1_group = h1.dominant_group if h1 is not None else None
    h4_group = h4.dominant_group if h4 is not None else None
    transition = _transition_pattern(h1_group, h4_group)

    return {
        "h1_dominant_group": h1_group,
        "h4_dominant_group": h4_group,
        "transition_pattern": transition,
        "h1_top_features": [feature.internal_name for feature in h1.features[:3]] if h1 else [],
        "h4_top_features": [feature.internal_name for feature in h4.features[:3]] if h4 else [],
        "h1_unreliable_direction_features": h1.unreliable_direction_feature_names if h1 else [],
        "h4_unreliable_direction_features": h4.unreliable_direction_feature_names if h4 else [],
    }


def build_shap_key_insight(horizon_summaries: Sequence[ShapHorizonSummary]) -> str:
    """Сформировать детерминированный ключевой инсайт по h=1 и h=4."""

    by_h = {item.horizon_weeks: item for item in horizon_summaries}
    h1 = by_h.get(1)
    h4 = by_h.get(4) or by_h.get(max(by_h) if by_h else 0)
    if h1 is None or h4 is None:
        return "SHAP-сводка построена по доступным горизонтам прогноза."

    h1_group = h1.dominant_group
    h4_group = h4.dominant_group
    if h1_group in {"target_lags", "target_rolling", "target_dynamics"} and h4_group in {"weather_lags", "weather_rolling", "calendar"}:
        return (
            "На коротком горизонте (1 нед.) доминируют признаки недавней заболеваемости "
            "и динамики эпидемического процесса. На длинном горизонте возрастает роль "
            "температурных, погодных или сезонных факторов."
        )
    if h1_group == h4_group and h1_group != "mixed":
        group_ru = FEATURE_GROUPS_RU.get(h1_group, h1_group)
        return f"На коротком и длинном горизонтах доминирует одна группа признаков: {group_ru}."

    h1_names = ", ".join(feature.name_ru for feature in h1.features[:3])
    h4_names = ", ".join(feature.name_ru for feature in h4.features[:3])
    return f"Top-3 факторы для h=1: {h1_names}. Top-3 факторы для h={h4.horizon_weeks}: {h4_names}."


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _ExplainSample:
    X_explain: pd.DataFrame
    y_test_explain: np.ndarray
    y_pred_explain: np.ndarray
    dates_explain: pd.Series


def _import_shap() -> Any:
    try:
        import shap  # type: ignore[import-untyped]
    except ImportError as exc:
        raise ShapDependencyError("Для SHAP-анализа требуется пакет shap. Установите его как проектную зависимость.") from exc
    return shap


def _validate_forecast_result_for_shap(result: ForecastRunResult) -> None:
    if not result.models_eval:
        raise ExplainabilityError("ForecastRunResult.models_eval пуст: нечего объяснять.")
    if len(result.models_eval) != len(result.target_cols):
        raise ExplainabilityError(
            f"Число eval-моделей ({len(result.models_eval)}) не совпадает с числом target-горизонтов ({len(result.target_cols)})."
        )
    X_test = _validate_2d_array(result.X_test, name="X_test")
    if X_test.shape[1] != len(result.feature_cols):
        raise ExplainabilityError(
            f"X_test содержит {X_test.shape[1]} признаков, а feature_cols содержит {len(result.feature_cols)}."
        )
    y_test = _validate_2d_array(result.y_test_mat, name="y_test_mat")
    y_pred = _validate_2d_array(result.y_pred_mat, name="y_pred_mat")
    if y_test.shape != y_pred.shape:
        raise ExplainabilityError(f"Формы y_test_mat и y_pred_mat должны совпадать: {y_test.shape} != {y_pred.shape}.")
    if y_test.shape[0] != X_test.shape[0]:
        raise ExplainabilityError("Число строк X_test не совпадает с числом строк y_test_mat.")


def _resolve_horizons(shap_horizons: tuple[int, ...] | None, *, n_horizons: int) -> tuple[int, ...]:
    horizons = tuple(range(1, n_horizons + 1)) if shap_horizons is None else tuple(shap_horizons)
    invalid = [h for h in horizons if h < 1 or h > n_horizons]
    if invalid:
        raise ExplainabilityError(f"Запрошены недоступные горизонты SHAP: {invalid!r}; доступно 1..{n_horizons}.")
    return horizons


def _make_explain_sample(
    *,
    X_test_df: pd.DataFrame,
    y_test_mat: np.ndarray,
    y_pred_mat: np.ndarray,
    dates_test: pd.Series,
    max_test_samples: int | None,
    random_state: int,
) -> _ExplainSample:
    if max_test_samples is not None and max_test_samples < len(X_test_df):
        sample_idx = (
            pd.RangeIndex(len(X_test_df))
            .to_series()
            .sample(int(max_test_samples), random_state=random_state)
            .sort_index()
            .index.to_numpy(dtype=int)
        )
        return _ExplainSample(
            X_explain=X_test_df.iloc[sample_idx].reset_index(drop=True),
            y_test_explain=np.asarray(y_test_mat, dtype=float)[sample_idx],
            y_pred_explain=np.asarray(y_pred_mat, dtype=float)[sample_idx],
            dates_explain=dates_test.iloc[sample_idx].reset_index(drop=True),
        )
    return _ExplainSample(
        X_explain=X_test_df.reset_index(drop=True),
        y_test_explain=np.asarray(y_test_mat, dtype=float),
        y_pred_explain=np.asarray(y_pred_mat, dtype=float),
        dates_explain=dates_test.reset_index(drop=True),
    )


def _extract_values_and_base(explanation: Any) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(explanation.values, dtype=float)
    if values.ndim == 3:
        values = values[:, :, 0]
    if values.ndim != 2:
        raise ShapComputationError(f"Ожидались 2D SHAP-значения, получена форма {values.shape}.")
    base_values = np.asarray(explanation.base_values, dtype=float).reshape(-1)
    return values, base_values


def _validate_shap_matrix(values: np.ndarray, *, expected_features: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.ndim != 2:
        raise ShapComputationError(f"Ожидалась 2D SHAP-матрица, получена форма {arr.shape}.")
    if arr.shape[1] != expected_features:
        raise ShapComputationError(f"Ожидалось {expected_features} признаков, получено {arr.shape[1]}.")
    if not np.isfinite(arr).all():
        raise ShapComputationError("SHAP-матрица содержит NaN или бесконечные значения.")
    return arr


def _normalize_base_values(base_values: np.ndarray, *, n_rows: int) -> np.ndarray:
    base = np.asarray(base_values, dtype=float).reshape(-1)
    if len(base) == 0:
        return np.zeros(n_rows, dtype=float)
    if len(base) == 1:
        return np.full(n_rows, float(base[0]), dtype=float)
    if len(base) != n_rows:
        return np.full(n_rows, float(base[0]), dtype=float)
    return base


def _validate_2d_array(value: Any, *, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2:
        raise ShapComputationError(f"{name} должен быть двумерной матрицей, получена форма {arr.shape}.")
    return arr


def _direction_text(mean_shap: float, *, direction_reliable: bool) -> str:
    if not direction_reliable:
        return (
            "направление влияния НЕУСТОЙЧИВОЕ — в разных эпидемических ситуациях этот фактор "
            "может как повышать, так и понижать прогноз. Важна сила влияния, а не знак"
        )
    if mean_shap > 0:
        return "при увеличении этого фактора прогноз заболеваемости устойчиво РАСТЁТ"
    return "при увеличении этого фактора прогноз заболеваемости устойчиво СНИЖАЕТСЯ"


def _transition_pattern(h1_group: str | None, h4_group: str | None) -> str | None:
    if h1_group is None or h4_group is None:
        return None
    short_memory = {"target_lags", "target_rolling", "target_dynamics"}
    longer_external = {"weather_lags", "weather_rolling", "calendar"}
    if h1_group in short_memory and h4_group in longer_external:
        return "lags_to_temp_seasonality"
    if h1_group == h4_group:
        return "stable_feature_regime"
    return "mixed_transition"


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ExplainabilityError(f"В {frame_name} отсутствуют колонки: {missing!r}.")


def _empty_global_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["horizon_weeks", "feature", "mean_abs_shap", "mean_shap", "mean_feature_value", "explainer", "rank"]
    )


def _empty_local_table() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "row_id",
            "feature",
            "shap_value",
            "feature_value",
            "horizon_weeks",
            "date",
            "y_true",
            "y_pred",
            "abs_err",
            "base_value",
            "base_plus_shap_sum",
            "abs_shap",
        ]
    )


def _empty_worst_table() -> pd.DataFrame:
    return pd.DataFrame(columns=["horizon_weeks", "row_id", "date", "y_true", "y_pred", "abs_err"])

