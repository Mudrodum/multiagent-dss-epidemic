"""
Seasonal age-group analysis for ai4epi.

The module converts an epidemiological dataframe with ARI/SARS age-group
columns into the structured ``age_group_season`` payload consumed by
``GlobalContext`` and section narrators. It does not call LLMs, does not build
forecasts and does not render bulletins. Its responsibility is limited to
season selection, age-group incidence calculations, peak extraction and
50%-prominence peak-width estimation on a smoothed weekly curve.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4epi.analysis.epidemic_waves import (
    DEFAULT_SEASON_START_WEEK,
    DEFAULT_WAVE_SMOOTH_WINDOW,
    season_start_and_label,
    season_week_order,
)

try:  # pragma: no cover - checked by integration environment.
    from scipy.signal import peak_prominences, peak_widths
except ImportError:  # pragma: no cover
    peak_prominences = None  # type: ignore[assignment]
    peak_widths = None  # type: ignore[assignment]


JsonObject = dict[str, Any]


AGE_GROUP_SOURCE_RENAME: dict[str, str] = {
    "sars_total_cases": "ari_total_cases",
    "sars_cases_age_group_0": "ari_cases_age_group_0",
    "sars_cases_age_group_1": "ari_cases_age_group_1",
    "sars_cases_age_group_2": "ari_cases_age_group_2",
    "sars_cases_age_group_4": "ari_cases_age_group_4",
    "sars_cases_age_group_5": "ari_cases_age_group_5",
}

AGE_GROUP_CODE_GLOSSARY: dict[str, str] = {
    "total": "всё население (агрегат, не возрастная группа)",
    "age_0_2": "дети 0–2 года",
    "age_3_6": "дети 3–6 лет",
    "age_7_14": "дети 7–14 лет",
    "age_15_64": "лица 15–64 года",
    "age_65_plus": "лица 65 лет и старше",
}

AGE_GROUP_WIDTH_UNDEFINED_NOTE = (
    "Примечание. Знак «—» означает, что ширина главного пика на уровне 50% prominence "
    "не определяется по сглаженному ряду текущего сезона."
)


class AgeGroupSeasonError(ValueError):
    """Base error for age-group seasonal analysis."""


class AgeGroupSeasonConfigError(AgeGroupSeasonError):
    """Invalid age-group seasonal configuration."""


class AgeGroupSeasonInputError(AgeGroupSeasonError):
    """Input dataframe does not match the expected age-group contract."""


class StrictModel(BaseModel):
    """Base pydantic model with a closed contract."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )


class AgeGroupSpec(StrictModel):
    """One age group in the seasonal burden table."""

    age_group_code: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    age_group_label: str = Field(min_length=1)
    cases_col: str = Field(min_length=1)
    population_col: str = Field(min_length=1)
    share_base: bool = True
    is_total: bool = False


DEFAULT_AGE_GROUP_SPECS: tuple[AgeGroupSpec, ...] = (
    AgeGroupSpec(
        age_group_code="total",
        age_group_label="Все население",
        cases_col="ari_total_cases",
        population_col="total_population",
        share_base=False,
        is_total=True,
    ),
    AgeGroupSpec(
        age_group_code="age_0_2",
        age_group_label="0–2 года",
        cases_col="ari_cases_age_group_0",
        population_col="population_age_group_0",
    ),
    AgeGroupSpec(
        age_group_code="age_3_6",
        age_group_label="3–6 лет",
        cases_col="ari_cases_age_group_1",
        population_col="population_age_group_1",
    ),
    AgeGroupSpec(
        age_group_code="age_7_14",
        age_group_label="7–14 лет",
        cases_col="ari_cases_age_group_2",
        population_col="population_age_group_2",
    ),
    AgeGroupSpec(
        age_group_code="age_15_64",
        age_group_label="15–64 года",
        cases_col="ari_cases_age_group_4",
        population_col="population_age_group_4",
    ),
    AgeGroupSpec(
        age_group_code="age_65_plus",
        age_group_label="65+ лет",
        cases_col="ari_cases_age_group_5",
        population_col="population_age_group_5",
    ),
)


class AgeGroupSeasonConfig(StrictModel):
    """
    Configuration of the current-season age-group analysis.

    Defaults match the age-group block in the research notebook: the epidemic
    season starts at ISO week 40, the weekly incidence curves are smoothed by a
    centered three-week rolling mean, and peak width is measured at 50% of peak
    prominence on that smoothed curve.
    """

    datetime_col: str = Field(default="datetime", min_length=1)
    season_start_week: int = Field(default=DEFAULT_SEASON_START_WEEK, ge=1, le=53)
    smooth_window: int = Field(default=DEFAULT_WAVE_SMOOTH_WINDOW, ge=1)
    age_groups: tuple[AgeGroupSpec, ...] = Field(default_factory=lambda: DEFAULT_AGE_GROUP_SPECS)
    metric_label_ru: str = Field(default="зарегистрированная заболеваемость ОРВИ по возрастным группам", min_length=1)
    round_digits: int = Field(default=2, ge=0, le=8)
    width_round_digits: int = Field(default=2, ge=0, le=8)
    include_plot_points: bool = True
    require_unique_dates: bool = False
    require_monday_week_start: bool = False
    require_regular_weekly_index: bool = False

    @field_validator("age_groups")
    @classmethod
    def validate_age_group_specs(cls, value: tuple[AgeGroupSpec, ...]) -> tuple[AgeGroupSpec, ...]:
        if not value:
            raise ValueError("age_groups must contain at least one group.")
        codes = [spec.age_group_code for spec in value]
        duplicates = sorted({code for code in codes if codes.count(code) > 1})
        if duplicates:
            raise ValueError(f"Duplicated age_group_code values: {duplicates!r}.")
        return value

    @model_validator(mode="after")
    def validate_smooth_window(self) -> "AgeGroupSeasonConfig":
        if self.smooth_window > 53:
            raise ValueError("smooth_window must not exceed one epidemic season.")
        return self


class AgeGroupSeasonOutputConfig(StrictModel):
    """Persistence settings for age-group seasonal artefacts."""

    output_dir: Path = Path("results_csv")
    bundle_filename: str = Field(default="age_group_season_table.json", min_length=1)
    rows_filename: str = Field(default="age_group_season_rows.csv", min_length=1)
    points_filename: str = Field(default="age_group_season_points.csv", min_length=1)
    save_bundle_json: bool = True
    save_rows_csv: bool = True
    save_points_csv: bool = True
    csv_encoding: str = Field(default="utf-8-sig", min_length=1)

    @field_validator("bundle_filename", "rows_filename", "points_filename")
    @classmethod
    def validate_simple_filename(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or len(path.parts) != 1:
            raise ValueError("Output filenames must be simple relative filenames.")
        return value


class AgePeakGeometry(StrictModel):
    """Peak-width geometry for one age-group curve."""

    peak_width_weeks: float | None = None
    left_crossing_pos: float | None = None
    right_crossing_pos: float | None = None
    width_level: float | None = None
    peak_value: float | None = None
    peak_prominence: float | None = None
    base_level: float | None = None
    peak_width_defined: bool = False
    peak_width_reason: str | None = None

    def to_public_dict(self) -> JsonObject:
        return self.model_dump(mode="json", exclude_none=False)


class AgeGroupSeasonTables(StrictModel):
    """Tabular artefacts of age-group seasonal analysis."""

    current_season: Any
    rows: Any
    points: Any

    @field_validator("current_season", "rows", "points")
    @classmethod
    def validate_dataframe(cls, value: Any) -> pd.DataFrame:
        if not isinstance(value, pd.DataFrame):
            raise TypeError("AgeGroupSeasonTables fields must be pandas.DataFrame objects.")
        return value


class AgeGroupSeasonRunResult(StrictModel):
    """Result of current-season age-group analysis."""

    config: AgeGroupSeasonConfig
    bundle: JsonObject
    tables: AgeGroupSeasonTables
    artifacts: dict[str, Path] = Field(default_factory=dict)

    def to_context_payload(self) -> JsonObject:
        """Return a JSON-serializable payload for GlobalContext.age_group_season."""

        return json.loads(json.dumps(self.bundle, ensure_ascii=False, default=_json_default))

    def to_public_dict(self) -> JsonObject:
        """Return a compact JSON-serializable representation of the run."""

        return {
            "config": self.config.model_dump(mode="json"),
            "bundle": self.to_context_payload(),
            "artifacts": {key: str(value) for key, value in self.artifacts.items()},
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_age_group_season_bundle(
    frame: pd.DataFrame,
    *,
    config: AgeGroupSeasonConfig | None = None,
) -> JsonObject:
    """Build the structured current-season age-group payload."""

    result = run_age_group_season_analysis(frame, config=config, output=None)
    return result.to_context_payload()


def run_age_group_season_analysis(
    frame: pd.DataFrame,
    *,
    config: AgeGroupSeasonConfig | None = None,
    output: AgeGroupSeasonOutputConfig | None = None,
) -> AgeGroupSeasonRunResult:
    """Run deterministic current-season age-group analysis and optionally save artefacts."""

    cfg = config or AgeGroupSeasonConfig()
    current_season = prepare_age_group_current_season_frame(frame, config=cfg)
    rows_payload, points = extract_age_group_rows(current_season, config=cfg)
    bundle, tables = build_age_group_bundle_and_tables(
        current_season=current_season,
        rows_payload=rows_payload,
        points=points,
        config=cfg,
    )
    result = AgeGroupSeasonRunResult(config=cfg, bundle=bundle, tables=tables)
    artifacts = save_age_group_season_artifacts(result, output) if output else {}
    return result.model_copy(update={"artifacts": artifacts})


def prepare_age_group_current_season_frame(
    frame: pd.DataFrame,
    *,
    config: AgeGroupSeasonConfig | None = None,
) -> pd.DataFrame:
    """Normalize an age-group dataframe and select the latest epidemic season."""

    cfg = config or AgeGroupSeasonConfig()
    if frame is None or len(frame) == 0:
        raise AgeGroupSeasonInputError("A non-empty DataFrame is required for age-group seasonal analysis.")

    data = frame.copy()
    rename_map = {
        old: new
        for old, new in AGE_GROUP_SOURCE_RENAME.items()
        if old in data.columns and new not in data.columns
    }
    if rename_map:
        data = data.rename(columns=rename_map)

    required_columns = [cfg.datetime_col]
    for spec in cfg.age_groups:
        required_columns.extend([spec.cases_col, spec.population_col])
    _require_columns(data, required_columns, frame_name="age-group input")

    data = data[required_columns].copy()
    data[cfg.datetime_col] = pd.to_datetime(data[cfg.datetime_col], errors="coerce")
    for spec in cfg.age_groups:
        data[spec.cases_col] = pd.to_numeric(data[spec.cases_col], errors="coerce")
        data[spec.population_col] = pd.to_numeric(data[spec.population_col], errors="coerce")

    data = data.dropna(subset=[cfg.datetime_col]).sort_values(cfg.datetime_col).reset_index(drop=True)
    if data.empty:
        raise AgeGroupSeasonInputError("No valid rows remain after datetime normalization.")
    if cfg.require_unique_dates and data[cfg.datetime_col].duplicated().any():
        duplicates = data.loc[data[cfg.datetime_col].duplicated(keep=False), cfg.datetime_col].head(10).tolist()
        raise AgeGroupSeasonInputError(f"Duplicate weekly dates are not allowed: {duplicates!r}.")
    if cfg.require_monday_week_start:
        non_monday = data.loc[data[cfg.datetime_col].dt.dayofweek != 0, cfg.datetime_col].head(10).tolist()
        if non_monday:
            raise AgeGroupSeasonInputError(f"Expected Monday week starts. Examples: {non_monday!r}.")
    if cfg.require_regular_weekly_index and len(data) > 1:
        diffs = data[cfg.datetime_col].diff().dropna().dt.days
        bad = diffs.loc[diffs != 7]
        if not bad.empty:
            idx = int(bad.index[0])
            raise AgeGroupSeasonInputError(
                "Expected a regular weekly index with a 7-day step. "
                f"Violation: {data.loc[idx - 1, cfg.datetime_col]} -> {data.loc[idx, cfg.datetime_col]}."
            )

    season_info = data[cfg.datetime_col].apply(lambda value: season_start_and_label(value, cfg.season_start_week))
    iso = data[cfg.datetime_col].dt.isocalendar()
    week_order = season_week_order(cfg.season_start_week)
    week_to_pos = {week: idx + 1 for idx, week in enumerate(week_order)}

    data = data.assign(
        season_start_year=[int(item[0]) for item in season_info],
        season_label=[str(item[1]) for item in season_info],
        iso_year=iso.year.astype(int),
        iso_week=iso.week.astype(int),
    )
    data["season_week_pos"] = data["iso_week"].map(week_to_pos).astype(int)

    latest_season_start = int(data["season_start_year"].max())
    current = data.loc[data["season_start_year"].eq(latest_season_start)].copy()
    current = current.sort_values(cfg.datetime_col).reset_index(drop=True)
    if current.empty:
        raise AgeGroupSeasonInputError("Latest epidemic season contains no rows.")

    current.attrs["season_start_year"] = latest_season_start
    current.attrs["season_label"] = str(current["season_label"].iloc[0])
    return current


def extract_age_group_rows(
    current_season: pd.DataFrame,
    *,
    config: AgeGroupSeasonConfig | None = None,
) -> tuple[list[JsonObject], pd.DataFrame]:
    """Extract summary rows and long-form weekly points for all configured age groups."""

    cfg = config or AgeGroupSeasonConfig()
    _require_scipy_signal()
    _require_columns(current_season, [cfg.datetime_col, "iso_week", "season_week_pos"], frame_name="current-season frame")

    total_cases_for_shares = 0.0
    for spec in cfg.age_groups:
        if spec.share_base:
            series = pd.to_numeric(current_season[spec.cases_col], errors="coerce").fillna(0.0)
            total_cases_for_shares += float(series.sum())

    rows: list[JsonObject] = []
    point_rows: list[JsonObject] = []

    for spec in cfg.age_groups:
        tmp = current_season[[cfg.datetime_col, "iso_week", "season_week_pos", spec.cases_col, spec.population_col]].copy()
        tmp[spec.cases_col] = pd.to_numeric(tmp[spec.cases_col], errors="coerce")
        tmp[spec.population_col] = pd.to_numeric(tmp[spec.population_col], errors="coerce")
        tmp = tmp.dropna(subset=[spec.cases_col, spec.population_col]).sort_values(cfg.datetime_col).reset_index(drop=True)
        tmp = tmp.loc[tmp[spec.population_col] > 0].copy().reset_index(drop=True)
        if tmp.empty:
            continue

        tmp["inc_per_10k"] = tmp[spec.cases_col] / tmp[spec.population_col] * 10000.0
        tmp["inc_per_10k_smooth"] = tmp["inc_per_10k"].rolling(
            window=cfg.smooth_window,
            center=True,
            min_periods=1,
        ).mean()

        y_smooth = tmp["inc_per_10k_smooth"].to_numpy(dtype=float)
        if not np.isfinite(y_smooth).any():
            continue
        peak_pos_idx = int(np.nanargmax(y_smooth))
        peak_row = tmp.iloc[peak_pos_idx]
        pop_ref = latest_nonnull(tmp[spec.population_col])
        season_cases = float(tmp[spec.cases_col].fillna(0.0).sum())
        cumulative_incidence_pct = (
            _round_float(season_cases / pop_ref * 100.0, cfg.round_digits)
            if pop_ref is not None and pop_ref > 0 else None
        )
        if cumulative_incidence_pct is None:
            continue

        mean_weekly_inc_per_10k = _round_float(float(tmp["inc_per_10k"].mean()), cfg.round_digits)
        peak_inc_per_10k = _round_float(float(peak_row["inc_per_10k_smooth"]), cfg.round_digits)
        if peak_inc_per_10k is None:
            continue

        geometry = build_age_peak_width_geometry(
            tmp["season_week_pos"].to_numpy(dtype=float),
            y_smooth,
            round_digits=cfg.width_round_digits,
        )
        geom = geometry.to_public_dict()

        share_pct = (
            100.0
            if spec.age_group_code == "total"
            else (
                _round_float(season_cases / total_cases_for_shares * 100.0, cfg.round_digits)
                if total_cases_for_shares > 0 else None
            )
        )

        row: JsonObject = {
            "age_group_code": spec.age_group_code,
            "age_group_label": spec.age_group_label,
            "is_total_row": bool(spec.is_total),
            "season_cases": _round_float(season_cases, cfg.round_digits),
            "cumulative_incidence_pct": cumulative_incidence_pct,
            "peak_week": int(peak_row["iso_week"]),
            "peak_date": pd.Timestamp(peak_row[cfg.datetime_col]).date().isoformat(),
            "peak_season_position": int(peak_row["season_week_pos"]),
            "peak_inc_per_10k": peak_inc_per_10k,
            "mean_weekly_inc_per_10k": mean_weekly_inc_per_10k,
            "peak_width_weeks": geom["peak_width_weeks"],
            "peak_width_defined": geom["peak_width_defined"],
            "peak_width_reason": geom["peak_width_reason"],
            "left_crossing_pos": geom["left_crossing_pos"],
            "right_crossing_pos": geom["right_crossing_pos"],
            "width_level": geom["width_level"],
            "peak_prominence": geom["peak_prominence"],
            "base_level": geom["base_level"],
            "fwhm_weeks": geom["peak_width_weeks"],
            "fwhm_defined": geom["peak_width_defined"],
            "fwhm_reason": geom["peak_width_reason"],
            "half_height": geom["width_level"],
            "share_of_total_cases_pct": share_pct,
        }
        rows.append(row)

        if cfg.include_plot_points:
            for _, point in tmp.iterrows():
                point_rows.append(
                    {
                        "age_group_code": spec.age_group_code,
                        "age_group_label": spec.age_group_label,
                        "date": pd.Timestamp(point[cfg.datetime_col]).date().isoformat(),
                        "iso_week": int(point["iso_week"]),
                        "season_week_pos": int(point["season_week_pos"]),
                        "cases": _round_float(float(point[spec.cases_col]), cfg.round_digits),
                        "population": _round_float(float(point[spec.population_col]), cfg.round_digits),
                        "inc_per_10k": _round_float(float(point["inc_per_10k"]), cfg.round_digits),
                        "inc_per_10k_smooth": _round_float(float(point["inc_per_10k_smooth"]), cfg.round_digits),
                    }
                )

    if not rows:
        raise AgeGroupSeasonInputError("No valid age-group rows could be extracted from the current season.")
    points = pd.DataFrame(point_rows)
    return rows, points


def build_age_group_bundle_and_tables(
    *,
    current_season: pd.DataFrame,
    rows_payload: Sequence[Mapping[str, Any]],
    points: pd.DataFrame,
    config: AgeGroupSeasonConfig,
) -> tuple[JsonObject, AgeGroupSeasonTables]:
    """Build context payload and tabular artefacts from age-group summary rows."""

    rows = [dict(item) for item in rows_payload]
    age_rows = [row for row in rows if not bool(row.get("is_total_row"))]
    if not age_rows:
        raise AgeGroupSeasonInputError("At least one non-total age group row is required.")

    highest_peak_row = max(age_rows, key=lambda row: _score_number(row.get("peak_inc_per_10k"), missing=-math.inf))
    largest_cumulative_row = max(age_rows, key=lambda row: _score_number(row.get("cumulative_incidence_pct"), missing=-math.inf))
    smallest_share_row = min(age_rows, key=lambda row: _score_number(row.get("share_of_total_cases_pct"), missing=math.inf))
    width_candidates = [row for row in age_rows if _optional_float(row.get("peak_width_weeks")) is not None]
    widest_wave_row = max(width_candidates, key=lambda row: float(row["peak_width_weeks"])) if width_candidates else None

    max_peak_pos = max(int(row["peak_season_position"]) for row in age_rows)
    latest_peak_candidates = [row for row in age_rows if int(row["peak_season_position"]) == max_peak_pos]
    latest_peak_row = latest_peak_candidates[0] if len(latest_peak_candidates) == 1 else None

    derived_findings = {
        "highest_peak_group": {
            "age_group_code": highest_peak_row["age_group_code"],
            "age_group_label": highest_peak_row["age_group_label"],
            "peak_inc_per_10k": highest_peak_row["peak_inc_per_10k"],
            "peak_week": highest_peak_row["peak_week"],
            "peak_date": highest_peak_row["peak_date"],
        },
        "largest_cumulative_incidence_group": {
            "age_group_code": largest_cumulative_row["age_group_code"],
            "age_group_label": largest_cumulative_row["age_group_label"],
            "cumulative_incidence_pct": largest_cumulative_row["cumulative_incidence_pct"],
        },
        "widest_wave_group": None if widest_wave_row is None else {
            "age_group_code": widest_wave_row["age_group_code"],
            "age_group_label": widest_wave_row["age_group_label"],
            "peak_width_weeks": widest_wave_row["peak_width_weeks"],
        },
        "latest_peak_group": None if latest_peak_row is None else {
            "age_group_code": latest_peak_row["age_group_code"],
            "age_group_label": latest_peak_row["age_group_label"],
            "peak_week": latest_peak_row["peak_week"],
            "peak_date": latest_peak_row["peak_date"],
            "peak_season_position": latest_peak_row["peak_season_position"],
        },
        "smallest_share_group": {
            "age_group_code": smallest_share_row["age_group_code"],
            "age_group_label": smallest_share_row["age_group_label"],
            "share_of_total_cases_pct": smallest_share_row["share_of_total_cases_pct"],
        },
    }

    semantic = {
        "age_group_codes": AGE_GROUP_CODE_GLOSSARY,
        "is_total_row_semantics": "Строка «Все население» — агрегат по популяции и не считается возрастной группой в сравнительных формулировках narrator.",
        "highest_peak_group_code": highest_peak_row["age_group_code"],
        "largest_cumulative_incidence_group_code": largest_cumulative_row["age_group_code"],
        "widest_wave_group_code": None if widest_wave_row is None else widest_wave_row["age_group_code"],
        "latest_peak_group_code": None if latest_peak_row is None else latest_peak_row["age_group_code"],
    }

    season_start_year = int(current_season.attrs.get("season_start_year"))
    season_label = str(current_season.attrs.get("season_label"))
    bundle: JsonObject = {
        "season_start_year": season_start_year,
        "season_label": season_label,
        "season_definition": f"эпидемический сезон с {config.season_start_week}-й недели по {config.season_start_week - 1}-ю неделю следующего года",
        "metric_label_ru": config.metric_label_ru,
        "source_semantics": "Возрастной раздел относится к зарегистрированным случаям ARI/ОРВИ и не тождественен модельной цели inc_per_10k.",
        "width_definition": "Ширина главного пика по сглаженному недельному ряду на уровне 50% prominence",
        "smooth_window_weeks": int(config.smooth_window),
        "peak_width_undefined_note": AGE_GROUP_WIDTH_UNDEFINED_NOTE,
        "fwhm_undefined_note": AGE_GROUP_WIDTH_UNDEFINED_NOTE,
        "comparison_scope_note": "Сравнительные выводы о ширине и времени пика относятся только к возрастным группам и не включают строку «Все население».",
        "rows": rows,
        "derived_findings": derived_findings,
        "semantic": semantic,
    }

    tables = AgeGroupSeasonTables(
        current_season=current_season.copy(),
        rows=pd.DataFrame(rows),
        points=points.copy(),
    )
    return bundle, tables


def save_age_group_season_artifacts(
    result: AgeGroupSeasonRunResult,
    output: AgeGroupSeasonOutputConfig | None = None,
) -> dict[str, Path]:
    """Save age-group seasonal JSON/CSV artefacts and return written paths."""

    cfg = output or AgeGroupSeasonOutputConfig()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    if cfg.save_bundle_json:
        path = out_dir / cfg.bundle_filename
        with path.open("w", encoding="utf-8") as stream:
            json.dump(result.to_context_payload(), stream, ensure_ascii=False, indent=2, default=_json_default)
        artifacts["bundle_json"] = path.resolve()
    if cfg.save_rows_csv:
        path = out_dir / cfg.rows_filename
        result.tables.rows.to_csv(path, index=False, encoding=cfg.csv_encoding)
        artifacts["rows_csv"] = path.resolve()
    if cfg.save_points_csv:
        path = out_dir / cfg.points_filename
        result.tables.points.to_csv(path, index=False, encoding=cfg.csv_encoding)
        artifacts["points_csv"] = path.resolve()
    return artifacts


# ---------------------------------------------------------------------------
# Optional plotting helpers for later rendering/PDF layers
# ---------------------------------------------------------------------------


def plot_age_group_season_graphs(
    frame: pd.DataFrame,
    bundle: Mapping[str, Any],
    output_dir: str | Path,
    *,
    config: AgeGroupSeasonConfig | None = None,
    overlay_filename: str = "age_group_season_overlay_plot.png",
    panel_filename: str = "age_group_season_panels_plot.png",
    dpi: int = 180,
) -> tuple[Path, Path]:
    """Render overlay and panel plots for the current age-group season.

    The plotting helper is intentionally outside the core payload builder. It
    is suitable for a future rendering/PDF layer and returns file paths only.
    """

    cfg = config or AgeGroupSeasonConfig()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("plot_age_group_season_graphs requires matplotlib.") from exc

    current = prepare_age_group_current_season_frame(frame, config=cfg)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    row_by_code = {str(row.get("age_group_code")): dict(row) for row in bundle.get("rows", []) or []}
    season_label = str(bundle.get("season_label") or current.attrs.get("season_label") or "")
    max_pos = int(current["season_week_pos"].max())
    tick_positions, tick_labels = age_group_tick_positions_labels(max_pos, season_start_week=cfg.season_start_week)

    overlay_path = out_dir / overlay_filename
    fig, ax = plt.subplots(figsize=(12, 5.8))
    for spec in cfg.age_groups:
        row = row_by_code.get(spec.age_group_code)
        if row is None:
            continue
        tmp = _age_group_curve_frame(current, spec, cfg)
        if tmp.empty:
            continue
        linewidth = 2.5 if spec.age_group_code == "total" else 1.8
        ax.plot(tmp["season_week_pos"], tmp["inc_per_10k_smooth"], linewidth=linewidth, label=spec.age_group_label)
        ax.scatter([row["peak_season_position"]], [row["peak_inc_per_10k"]], s=26, zorder=3)
    ax.set_xlim(1, max_pos + 0.5)
    ax.set_title(f"Возрастная динамика зарегистрированной заболеваемости ОРВИ\nэпидемический сезон {season_label}")
    ax.set_xlabel("Календарные недели сезона")
    ax.set_ylabel("На 10 тыс. населения")
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(overlay_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    panel_path = out_dir / panel_filename
    n_groups = len(cfg.age_groups)
    n_cols = 2
    n_rows = int(math.ceil(n_groups / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 3.6 * n_rows), sharex=True)
    axes_array = np.asarray(axes).reshape(-1)

    for ax, spec in zip(axes_array, cfg.age_groups):
        row = row_by_code.get(spec.age_group_code)
        if row is None:
            ax.axis("off")
            continue
        tmp = _age_group_curve_frame(current, spec, cfg)
        if tmp.empty:
            ax.axis("off")
            continue
        ax.plot(tmp["season_week_pos"], tmp["inc_per_10k"], alpha=0.35, linewidth=1.0)
        ax.plot(tmp["season_week_pos"], tmp["inc_per_10k_smooth"], linewidth=2.0)
        ax.scatter([row["peak_season_position"]], [row["peak_inc_per_10k"]], s=24, zorder=3)
        if row.get("width_level") is not None:
            ax.axhline(y=row["width_level"], linestyle="--", linewidth=0.9, alpha=0.65)
        if row.get("left_crossing_pos") is not None and row.get("width_level") is not None:
            ax.scatter([row["left_crossing_pos"]], [row["width_level"]], s=18, zorder=4)
        if row.get("right_crossing_pos") is not None and row.get("width_level") is not None:
            ax.scatter([row["right_crossing_pos"]], [row["width_level"]], s=18, zorder=4)
        if row.get("left_crossing_pos") is not None and row.get("right_crossing_pos") is not None and row.get("width_level") is not None:
            ax.hlines(
                y=row["width_level"],
                xmin=row["left_crossing_pos"],
                xmax=row["right_crossing_pos"],
                linewidth=1.6,
                alpha=0.9,
            )
        ax.set_title(spec.age_group_label, fontsize=9)
        ax.set_xlim(1, max_pos + 0.5)
        ax.grid(True, alpha=0.2)
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("Кал. недели", fontsize=7)
        ax.set_ylabel("/10 тыс.", fontsize=7)

    for ax in axes_array[n_groups:]:
        ax.axis("off")

    fig.suptitle(
        f"Возрастные профили зарегистрированной заболеваемости ОРВИ\nэпидемический сезон {season_label}",
        fontsize=12,
        y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(panel_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return overlay_path.resolve(), panel_path.resolve()


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def build_age_peak_width_geometry(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    round_digits: int = 2,
) -> AgePeakGeometry:
    """Compute width at 50% prominence for one smoothed age-group curve."""

    _require_scipy_signal()
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    if len(x) == 0 or len(y) == 0 or np.all(~np.isfinite(y)):
        return AgePeakGeometry(peak_width_reason="empty_or_nonfinite")
    if len(x) != len(y):
        raise AgeGroupSeasonInputError(f"xs and ys must have equal length: {len(x)} != {len(y)}.")

    peak_idx = int(np.nanargmax(y))
    peak_value = float(y[peak_idx])
    if not math.isfinite(peak_value) or peak_value <= 0:
        return AgePeakGeometry(
            peak_value=_round_float(peak_value, round_digits),
            peak_width_reason="nonpositive_peak",
        )

    try:
        prominences, left_bases, right_bases = peak_prominences(y, [peak_idx])  # type: ignore[misc]
        prominence = float(prominences[0])
    except Exception as exc:
        return AgePeakGeometry(
            peak_value=_round_float(peak_value, round_digits),
            peak_width_reason=f"prominence_error:{type(exc).__name__}",
        )

    if not math.isfinite(prominence) or prominence <= 0:
        return AgePeakGeometry(
            peak_value=_round_float(peak_value, round_digits),
            peak_prominence=_round_float(prominence, round_digits) if math.isfinite(prominence) else None,
            peak_width_reason="nonpositive_prominence",
        )

    try:
        widths, width_heights, left_ips, right_ips = peak_widths(  # type: ignore[misc]
            y,
            [peak_idx],
            rel_height=0.5,
            prominence_data=(prominences, left_bases, right_bases),
        )
        del widths
        left_ip = float(left_ips[0])
        right_ip = float(right_ips[0])
        width_level = float(width_heights[0])
    except Exception as exc:
        return AgePeakGeometry(
            peak_value=_round_float(peak_value, round_digits),
            peak_prominence=_round_float(prominence, round_digits),
            base_level=_round_float(peak_value - prominence, round_digits),
            peak_width_reason=f"width_error:{type(exc).__name__}",
        )

    left_x = interp_x_at_fractional_index(x, left_ip)
    right_x = interp_x_at_fractional_index(x, right_ip)
    width_weeks = right_x - left_x
    if not math.isfinite(width_weeks) or width_weeks <= 0:
        return AgePeakGeometry(
            peak_width_weeks=None,
            left_crossing_pos=_round_float(left_x, 3),
            right_crossing_pos=_round_float(right_x, 3),
            width_level=_round_float(width_level, round_digits),
            peak_value=_round_float(peak_value, round_digits),
            peak_prominence=_round_float(prominence, round_digits),
            base_level=_round_float(peak_value - prominence, round_digits),
            peak_width_defined=False,
            peak_width_reason="nonpositive_width",
        )

    return AgePeakGeometry(
        peak_width_weeks=_round_float(width_weeks, round_digits),
        left_crossing_pos=_round_float(left_x, 3),
        right_crossing_pos=_round_float(right_x, 3),
        width_level=_round_float(width_level, round_digits),
        peak_value=_round_float(peak_value, round_digits),
        peak_prominence=_round_float(prominence, round_digits),
        base_level=_round_float(peak_value - prominence, round_digits),
        peak_width_defined=True,
        peak_width_reason=None,
    )


def latest_nonnull(series: pd.Series) -> float | None:
    """Return the latest finite numeric value in a series."""

    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return None
    value = float(s.iloc[-1])
    return value if math.isfinite(value) else None


def interp_x_at_fractional_index(xs: np.ndarray, idx_float: float) -> float:
    """Interpolate x-coordinate at a fractional array index."""

    axis = np.arange(len(xs), dtype=float)
    return float(np.interp(float(idx_float), axis, np.asarray(xs, dtype=float)))


def age_group_tick_positions_labels(max_pos: int, *, season_start_week: int = DEFAULT_SEASON_START_WEEK) -> tuple[list[int], list[str]]:
    """Return tick positions and ISO-week labels for age-group season plots."""

    week_order = season_week_order(season_start_week)
    positions = list(range(1, int(max_pos) + 1, 4))
    labels = [str(week_order[position - 1]) for position in positions]
    return positions, labels


def load_age_group_season_bundle(path: str | Path) -> JsonObject:
    """Load a previously saved age-group season context payload."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise AgeGroupSeasonInputError("Age-group bundle JSON root must be an object.")
    return data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _age_group_curve_frame(current_season: pd.DataFrame, spec: AgeGroupSpec, cfg: AgeGroupSeasonConfig) -> pd.DataFrame:
    tmp = current_season[["season_week_pos", "iso_week", spec.cases_col, spec.population_col]].copy()
    tmp[spec.cases_col] = pd.to_numeric(tmp[spec.cases_col], errors="coerce")
    tmp[spec.population_col] = pd.to_numeric(tmp[spec.population_col], errors="coerce")
    tmp = tmp.dropna().reset_index(drop=True)
    tmp = tmp.loc[tmp[spec.population_col] > 0].copy().reset_index(drop=True)
    if tmp.empty:
        return tmp
    tmp["inc_per_10k"] = tmp[spec.cases_col] / tmp[spec.population_col] * 10000.0
    tmp["inc_per_10k_smooth"] = tmp["inc_per_10k"].rolling(window=cfg.smooth_window, center=True, min_periods=1).mean()
    return tmp


def _require_scipy_signal() -> None:
    if peak_prominences is None or peak_widths is None:
        raise AgeGroupSeasonConfigError(
            "age_group_season peak-width calculation requires scipy.signal. "
            "Install scipy or disable this module in the calling pipeline."
        )


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise AgeGroupSeasonInputError(f"{frame_name} misses required columns: {missing!r}.")


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _score_number(value: Any, *, missing: float) -> float:
    out = _optional_float(value)
    return missing if out is None else out


def _round_float(value: float | int | np.floating[Any] | None, digits: int) -> float | None:
    if value is None:
        return None
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(x):
        return None
    return round(x, int(digits))


def _json_default(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")

