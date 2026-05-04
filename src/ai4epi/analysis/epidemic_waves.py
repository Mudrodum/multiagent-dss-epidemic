"""
Analysis of recent epidemic waves for ai4epi.

The module converts a weekly epidemiological time series into the structured
``epidemic_wave_comparison`` payload consumed by ``GlobalContext`` and section
narrators. It does not call LLMs, does not build bulletins and does not render
PDF reports. Its responsibility is limited to deterministic wave extraction,
FWHM-like width calculation and optional persistence of the resulting artefacts.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


JsonObject = dict[str, Any]
WaveStatus = Literal["complete", "left_censored", "right_censored", "both_censored"]


DEFAULT_SEASON_START_WEEK = 40
DEFAULT_WAVE_SMOOTH_WINDOW = 3
DEFAULT_N_LAST_SEASONS = 3


class EpidemicWaveError(ValueError):
    """Base error for epidemic-wave analysis."""


class EpidemicWaveConfigError(EpidemicWaveError):
    """Invalid epidemic-wave configuration."""


class EpidemicWaveInputError(EpidemicWaveError):
    """Input frame does not match the expected weekly-series contract."""


class StrictModel(BaseModel):
    """Base pydantic model with a closed contract."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        arbitrary_types_allowed=True,
        populate_by_name=True,
    )


class EpidemicWaveConfig(StrictModel):
    """
    Configuration of recent-wave extraction.

    The defaults match the wave-comparison block in the research notebook:
    the epidemic season starts at ISO week 40, the series is smoothed by a
    centered rolling mean with a three-week window, and the three latest
    seasons are compared.
    """

    target_col: str = Field(default="inc_per_10k", min_length=1)
    datetime_col: str = Field(default="datetime", min_length=1)
    season_start_week: int = Field(default=DEFAULT_SEASON_START_WEEK, ge=1, le=53)
    smooth_window: int = Field(default=DEFAULT_WAVE_SMOOTH_WINDOW, ge=1)
    n_last_seasons: int = Field(default=DEFAULT_N_LAST_SEASONS, ge=2)
    series_label_ru: str = Field(default="заболеваемость на 10 тыс. населения", min_length=1)
    comparison_mode: str = Field(default="three_last_seasons", min_length=1)
    require_regular_weekly_index: bool = False
    require_monday_week_start: bool = False
    round_digits: int = Field(default=3, ge=0, le=8)
    include_plot_points: bool = True

    @model_validator(mode="after")
    def validate_smoothing_window(self) -> "EpidemicWaveConfig":
        if self.smooth_window > 53:
            raise ValueError("smooth_window must not exceed one epidemic season.")
        return self


class EpidemicWaveOutputConfig(StrictModel):
    """Persistence settings for epidemic-wave artefacts."""

    output_dir: Path = Path("results_csv")
    bundle_filename: str = Field(default="epidemic_wave_comparison.json", min_length=1)
    waves_filename: str = Field(default="epidemic_waves.csv", min_length=1)
    points_filename: str = Field(default="epidemic_wave_points.csv", min_length=1)
    latest_vs_previous_filename: str = Field(default="epidemic_wave_latest_vs_previous.csv", min_length=1)
    save_bundle_json: bool = True
    save_waves_csv: bool = True
    save_points_csv: bool = True
    save_latest_vs_previous_csv: bool = True
    csv_encoding: str = Field(default="utf-8-sig", min_length=1)

    @field_validator(
        "bundle_filename",
        "waves_filename",
        "points_filename",
        "latest_vs_previous_filename",
    )
    @classmethod
    def validate_simple_filename(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or len(path.parts) != 1:
            raise ValueError("Output filenames must be simple relative filenames.")
        return value


class EpidemicWaveTables(StrictModel):
    """Tabular artefacts of wave analysis."""

    weekly: Any
    selected_weekly: Any
    waves: Any
    plot_points: Any
    latest_vs_previous: Any
    peak_ranking: Any
    width_ranking_complete: Any

    @field_validator(
        "weekly",
        "selected_weekly",
        "waves",
        "plot_points",
        "latest_vs_previous",
        "peak_ranking",
        "width_ranking_complete",
    )
    @classmethod
    def validate_dataframe(cls, value: Any) -> pd.DataFrame:
        if not isinstance(value, pd.DataFrame):
            raise TypeError("EpidemicWaveTables fields must be pandas.DataFrame objects.")
        return value


class EpidemicWaveRunResult(StrictModel):
    """Result of recent epidemic-wave comparison."""

    config: EpidemicWaveConfig
    bundle: JsonObject
    tables: EpidemicWaveTables
    artifacts: dict[str, Path] = Field(default_factory=dict)

    def to_context_payload(self) -> JsonObject:
        """Return a JSON-serializable payload for GlobalContext.epidemic_wave_comparison."""

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


def build_epidemic_wave_comparison_bundle(
    frame: pd.DataFrame,
    *,
    config: EpidemicWaveConfig | None = None,
) -> JsonObject:
    """Build the structured epidemic-wave comparison payload.

    Parameters
    ----------
    frame:
        Weekly time series with at least ``datetime_col`` and ``target_col``.
    config:
        Wave extraction configuration. Defaults match the source notebook.
    """

    result = run_epidemic_wave_analysis(frame, config=config, output=None)
    return result.to_context_payload()


def run_epidemic_wave_analysis(
    frame: pd.DataFrame,
    *,
    config: EpidemicWaveConfig | None = None,
    output: EpidemicWaveOutputConfig | None = None,
) -> EpidemicWaveRunResult:
    """Run deterministic recent-wave analysis and optionally save artefacts."""

    cfg = config or EpidemicWaveConfig()
    weekly = prepare_seasonal_weekly_frame(frame, config=cfg)
    selected_weekly = select_last_seasons(weekly, config=cfg)
    waves_payload = extract_wave_records(selected_weekly, config=cfg)

    if len(waves_payload) < cfg.n_last_seasons:
        raise EpidemicWaveInputError(
            "Not enough non-empty seasons after selection: "
            f"got {len(waves_payload)}, expected {cfg.n_last_seasons}."
        )

    bundle, tables = build_wave_bundle_and_tables(
        weekly=weekly,
        selected_weekly=selected_weekly,
        waves_payload=waves_payload,
        config=cfg,
    )
    artifacts = save_epidemic_wave_artifacts(EpidemicWaveRunResult(config=cfg, bundle=bundle, tables=tables), output) if output else {}
    return EpidemicWaveRunResult(config=cfg, bundle=bundle, tables=tables, artifacts=artifacts)


def prepare_seasonal_weekly_frame(frame: pd.DataFrame, *, config: EpidemicWaveConfig | None = None) -> pd.DataFrame:
    """Normalize a weekly series and add epidemic-season coordinates."""

    cfg = config or EpidemicWaveConfig()
    if frame is None or len(frame) == 0:
        raise EpidemicWaveInputError("A non-empty DataFrame is required for wave analysis.")
    _require_columns(frame, [cfg.datetime_col, cfg.target_col], frame_name="epidemic-wave input")

    data = frame[[cfg.datetime_col, cfg.target_col]].copy()
    data[cfg.datetime_col] = pd.to_datetime(data[cfg.datetime_col], errors="coerce")
    data[cfg.target_col] = pd.to_numeric(data[cfg.target_col], errors="coerce")
    data = data.dropna(subset=[cfg.datetime_col, cfg.target_col]).sort_values(cfg.datetime_col).reset_index(drop=True)

    if data.empty:
        raise EpidemicWaveInputError("No valid rows remain after datetime/target normalization.")
    if data[cfg.datetime_col].duplicated().any():
        duplicates = data.loc[data[cfg.datetime_col].duplicated(keep=False), cfg.datetime_col].head(10).tolist()
        raise EpidemicWaveInputError(f"Duplicate weekly dates are not allowed: {duplicates!r}.")
    if cfg.require_monday_week_start:
        non_monday = data.loc[data[cfg.datetime_col].dt.dayofweek != 0, cfg.datetime_col].head(10).tolist()
        if non_monday:
            raise EpidemicWaveInputError(f"Expected Monday week starts. Examples: {non_monday!r}.")
    if cfg.require_regular_weekly_index and len(data) > 1:
        diffs = data[cfg.datetime_col].diff().dropna().dt.days
        bad = diffs.loc[diffs != 7]
        if not bad.empty:
            idx = int(bad.index[0])
            raise EpidemicWaveInputError(
                "Expected a regular weekly index with a 7-day step. "
                f"Violation: {data.loc[idx - 1, cfg.datetime_col]} -> {data.loc[idx, cfg.datetime_col]}."
            )

    season_info = data[cfg.datetime_col].apply(lambda value: season_start_and_label(value, cfg.season_start_week))
    iso = data[cfg.datetime_col].dt.isocalendar()
    week_order = season_week_order(cfg.season_start_week)
    week_to_pos = {week: idx for idx, week in enumerate(week_order)}

    data = data.assign(
        season_start_year=[int(item[0]) for item in season_info],
        season_label=[str(item[1]) for item in season_info],
        iso_year=iso.year.astype(int),
        iso_week=iso.week.astype(int),
    )
    data["season_week_pos"] = data["iso_week"].map(week_to_pos).astype(float)

    weekly = (
        data.groupby(
            ["season_start_year", "season_label", "iso_week", "season_week_pos"],
            as_index=False,
        )
        .agg(
            value=(cfg.target_col, "mean"),
            date=(cfg.datetime_col, "min"),
        )
        .sort_values(["season_start_year", "season_week_pos"])
        .reset_index(drop=True)
    )
    weekly["smoothed_value"] = weekly.groupby("season_label")["value"].transform(
        lambda series: series.rolling(window=cfg.smooth_window, center=True, min_periods=1).mean()
    )
    return weekly


def select_last_seasons(weekly: pd.DataFrame, *, config: EpidemicWaveConfig | None = None) -> pd.DataFrame:
    """Select the last ``n_last_seasons`` epidemic seasons."""

    cfg = config or EpidemicWaveConfig()
    _require_columns(weekly, ["season_start_year", "season_label"], frame_name="seasonal weekly frame")
    season_years = sorted(int(item) for item in weekly["season_start_year"].dropna().unique().tolist())
    if len(season_years) < cfg.n_last_seasons:
        raise EpidemicWaveInputError(
            "Not enough epidemic seasons for comparison: "
            f"found {len(season_years)}, expected at least {cfg.n_last_seasons}."
        )
    selected_years = season_years[-cfg.n_last_seasons :]
    return weekly.loc[weekly["season_start_year"].isin(selected_years)].copy().reset_index(drop=True)


def extract_wave_records(selected_weekly: pd.DataFrame, *, config: EpidemicWaveConfig | None = None) -> list[JsonObject]:
    """Extract one wave record for each selected season."""

    cfg = config or EpidemicWaveConfig()
    _require_columns(
        selected_weekly,
        ["season_start_year", "season_label", "iso_week", "season_week_pos", "value", "date", "smoothed_value"],
        frame_name="selected seasonal weekly frame",
    )

    waves: list[JsonObject] = []
    season_years = sorted(int(item) for item in selected_weekly["season_start_year"].dropna().unique().tolist())
    for season_year in season_years:
        season_df = selected_weekly.loc[selected_weekly["season_start_year"].eq(season_year)].copy()
        season_df = season_df.sort_values("season_week_pos").reset_index(drop=True)
        if season_df.empty:
            continue
        waves.append(extract_one_wave_record(season_df, config=cfg))
    return sorted(waves, key=lambda item: int(item["season_start_year"]))


def extract_one_wave_record(season_df: pd.DataFrame, *, config: EpidemicWaveConfig | None = None) -> JsonObject:
    """Extract peak, FWHM-like width and burden metrics for one season."""

    cfg = config or EpidemicWaveConfig()
    xs = season_df["season_week_pos"].to_numpy(dtype=float)
    ys = season_df["smoothed_value"].to_numpy(dtype=float)
    ys_raw = season_df["value"].to_numpy(dtype=float)
    if len(xs) == 0 or len(ys) == 0:
        raise EpidemicWaveInputError("Season frame is empty.")
    if not np.isfinite(ys).any():
        raise EpidemicWaveInputError("Season smoothed values are all non-finite.")

    peak_idx = int(np.nanargmax(ys))
    peak_y = float(ys[peak_idx])
    half_height = float(peak_y * 0.5)

    left_x = nearest_half_height_crossing(xs, ys, peak_idx, half_height, side="left")
    right_x = nearest_half_height_crossing(xs, ys, peak_idx, half_height, side="right")
    wave_status = classify_wave_status(left_x, right_x)

    if left_x is not None and right_x is not None:
        fwhm_weeks = _round_float(right_x - left_x, cfg.round_digits)
        left_span = _round_float(xs[peak_idx] - left_x, cfg.round_digits)
        right_span = _round_float(right_x - xs[peak_idx], cfg.round_digits)
        asymmetry_ratio = _round_float(right_span / left_span, cfg.round_digits) if left_span and left_span > 0 else None
        fwhm_lower_bound = None
    else:
        fwhm_weeks = None
        left_span = _round_float(xs[peak_idx] - left_x, cfg.round_digits) if left_x is not None else None
        right_span = _round_float(right_x - xs[peak_idx], cfg.round_digits) if right_x is not None else None
        asymmetry_ratio = None
        fwhm_lower_bound = _round_float(xs[-1] - left_x, cfg.round_digits) if left_x is not None and right_x is None else None

    wave: JsonObject = {
        "season_start_year": int(season_df["season_start_year"].iloc[0]),
        "season_label": str(season_df["season_label"].iloc[0]),
        "wave_status": wave_status,
        "observed_until": pd.Timestamp(season_df["date"].max()).date().isoformat(),
        "n_observed_weeks": int(len(season_df)),
        "peak_week": int(season_df.loc[peak_idx, "iso_week"]),
        "peak_date": pd.Timestamp(season_df.loc[peak_idx, "date"]).date().isoformat(),
        "peak_value": _round_float(peak_y, cfg.round_digits),
        "half_height_value": _round_float(half_height, cfg.round_digits),
        "left_half_cross_pos": _round_float(left_x, cfg.round_digits) if left_x is not None else None,
        "right_half_cross_pos": _round_float(right_x, cfg.round_digits) if right_x is not None else None,
        "fwhm_weeks": fwhm_weeks,
        "fwhm_lower_bound_weeks": fwhm_lower_bound,
        "left_span_weeks": left_span,
        "right_span_weeks": right_span,
        "asymmetry_ratio": asymmetry_ratio,
        "season_area": _round_float(float(np.nansum(ys_raw)), cfg.round_digits),
        "rise_slope_half_to_peak": _round_float((peak_y - half_height) / left_span, cfg.round_digits) if left_span not in (None, 0) else None,
        "fall_slope_peak_to_half": _round_float((peak_y - half_height) / right_span, cfg.round_digits) if right_span not in (None, 0) else None,
        "secondary_peak_ratio": secondary_peak_ratio(ys, peak_idx, digits=cfg.round_digits),
    }
    if cfg.include_plot_points:
        wave["plot_points"] = [
            {
                "week": int(row["iso_week"]),
                "pos": float(row["season_week_pos"]),
                "date": pd.Timestamp(row["date"]).date().isoformat(),
                "value_raw": _round_float(float(row["value"]), cfg.round_digits),
                "value_smooth": _round_float(float(row["smoothed_value"]), cfg.round_digits),
            }
            for _, row in season_df.iterrows()
        ]
    return wave


def build_wave_bundle_and_tables(
    *,
    weekly: pd.DataFrame,
    selected_weekly: pd.DataFrame,
    waves_payload: Sequence[Mapping[str, Any]],
    config: EpidemicWaveConfig,
) -> tuple[JsonObject, EpidemicWaveTables]:
    """Build context payload and tabular artefacts from wave records."""

    waves = [dict(item) for item in waves_payload]
    if not waves:
        raise EpidemicWaveInputError("At least one wave record is required.")
    latest = waves[-1]
    previous = waves[:-1]

    peak_ranking = sorted(
        [{"season_label": wave["season_label"], "peak_value": wave["peak_value"]} for wave in waves],
        key=lambda item: float(item["peak_value"]),
        reverse=True,
    )
    width_ranking_complete = sorted(
        [
            {"season_label": wave["season_label"], "fwhm_weeks": wave["fwhm_weeks"]}
            for wave in waves
            if wave.get("fwhm_weeks") is not None
        ],
        key=lambda item: float(item["fwhm_weeks"]),
        reverse=True,
    )
    latest_vs_previous = build_latest_vs_previous(latest, previous, digits=config.round_digits)

    bundle: JsonObject = {
        "series_name": config.target_col,
        "series_label_ru": config.series_label_ru,
        "comparison_mode": config.comparison_mode,
        "season_definition": f"ISO weeks {config.season_start_week}..{config.season_start_week - 1}",
        "smoothing": {
            "method": "centered_rolling_mean",
            "window_weeks": int(config.smooth_window),
        },
        "width_definition": "FWHM-like on smoothed weekly curve",
        "season_labels": [str(wave["season_label"]) for wave in waves],
        "latest_wave_status": latest.get("wave_status"),
        "waves": waves,
        "peak_ranking": peak_ranking,
        "width_ranking_complete": width_ranking_complete,
        "latest_vs_previous": latest_vs_previous,
        "allowed_claims": [
            "peak height comparison",
            "peak timing comparison",
            "FWHM width comparison when complete",
            "right-censored width caveat for incomplete latest season",
            "asymmetry comparison when complete",
            "season burden comparison",
        ],
        "forbidden_claims": [
            "epidemic thresholds",
            "MEM intensity labels",
            "formal epidemic start/end declarations",
            "causal explanations absent from evidence",
        ],
    }

    tables = EpidemicWaveTables(
        weekly=weekly.copy(),
        selected_weekly=selected_weekly.copy(),
        waves=waves_to_frame(waves),
        plot_points=wave_points_to_frame(waves),
        latest_vs_previous=pd.DataFrame(latest_vs_previous),
        peak_ranking=pd.DataFrame(peak_ranking),
        width_ranking_complete=pd.DataFrame(width_ranking_complete),
    )
    return bundle, tables


def build_latest_vs_previous(latest: Mapping[str, Any], previous: Sequence[Mapping[str, Any]], *, digits: int = 3) -> list[JsonObject]:
    """Compare the latest wave with previous selected seasons."""

    comparisons: list[JsonObject] = []
    for prev in previous:
        latest_peak = _optional_float(latest.get("peak_value"))
        prev_peak = _optional_float(prev.get("peak_value"))
        latest_area = _optional_float(latest.get("season_area"))
        prev_area = _optional_float(prev.get("season_area"))
        latest_width = _optional_float(latest.get("fwhm_weeks"))
        prev_width = _optional_float(prev.get("fwhm_weeks"))
        comparisons.append(
            {
                "latest_season": latest.get("season_label"),
                "previous_season": prev.get("season_label"),
                "peak_diff_abs": _round_float(latest_peak - prev_peak, digits) if latest_peak is not None and prev_peak is not None else None,
                "peak_diff_pct": _round_float((latest_peak - prev_peak) / prev_peak * 100.0, 1) if latest_peak is not None and prev_peak not in (None, 0) else None,
                "peak_week_diff": int(latest["peak_week"] - prev["peak_week"]) if latest.get("peak_week") is not None and prev.get("peak_week") is not None else None,
                "season_area_diff_pct": _round_float((latest_area - prev_area) / prev_area * 100.0, 1) if latest_area is not None and prev_area not in (None, 0) else None,
                "width_diff_weeks": _round_float(latest_width - prev_width, digits) if latest_width is not None and prev_width is not None else None,
                "latest_width_censored": latest.get("fwhm_weeks") is None,
            }
        )
    return comparisons


def save_epidemic_wave_artifacts(
    result: EpidemicWaveRunResult,
    output: EpidemicWaveOutputConfig | None = None,
) -> dict[str, Path]:
    """Save wave-analysis JSON/CSV artefacts and return written paths."""

    cfg = output or EpidemicWaveOutputConfig()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, Path] = {}

    if cfg.save_bundle_json:
        path = out_dir / cfg.bundle_filename
        with path.open("w", encoding="utf-8") as stream:
            json.dump(result.to_context_payload(), stream, ensure_ascii=False, indent=2, default=_json_default)
        artifacts["bundle_json"] = path.resolve()
    if cfg.save_waves_csv:
        path = out_dir / cfg.waves_filename
        result.tables.waves.to_csv(path, index=False, encoding=cfg.csv_encoding)
        artifacts["waves_csv"] = path.resolve()
    if cfg.save_points_csv:
        path = out_dir / cfg.points_filename
        result.tables.plot_points.to_csv(path, index=False, encoding=cfg.csv_encoding)
        artifacts["points_csv"] = path.resolve()
    if cfg.save_latest_vs_previous_csv:
        path = out_dir / cfg.latest_vs_previous_filename
        result.tables.latest_vs_previous.to_csv(path, index=False, encoding=cfg.csv_encoding)
        artifacts["latest_vs_previous_csv"] = path.resolve()
    return artifacts


# ---------------------------------------------------------------------------
# Plotting helper kept separate from the core payload builder
# ---------------------------------------------------------------------------


def plot_epidemic_wave_comparison_bundle(
    bundle: Mapping[str, Any],
    out_path: str | Path,
    *,
    figsize: tuple[float, float] = (13.5, 5.8),
    dpi: int = 220,
) -> Path:
    """Render a PNG plot of the three-season wave comparison.

    Plotting is optional and is kept out of the context-building contract. The
    function imports matplotlib lazily so the numerical pipeline can run without
    a plotting backend when figures are not needed.
    """

    import matplotlib.pyplot as plt

    season_start_week = _infer_season_start_week_from_bundle(bundle) or DEFAULT_SEASON_START_WEEK
    week_order = season_week_order(season_start_week)
    xticks = np.arange(len(week_order))
    xticklabels = [str(week) for week in week_order]

    colors = ["#163A5F", "#4C956C", "#E07A5F", "#6D597A", "#355070"]
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#F7F9FB")

    for idx, wave in enumerate(bundle.get("waves", []) or []):
        points = pd.DataFrame(wave.get("plot_points", []) or [])
        if points.empty:
            continue
        points = points.sort_values("pos").reset_index(drop=True)
        x = points["pos"].to_numpy(dtype=float)
        y = points["value_smooth"].to_numpy(dtype=float)
        label = str(wave.get("season_label") or f"season_{idx + 1}")
        color = colors[idx % len(colors)]
        ax.plot(x, y, linewidth=2.4, marker="o", markersize=3.2, color=color, label=label)

        peak_idx = int(np.nanargmax(y))
        peak_pos = float(points.loc[peak_idx, "pos"])
        peak_value = _optional_float(wave.get("peak_value"))
        if peak_value is not None:
            ax.scatter([peak_pos], [peak_value], color=color, s=38, zorder=5)
            ax.annotate(
                f"пик {peak_value:.2f}",
                xy=(peak_pos, peak_value),
                xytext=(5, 8),
                textcoords="offset points",
                fontsize=8.5,
                color=color,
            )

        half_height = _optional_float(wave.get("half_height_value"))
        left_x = _optional_float(wave.get("left_half_cross_pos"))
        right_x = _optional_float(wave.get("right_half_cross_pos"))
        if half_height is not None and left_x is not None:
            end_x = right_x if right_x is not None else float(points["pos"].max())
            ax.hlines(half_height, left_x, end_x, colors=color, linestyles="--", linewidth=1.4, alpha=0.9)
            ax.scatter([left_x], [half_height], marker="x", s=48, color=color, zorder=6)
            if right_x is not None:
                ax.scatter([right_x], [half_height], marker="x", s=48, color=color, zorder=6)
                mid_x = (left_x + right_x) / 2.0
                fwhm = _optional_float(wave.get("fwhm_weeks"))
                if fwhm is not None:
                    ax.text(mid_x, half_height, f"FWHM={fwhm:.2f}", fontsize=8.2, color=color, ha="center", va="bottom")
            else:
                lower = _optional_float(wave.get("fwhm_lower_bound_weeks"))
                if lower is not None:
                    ax.text(end_x, half_height, f"FWHM ≥ {lower:.2f}", fontsize=8.2, color=color, ha="right", va="bottom")

    ax.set_title("Сравнение трёх последних эпидемических волн", fontsize=13, pad=10)
    ax.set_xlabel(f"Эпидемиологические недели сезона ({season_start_week} … {season_start_week - 1})", fontsize=11)
    ax.set_ylabel(str(bundle.get("series_label_ru") or "Значение"), fontsize=11)
    ax.set_xlim(0, len(week_order) - 1)
    ax.set_xticks(xticks[::2])
    ax.set_xticklabels([xticklabels[i] for i in range(0, len(week_order), 2)], fontsize=8)
    ax.grid(True, which="major", alpha=0.22, linewidth=0.8)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Deterministic helpers
# ---------------------------------------------------------------------------


def season_week_order(season_start_week: int = DEFAULT_SEASON_START_WEEK) -> list[int]:
    """Return ISO-week order inside an epidemic season."""

    if not 1 <= int(season_start_week) <= 53:
        raise EpidemicWaveConfigError("season_start_week must be in [1, 53].")
    return list(range(int(season_start_week), 54)) + list(range(1, int(season_start_week)))


def season_start_and_label(timestamp: Any, season_start_week: int = DEFAULT_SEASON_START_WEEK) -> tuple[int, str]:
    """Return epidemic-season start year and label for a timestamp."""

    ts = pd.Timestamp(timestamp)
    iso = ts.isocalendar()
    iso_year = int(iso.year)
    iso_week = int(iso.week)
    start_year = iso_year if iso_week >= int(season_start_week) else iso_year - 1
    return start_year, f"{start_year}-{start_year + 1}"


def interpolate_x_at_y(x0: float, y0: float, x1: float, y1: float, target_y: float) -> float:
    """Linear interpolation of x where the segment reaches target_y."""

    if abs(float(y1) - float(y0)) < 1e-12:
        return float(x0)
    return float(x0 + (target_y - y0) * (x1 - x0) / (y1 - y0))


def nearest_half_height_crossing(
    xs: np.ndarray,
    ys: np.ndarray,
    peak_idx: int,
    half_height: float,
    *,
    side: Literal["left", "right"],
) -> float | None:
    """Return nearest left/right half-height crossing around a peak."""

    if side not in {"left", "right"}:
        raise ValueError("side must be 'left' or 'right'.")
    if not 0 <= int(peak_idx) < len(xs):
        raise ValueError("peak_idx is out of bounds.")

    if side == "left":
        pairs = [(j, j + 1) for j in range(int(peak_idx) - 1, -1, -1)]
    else:
        pairs = [(j, j + 1) for j in range(int(peak_idx), len(xs) - 1)]

    for i0, i1 in pairs:
        y0 = float(ys[i0])
        y1 = float(ys[i1])
        if not (math.isfinite(y0) and math.isfinite(y1)):
            continue
        if (y0 - half_height) == 0:
            return float(xs[i0])
        if (y1 - half_height) == 0:
            return float(xs[i1])
        if (y0 - half_height) * (y1 - half_height) < 0:
            return interpolate_x_at_y(float(xs[i0]), y0, float(xs[i1]), y1, float(half_height))
    return None


def classify_wave_status(left_x: float | None, right_x: float | None) -> WaveStatus:
    """Classify whether FWHM boundaries are observed or censored."""

    if left_x is None and right_x is None:
        return "both_censored"
    if left_x is None:
        return "left_censored"
    if right_x is None:
        return "right_censored"
    return "complete"


def secondary_peak_ratio(ys: np.ndarray, peak_idx: int, *, digits: int = 3) -> float | None:
    """Return secondary-peak-to-main-peak ratio for a smoothed curve."""

    values = np.asarray(ys, dtype=float)
    if len(values) < 3:
        return None
    local_peaks: list[float] = []
    for i in range(1, len(values) - 1):
        if i == int(peak_idx):
            continue
        if values[i] >= values[i - 1] and values[i] >= values[i + 1]:
            local_peaks.append(float(values[i]))
    if local_peaks:
        second_value = max(local_peaks)
    else:
        candidates = [float(value) for i, value in enumerate(values) if i != int(peak_idx)]
        if not candidates:
            return None
        second_value = max(candidates)
    peak_value = float(values[int(peak_idx)])
    if peak_value <= 0 or not math.isfinite(peak_value):
        return None
    return _round_float(second_value / peak_value, digits)


def waves_to_frame(waves: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Convert wave records to a flat DataFrame without plot points."""

    rows: list[JsonObject] = []
    for wave in waves:
        row = {key: value for key, value in wave.items() if key != "plot_points"}
        rows.append(row)
    return pd.DataFrame(rows)


def wave_points_to_frame(waves: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Convert nested wave plot points to a flat DataFrame."""

    rows: list[JsonObject] = []
    for wave in waves:
        season_label = str(wave.get("season_label") or "")
        season_start_year = wave.get("season_start_year")
        for point in wave.get("plot_points", []) or []:
            rows.append(
                {
                    "season_start_year": season_start_year,
                    "season_label": season_label,
                    **dict(point),
                }
            )
    return pd.DataFrame(rows)


def load_epidemic_wave_bundle(path: str | Path) -> JsonObject:
    """Load a previously saved epidemic-wave context payload."""

    file_path = Path(path)
    with file_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise EpidemicWaveInputError("Epidemic-wave bundle JSON root must be an object.")
    return data


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], *, frame_name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise EpidemicWaveInputError(f"{frame_name} misses required columns: {missing!r}.")


def _round_float(value: float | int | np.floating[Any] | None, digits: int) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return round(numeric, int(digits))


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


def _infer_season_start_week_from_bundle(bundle: Mapping[str, Any]) -> int | None:
    raw = str(bundle.get("season_definition") or "")
    # Expected form: "ISO weeks 40..39".
    tokens = [token for token in raw.replace(".", " ").split() if token.isdigit()]
    if not tokens:
        return None
    try:
        week = int(tokens[0])
    except ValueError:
        return None
    return week if 1 <= week <= 53 else None

