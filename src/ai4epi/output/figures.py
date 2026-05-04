"""
Publication figure generation for ai4epi bulletins.

This module builds deterministic PNG assets from already computed analysis
outputs and ``GlobalContext`` payloads. It does not call LLMs and does not alter
analysis results.

The plotting code intentionally follows the reference notebook output:

1. forecast plot with two panels: RU and EN;
2. three-wave comparison plot with peak labels and FWHM markers;
3. age-group season overlay plot;
4. age-group panel plot with raw/smoothed curves and width markers.

The renderer/PDF layer decides where these assets are placed in the final
bulletin. This module is responsible only for making the images themselves.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:  # Package import.
    from ai4epi.core.context import GlobalContext
except ImportError:  # pragma: no cover
    from ai4epi.core.context import GlobalContext  # type: ignore[no-redef]


JsonObject = dict[str, Any]

SEASON_START_WEEK = 40
WAVE_SMOOTH_WINDOW = 3
DATA_COLOR = "#1f77b4"
FORECAST_COLOR = "#ff7f0e"
PI_ALPHA = 0.20

AGE_GROUP_ORDER = (
    "total",
    "0_2",
    "3_6",
    "7_14",
    "15_64",
    "65_plus",
)

AGE_LABEL_TO_CODE = {
    "Все население": "total",
    "Все население ": "total",
    "0–2 года": "0_2",
    "0-2 года": "0_2",
    "3–6 лет": "3_6",
    "3-6 лет": "3_6",
    "7–14 лет": "7_14",
    "7-14 лет": "7_14",
    "15–64 года": "15_64",
    "15-64 года": "15_64",
    "65+ лет": "65_plus",
    "65 лет и старше": "65_plus",
}


class FigureGenerationError(RuntimeError):
    """Base error for publication figure generation."""


def build_publication_figure_assets(
    *,
    context: GlobalContext | Mapping[str, Any],
    analysis_artifacts: Mapping[str, str] | None,
    output_dir: str | Path,
    dpi: int = 180,
) -> dict[str, JsonObject]:
    """Generate notebook-compatible bulletin figure assets."""

    ctx = _context_payload(context)
    artifacts = {str(k): str(v) for k, v in dict(analysis_artifacts or {}).items()}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    assets: dict[str, JsonObject] = {}

    forecast_path = out / "forecast_plot.png"
    if _plot_forecast_like_notebook(ctx, artifacts, forecast_path, dpi=dpi):
        assets["forecast_plot"] = _image_asset(
            name="forecast_plot",
            path=forecast_path,
            title="Рис. 1. Результаты моделирования заболеваемости гриппом с начала сезона по текущую неделю.",
            description=(
                "Исторические значения, центральная прогнозная траектория и "
                "80% прогнозные интервалы на ближайшие четыре недели."
            ),
            section_id="current_situation",
            order=10,
            placement="before_section",
        )

    wave_path = out / "epidemic_wave_comparison_plot.png"
    if _plot_epidemic_wave_comparison_like_notebook(ctx, wave_path, dpi=dpi):
        assets["epidemic_wave_comparison_plot"] = _image_asset(
            name="epidemic_wave_comparison_plot",
            path=wave_path,
            title="Рис. 2. Сравнение трёх последних эпидемических волн по показателю заболеваемости на 10 тыс. населения.",
            description="Сглаженные сезонные кривые по показателю заболеваемости на 10 тыс. населения; отмечены положения пиков и FWHM-маркеры.",
            section_id="epidemic_wave_comparison",
            order=20,
            placement="before_section",
        )

    age_points_path = _artifact_path(artifacts, "age_group_season.points_csv")
    age_rows_path = _artifact_path(artifacts, "age_group_season.rows_csv")
    age_block = ctx.get("age_group_season") or {}

    age_overlay_path = out / "age_group_season_overlay_plot.png"
    if age_points_path and age_rows_path and _plot_age_group_season_overlay_like_notebook(
        points_path=age_points_path,
        rows_path=age_rows_path,
        context_block=age_block,
        path=age_overlay_path,
        dpi=dpi,
    ):
        assets["age_group_season_overlay_plot"] = _image_asset(
            name="age_group_season_overlay_plot",
            path=age_overlay_path,
            title="Рис. 3. Сглаженная возрастная динамика зарегистрированной заболеваемости ОРВИ в текущем эпидемическом сезоне.",
            description="Сглаженные недельные кривые текущего эпидемического сезона по возрастным группам.",
            section_id="age_group_season_overview",
            order=30,
            placement="before_section",
        )

    age_panels_path = out / "age_group_season_panels_plot.png"
    if age_points_path and age_rows_path and _plot_age_group_season_panels_like_notebook(
        points_path=age_points_path,
        rows_path=age_rows_path,
        context_block=age_block,
        path=age_panels_path,
        dpi=dpi,
    ):
        assets["age_group_season_panels_plot"] = _image_asset(
            name="age_group_season_panels_plot",
            path=age_panels_path,
            title="Рис. 4. Профили возрастных групп: сырые и сглаженные ряды, точки пика и маркеры ширины главного пика на уровне 50% prominence.",
            description="Профили возрастных групп: сырые и сглаженные ряды, точки пика и маркеры ширины главного пика на уровне 50% prominence.",
            section_id="age_group_season_overview",
            order=31,
            placement="before_section",
        )

    return assets


def _context_payload(context: GlobalContext | Mapping[str, Any]) -> JsonObject:
    if isinstance(context, GlobalContext):
        return context.to_public_dict()
    return dict(context)


def _artifact_path(artifacts: Mapping[str, str], key: str) -> Path | None:
    value = artifacts.get(key)
    if not value:
        return None
    path = Path(value)
    return path if path.is_file() else None


def _image_asset(
    *,
    name: str,
    path: Path,
    title: str,
    description: str,
    section_id: str,
    order: int,
    placement: str,
) -> JsonObject:
    return {
        "name": name,
        "kind": "image",
        "path": str(path.resolve()),
        "title": title,
        "description": description,
        "metadata": {
            "section_id": section_id,
            "order": order,
            "placement": placement,
        },
    }


def _prepare_matplotlib() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        raise FigureGenerationError(f"matplotlib is required for figure generation: {exc}") from exc

    plt.rcParams.update(
        {
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _iso_week_labels(dates: pd.DatetimeIndex) -> list[str]:
    weeks = pd.DatetimeIndex(dates).isocalendar().week.to_numpy()
    return [f"{int(week):02d}" for week in weeks]


def _plot_forecast_like_notebook(
    ctx: Mapping[str, Any],
    artifacts: Mapping[str, str],
    path: Path,
    *,
    dpi: int,
    window_back: int = 24,
    tick_step: int = 2,
) -> bool:
    forecast = ctx.get("forecast") or {}
    horizons = list(forecast.get("horizons") or [])
    if not horizons:
        return False

    hist_path = _artifact_path(artifacts, "forecasting.history_plus_forecast")
    if hist_path is None:
        return False

    table = pd.read_csv(hist_path)
    if not {"date", "value"}.issubset(table.columns):
        return False

    table = table.copy()
    table["date"] = pd.to_datetime(table["date"])
    table["value"] = pd.to_numeric(table["value"], errors="coerce")

    if "row_type" in table.columns:
        history = table[table["row_type"].astype(str).eq("history")].copy()
        forecast_rows = table[table["row_type"].astype(str).eq("forecast")].copy()
    else:
        history = table.iloc[:-len(horizons)].copy()
        forecast_rows = table.iloc[-len(horizons):].copy()

    history = history.dropna(subset=["date", "value"]).sort_values("date").tail(window_back)
    forecast_rows = forecast_rows.dropna(subset=["date", "value"]).sort_values("date")
    if history.empty or forecast_rows.empty:
        return False

    y_hat = forecast_rows["value"].to_numpy(dtype=float)
    future_dates = pd.DatetimeIndex(forecast_rows["date"])
    origin_date = pd.Timestamp(history["date"].max())

    y_lo: np.ndarray | None = None
    y_hi: np.ndarray | None = None
    if {"q_lo", "q_hi"}.issubset(forecast_rows.columns):
        y_lo = pd.to_numeric(forecast_rows["q_lo"], errors="coerce").to_numpy(dtype=float)
        y_hi = pd.to_numeric(forecast_rows["q_hi"], errors="coerce").to_numpy(dtype=float)
        y_lo, y_hi = np.minimum(y_lo, y_hi), np.maximum(y_lo, y_hi)
        if not (np.isfinite(y_lo).all() and np.isfinite(y_hi).all()):
            y_lo = None
            y_hi = None

    hist_dt = pd.DatetimeIndex(history["date"])
    all_dates = pd.DatetimeIndex(list(hist_dt) + list(future_dates))
    x_all = np.arange(len(all_dates))
    x_hist = x_all[: len(history)]
    x_fut = x_all[len(history) :]
    labels = _iso_week_labels(all_dates)
    h = len(y_hat)

    plt = _prepare_matplotlib()
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=True)

    def draw(ax: Any, lang: str) -> None:
        ax.plot(
            x_hist,
            history["value"].to_numpy(dtype=float),
            color=DATA_COLOR,
            marker="o",
            linewidth=2,
            markersize=5,
            label="Данные" if lang == "ru" else "Data",
        )

        y_last = float(history["value"].iloc[-1])
        x_bridge = np.concatenate(([x_hist[-1]], x_fut))
        y_bridge = np.concatenate(([y_last], y_hat))
        ax.plot(
            x_bridge,
            y_bridge,
            color=FORECAST_COLOR,
            linestyle="--",
            marker="o",
            markevery=range(1, len(x_bridge)),
            linewidth=2,
            markersize=5,
            label="Прогноз" if lang == "ru" else "Forecast",
        )

        if y_lo is not None and y_hi is not None:
            ax.fill_between(
                x_fut,
                y_lo[:h],
                y_hi[:h],
                color=FORECAST_COLOR,
                alpha=PI_ALPHA,
                label="Предиктивный интервал 80%" if lang == "ru" else "Prediction interval 80%",
            )

        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
        if lang == "ru":
            ax.set_title(f"Прогноз на {h} недели вперёд")
            ax.set_xlabel("Недели")
            ax.set_ylabel("Заболеваемость гриппом\nна 10 тыс населения")
        else:
            ax.set_title(f"{h}-week forecast ahead")
            ax.set_xlabel("Weeks")
            ax.set_ylabel("Influenza morbidity per 10000 population")

    draw(axes[0], "ru")
    draw(axes[1], "en")

    tick_idx = np.arange(0, len(x_all), tick_step)
    for ax in axes:
        ax.set_xticks(tick_idx)
        ax.set_xticklabels([labels[i] for i in tick_idx])
        ax.tick_params(axis="x", labelbottom=True)

    _save_figure(fig, path, dpi=dpi)
    return True


def _season_week_order(season_start_week: int = SEASON_START_WEEK) -> list[int]:
    return list(range(season_start_week, 54)) + list(range(1, season_start_week))


def _plot_epidemic_wave_comparison_like_notebook(
    ctx: Mapping[str, Any],
    path: Path,
    *,
    dpi: int,
    figsize: tuple[float, float] = (13.5, 5.8),
) -> bool:
    bundle = ctx.get("epidemic_wave_comparison") or {}
    waves = list(bundle.get("waves") or [])
    if not waves:
        return False

    week_order = _season_week_order(SEASON_START_WEEK)
    xticks = np.arange(len(week_order))
    xticklabels = [str(week) for week in week_order]
    colors = ["#163A5F", "#4C956C", "#E07A5F"]

    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_facecolor("#F7F9FB")

    plotted = False
    for idx, wave in enumerate(waves):
        points = pd.DataFrame(wave.get("plot_points") or [])
        if points.empty or not {"pos", "value_smooth"}.issubset(points.columns):
            continue
        points = points.sort_values("pos").reset_index(drop=True)
        x = points["pos"].to_numpy(dtype=float)
        y = points["value_smooth"].to_numpy(dtype=float)
        label = str(wave.get("season_label") or f"season-{idx + 1}")
        color = colors[idx % len(colors)]

        ax.plot(x, y, linewidth=2.4, marker="o", markersize=3.2, color=color, label=label)
        plotted = True

        if len(y) == 0:
            continue
        peak_idx = int(np.nanargmax(y))
        peak_pos = float(points.loc[peak_idx, "pos"])
        peak_value = _safe_float(wave.get("peak_value"), default=float(y[peak_idx]))
        ax.scatter([peak_pos], [peak_value], color=color, s=38, zorder=5)
        ax.annotate(
            f"пик {peak_value:.2f}",
            xy=(peak_pos, peak_value),
            xytext=(5, 8),
            textcoords="offset points",
            fontsize=8.5,
            color=color,
        )

        half_h = _safe_float(wave.get("half_height_value"), default=math.nan)
        left_x = wave.get("left_half_cross_pos")
        right_x = wave.get("right_half_cross_pos")
        if math.isfinite(half_h) and left_x is not None:
            left = float(left_x)
            end_x = float(right_x) if right_x is not None else float(points["pos"].max())
            ax.hlines(half_h, left, end_x, colors=color, linestyles="--", linewidth=1.4, alpha=0.9)
            ax.scatter([left], [half_h], marker="x", s=48, color=color, zorder=6)
            if right_x is not None:
                right = float(right_x)
                ax.scatter([right], [half_h], marker="x", s=48, color=color, zorder=6)
                mid_x = (left + right) / 2.0
                fwhm = wave.get("fwhm_weeks")
                if fwhm is not None:
                    ax.text(
                        mid_x,
                        half_h,
                        f"FWHM={float(fwhm):.2f}",
                        fontsize=8.2,
                        color=color,
                        ha="center",
                        va="bottom",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.7),
                    )
            else:
                fwhm_lb = wave.get("fwhm_lower_bound_weeks")
                if fwhm_lb is not None:
                    ax.text(
                        end_x,
                        half_h,
                        f"FWHM ≥ {float(fwhm_lb):.2f}",
                        fontsize=8.2,
                        color=color,
                        ha="right",
                        va="bottom",
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec=color, alpha=0.7),
                    )

    if not plotted:
        return False

    ax.set_title("Сравнение трёх последних эпидемических волн", fontsize=13, pad=10)
    ax.set_xlabel("Эпидемиологические недели сезона (40 … 39)", fontsize=11)
    ax.set_ylabel("Заболеваемость на 10 тыс. населения", fontsize=11)
    ax.set_xlim(0, len(week_order) - 1)
    ax.set_xticks(xticks[::2])
    ax.set_xticklabels([xticklabels[i] for i in range(0, len(week_order), 2)], fontsize=8)
    ax.grid(True, which="major", alpha=0.22, linewidth=0.8)
    ax.legend(loc="upper right", frameon=True)
    _save_figure(fig, path, dpi=220)
    return True


def _plot_age_group_season_overlay_like_notebook(
    *,
    points_path: Path,
    rows_path: Path,
    context_block: Mapping[str, Any],
    path: Path,
    dpi: int,
) -> bool:
    points, rows = _load_age_points_and_rows(points_path, rows_path)
    if points.empty or rows.empty:
        return False

    season_label = str(context_block.get("season_label") or "")
    max_pos = int(pd.to_numeric(points["season_week_pos"], errors="coerce").max())
    positions, labels = _age_group_tick_positions_labels(max_pos=max_pos)
    row_by_code = {str(row["age_group_code"]): row for _, row in rows.iterrows()}

    plt = _prepare_matplotlib()
    fig, ax = plt.subplots(figsize=(12, 5.8))

    plotted = False
    for code in _age_group_codes_in_plot_order(points, rows):
        row = row_by_code.get(code)
        group = points[points["age_group_code"].astype(str).eq(code)].copy()
        if group.empty:
            continue
        group = group.sort_values("season_week_pos")
        label = _age_group_label_for_code(code, group, row)
        y_col = "inc_per_10k_smooth" if "inc_per_10k_smooth" in group.columns else "inc_per_10k"
        if y_col not in group.columns:
            continue
        ax.plot(
            group["season_week_pos"],
            pd.to_numeric(group[y_col], errors="coerce"),
            linewidth=2.5 if code == "total" else 1.8,
            label=label,
        )
        if row is not None and _not_null(row.get("peak_season_position")) and _not_null(row.get("peak_inc_per_10k")):
            ax.scatter([float(row["peak_season_position"])], [float(row["peak_inc_per_10k"])], s=26, zorder=3)
        plotted = True

    if not plotted:
        return False

    ax.set_xlim(1, max_pos + 0.5)
    title = "Возрастная динамика зарегистрированной заболеваемости ОРВИ"
    if season_label:
        title += f"\nэпидемический сезон {season_label}"
    ax.set_title(title)
    ax.set_xlabel("Календарные недели сезона")
    ax.set_ylabel("На 10 тыс. населения")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.grid(True, alpha=0.25)
    ax.legend(ncol=3, fontsize=8)
    _save_figure(fig, path, dpi=dpi)
    return True


def _plot_age_group_season_panels_like_notebook(
    *,
    points_path: Path,
    rows_path: Path,
    context_block: Mapping[str, Any],
    path: Path,
    dpi: int,
) -> bool:
    points, rows = _load_age_points_and_rows(points_path, rows_path)
    if points.empty or rows.empty:
        return False

    season_label = str(context_block.get("season_label") or "")
    max_pos = int(pd.to_numeric(points["season_week_pos"], errors="coerce").max())
    positions, labels = _age_group_tick_positions_labels(max_pos=max_pos)
    row_by_code = {str(row["age_group_code"]): row for _, row in rows.iterrows()}
    codes = _age_group_codes_in_plot_order(points, rows)

    plt = _prepare_matplotlib()
    fig, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
    flat_axes = axes.flatten()

    plotted_any = False
    for ax, code in zip(flat_axes, codes):
        row = row_by_code.get(code)
        group = points[points["age_group_code"].astype(str).eq(code)].copy()
        if row is None or group.empty:
            ax.axis("off")
            continue

        group = group.sort_values("season_week_pos")
        label = _age_group_label_for_code(code, group, row)
        raw_col = "inc_per_10k" if "inc_per_10k" in group.columns else "inc_per_10k_smooth"
        smooth_col = "inc_per_10k_smooth" if "inc_per_10k_smooth" in group.columns else raw_col
        if raw_col not in group.columns or smooth_col not in group.columns:
            ax.axis("off")
            continue

        ax.plot(group["season_week_pos"], pd.to_numeric(group[raw_col], errors="coerce"), alpha=0.35, linewidth=1.0)
        ax.plot(group["season_week_pos"], pd.to_numeric(group[smooth_col], errors="coerce"), linewidth=2.0)

        if _not_null(row.get("peak_season_position")) and _not_null(row.get("peak_inc_per_10k")):
            ax.scatter([float(row["peak_season_position"])], [float(row["peak_inc_per_10k"])], s=24, zorder=3)

        width_level = row.get("width_level") if _not_null(row.get("width_level")) else row.get("half_height")
        if _not_null(width_level):
            ax.axhline(y=float(width_level), linestyle="--", linewidth=0.9, alpha=0.65)

        left = row.get("left_crossing_pos")
        right = row.get("right_crossing_pos")
        if _not_null(left) and _not_null(width_level):
            ax.scatter([float(left)], [float(width_level)], s=18, zorder=4)
        if _not_null(right) and _not_null(width_level):
            ax.scatter([float(right)], [float(width_level)], s=18, zorder=4)
        if _not_null(left) and _not_null(right) and _not_null(width_level):
            ax.hlines(y=float(width_level), xmin=float(left), xmax=float(right), linewidth=1.6, alpha=0.9)

        ax.set_title(label, fontsize=9)
        ax.set_xlim(1, max_pos + 0.5)
        ax.grid(True, alpha=0.2)
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("Кал. недели", fontsize=7)
        ax.set_ylabel("/10 тыс.", fontsize=7)
        plotted_any = True

    for ax in flat_axes[len(codes) :]:
        ax.axis("off")

    if not plotted_any:
        return False

    title = "Возрастные профили зарегистрированной заболеваемости ОРВИ"
    if season_label:
        title += f"\nэпидемический сезон {season_label}"
    fig.suptitle(title, fontsize=12, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    _save_figure(fig, path, dpi=dpi)
    return True


def _load_age_points_and_rows(points_path: Path, rows_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    points = pd.read_csv(points_path)
    rows = pd.read_csv(rows_path)

    if "age_group_code" not in points.columns:
        if "age_group_label" in points.columns:
            points["age_group_code"] = points["age_group_label"].map(lambda x: AGE_LABEL_TO_CODE.get(str(x), str(x)))
        else:
            return pd.DataFrame(), pd.DataFrame()
    if "age_group_code" not in rows.columns:
        if "age_group_label" in rows.columns:
            rows["age_group_code"] = rows["age_group_label"].map(lambda x: AGE_LABEL_TO_CODE.get(str(x), str(x)))
        else:
            return pd.DataFrame(), pd.DataFrame()

    if "season_week_pos" not in points.columns:
        return pd.DataFrame(), pd.DataFrame()
    if "age_group_label" not in points.columns:
        points["age_group_label"] = points["age_group_code"].astype(str)
    if "age_group_label" not in rows.columns:
        rows["age_group_label"] = rows["age_group_code"].astype(str)

    points["season_week_pos"] = pd.to_numeric(points["season_week_pos"], errors="coerce")
    points = points.dropna(subset=["season_week_pos"]).copy()
    return points, rows


def _age_group_codes_in_plot_order(points: pd.DataFrame, rows: pd.DataFrame) -> list[str]:
    existing = set(points["age_group_code"].astype(str)) | set(rows["age_group_code"].astype(str))
    ordered = [code for code in AGE_GROUP_ORDER if code in existing]
    extras = sorted(code for code in existing if code not in set(AGE_GROUP_ORDER))
    return ordered + extras


def _age_group_label_for_code(code: str, group: pd.DataFrame, row: pd.Series | None) -> str:
    if row is not None and "age_group_label" in row and _not_null(row.get("age_group_label")):
        return str(row["age_group_label"])
    if "age_group_label" in group.columns and not group.empty:
        return str(group["age_group_label"].iloc[0])
    return code


def _age_group_tick_positions_labels(max_pos: int, season_start_week: int = SEASON_START_WEEK) -> tuple[list[int], list[str]]:
    week_order = _season_week_order(season_start_week)
    positions = list(range(1, int(max_pos) + 1, 4))
    labels = [str(week_order[position - 1]) for position in positions]
    return positions, labels


def _safe_float(value: Any, *, default: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _not_null(value: Any) -> bool:
    if value is None:
        return False
    try:
        return bool(pd.notna(value))
    except Exception:
        return True


def _save_figure(fig: Any, path: Path, *, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    try:
        import matplotlib.pyplot as plt

        plt.close(fig)
    except Exception:  # pragma: no cover
        pass
