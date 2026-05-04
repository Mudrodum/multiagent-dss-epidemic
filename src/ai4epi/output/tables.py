"""
Publication table generation for ai4epi bulletins.

This module builds deterministic table payloads from the same analysis context
that is used by the bulletin narrators and figure layer. It does not call LLMs,
does not alter numerical results, and does not depend on ReportLab.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


JsonObject = dict[str, Any]


@dataclass(frozen=True)
class PublicationTable:
    """A deterministic publication table attached to a bulletin section."""

    name: str
    title: str
    section_id: str
    placement: str
    order: int
    columns: tuple[str, ...]
    rows: tuple[tuple[str, ...], ...]
    note: str | None = None


def build_publication_tables_from_assets(raw_assets: Mapping[str, Any] | Sequence[Any] | None) -> tuple[PublicationTable, ...]:
    """Build standard notebook-compatible tables from bulletin assets.

    The function locates the run directory from figure asset paths and then reads
    ``analysis/context_relevant.json``. This keeps the public ``Bulletin`` schema
    unchanged while allowing renderers to recover deterministic tables from the
    analysis artefacts produced by the same run.
    """

    run_dir = _infer_run_dir_from_assets(raw_assets)
    if run_dir is None:
        return ()

    context_path = run_dir / "analysis" / "context_relevant.json"
    if not context_path.is_file():
        return ()

    try:
        context = json.loads(context_path.read_text(encoding="utf-8"))
    except Exception:
        return ()

    tables: list[PublicationTable] = []
    forecast = _forecast_table(context)
    if forecast is not None:
        tables.append(forecast)

    age = _age_group_table(context)
    if age is not None:
        tables.append(age)

    quality = _model_quality_table(context)
    if quality is not None:
        tables.append(quality)

    return tuple(sorted(tables, key=lambda table: (table.section_id, table.order, table.name)))


def tables_for_section(
    tables: Sequence[PublicationTable],
    section_id: str,
    *,
    placement: str | None = None,
) -> tuple[PublicationTable, ...]:
    """Return tables attached to one section, sorted by publication order."""

    selected = [table for table in tables if table.section_id == section_id]
    if placement is not None:
        selected = [table for table in selected if table.placement == placement]
    return tuple(sorted(selected, key=lambda table: (table.order, table.name)))


def table_to_markdown(table: PublicationTable, *, include_title: bool = True) -> str:
    """Render one publication table as Markdown."""

    lines: list[str] = []
    if include_title and table.title:
        lines.extend([f"**{table.title}**", ""])
    lines.append("| " + " | ".join(_escape_markdown_cell(col) for col in table.columns) + " |")
    lines.append("| " + " | ".join("---" for _ in table.columns) + " |")
    for row in table.rows:
        lines.append("| " + " | ".join(_escape_markdown_cell(cell) for cell in row) + " |")
    if table.note:
        lines.extend(["", table.note])
    return "\n".join(lines).strip()


def table_to_html(table: PublicationTable, *, include_title: bool = True) -> str:
    """Render one publication table as HTML."""

    from html import escape

    title = f"<p><strong>{escape(table.title)}</strong></p>\n" if include_title and table.title else ""
    header = "".join(f"<th>{escape(col)}</th>" for col in table.columns)
    body_rows = []
    for row in table.rows:
        body_rows.append("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>")
    note = f"\n<p><em>{escape(table.note)}</em></p>" if table.note else ""
    return f'{title}<table class="publication-table"><thead><tr>{header}</tr></thead><tbody>{"".join(body_rows)}</tbody></table>{note}'


def _infer_run_dir_from_assets(raw_assets: Mapping[str, Any] | Sequence[Any] | None) -> Path | None:
    for path in _iter_asset_paths(raw_assets):
        candidate = _run_dir_from_path(path)
        if candidate is not None:
            return candidate
    return None


def _iter_asset_paths(raw_assets: Mapping[str, Any] | Sequence[Any] | None) -> list[Path]:
    if raw_assets is None:
        return []
    values: Sequence[Any]
    if isinstance(raw_assets, Mapping):
        values = list(raw_assets.values())
    elif isinstance(raw_assets, Sequence) and not isinstance(raw_assets, (str, bytes, bytearray)):
        values = list(raw_assets)
    else:
        return []

    paths: list[Path] = []
    for value in values:
        path_value: Any = None
        if isinstance(value, Mapping):
            path_value = value.get("path")
        else:
            path_value = getattr(value, "path", None)
        if path_value:
            paths.append(Path(str(path_value)).expanduser())
    return paths


def _run_dir_from_path(path: Path) -> Path | None:
    # Typical path: <run>/rendered/figures/forecast_plot.png.
    candidates = [path]
    candidates.extend(path.parents)
    for candidate in candidates:
        if (candidate / "analysis" / "context_relevant.json").is_file():
            return candidate
    for candidate in path.parents:
        if candidate.name == "rendered":
            run_dir = candidate.parent
            if (run_dir / "analysis" / "context_relevant.json").is_file():
                return run_dir
    return None


def _forecast_table(context: Mapping[str, Any]) -> PublicationTable | None:
    forecast = context.get("forecast")
    if not isinstance(forecast, Mapping):
        return None
    horizons = forecast.get("horizons")
    if not isinstance(horizons, Sequence) or isinstance(horizons, (str, bytes, bytearray)):
        return None

    rows: list[tuple[str, ...]] = []
    for item in horizons:
        if not isinstance(item, Mapping):
            continue
        h = _safe_int(item.get("horizon_weeks") or item.get("horizon") or item.get("h"))
        date = str(item.get("target_date") or item.get("date") or "—")
        point = _fmt_number(item.get("point_forecast"), digits=3)
        lo = _fmt_number(item.get("q_lo"), digits=3)
        hi = _fmt_number(item.get("q_hi"), digits=3)
        width = _fmt_number(item.get("interval_width", item.get("pi_width")), digits=3)
        label = f"h={h}" if h is not None else "—"
        rows.append((label, date, point, f"[{lo}; {hi}]", width))

    if not rows:
        return None
    return PublicationTable(
        name="forecast_table",
        title="Таблица с прогнозом",
        section_id="forecast_risks",
        placement="before_section",
        order=39,
        columns=("Горизонт", "Дата", "Прогноз", "80% интервал", "Ширина интервала"),
        rows=tuple(rows),
    )


def _age_group_table(context: Mapping[str, Any]) -> PublicationTable | None:
    block = context.get("age_group_season")
    if not isinstance(block, Mapping):
        return None
    raw_rows = block.get("rows")
    if not isinstance(raw_rows, Sequence) or isinstance(raw_rows, (str, bytes, bytearray)):
        return None

    rows: list[tuple[str, ...]] = []
    for item in raw_rows:
        if not isinstance(item, Mapping):
            continue
        rows.append(
            (
                str(item.get("age_group_label") or "—"),
                _fmt_number(item.get("season_cases"), digits=2),
                _fmt_number(item.get("cumulative_incidence_pct"), digits=2),
                _fmt_int_or_dash(item.get("peak_week")),
                _fmt_number(item.get("peak_inc_per_10k"), digits=2),
                _fmt_number(item.get("mean_weekly_inc_per_10k"), digits=2),
                _fmt_width(item),
                _fmt_number(item.get("share_of_total_cases_pct"), digits=2),
            )
        )

    if not rows:
        return None
    note = str(block.get("peak_width_undefined_note") or block.get("fwhm_undefined_note") or "").strip() or None
    return PublicationTable(
        name="age_group_table",
        title="Сводная таблица по возрастным группам",
        section_id="age_group_season_overview",
        placement="before_section",
        order=32,
        columns=("Группа", "Случаи", "Накопл., %", "Пик, нед.", "Пик /10 тыс.", "Средняя /10 тыс.", "Ширина пика", "Доля, %"),
        rows=tuple(rows),
        note=note,
    )


def _model_quality_table(context: Mapping[str, Any]) -> PublicationTable | None:
    block = context.get("model_quality")
    if not isinstance(block, Mapping):
        return None
    metrics = block.get("metrics")
    if not isinstance(metrics, Sequence) or isinstance(metrics, (str, bytes, bytearray)):
        return None

    rows: list[tuple[str, ...]] = []
    for item in metrics:
        if not isinstance(item, Mapping):
            continue
        h = _safe_int(item.get("horizon_weeks") or item.get("horizon") or item.get("h"))
        rows.append(
            (
                f"h={h}" if h is not None else "—",
                _fmt_number(item.get("mae"), digits=3),
                _fmt_number(item.get("rmse"), digits=3),
                _fmt_number(item.get("r2"), digits=3),
                _fmt_int_or_dash(item.get("n_test")),
            )
        )

    if not rows:
        return None
    return PublicationTable(
        name="model_quality_table",
        title="Качество на тестовой выборке",
        section_id="model_quality",
        placement="after_section",
        order=61,
        columns=("Горизонт", "MAE", "RMSE", "R²", "n_test"),
        rows=tuple(rows),
    )


def _fmt_width(item: Mapping[str, Any]) -> str:
    defined = item.get("peak_width_defined", item.get("fwhm_defined", True))
    if defined is False:
        return "—"
    value = item.get("peak_width_weeks", item.get("fwhm_weeks"))
    return _fmt_number(value, digits=2)


def _fmt_int_or_dash(value: Any) -> str:
    number = _safe_int(value)
    return str(number) if number is not None else "—"


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _fmt_number(value: Any, *, digits: int) -> str:
    try:
        if value is None or value == "":
            return "—"
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:.{digits}f}"


def _escape_markdown_cell(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").strip()
