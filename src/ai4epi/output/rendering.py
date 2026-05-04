"""
Публикационный rendering-слой для бюллетеней ai4epi.

Модуль преобразует уже собранный ``Bulletin`` в Markdown и HTML. Он не вызывает
LLM, не редактирует текст, не выполняет runtime/evaluation проверки и не строит
PDF. Его задача — дать устойчивое промежуточное представление, на которое затем
можно опереться в ``pdf.py``, веб-интерфейсе или документации.
"""

from __future__ import annotations

from datetime import datetime
from html import escape as html_escape
import json
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

try:  # Пакетный импорт.
    from ai4epi.generation.bulletin import Bulletin, BulletinSection, flatten_text, load_bulletin
    from ai4epi.output.tables import (
        PublicationTable,
        build_publication_tables_from_assets,
        table_to_html,
        table_to_markdown,
        tables_for_section,
    )
except ImportError:  # pragma: no cover - поддержка запуска как набора отдельных файлов.
    from ai4epi.generation.bulletin import Bulletin, BulletinSection, flatten_text, load_bulletin  # type: ignore[no-redef]
    try:
        from ai4epi.output.tables import (  # type: ignore[no-redef]
            PublicationTable,
            build_publication_tables_from_assets,
            table_to_html,
            table_to_markdown,
            tables_for_section,
        )
    except ImportError:
        from tables import (  # type: ignore[no-redef]
            PublicationTable,
            build_publication_tables_from_assets,
            table_to_html,
            table_to_markdown,
            tables_for_section,
        )


JsonObject = dict[str, Any]
RenderFormat = Literal["markdown", "html"]
SectionRenderMode = Literal["auto", "paragraphs", "key_value", "table", "json"]
AssetKind = Literal["image", "table", "file", "html"]


class RenderingError(ValueError):
    """Базовая ошибка rendering-слоя."""


class StrictModel(BaseModel):
    """Базовая pydantic-модель с закрытым контрактом."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


class SectionRenderSpec(StrictModel):
    """Настройки отображения одной секции."""

    section_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    title: str | None = None
    mode: SectionRenderMode = "auto"
    field_order: tuple[str, ...] = Field(default_factory=tuple)
    hidden_fields: tuple[str, ...] = Field(default_factory=tuple)
    table_columns: tuple[str, ...] = Field(default_factory=tuple)
    include_title: bool = True


class RenderAsset(StrictModel):
    """Один внешний asset, который может быть включён в Markdown/HTML."""

    name: str = Field(min_length=1)
    kind: AssetKind
    path: Path | None = None
    title: str | None = None
    description: str | None = None
    html: str | None = None
    metadata: JsonObject = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_asset_payload(self) -> "RenderAsset":
        if self.kind == "html" and not self.html:
            raise ValueError("HTML asset requires the html field.")
        if self.kind != "html" and self.path is None:
            raise ValueError("Non-HTML asset requires the path field.")
        return self


class RenderSettings(StrictModel):
    """Настройки преобразования Bulletin в публикационный документ."""

    document_title: str = "Еженедельный бюллетень по гриппу и ОРВИ"
    language: str = "ru"
    include_metadata: bool = False
    include_status: bool = True
    include_runtime_report_summary: bool = True
    include_editorial_notes: bool = False
    include_assets: bool = True
    include_table_of_contents: bool = True
    heading_level: int = Field(default=1, ge=1, le=4)
    section_heading_level: int = Field(default=2, ge=2, le=5)
    max_json_block_chars: int = Field(default=6000, ge=500)
    section_specs: tuple[SectionRenderSpec, ...] = Field(default_factory=tuple)
    assets: tuple[RenderAsset, ...] = Field(default_factory=tuple)
    css: str | None = None

    @field_validator("section_specs")
    @classmethod
    def validate_unique_section_specs(cls, value: tuple[SectionRenderSpec, ...]) -> tuple[SectionRenderSpec, ...]:
        ids = [item.section_id for item in value]
        duplicates = sorted({section_id for section_id in ids if ids.count(section_id) > 1})
        if duplicates:
            raise ValueError(f"section_specs содержит повторяющиеся section_id: {duplicates!r}.")
        return value

    def spec_for(self, section_id: str) -> SectionRenderSpec | None:
        """Вернуть пользовательскую настройку отображения секции."""

        for spec in self.section_specs:
            if spec.section_id == section_id:
                return spec
        return None


class RenderedSection(StrictModel):
    """Отрендеренная секция в Markdown и HTML."""

    section_id: str
    title: str
    order: int
    markdown: str
    html: str
    plain_text: str


class RenderedBulletin(StrictModel):
    """Результат rendering-pass."""

    markdown: str
    html: str
    sections: list[RenderedSection]
    metadata: JsonObject
    assets: list[JsonObject] = Field(default_factory=list)
    output_paths: dict[str, str] = Field(default_factory=dict)

    def to_public_dict(self) -> JsonObject:
        """Вернуть JSON-сериализуемое описание результата."""

        return self.model_dump(mode="json", exclude_none=False)


class RenderOutputConfig(StrictModel):
    """Настройки сохранения rendering-артефактов."""

    output_dir: Path
    markdown_filename: str = Field(default="bulletin_rendered.md", min_length=1)
    html_filename: str = Field(default="bulletin_rendered.html", min_length=1)
    manifest_filename: str = Field(default="render_manifest.json", min_length=1)
    save_markdown: bool = True
    save_html: bool = True
    save_manifest: bool = True

    @field_validator("markdown_filename", "html_filename", "manifest_filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or len(path.parts) != 1:
            raise ValueError("Имена output-файлов должны быть простыми относительными именами.")
        return value

    @model_validator(mode="after")
    def validate_any_output(self) -> "RenderOutputConfig":
        if not (self.save_markdown or self.save_html or self.save_manifest):
            raise ValueError("Должен быть включён хотя бы один output-артефакт.")
        return self


def render_bulletin(
    bulletin: Bulletin | Mapping[str, Any],
    *,
    settings: RenderSettings | None = None,
) -> RenderedBulletin:
    """Отрендерить Bulletin одновременно в Markdown и HTML."""

    b = _coerce_bulletin(bulletin)
    cfg = settings or RenderSettings()
    bulletin_assets = _render_assets_from_bulletin(b.assets)
    if bulletin_assets:
        cfg = cfg.model_copy(update={"assets": tuple(bulletin_assets) + tuple(cfg.assets)})

    publication_tables = build_publication_tables_from_assets(b.assets)
    sections = [_render_section(section, cfg, publication_tables=publication_tables) for section in b.ordered_sections]
    metadata = _render_metadata_dict(b)

    markdown = _assemble_markdown(b, sections, cfg)
    html = _assemble_html(b, sections, cfg)

    return RenderedBulletin(
        markdown=markdown,
        html=html,
        sections=sections,
        metadata=metadata,
        assets=[asset.model_dump(mode="json", exclude_none=True) for asset in cfg.assets],
    )


def render_bulletin_markdown(
    bulletin: Bulletin | Mapping[str, Any],
    *,
    settings: RenderSettings | None = None,
) -> str:
    """Вернуть публикационный Markdown."""

    return render_bulletin(bulletin, settings=settings).markdown


def render_bulletin_html(
    bulletin: Bulletin | Mapping[str, Any],
    *,
    settings: RenderSettings | None = None,
) -> str:
    """Вернуть standalone HTML-документ."""

    return render_bulletin(bulletin, settings=settings).html


def render_bulletin_to_files(
    bulletin: Bulletin | Mapping[str, Any],
    *,
    settings: RenderSettings | None = None,
    output: RenderOutputConfig,
) -> RenderedBulletin:
    """Отрендерить Bulletin и сохранить Markdown/HTML/manifest."""

    rendered = render_bulletin(bulletin, settings=settings)
    output.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}

    if output.save_markdown:
        path = _ensure_parent_dir(output.output_dir / output.markdown_filename)
        path.write_text(rendered.markdown, encoding="utf-8")
        output_paths["markdown"] = str(path.resolve())

    if output.save_html:
        path = _ensure_parent_dir(output.output_dir / output.html_filename)
        path.write_text(rendered.html, encoding="utf-8")
        output_paths["html"] = str(path.resolve())

    if output.save_manifest:
        manifest = rendered.to_public_dict()
        manifest["markdown"] = None
        manifest["html"] = None
        manifest["output_paths"] = dict(output_paths)
        path = _write_json(manifest, output.output_dir / output.manifest_filename)
        output_paths["manifest"] = str(Path(path).resolve())

    rendered.output_paths = output_paths
    return rendered


def render_bulletin_file(
    bulletin_path: str | Path,
    *,
    settings: RenderSettings | None = None,
    output: RenderOutputConfig | None = None,
) -> RenderedBulletin:
    """Загрузить Bulletin из JSON и отрендерить его."""

    bulletin = load_bulletin(bulletin_path)
    if output is None:
        return render_bulletin(bulletin, settings=settings)
    return render_bulletin_to_files(bulletin, settings=settings, output=output)


def make_default_render_settings(*, include_assets: bool = True) -> RenderSettings:
    """Вернуть стандартные настройки публикационного рендера."""

    return RenderSettings(
        include_assets=include_assets,
        section_specs=(
            SectionRenderSpec(section_id="current_situation", mode="paragraphs"),
            SectionRenderSpec(section_id="epidemic_wave_comparison", mode="paragraphs"),
            SectionRenderSpec(section_id="age_group_season_overview", mode="paragraphs"),
            SectionRenderSpec(section_id="forecast_risks", mode="paragraphs"),
            SectionRenderSpec(section_id="shap_interpretation", mode="paragraphs"),
            SectionRenderSpec(section_id="model_quality", mode="paragraphs"),
            SectionRenderSpec(section_id="model_description", mode="paragraphs"),
        ),
    )


def _coerce_bulletin(value: Bulletin | Mapping[str, Any]) -> Bulletin:
    if isinstance(value, Bulletin):
        return value
    if isinstance(value, Mapping):
        return Bulletin.model_validate(value)
    raise RenderingError(f"Ожидался Bulletin или mapping, получено {type(value).__name__}.")


def _render_metadata_dict(bulletin: Bulletin) -> JsonObject:
    meta = bulletin.metadata.model_dump(mode="json", exclude_none=False)
    meta["status"] = bulletin.status
    meta["section_count"] = len(bulletin.sections)
    return meta


def _render_section(
    section: BulletinSection,
    settings: RenderSettings,
    *,
    publication_tables: Sequence[PublicationTable] = (),
) -> RenderedSection:
    spec = settings.spec_for(section.section_id)
    title = spec.title if spec and spec.title else section.title
    mode = spec.mode if spec else "auto"

    markdown_body = render_payload_markdown(
        section.content,
        mode=mode,
        field_order=spec.field_order if spec else (),
        hidden_fields=spec.hidden_fields if spec else (),
        table_columns=spec.table_columns if spec else (),
        max_json_block_chars=settings.max_json_block_chars,
    )
    html_body = render_payload_html(
        section.content,
        mode=mode,
        field_order=spec.field_order if spec else (),
        hidden_fields=spec.hidden_fields if spec else (),
        table_columns=spec.table_columns if spec else (),
        max_json_block_chars=settings.max_json_block_chars,
    )

    before_assets = _assets_for_section(settings.assets, section.section_id, placement="before_section")
    after_assets = _assets_for_section(settings.assets, section.section_id, placement="after_section")
    before_tables = tables_for_section(publication_tables, section.section_id, placement="before_section")
    after_tables = tables_for_section(publication_tables, section.section_id, placement="after_section")

    before_markdown_parts: list[str] = []
    if before_assets:
        before_markdown_parts.append("\n".join(_assets_markdown_lines(before_assets, include_heading=False)))
    if before_tables:
        before_markdown_parts.extend(table_to_markdown(table) for table in before_tables)

    after_markdown_parts: list[str] = []
    if after_assets:
        after_markdown_parts.append("\n".join(_assets_markdown_lines(after_assets, include_heading=False)))
    if after_tables:
        after_markdown_parts.extend(table_to_markdown(table) for table in after_tables)

    before_markdown = "\n\n".join(part.strip() for part in before_markdown_parts if part.strip())
    after_markdown = "\n\n".join(part.strip() for part in after_markdown_parts if part.strip())
    if before_markdown.strip():
        markdown_body = f"{before_markdown.strip()}\n\n{markdown_body.strip()}".strip()
    if after_markdown.strip():
        markdown_body = f"{markdown_body.rstrip()}\n\n{after_markdown.strip()}".strip()

    before_html_parts: list[str] = []
    if before_assets:
        before_html_parts.append(_assets_html(before_assets, include_heading=False))
    if before_tables:
        before_html_parts.extend(table_to_html(table) for table in before_tables)

    after_html_parts: list[str] = []
    if after_assets:
        after_html_parts.append(_assets_html(after_assets, include_heading=False))
    if after_tables:
        after_html_parts.extend(table_to_html(table) for table in after_tables)

    before_html = "\n".join(part for part in before_html_parts if part.strip())
    after_html = "\n".join(part for part in after_html_parts if part.strip())
    if before_html.strip():
        html_body = f"{before_html}\n{html_body}"
    if after_html.strip():
        html_body = f"{html_body}\n{after_html}"

    if spec is None or spec.include_title:
        heading = "#" * settings.section_heading_level
        markdown = f"{heading} {title}\n\n{markdown_body}".strip() + "\n"
        html = f'<section id="{html_escape(section.section_id)}">\n<h{settings.section_heading_level}>{html_escape(title)}</h{settings.section_heading_level}>\n{html_body}\n</section>'
    else:
        markdown = markdown_body.strip() + "\n"
        html = html_body

    return RenderedSection(
        section_id=section.section_id,
        title=title,
        order=section.order,
        markdown=markdown,
        html=html,
        plain_text=section.plain_text or flatten_text(section.content),
    )


def render_payload_markdown(
    payload: Mapping[str, Any],
    *,
    mode: SectionRenderMode = "auto",
    field_order: Sequence[str] = (),
    hidden_fields: Sequence[str] = (),
    table_columns: Sequence[str] = (),
    max_json_block_chars: int = 6000,
) -> str:
    """Отрендерить payload секции в Markdown."""

    effective_mode = _resolve_mode(payload, mode)
    visible_items = _ordered_visible_items(payload, field_order=field_order, hidden_fields=hidden_fields)

    if effective_mode == "paragraphs":
        parts = [_value_to_markdown(value) for _, value in visible_items]
        return "\n\n".join(part for part in parts if part.strip()).strip()

    if effective_mode == "key_value":
        lines = []
        for key, value in visible_items:
            lines.append(f"- **{_humanize_key(key)}:** {_inline_markdown_value(value)}")
        return "\n".join(lines)

    if effective_mode == "table":
        return _mapping_or_rows_to_markdown_table(payload, table_columns=table_columns)

    return _json_code_block(payload, max_chars=max_json_block_chars)


def render_payload_html(
    payload: Mapping[str, Any],
    *,
    mode: SectionRenderMode = "auto",
    field_order: Sequence[str] = (),
    hidden_fields: Sequence[str] = (),
    table_columns: Sequence[str] = (),
    max_json_block_chars: int = 6000,
) -> str:
    """Отрендерить payload секции в HTML."""

    effective_mode = _resolve_mode(payload, mode)
    visible_items = _ordered_visible_items(payload, field_order=field_order, hidden_fields=hidden_fields)

    if effective_mode == "paragraphs":
        blocks = []
        for _, value in visible_items:
            blocks.extend(_value_to_html_blocks(value))
        return "\n".join(blocks)

    if effective_mode == "key_value":
        rows = "\n".join(
            f"<li><strong>{html_escape(_humanize_key(key))}:</strong> {_inline_html_value(value)}</li>"
            for key, value in visible_items
        )
        return f"<ul>\n{rows}\n</ul>"

    if effective_mode == "table":
        return _mapping_or_rows_to_html_table(payload, table_columns=table_columns)

    text = _truncate(_json_dumps(payload), max_json_block_chars)
    return f"<pre><code>{html_escape(text)}</code></pre>"


def _assemble_markdown(bulletin: Bulletin, sections: Sequence[RenderedSection], settings: RenderSettings) -> str:
    lines: list[str] = []
    h = "#" * settings.heading_level
    lines.append(f"{h} {settings.document_title}")
    lines.append("")

    if settings.include_metadata:
        lines.extend(_metadata_markdown_lines(bulletin, include_status=settings.include_status))
        lines.append("")

    if settings.include_table_of_contents:
        lines.append("## Содержание" if settings.heading_level == 1 else "#" * (settings.heading_level + 1) + " Содержание")
        lines.append("")
        for section in sections:
            anchor = _markdown_anchor(section.title)
            lines.append(f"- [{section.title}](#{anchor})")
        lines.append("")

    for section in sections:
        lines.append(section.markdown.rstrip())
        lines.append("")

    if settings.include_assets:
        unplaced_assets = _unplaced_assets(settings.assets)
        if unplaced_assets:
            lines.extend(_assets_markdown_lines(unplaced_assets, include_heading=False))
            lines.append("")

    if settings.include_runtime_report_summary and bulletin.runtime_check_report is not None:
        lines.extend(_runtime_report_markdown_lines(bulletin))
        lines.append("")

    if settings.include_editorial_notes and bulletin.editorial_notes:
        lines.append("## Редакторские примечания")
        lines.append("")
        for note in bulletin.editorial_notes:
            lines.append(f"- {note}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"

def _assemble_html(bulletin: Bulletin, sections: Sequence[RenderedSection], settings: RenderSettings) -> str:
    title = html_escape(settings.document_title)
    css = settings.css or default_css()
    body_parts: list[str] = [f"<h1>{title}</h1>"]

    if settings.include_metadata:
        body_parts.append(_metadata_html(bulletin, include_status=settings.include_status))

    if settings.include_table_of_contents:
        items = "\n".join(
            f'<li><a href="#{html_escape(section.section_id)}">{html_escape(section.title)}</a></li>'
            for section in sections
        )
        body_parts.append(f"<nav class=\"toc\"><h2>Содержание</h2><ul>{items}</ul></nav>")

    for section in sections:
        body_parts.append(section.html)

    if settings.include_assets:
        unplaced_assets = _unplaced_assets(settings.assets)
        if unplaced_assets:
            body_parts.append(_assets_html(unplaced_assets, include_heading=False))

    if settings.include_runtime_report_summary and bulletin.runtime_check_report is not None:
        body_parts.append(_runtime_report_html(bulletin))

    if settings.include_editorial_notes and bulletin.editorial_notes:
        notes = "\n".join(f"<li>{html_escape(note)}</li>" for note in bulletin.editorial_notes)
        body_parts.append(f"<section><h2>Редакторские примечания</h2><ul>{notes}</ul></section>")

    body = "\n".join(body_parts)
    return (
        "<!doctype html>\n"
        f'<html lang="{html_escape(settings.language)}">\n'
        "<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{title}</title>\n"
        f"<style>\n{css}\n</style>\n"
        "</head>\n"
        f"<body>\n<main>\n{body}\n</main>\n</body>\n</html>\n"
    )

def _metadata_markdown_lines(bulletin: Bulletin, *, include_status: bool) -> list[str]:
    meta = bulletin.metadata
    lines = ["## Метаданные", ""]
    lines.append(f"- Идентификатор выпуска: `{meta.bulletin_id}`")
    lines.append(f"- Дата прогноза: {meta.origin_date}")
    lines.append(f"- ISO-неделя: {meta.iso_year}-W{meta.iso_week:02d}")
    lines.append(f"- Единица измерения: {meta.unit}")
    lines.append(f"- Расчёт на население: {meta.per_population:g}")
    if include_status:
        lines.append(f"- Статус: `{bulletin.status}`")
    generated_at = _format_datetime(meta.generated_at)
    if generated_at:
        lines.append(f"- Сформировано: {generated_at}")
    return lines


def _metadata_html(bulletin: Bulletin, *, include_status: bool) -> str:
    meta = bulletin.metadata
    rows = [
        ("Идентификатор выпуска", f"<code>{html_escape(meta.bulletin_id)}</code>"),
        ("Дата прогноза", html_escape(str(meta.origin_date))),
        ("ISO-неделя", html_escape(f"{meta.iso_year}-W{meta.iso_week:02d}")),
        ("Единица измерения", html_escape(meta.unit)),
        ("Расчёт на население", html_escape(f"{meta.per_population:g}")),
    ]
    if include_status:
        rows.append(("Статус", f"<code>{html_escape(bulletin.status)}</code>"))
    generated_at = _format_datetime(meta.generated_at)
    if generated_at:
        rows.append(("Сформировано", html_escape(generated_at)))
    body = "\n".join(f"<tr><th>{html_escape(k)}</th><td>{v}</td></tr>" for k, v in rows)
    return f"<section class=\"metadata\"><h2>Метаданные</h2><table>{body}</table></section>"


def _runtime_report_markdown_lines(bulletin: Bulletin) -> list[str]:
    report = bulletin.runtime_check_report
    if report is None:
        return []
    lines = ["## Runtime-проверки", ""]
    lines.append(f"- Статус: `{'ok' if report.ok else 'failed'}`")
    lines.append(f"- Всего проверок: {report.checks_total}")
    lines.append(f"- Критических нарушений: {len(report.errors)}")
    lines.append(f"- Предупреждений: {len(report.warnings)}")
    if report.errors:
        lines.append("")
        lines.append("Критические нарушения:")
        for issue in report.errors[:10]:
            lines.append(f"- `{issue.section_id}` / `{issue.check_name}`: {issue.message}")
    return lines


def _runtime_report_html(bulletin: Bulletin) -> str:
    report = bulletin.runtime_check_report
    if report is None:
        return ""
    rows = [
        ("Статус", "ok" if report.ok else "failed"),
        ("Всего проверок", str(report.checks_total)),
        ("Критических нарушений", str(len(report.errors))),
        ("Предупреждений", str(len(report.warnings))),
    ]
    table_rows = "\n".join(f"<tr><th>{html_escape(k)}</th><td>{html_escape(v)}</td></tr>" for k, v in rows)
    error_items = ""
    if report.errors:
        items = "\n".join(
            f"<li><code>{html_escape(issue.section_id)}</code> / <code>{html_escape(issue.check_name)}</code>: {html_escape(issue.message)}</li>"
            for issue in report.errors[:10]
        )
        error_items = f"<h3>Критические нарушения</h3><ul>{items}</ul>"
    return f"<section><h2>Runtime-проверки</h2><table>{table_rows}</table>{error_items}</section>"


def _render_assets_from_bulletin(raw_assets: Mapping[str, Any] | Sequence[Any] | None) -> tuple[RenderAsset, ...]:
    """Convert Bulletin.assets payload to RenderAsset objects.

    Supported contracts:
    - {asset_name: {kind, path, title, description, metadata}}
    - [{name, kind, path, title, description, metadata}, ...]
    """

    if not raw_assets:
        return ()

    items: list[Mapping[str, Any]] = []
    if isinstance(raw_assets, Mapping):
        for name, value in raw_assets.items():
            if isinstance(value, Mapping):
                payload = dict(value)
                payload.setdefault("name", str(name))
                items.append(payload)
    elif isinstance(raw_assets, Sequence) and not isinstance(raw_assets, (str, bytes)):
        for value in raw_assets:
            if isinstance(value, Mapping):
                items.append(dict(value))

    assets: list[RenderAsset] = []
    for item in items:
        try:
            assets.append(RenderAsset.model_validate(item))
        except Exception:
            # Rendering must remain robust: malformed optional assets are ignored.
            continue
    return tuple(assets)


def _asset_section_id(asset: RenderAsset) -> str | None:
    value = asset.metadata.get("section_id") if asset.metadata else None
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _asset_placement(asset: RenderAsset) -> str:
    value = asset.metadata.get("placement") if asset.metadata else None
    text = str(value).strip() if value is not None else "before_section"
    return text or "before_section"


def _assets_for_section(
    assets: Sequence[RenderAsset],
    section_id: str,
    *,
    placement: str | None = None,
) -> tuple[RenderAsset, ...]:
    selected = [asset for asset in assets if _asset_section_id(asset) == section_id]
    if placement is not None:
        selected = [asset for asset in selected if _asset_placement(asset) == placement]
    selected.sort(key=lambda asset: (asset.metadata.get("order", 10_000), asset.name))
    return tuple(selected)


def _unplaced_assets(assets: Sequence[RenderAsset]) -> tuple[RenderAsset, ...]:
    selected = [asset for asset in assets if _asset_section_id(asset) is None]
    selected.sort(key=lambda asset: (asset.metadata.get("order", 10_000), asset.name))
    return tuple(selected)


def _assets_markdown_lines(assets: Sequence[RenderAsset], *, include_heading: bool = False) -> list[str]:
    lines: list[str] = []
    if include_heading:
        lines.extend(["## Материалы", ""])
    for asset in assets:
        title = asset.title or asset.name
        if asset.kind == "image" and asset.path is not None:
            description = asset.description or title
            lines.append(f"![{description}]({asset.path})")
        elif asset.path is not None:
            lines.append(f"- [{title}]({asset.path})")
        elif asset.kind == "html" and asset.html:
            lines.append(asset.html)
        if asset.description and asset.kind != "image":
            lines.append(f"  - {asset.description}")
    return lines


def _assets_html(assets: Sequence[RenderAsset], *, include_heading: bool = True) -> str:
    blocks = ["<section>"]
    if include_heading:
        blocks.append("<h2>Материалы</h2>")
    for asset in assets:
        title = html_escape(asset.title or asset.name)
        description = html_escape(asset.description or "")
        if asset.kind == "image" and asset.path is not None:
            caption = description or title
            blocks.append(f'<figure><img src="{html_escape(str(asset.path))}" alt="{caption}"><figcaption>{caption}</figcaption></figure>')
        elif asset.kind == "html" and asset.html:
            blocks.append(asset.html)
        elif asset.path is not None:
            blocks.append(f'<p><a href="{html_escape(str(asset.path))}">{title}</a></p>')
        if description and asset.kind != "image":
            blocks.append(f"<p>{description}</p>")
    blocks.append("</section>")
    return "\n".join(blocks)


def _resolve_mode(payload: Mapping[str, Any], mode: SectionRenderMode) -> SectionRenderMode:
    if mode != "auto":
        return mode
    values = list(payload.values())
    if values and all(isinstance(value, str) for value in values):
        return "paragraphs"
    if _looks_like_table_payload(payload):
        return "table"
    return "key_value"


def _ordered_visible_items(
    payload: Mapping[str, Any],
    *,
    field_order: Sequence[str],
    hidden_fields: Sequence[str],
) -> list[tuple[str, Any]]:
    hidden = set(hidden_fields)
    ordered: list[tuple[str, Any]] = []
    seen: set[str] = set()
    for key in field_order:
        if key in payload and key not in hidden:
            ordered.append((key, payload[key]))
            seen.add(key)
    for key, value in payload.items():
        if key not in seen and key not in hidden:
            ordered.append((str(key), value))
    return ordered


def _value_to_markdown(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        return render_payload_markdown(value, mode="key_value")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if all(isinstance(item, Mapping) for item in value):
            return _rows_to_markdown_table(value)  # type: ignore[arg-type]
        return "\n".join(f"- {_inline_markdown_value(item)}" for item in value)
    return str(value)


def _value_to_html_blocks(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        paragraphs = [part.strip() for part in value.split("\n") if part.strip()]
        return [f"<p>{html_escape(paragraph)}</p>" for paragraph in paragraphs]
    if isinstance(value, Mapping):
        return [render_payload_html(value, mode="key_value")]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if all(isinstance(item, Mapping) for item in value):
            return [_rows_to_html_table(value)]  # type: ignore[arg-type]
        items = "\n".join(f"<li>{_inline_html_value(item)}</li>" for item in value)
        return [f"<ul>{items}</ul>"]
    return [f"<p>{html_escape(str(value))}</p>"]


def _inline_markdown_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return f"`{_truncate(_json_dumps(value), 300)}`"


def _inline_html_value(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, (str, int, float, bool)):
        return html_escape(str(value))
    return f"<code>{html_escape(_truncate(_json_dumps(value), 300))}</code>"


def _looks_like_table_payload(payload: Mapping[str, Any]) -> bool:
    return len(payload) == 1 and any(_is_rows(value) for value in payload.values())


def _is_rows(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)) and all(isinstance(item, Mapping) for item in value)


def _mapping_or_rows_to_markdown_table(payload: Mapping[str, Any], *, table_columns: Sequence[str]) -> str:
    rows_value = None
    if _is_rows(payload):
        rows_value = payload
    elif _looks_like_table_payload(payload):
        rows_value = next(iter(payload.values()))
    if rows_value is None:
        return render_payload_markdown(payload, mode="key_value")
    return _rows_to_markdown_table(rows_value, table_columns=table_columns)  # type: ignore[arg-type]


def _mapping_or_rows_to_html_table(payload: Mapping[str, Any], *, table_columns: Sequence[str]) -> str:
    rows_value = None
    if _is_rows(payload):
        rows_value = payload
    elif _looks_like_table_payload(payload):
        rows_value = next(iter(payload.values()))
    if rows_value is None:
        return render_payload_html(payload, mode="key_value")
    return _rows_to_html_table(rows_value, table_columns=table_columns)  # type: ignore[arg-type]


def _rows_to_markdown_table(rows: Sequence[Mapping[str, Any]], *, table_columns: Sequence[str] = ()) -> str:
    if not rows:
        return ""
    columns = list(table_columns) if table_columns else _union_columns(rows)
    header = "| " + " | ".join(_humanize_key(col) for col in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(_escape_markdown_table_cell(_inline_markdown_value(row.get(col))) for col in columns) + " |")
    return "\n".join([header, sep, *body])


def _rows_to_html_table(rows: Sequence[Mapping[str, Any]], *, table_columns: Sequence[str] = ()) -> str:
    if not rows:
        return ""
    columns = list(table_columns) if table_columns else _union_columns(rows)
    thead = "".join(f"<th>{html_escape(_humanize_key(col))}</th>" for col in columns)
    body_rows = []
    for row in rows:
        cells = "".join(f"<td>{_inline_html_value(row.get(col))}</td>" for col in columns)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{thead}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _union_columns(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    columns: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key_str = str(key)
            if key_str not in seen:
                columns.append(key_str)
                seen.add(key_str)
    return columns


def _json_code_block(value: Any, *, max_chars: int) -> str:
    return "```json\n" + _truncate(_json_dumps(value), max_chars) + "\n```"


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def _humanize_key(key: str) -> str:
    return str(key).replace("_", " ")


def _escape_markdown_table_cell(value: str) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_anchor(title: str) -> str:
    anchor = title.strip().lower()
    anchor = anchor.replace(" ", "-")
    anchor = "".join(ch for ch in anchor if ch.isalnum() or ch in "-_")
    return anchor


def _format_datetime(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    if value is None:
        return ""
    return str(value)



def _ensure_parent_dir(path: str | Path) -> Path:
    file_path = Path(path).expanduser()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def _write_json(data: Any, path: str | Path) -> Path:
    output_path = _ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, default=str)
    return output_path.resolve()

def default_css() -> str:
    """Минимальный CSS для standalone HTML."""

    return """
:root {
  color-scheme: light;
  --text: #111827;
  --muted: #4b5563;
  --border: #d1d5db;
  --background: #ffffff;
  --panel: #f9fafb;
}
body {
  margin: 0;
  background: var(--background);
  color: var(--text);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  line-height: 1.55;
}
main {
  max-width: 920px;
  margin: 0 auto;
  padding: 40px 28px 64px;
}
h1, h2, h3 {
  line-height: 1.2;
}
section {
  margin-top: 32px;
}
.metadata table,
table {
  width: 100%;
  border-collapse: collapse;
  margin: 16px 0;
}
th, td {
  border: 1px solid var(--border);
  padding: 8px 10px;
  vertical-align: top;
}
th {
  background: var(--panel);
  text-align: left;
}
code, pre {
  background: var(--panel);
}
pre {
  padding: 12px;
  overflow-x: auto;
}
.toc {
  border: 1px solid var(--border);
  background: var(--panel);
  padding: 16px 20px;
}
figure {
  margin: 20px 0;
}
img {
  max-width: 100%;
  height: auto;
}
figcaption {
  color: var(--muted);
  font-size: 0.95rem;
  margin-top: 8px;
}
""".strip()

