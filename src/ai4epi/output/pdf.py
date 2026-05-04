"""
PDF-rendering слой для бюллетеней ai4epi.

Модуль преобразует уже готовый ``RenderedBulletin`` или ``Bulletin`` в PDF.
Он не вызывает LLM, не редактирует текст, не выполняет runtime/evaluation
проверки и не собирает ``GlobalContext``. Структурирование содержания остаётся
за ``rendering.py``; этот модуль отвечает только за типографику PDF,
подключение шрифтов, изображения, таблицы, пагинацию и манифест результата.

Для генерации используется ReportLab. Кириллица требует TTF-шрифт с поддержкой
Unicode; если такой шрифт не найден, renderer завершится явной ошибкой, а не
создаст PDF с повреждёнными символами.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

try:  # Пакетный импорт.
    from ai4epi.generation.bulletin import Bulletin, load_bulletin
    from ai4epi.output.rendering import RenderSettings, RenderedBulletin, render_bulletin
except ImportError:  # pragma: no cover - поддержка запуска как набора отдельных файлов.
    from ai4epi.generation.bulletin import Bulletin, load_bulletin  # type: ignore[no-redef]
    from ai4epi.output.rendering import RenderSettings, RenderedBulletin, render_bulletin  # type: ignore[no-redef]


JsonObject = dict[str, Any]
PageSizeName = Literal["A4", "LETTER"]
PdfAssetKind = Literal["image", "file", "html", "table"]


class PdfRenderingError(RuntimeError):
    """Базовая ошибка PDF-rendering слоя."""


class PdfDependencyError(PdfRenderingError):
    """Не установлена обязательная зависимость PDF-renderer-а."""


class PdfFontError(PdfRenderingError):
    """Не удалось зарегистрировать кириллический шрифт для PDF."""


class PdfAssetError(PdfRenderingError):
    """Некорректный asset для PDF-рендеринга."""


class StrictModel(BaseModel):
    """Базовая pydantic-модель с закрытым контрактом."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


class PdfFontConfig(StrictModel):
    """Настройки шрифтов PDF.

    Если пути не заданы, renderer ищет системные DejaVu/Liberation/Noto шрифты.
    Для русскоязычного бюллетеня ``require_unicode_font=True`` должен оставаться
    включённым.
    """

    family_name: str = Field(default="Ai4EpiSans", min_length=1)
    regular_path: Path | None = None
    bold_path: Path | None = None
    require_unicode_font: bool = True

    @model_validator(mode="after")
    def validate_font_pair(self) -> "PdfFontConfig":
        if (self.regular_path is None) != (self.bold_path is None):
            raise ValueError("regular_path и bold_path должны задаваться совместно.")
        return self


class PdfSettings(StrictModel):
    """Настройки PDF-типографики."""

    document_title: str = Field(default="Еженедельный бюллетень по гриппу и ОРВИ", min_length=1)
    page_size: PageSizeName = "A4"
    margin_left_mm: float = Field(default=18.0, ge=0.0)
    margin_right_mm: float = Field(default=18.0, ge=0.0)
    margin_top_mm: float = Field(default=16.0, ge=0.0)
    margin_bottom_mm: float = Field(default=16.0, ge=0.0)
    font: PdfFontConfig = Field(default_factory=PdfFontConfig)
    body_font_size: float = Field(default=10.2, gt=0.0)
    body_leading: float = Field(default=13.2, gt=0.0)
    title_font_size: float = Field(default=17.0, gt=0.0)
    section_font_size: float = Field(default=13.0, gt=0.0)
    small_font_size: float = Field(default=8.5, gt=0.0)
    table_font_size: float = Field(default=8.2, gt=0.0)
    include_metadata: bool = False
    include_table_of_contents: bool = True
    include_runtime_summary: bool = True
    include_assets: bool = True
    include_page_numbers: bool = True
    max_image_width_mm: float = Field(default=165.0, gt=0.0)
    max_image_height_mm: float = Field(default=105.0, gt=0.0)
    split_long_tables: bool = True
    paragraph_spacing_pt: float = Field(default=6.0, ge=0.0)


class PdfOutputConfig(StrictModel):
    """Настройки сохранения PDF-артефактов."""

    output_dir: Path
    pdf_filename: str = Field(default="bulletin.pdf", min_length=1)
    manifest_filename: str = Field(default="pdf_manifest.json", min_length=1)
    save_manifest: bool = True

    @field_validator("pdf_filename", "manifest_filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or len(path.parts) != 1:
            raise ValueError("Имена PDF output-файлов должны быть простыми относительными именами.")
        return value


class PdfRenderResult(StrictModel):
    """Результат PDF-rendering pass."""

    pdf_path: Path
    manifest_path: Path | None = None
    page_count: int | None = Field(default=None, ge=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    font_family: str = Field(min_length=1)
    font_regular_path: Path | None = None
    font_bold_path: Path | None = None
    metadata: JsonObject = Field(default_factory=dict)

    def to_public_dict(self) -> JsonObject:
        """Вернуть JSON-сериализуемое описание PDF-результата."""

        return self.model_dump(mode="json", exclude_none=False)


@dataclass(frozen=True)
class RegisteredFontSet:
    """Имена зарегистрированных ReportLab-шрифтов."""

    regular_name: str
    bold_name: str
    regular_path: Path | None
    bold_path: Path | None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render_pdf(
    bulletin_or_rendered: Bulletin | RenderedBulletin | Mapping[str, Any],
    *,
    output: PdfOutputConfig,
    pdf_settings: PdfSettings | None = None,
    render_settings: RenderSettings | None = None,
) -> PdfRenderResult:
    """Сформировать PDF-файл из Bulletin или RenderedBulletin."""

    settings = pdf_settings or PdfSettings()
    rendered = _coerce_rendered_bulletin(bulletin_or_rendered, render_settings=render_settings)
    output.output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output.output_dir / output.pdf_filename

    font_set = _register_pdf_fonts(settings.font)
    _build_reportlab_pdf(rendered, pdf_path=pdf_path, settings=settings, font_set=font_set)
    page_count = _count_pdf_pages(pdf_path)

    result = PdfRenderResult(
        pdf_path=pdf_path.resolve(),
        page_count=page_count,
        font_family=settings.font.family_name,
        font_regular_path=font_set.regular_path,
        font_bold_path=font_set.bold_path,
        metadata={
            "document_title": settings.document_title,
            "source_section_count": len(rendered.sections),
            "rendered_metadata": rendered.metadata,
        },
    )

    if output.save_manifest:
        manifest_path = output.output_dir / output.manifest_filename
        _write_json(result.to_public_dict(), manifest_path)
        result.manifest_path = manifest_path.resolve()

    return result


def render_bulletin_pdf(
    bulletin: Bulletin | Mapping[str, Any],
    *,
    output: PdfOutputConfig,
    pdf_settings: PdfSettings | None = None,
    render_settings: RenderSettings | None = None,
) -> PdfRenderResult:
    """Отрендерить Bulletin в PDF."""

    return render_pdf(
        bulletin,
        output=output,
        pdf_settings=pdf_settings,
        render_settings=render_settings,
    )


def render_bulletin_pdf_file(
    bulletin_path: str | Path,
    *,
    output: PdfOutputConfig,
    pdf_settings: PdfSettings | None = None,
    render_settings: RenderSettings | None = None,
) -> PdfRenderResult:
    """Загрузить Bulletin из JSON и отрендерить его в PDF."""

    bulletin = load_bulletin(bulletin_path)
    return render_bulletin_pdf(
        bulletin,
        output=output,
        pdf_settings=pdf_settings,
        render_settings=render_settings,
    )


def make_default_pdf_settings() -> PdfSettings:
    """Вернуть стандартные настройки PDF для русскоязычного бюллетеня."""

    return PdfSettings()


# ---------------------------------------------------------------------------
# ReportLab backend
# ---------------------------------------------------------------------------


def _build_reportlab_pdf(
    rendered: RenderedBulletin,
    *,
    pdf_path: Path,
    settings: PdfSettings,
    font_set: RegisteredFontSet,
) -> None:
    reportlab = _require_reportlab()
    pagesizes = reportlab["pagesizes"]
    units = reportlab["units"]
    platypus = reportlab["platypus"]

    page_size = pagesizes.A4 if settings.page_size == "A4" else pagesizes.letter
    mm = units.mm
    doc = platypus.SimpleDocTemplate(
        str(pdf_path),
        pagesize=page_size,
        leftMargin=settings.margin_left_mm * mm,
        rightMargin=settings.margin_right_mm * mm,
        topMargin=settings.margin_top_mm * mm,
        bottomMargin=settings.margin_bottom_mm * mm,
        title=settings.document_title,
        author="ai4epi",
        subject="Influenza and ARI weekly bulletin",
    )

    styles = _make_styles(settings, font_set)
    story: list[Any] = []
    story.extend(_title_flowables(rendered, settings=settings, styles=styles))

    if settings.include_metadata:
        story.extend(_metadata_flowables(rendered, styles=styles))

    if settings.include_table_of_contents:
        story.extend(_toc_flowables(rendered, styles=styles))

    story.extend(_body_flowables(rendered, styles=styles, settings=settings))

    if settings.include_runtime_summary:
        runtime = _runtime_summary_from_metadata(rendered.metadata)
        if runtime:
            story.extend(_runtime_summary_flowables(runtime, styles=styles))

    def on_page(canvas: Any, doc_obj: Any) -> None:
        if settings.include_page_numbers:
            canvas.saveState()
            canvas.setFont(font_set.regular_name, settings.small_font_size)
            canvas.drawRightString(
                page_size[0] - settings.margin_right_mm * mm,
                settings.margin_bottom_mm * mm * 0.55,
                str(doc_obj.page),
            )
            canvas.restoreState()

    try:
        doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    except Exception as exc:
        raise PdfRenderingError(f"Не удалось построить PDF: {exc}") from exc


def _require_reportlab() -> dict[str, Any]:
    try:
        from reportlab.lib import colors, pagesizes, units
        from reportlab.lib.enums import TA_CENTER, TA_LEFT
        from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
        from reportlab.lib.utils import ImageReader
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        from reportlab.platypus import (
            Image,
            ListFlowable,
            ListItem,
            Paragraph,
            Preformatted,
            SimpleDocTemplate,
            Spacer,
            Table,
            TableStyle,
        )
    except ImportError as exc:  # pragma: no cover - зависит от окружения установки.
        raise PdfDependencyError("Для PDF-rendering требуется пакет reportlab.") from exc

    return {
        "colors": colors,
        "pagesizes": pagesizes,
        "units": units,
        "TA_CENTER": TA_CENTER,
        "TA_LEFT": TA_LEFT,
        "ParagraphStyle": ParagraphStyle,
        "getSampleStyleSheet": getSampleStyleSheet,
        "ImageReader": ImageReader,
        "pdfmetrics": pdfmetrics,
        "TTFont": TTFont,
        "platypus": type(
            "Platypus",
            (),
            {
                "Image": Image,
                "ListFlowable": ListFlowable,
                "ListItem": ListItem,
                "Paragraph": Paragraph,
                "Preformatted": Preformatted,
                "SimpleDocTemplate": SimpleDocTemplate,
                "Spacer": Spacer,
                "Table": Table,
                "TableStyle": TableStyle,
            },
        ),
    }


def _register_pdf_fonts(config: PdfFontConfig) -> RegisteredFontSet:
    reportlab = _require_reportlab()
    pdfmetrics = reportlab["pdfmetrics"]
    TTFont = reportlab["TTFont"]

    regular_path, bold_path = _resolve_font_paths(config)
    regular_name = f"{config.family_name}-Regular"
    bold_name = f"{config.family_name}-Bold"

    if regular_path is None or bold_path is None:
        if config.require_unicode_font:
            raise PdfFontError(
                "Не найден системный TTF-шрифт с поддержкой кириллицы. "
                "Передайте PdfFontConfig(regular_path=..., bold_path=...)."
            )
        return RegisteredFontSet(
            regular_name="Helvetica",
            bold_name="Helvetica-Bold",
            regular_path=None,
            bold_path=None,
        )

    try:
        pdfmetrics.registerFont(TTFont(regular_name, str(regular_path)))
        pdfmetrics.registerFont(TTFont(bold_name, str(bold_path)))
    except Exception as exc:
        raise PdfFontError(f"Не удалось зарегистрировать TTF-шрифты для PDF: {exc}") from exc

    return RegisteredFontSet(
        regular_name=regular_name,
        bold_name=bold_name,
        regular_path=regular_path.resolve(),
        bold_path=bold_path.resolve(),
    )


def _resolve_font_paths(config: PdfFontConfig) -> tuple[Path | None, Path | None]:
    if config.regular_path is not None and config.bold_path is not None:
        regular = Path(config.regular_path).expanduser().resolve()
        bold = Path(config.bold_path).expanduser().resolve()
        if not regular.is_file():
            raise PdfFontError(f"Файл regular-шрифта не найден: {regular}")
        if not bold.is_file():
            raise PdfFontError(f"Файл bold-шрифта не найден: {bold}")
        return regular, bold

    candidates = [
        (
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("C:/Windows/Fonts/arialbd.ttf"),
        ),
        (
            Path("C:/Windows/Fonts/segoeui.ttf"),
            Path("C:/Windows/Fonts/segoeuib.ttf"),
        ),
        (
            Path("C:/Windows/Fonts/calibri.ttf"),
            Path("C:/Windows/Fonts/calibrib.ttf"),
        ),
        (
            Path("C:/Windows/Fonts/times.ttf"),
            Path("C:/Windows/Fonts/timesbd.ttf"),
        ),

        (
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
        ),
        (
            Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf"),
            Path("/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"),
        ),
        (
            Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"),
            Path("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"),
        ),
        (
            Path("/usr/local/share/fonts/DejaVuSans.ttf"),
            Path("/usr/local/share/fonts/DejaVuSans-Bold.ttf"),
        ),
    ]
    for regular, bold in candidates:
        if regular.is_file() and bold.is_file():
            return regular, bold
    return None, None


def _make_styles(settings: PdfSettings, font_set: RegisteredFontSet) -> dict[str, Any]:
    reportlab = _require_reportlab()
    ParagraphStyle = reportlab["ParagraphStyle"]
    getSampleStyleSheet = reportlab["getSampleStyleSheet"]
    TA_CENTER = reportlab["TA_CENTER"]
    TA_LEFT = reportlab["TA_LEFT"]

    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Ai4EpiTitle",
            parent=base["Title"],
            fontName=font_set.bold_name,
            fontSize=settings.title_font_size,
            leading=settings.title_font_size + 3,
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "Ai4EpiSubtitle",
            parent=base["BodyText"],
            fontName=font_set.regular_name,
            fontSize=settings.body_font_size,
            leading=settings.body_leading,
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "h1": ParagraphStyle(
            "Ai4EpiHeading1",
            parent=base["Heading1"],
            fontName=font_set.bold_name,
            fontSize=settings.section_font_size + 2,
            leading=settings.section_font_size + 5,
            spaceBefore=10,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Ai4EpiHeading2",
            parent=base["Heading2"],
            fontName=font_set.bold_name,
            fontSize=settings.section_font_size,
            leading=settings.section_font_size + 4,
            spaceBefore=9,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "Ai4EpiHeading3",
            parent=base["Heading3"],
            fontName=font_set.bold_name,
            fontSize=settings.body_font_size + 1,
            leading=settings.body_leading + 1,
            spaceBefore=7,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "Ai4EpiBody",
            parent=base["BodyText"],
            fontName=font_set.regular_name,
            fontSize=settings.body_font_size,
            leading=settings.body_leading,
            alignment=TA_LEFT,
            spaceAfter=settings.paragraph_spacing_pt,
        ),
        "small": ParagraphStyle(
            "Ai4EpiSmall",
            parent=base["BodyText"],
            fontName=font_set.regular_name,
            fontSize=settings.small_font_size,
            leading=settings.small_font_size + 2,
            spaceAfter=4,
        ),
        "table": ParagraphStyle(
            "Ai4EpiTableCell",
            parent=base["BodyText"],
            fontName=font_set.regular_name,
            fontSize=settings.table_font_size,
            leading=settings.table_font_size + 2,
            wordWrap="CJK",
        ),
        "table_header": ParagraphStyle(
            "Ai4EpiTableHeader",
            parent=base["BodyText"],
            fontName=font_set.bold_name,
            fontSize=settings.table_font_size,
            leading=settings.table_font_size + 2,
            wordWrap="CJK",
        ),
        "code": ParagraphStyle(
            "Ai4EpiCode",
            parent=base["Code"],
            fontName=font_set.regular_name,
            fontSize=settings.small_font_size,
            leading=settings.small_font_size + 2,
        ),
    }


# ---------------------------------------------------------------------------
# Flowable construction
# ---------------------------------------------------------------------------


def _title_flowables(rendered: RenderedBulletin, *, settings: PdfSettings, styles: Mapping[str, Any]) -> list[Any]:
    platypus = _require_reportlab()["platypus"]
    flowables: list[Any] = [platypus.Paragraph(_xml_escape(settings.document_title), styles["title"])]

    subtitle = _release_subtitle(rendered.metadata)
    if subtitle:
        flowables.append(platypus.Paragraph(_xml_escape(subtitle), styles["subtitle"]))
    else:
        flowables.append(platypus.Spacer(1, 6))
    return flowables


def _release_subtitle(metadata: Mapping[str, Any]) -> str:
    """Build notebook-style issue subtitle from bulletin metadata."""

    iso_year = metadata.get("iso_year")
    iso_week = metadata.get("iso_week")
    if iso_year is None or iso_week is None:
        return ""

    start_date = _parse_iso_date(metadata.get("origin_date"))
    if start_date is None:
        return f"за {iso_week} неделю {iso_year} года."

    end_date = start_date + timedelta(days=6)
    return f"за {iso_week} неделю {iso_year} года. ({_format_short_date(start_date)} - {_format_short_date(end_date)})"


def _parse_iso_date(value: Any) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text[:10])
    except ValueError:
        return None


def _format_short_date(value: datetime) -> str:
    return value.strftime("%d.%m.%y")

def _metadata_flowables(rendered: RenderedBulletin, *, styles: Mapping[str, Any]) -> list[Any]:
    meta = rendered.metadata or {}
    rows = []
    for label, key in (
        ("Идентификатор выпуска", "bulletin_id"),
        ("Дата прогноза", "origin_date"),
        ("ISO-год", "iso_year"),
        ("ISO-неделя", "iso_week"),
        ("Единица измерения", "unit"),
        ("Расчёт на население", "per_population"),
        ("Статус", "status"),
    ):
        if key in meta and meta[key] is not None:
            rows.append([label, str(meta[key])])
    if not rows:
        return []
    return [
        _heading("Метаданные", styles["h2"]),
        _table_from_rows(rows, styles=styles, header=False),
    ]


def _toc_flowables(rendered: RenderedBulletin, *, styles: Mapping[str, Any]) -> list[Any]:
    platypus = _require_reportlab()["platypus"]
    items = []
    for title in _toc_titles(rendered):
        items.append(platypus.ListItem(platypus.Paragraph(_xml_escape(title), styles["body"])))
    if not items:
        return []
    return [
        _heading("Содержание", styles["h2"]),
        platypus.ListFlowable(items, bulletType="bullet", leftIndent=16),
        platypus.Spacer(1, 4),
    ]


def _body_flowables(rendered: RenderedBulletin, *, styles: Mapping[str, Any], settings: PdfSettings) -> list[Any]:
    """Build PDF body with the agreed structural changes only.

    Agreed changes:
    1. number existing sections without renaming them;
    2. add the release subtitle in the title block;
    3. remove explicit page-break markers before/inside sections.
    """

    flowables: list[Any] = []
    for section_number, section in enumerate(sorted(rendered.sections, key=lambda item: (item.order, item.section_id)), start=1):
        markdown = _strip_explicit_page_break_markers(section.markdown)
        markdown = _replace_first_heading(markdown, f"{section_number}. {section.title}")
        flowables.extend(_section_flowables(markdown, styles=styles, settings=settings))

    return flowables

def _section_flowables(markdown: str, *, styles: Mapping[str, Any], settings: PdfSettings) -> list[Any]:
    flowables: list[Any] = []
    lines = markdown.splitlines()
    index = 0
    paragraph_buffer: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_buffer:
            return
        text = " ".join(part.strip() for part in paragraph_buffer if part.strip()).strip()
        paragraph_buffer.clear()
        if text:
            flowables.append(_paragraph(text, styles["body"]))

    while index < len(lines):
        line = lines[index].rstrip()
        stripped = line.strip()

        if not stripped:
            flush_paragraph()
            index += 1
            continue

        if stripped.startswith("```"):
            flush_paragraph()
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code_lines.append(lines[index])
                index += 1
            if index < len(lines):
                index += 1
            flowables.append(_preformatted("\n".join(code_lines), styles["code"]))
            continue

        if _is_markdown_table_start(lines, index):
            flush_paragraph()
            table_lines: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table_lines.append(lines[index].strip())
                index += 1
            flowables.append(_markdown_table_to_flowable(table_lines, styles=styles))
            continue

        if stripped.startswith("!["):
            flush_paragraph()
            image = _markdown_image_to_flowables(stripped, styles=styles, settings=settings)
            flowables.extend(image)
            index += 1
            continue

        if stripped.startswith("#"):
            flush_paragraph()
            level, title = _parse_heading(stripped)
            if title:
                style = styles["h1"] if level <= 1 else styles["h2"] if level == 2 else styles["h3"]
                flowables.append(_heading(title, style))
            index += 1
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            bullet_lines = []
            while index < len(lines) and lines[index].strip().startswith("- "):
                bullet_lines.append(lines[index].strip()[2:].strip())
                index += 1
            flowables.append(_bullet_list(bullet_lines, styles=styles))
            continue

        paragraph_buffer.append(stripped)
        index += 1

    flush_paragraph()
    return flowables


def _assets_flowables(
    assets: Sequence[Mapping[str, Any]],
    *,
    styles: Mapping[str, Any],
    settings: PdfSettings,
) -> list[Any]:
    flowables: list[Any] = [_heading("Материалы", styles["h2"])]
    for asset in assets:
        kind = str(asset.get("kind") or "")
        title = str(asset.get("title") or asset.get("name") or "Материал")
        path = asset.get("path")
        description = str(asset.get("description") or "")
        if kind == "image" and path:
            flowables.extend(_image_flowables(Path(path), title=title, description=description, styles=styles, settings=settings))
        elif kind == "file" and path:
            flowables.append(_paragraph(f"{title}: {path}", styles["body"]))
        elif kind == "html":
            flowables.append(_paragraph(f"{title}: HTML asset не встраивается в PDF напрямую.", styles["small"]))
        elif kind == "table":
            flowables.append(_paragraph(f"{title}: табличный asset ожидается в основном содержании бюллетеня.", styles["small"]))
        if description and kind != "image":
            flowables.append(_paragraph(description, styles["small"]))
    return flowables


def _runtime_summary_flowables(runtime: Mapping[str, Any], *, styles: Mapping[str, Any]) -> list[Any]:
    rows = [[str(key), str(value)] for key, value in runtime.items()]
    if not rows:
        return []
    return [
        _heading("Runtime-проверки", styles["h2"]),
        _table_from_rows(rows, styles=styles, header=False),
    ]


def _heading(text: str, style: Any) -> Any:
    return _require_reportlab()["platypus"].Paragraph(_xml_escape(text), style)


def _paragraph(text: str, style: Any) -> Any:
    return _require_reportlab()["platypus"].Paragraph(_markdown_inline_to_reportlab(text), style)


def _preformatted(text: str, style: Any) -> Any:
    return _require_reportlab()["platypus"].Preformatted(_xml_escape(text), style)


def _bullet_list(items: Sequence[str], *, styles: Mapping[str, Any]) -> Any:
    platypus = _require_reportlab()["platypus"]
    list_items = [platypus.ListItem(_paragraph(item, styles["body"])) for item in items if item.strip()]
    return platypus.ListFlowable(list_items, bulletType="bullet", leftIndent=16)


def _table_from_rows(rows: Sequence[Sequence[Any]], *, styles: Mapping[str, Any], header: bool) -> Any:
    reportlab = _require_reportlab()
    colors = reportlab["colors"]
    platypus = reportlab["platypus"]

    data = []
    for row_index, row in enumerate(rows):
        style = styles["table_header"] if header and row_index == 0 else styles["table"]
        data.append([platypus.Paragraph(_xml_escape(str(cell)), style) for cell in row])

    table = platypus.Table(data, repeatRows=1 if header else 0, hAlign="LEFT")
    table.setStyle(
        platypus.TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#d1d5db")),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f3f4f6") if header else colors.white),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ]
        )
    )
    return table


def _markdown_table_to_flowable(table_lines: Sequence[str], *, styles: Mapping[str, Any]) -> Any:
    rows: list[list[str]] = []
    for index, line in enumerate(table_lines):
        if index == 1 and re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", line):
            continue
        cells = [cell.strip().replace("\\|", "|") for cell in line.strip("|").split("|")]
        rows.append(cells)
    if not rows:
        return _paragraph("", styles["body"])
    return _table_from_rows(rows, styles=styles, header=True)


def _markdown_image_to_flowables(line: str, *, styles: Mapping[str, Any], settings: PdfSettings) -> list[Any]:
    match = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
    if not match:
        return [_paragraph(line, styles["body"])]
    description = match.group(1).strip()
    path = Path(match.group(2).strip())
    return _image_flowables(path, title=description or path.name, description=description, styles=styles, settings=settings)


def _image_flowables(
    path: Path,
    *,
    title: str,
    description: str,
    styles: Mapping[str, Any],
    settings: PdfSettings,
) -> list[Any]:
    reportlab = _require_reportlab()
    platypus = reportlab["platypus"]
    ImageReader = reportlab["ImageReader"]
    mm = reportlab["units"].mm

    image_path = Path(path).expanduser()
    if not image_path.is_file():
        raise PdfAssetError(f"Изображение не найдено: {image_path}")

    try:
        reader = ImageReader(str(image_path))
        width_px, height_px = reader.getSize()
    except Exception as exc:
        raise PdfAssetError(f"Не удалось прочитать изображение {image_path}: {exc}") from exc

    max_width = settings.max_image_width_mm * mm
    max_height = settings.max_image_height_mm * mm
    scale = min(max_width / width_px, max_height / height_px, 1.0)
    image = platypus.Image(str(image_path), width=width_px * scale, height=height_px * scale)
    flowables: list[Any] = [image]
    caption = description or title
    if caption:
        flowables.append(_paragraph(caption, styles["small"]))
    return flowables


def _toc_titles(rendered: RenderedBulletin) -> list[str]:
    """Return TOC titles with numbering, preserving original section names."""

    return [
        f"{index}. {section.title}"
        for index, section in enumerate(sorted(rendered.sections, key=lambda item: (item.order, item.section_id)), start=1)
    ]


def _section_pdf_title(section_id: str, *, fallback: str) -> str:
    """Backward-compatible helper: do not rename section titles here."""

    return fallback

def _replace_first_heading(markdown: str, title: str) -> str:
    lines = markdown.splitlines()
    for index, line in enumerate(lines):
        if line.strip().startswith("#"):
            level, _ = _parse_heading(line.strip())
            if level:
                lines[index] = f"{'#' * level} {title}"
                return "\n".join(lines).strip() + "\n"
    return f"## {title}\n\n{markdown.strip()}\n"


def _extract_forecast_table_block(markdown: str) -> tuple[str, str]:
    lines = markdown.splitlines()
    title_index: int | None = None
    for index, line in enumerate(lines):
        if "Таблица с прогнозом" in _strip_markdown_markup(line):
            title_index = index
            break
    if title_index is None:
        return "", markdown

    table_start: int | None = None
    for index in range(title_index + 1, len(lines)):
        if _is_markdown_table_start(lines, index):
            table_start = index
            break
    if table_start is None:
        return "", markdown

    table_end = table_start
    while table_end < len(lines) and lines[table_end].strip().startswith("|"):
        table_end += 1

    table_lines = lines[table_start:table_end]
    before = lines[:title_index]
    after = lines[table_end:]

    # Drop blank lines left by moving the table out of forecast_risks.
    while before and not before[-1].strip():
        before.pop()
    while after and not after[0].strip():
        after.pop(0)

    remaining = "\n".join(before + ([""] if before and after else []) + after).strip() + "\n"
    return "\n".join(table_lines).strip() + "\n", remaining


def _strip_markdown_markup(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned)
    cleaned = cleaned.strip("*_` ")
    return cleaned


def _strip_explicit_page_break_markers(markdown: str) -> str:
    lines = []
    for line in markdown.splitlines():
        lowered = line.lower()
        if "pagebreak" in lowered or "page-break-before" in lowered or "page-break-after" in lowered:
            continue
        lines.append(line)
    return "\n".join(lines).strip() + "\n"


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------


def _coerce_rendered_bulletin(
    value: Bulletin | RenderedBulletin | Mapping[str, Any],
    *,
    render_settings: RenderSettings | None,
) -> RenderedBulletin:
    if isinstance(value, RenderedBulletin):
        return value
    if isinstance(value, Bulletin):
        return render_bulletin(value, settings=render_settings)
    if isinstance(value, Mapping):
        if "markdown" in value and "html" in value and "sections" in value:
            return RenderedBulletin.model_validate(value)
        return render_bulletin(Bulletin.model_validate(value), settings=render_settings)
    raise PdfRenderingError(f"Ожидался Bulletin, RenderedBulletin или mapping, получено {type(value).__name__}.")


def _parse_heading(line: str) -> tuple[int, str]:
    match = re.match(r"^(#{1,6})\s+(.*)$", line)
    if not match:
        return 0, ""
    return len(match.group(1)), match.group(2).strip()


def _is_markdown_table_start(lines: Sequence[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False
    line = lines[index].strip()
    next_line = lines[index + 1].strip()
    return line.startswith("|") and "|" in line[1:] and re.fullmatch(r"\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?", next_line) is not None


def _markdown_inline_to_reportlab(text: str) -> str:
    escaped = _xml_escape(text)
    escaped = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<font name=\"Courier\">\1</font>", escaped)
    return escaped


def _xml_escape(text: Any) -> str:
    raw = str(text)
    return (
        raw.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _runtime_summary_from_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    # rendering.py пока не кладёт runtime_report в metadata, но поле оставлено
    # для будущего расширения без изменения публичного PDF-контракта.
    raw = metadata.get("runtime_check_report")
    if not isinstance(raw, Mapping):
        return {}
    return {
        "status": "ok" if raw.get("ok") else "failed",
        "checks_total": raw.get("checks_total"),
        "checks_failed": raw.get("checks_failed"),
    }


def _count_pdf_pages(path: Path) -> int | None:
    try:
        from pypdf import PdfReader  # type: ignore[import-untyped]
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore[import-untyped]
        except Exception:
            return None
    try:
        reader = PdfReader(str(path))
        return len(reader.pages)
    except Exception:
        return None


def _write_json(data: Any, path: str | Path) -> Path:
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(_jsonable(data), stream, ensure_ascii=False, indent=2, default=str)
    return output_path.resolve()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_jsonable(item) for item in value]
    return value

