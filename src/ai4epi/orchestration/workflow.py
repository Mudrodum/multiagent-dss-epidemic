"""
Верхнеуровневый workflow для полного запуска ai4epi.

Модуль связывает уже разделённые слои репозитория:

* ``analysis_pipeline.py`` — данные, признаки, прогноз, SHAP, аналитические
  блоки и ``GlobalContext``;
* ``pipeline.py`` — секционные narrator-агенты и базовый ``Bulletin``;
* ``editor.py`` — независимый editor-pass поверх готового бюллетеня;
* ``evaluation.py`` — downstream-оценка готового бюллетеня;
* ``rendering.py`` и ``pdf.py`` — публикационные артефакты.

Этот файл не содержит доменных эвристик, не обучает модели напрямую и не
формирует промпты. Он является application-level orchestration layer: одна
точка запуска для сценария «получить данные → построить прогноз → собрать
контекст → сгенерировать бюллетень → отредактировать → оценить → отрендерить».

Важный контракт источников: по умолчанию загрузка начинается с ISO-недели ``2011-W01``. Если ``AnalysisSourceConfig.end_date is None``,
workflow трактует конечную дату как последнюю фактически загруженную неделю
из influenza-ряда. Запрос к источнику может технически выполняться до текущей
недели, но в отчёте workflow и downstream-артефактах фиксируется именно
последняя доступная строка данных.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import time
from typing import Any, Literal, Mapping

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

try:  # package import
    from ai4epi.analysis.analysis_pipeline import (
        AnalysisOutputConfig,
        AnalysisPipelineSettings,
        AnalysisRunResult,
        AnalysisSourceConfig,
        run_analysis_pipeline,
        run_analysis_pipeline_from_sources,
    )
    from ai4epi.generation.bulletin import Bulletin, save_bulletin, save_bulletin_markdown
    from ai4epi.core.context import GlobalContext
    from ai4epi.generation.editor import EditorRunResult, EditorSettings, edit_bulletin
    from ai4epi.quality.evaluation import EvalConfig, EvalReport, evaluate_bulletin, save_eval_report
    from ai4epi.core.io import TableReadOptions, read_table, write_json
    from ai4epi.generation.narrator import NarratorSettings, make_chat_backend
    from ai4epi.output.pdf import PdfOutputConfig, PdfRenderResult, render_bulletin_pdf
    from ai4epi.output.figures import build_publication_figure_assets
    from ai4epi.generation.pipeline import (
        PipelineOutputConfig,
        PipelineRunResult,
        PipelineSettings,
        run_pipeline_from_files,
    )
    from ai4epi.output.rendering import (
        RenderOutputConfig,
        RenderedBulletin,
        make_default_render_settings,
        render_bulletin_to_files,
    )
except ImportError:  # pragma: no cover - поддержка плоского исследовательского каталога.
    from ai4epi.analysis.analysis_pipeline import (  # type: ignore[no-redef]
        AnalysisOutputConfig,
        AnalysisPipelineSettings,
        AnalysisRunResult,
        AnalysisSourceConfig,
        run_analysis_pipeline,
        run_analysis_pipeline_from_sources,
    )
    from ai4epi.generation.bulletin import Bulletin, save_bulletin, save_bulletin_markdown  # type: ignore[no-redef]
    from ai4epi.core.context import GlobalContext  # type: ignore[no-redef]
    from ai4epi.generation.editor import EditorRunResult, EditorSettings, edit_bulletin  # type: ignore[no-redef]
    from ai4epi.quality.evaluation import EvalConfig, EvalReport, evaluate_bulletin, save_eval_report  # type: ignore[no-redef]
    from ai4epi.core.io import TableReadOptions, read_table, write_json  # type: ignore[no-redef]
    from ai4epi.generation.narrator import NarratorSettings, make_chat_backend  # type: ignore[no-redef]
    from ai4epi.output.pdf import PdfOutputConfig, PdfRenderResult, render_bulletin_pdf  # type: ignore[no-redef]
    from ai4epi.output.figures import build_publication_figure_assets  # type: ignore[no-redef]
    from ai4epi.generation.pipeline import (  # type: ignore[no-redef]
        PipelineOutputConfig,
        PipelineRunResult,
        PipelineSettings,
        run_pipeline_from_files,
    )
    from ai4epi.output.rendering import (  # type: ignore[no-redef]
        RenderOutputConfig,
        RenderedBulletin,
        make_default_render_settings,
        render_bulletin_to_files,
    )


JsonObject = dict[str, Any]
WorkflowStatus = Literal["ok", "partial", "failed"]
WorkflowBulletinKind = Literal["none", "base", "edited"]


class WorkflowError(RuntimeError):
    """Базовая ошибка полного workflow."""


class WorkflowConfigurationError(ValueError):
    """Ошибка конфигурации полного workflow."""


class StrictModel(BaseModel):
    """Базовая pydantic-модель с закрытым контрактом."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


class WorkflowLLMConfig(StrictModel):
    """LLM-настройки верхнеуровневого workflow."""

    model: str = Field(min_length=1)
    backend: str = Field(default="ollama", min_length=1)
    base_url: str = Field(default="http://localhost:11434", min_length=1)
    timeout_sec: int = Field(default=180, gt=0)
    editor_model: str | None = Field(default=None, min_length=1)
    evaluator_model: str | None = Field(default=None, min_length=1)
    reuse_narrator_for_editor: bool = True
    reuse_narrator_for_evaluation: bool = True

    def narrator_backend(self) -> Any:
        """Создать backend для narrator-слоя."""

        return make_chat_backend(
            self.backend,
            model=self.model,
            base_url=self.base_url,
            default_timeout=self.timeout_sec,
        )

    def editor_backend(self) -> Any:
        """Создать backend для editor-pass."""

        model = self.editor_model or (self.model if self.reuse_narrator_for_editor else None)
        if not model:
            raise WorkflowConfigurationError(
                "Для run_editor=True задайте editor_model или reuse_narrator_for_editor=True."
            )
        return make_chat_backend(
            self.backend,
            model=model,
            base_url=self.base_url,
            default_timeout=self.timeout_sec,
        )

    def evaluator_backend(self) -> Any | None:
        """Создать backend для LLM-based evaluation или вернуть None."""

        model = self.evaluator_model or (self.model if self.reuse_narrator_for_evaluation else None)
        if not model:
            return None
        return make_chat_backend(
            self.backend,
            model=model,
            base_url=self.base_url,
            default_timeout=self.timeout_sec,
        )


class FullWorkflowSettings(StrictModel):
    """Настройки полного workflow."""

    run_analysis: bool = True
    run_bulletin: bool = True
    run_editor: bool = True
    run_evaluation: bool = True
    render_markdown_html: bool = True
    render_pdf: bool = False
    raise_on_error: bool = False
    fail_on_evaluation_errors: bool = False
    stop_after_analysis_failure: bool = True
    stop_after_bulletin_failure: bool = True

    @model_validator(mode="after")
    def validate_stage_dependencies(self) -> "FullWorkflowSettings":
        if self.run_editor and not self.run_bulletin:
            raise ValueError("run_editor=True требует run_bulletin=True.")
        if self.run_evaluation and not self.run_bulletin:
            raise ValueError("run_evaluation=True требует run_bulletin=True.")
        if (self.render_markdown_html or self.render_pdf) and not self.run_bulletin:
            raise ValueError("rendering требует run_bulletin=True.")
        return self


class FullWorkflowOutputConfig(StrictModel):
    """Каталоги и имена артефактов полного workflow."""

    output_dir: Path = Path("runs/ai4epi_run")
    analysis_subdir: str = Field(default="analysis", min_length=1)
    bulletin_subdir: str = Field(default="bulletin", min_length=1)
    evaluation_subdir: str = Field(default="evaluation", min_length=1)
    rendering_subdir: str = Field(default="rendered", min_length=1)
    pdf_subdir: str = Field(default="pdf", min_length=1)
    base_pdf_filename: str = Field(default="bulletin_base.pdf", min_length=1)
    edited_pdf_filename: str = Field(default="bulletin_edited.pdf", min_length=1)
    base_pdf_manifest_filename: str = Field(default="pdf_base_manifest.json", min_length=1)
    edited_pdf_manifest_filename: str = Field(default="pdf_edited_manifest.json", min_length=1)
    workflow_report_filename: str = Field(default="workflow_run.json", min_length=1)
    edited_bulletin_json_filename: str = Field(default="bulletin_edited.json", min_length=1)
    edited_bulletin_markdown_filename: str = Field(default="bulletin_edited.md", min_length=1)
    editor_report_filename: str = Field(default="editor_run.json", min_length=1)
    evaluation_report_filename: str = Field(default="evaluation_report.json", min_length=1)
    save_workflow_report: bool = True

    @field_validator(
        "analysis_subdir",
        "bulletin_subdir",
        "evaluation_subdir",
        "rendering_subdir",
        "pdf_subdir",
        "base_pdf_filename",
        "edited_pdf_filename",
        "base_pdf_manifest_filename",
        "edited_pdf_manifest_filename",
        "workflow_report_filename",
        "edited_bulletin_json_filename",
        "edited_bulletin_markdown_filename",
        "editor_report_filename",
        "evaluation_report_filename",
    )
    @classmethod
    def validate_simple_relative_name(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or len(path.parts) != 1:
            raise ValueError("Имена файлов и подкаталогов должны быть простыми относительными именами.")
        return value

    @property
    def analysis_dir(self) -> Path:
        return self.output_dir / self.analysis_subdir

    @property
    def bulletin_dir(self) -> Path:
        return self.output_dir / self.bulletin_subdir

    @property
    def evaluation_dir(self) -> Path:
        return self.output_dir / self.evaluation_subdir

    @property
    def rendering_dir(self) -> Path:
        return self.output_dir / self.rendering_subdir

    @property
    def pdf_dir(self) -> Path:
        return self.output_dir / self.pdf_subdir

    @property
    def workflow_report_path(self) -> Path:
        return self.output_dir / self.workflow_report_filename

    @property
    def edited_bulletin_json_path(self) -> Path:
        return self.bulletin_dir / self.edited_bulletin_json_filename

    @property
    def edited_bulletin_markdown_path(self) -> Path:
        return self.bulletin_dir / self.edited_bulletin_markdown_filename

    @property
    def editor_report_path(self) -> Path:
        return self.bulletin_dir / self.editor_report_filename

    @property
    def evaluation_report_path(self) -> Path:
        return self.evaluation_dir / self.evaluation_report_filename

    def make_analysis_output(self) -> AnalysisOutputConfig:
        return AnalysisOutputConfig(output_dir=self.analysis_dir)

    def make_bulletin_output(self) -> PipelineOutputConfig:
        return PipelineOutputConfig(
            output_dir=self.bulletin_dir,
            base_bulletin_json_filename="bulletin_base.json",
            base_bulletin_markdown_filename="bulletin_base.md",
            edited_bulletin_json_filename=self.edited_bulletin_json_filename,
            edited_bulletin_markdown_filename=self.edited_bulletin_markdown_filename,
            editor_run_filename=self.editor_report_filename,
        )

    def make_render_output(self) -> RenderOutputConfig:
        return RenderOutputConfig(output_dir=self.rendering_dir)

    def make_pdf_output(
        self,
        *,
        pdf_filename: str | None = None,
        manifest_filename: str | None = None,
    ) -> PdfOutputConfig:
        return PdfOutputConfig(
            output_dir=self.pdf_dir,
            pdf_filename=pdf_filename or "bulletin.pdf",
            manifest_filename=manifest_filename or "pdf_manifest.json",
        )

    def make_base_pdf_output(self) -> PdfOutputConfig:
        return self.make_pdf_output(
            pdf_filename=self.base_pdf_filename,
            manifest_filename=self.base_pdf_manifest_filename,
        )

    def make_edited_pdf_output(self) -> PdfOutputConfig:
        return self.make_pdf_output(
            pdf_filename=self.edited_pdf_filename,
            manifest_filename=self.edited_pdf_manifest_filename,
        )


class FullWorkflowRunResult(StrictModel):
    """Структурированный результат полного workflow."""

    status: WorkflowStatus
    duration_sec: float = Field(ge=0.0)
    final_bulletin_kind: WorkflowBulletinKind = "none"
    effective_end_date: date | None = None
    analysis_result: AnalysisRunResult | None = None
    bulletin_result: PipelineRunResult | None = None
    editor_result: EditorRunResult | None = None
    evaluation_report: EvalReport | None = None
    rendered: RenderedBulletin | None = None
    pdf_result: PdfRenderResult | None = None
    base_pdf_result: PdfRenderResult | None = None
    edited_pdf_result: PdfRenderResult | None = None
    context: GlobalContext | None = None
    bulletin: Bulletin | None = None
    artifacts: dict[str, str] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    error_message: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def partial(self) -> bool:
        return self.status == "partial"

    def raise_for_failure(self) -> None:
        """Выбросить исключение, если workflow завершился неуспешно."""

        if self.status == "failed":
            raise WorkflowError(self.error_message or "ai4epi full workflow failed.")

    def to_public_dict(self, *, include_heavy_objects: bool = False) -> JsonObject:
        """Вернуть JSON-сериализуемый отчёт workflow."""

        data: JsonObject = {
            "status": self.status,
            "duration_sec": self.duration_sec,
            "final_bulletin_kind": self.final_bulletin_kind,
            "effective_end_date": self.effective_end_date.isoformat() if self.effective_end_date else None,
            "artifacts": dict(self.artifacts),
            "warnings": list(self.warnings),
            "error_message": self.error_message,
            "has_analysis_result": self.analysis_result is not None,
            "has_bulletin_result": self.bulletin_result is not None,
            "has_editor_result": self.editor_result is not None,
            "has_evaluation_report": self.evaluation_report is not None,
            "has_rendered": self.rendered is not None,
            "has_pdf_result": self.pdf_result is not None,
            "has_base_pdf_result": self.base_pdf_result is not None,
            "has_edited_pdf_result": self.edited_pdf_result is not None,
            "has_context": self.context is not None,
            "has_bulletin": self.bulletin is not None,
        }
        if self.analysis_result is not None:
            data["analysis"] = self.analysis_result.to_public_dict(include_heavy_objects=False)
        if self.bulletin_result is not None:
            data["bulletin_generation"] = self.bulletin_result.to_public_dict(include_bulletins=False)
        if self.editor_result is not None:
            data["editor"] = self.editor_result.to_public_dict()
        if self.evaluation_report is not None:
            data["evaluation"] = self.evaluation_report.to_public_dict()
        if self.rendered is not None:
            data["rendering"] = {
                "output_paths": dict(self.rendered.output_paths),
                "section_count": len(self.rendered.sections),
            }
        if self.pdf_result is not None:
            data["pdf"] = self.pdf_result.to_public_dict()
        if self.base_pdf_result is not None:
            data["pdf_base"] = self.base_pdf_result.to_public_dict()
        if self.edited_pdf_result is not None:
            data["pdf_edited"] = self.edited_pdf_result.to_public_dict()
        if include_heavy_objects:
            data["context"] = self.context.to_public_dict() if self.context is not None else None
            data["bulletin"] = self.bulletin.to_public_dict() if self.bulletin is not None else None
        return data


# ---------------------------------------------------------------------------
# Public source-based workflow
# ---------------------------------------------------------------------------


def run_full_workflow_from_sources(
    *,
    source: AnalysisSourceConfig,
    llm: WorkflowLLMConfig,
    settings: FullWorkflowSettings | None = None,
    output: FullWorkflowOutputConfig | None = None,
    analysis_settings: AnalysisPipelineSettings | None = None,
    pipeline_settings: PipelineSettings | None = None,
    narrator_settings: NarratorSettings | None = None,
    editor_settings: EditorSettings | None = None,
    eval_config: EvalConfig | None = None,
    extra_analysis_kwargs: Mapping[str, Any] | None = None,
) -> FullWorkflowRunResult:
    """Выполнить полный workflow с загрузкой данных из источников.

    Если ``source.end_date is None``, фактическая конечная дата определяется
    после загрузки как последняя неделя в полученном influenza-ряде.
    """

    return _run_full_workflow(
        source=source,
        tables=None,
        llm=llm,
        settings=settings,
        output=output,
        analysis_settings=analysis_settings,
        pipeline_settings=pipeline_settings,
        narrator_settings=narrator_settings,
        editor_settings=editor_settings,
        eval_config=eval_config,
        extra_analysis_kwargs=extra_analysis_kwargs,
    )


def run_full_workflow_from_source_params(
    *,
    city: str,
    model: str,
    begin_year: int = 2011,
    begin_week: int = 1,
    end_date: date | datetime | pd.Timestamp | str | None = None,
    output_dir: str | Path = "runs/ai4epi_run",
    fetch_weather: bool = True,
    run_editor: bool = True,
    run_evaluation: bool = True,
    render_pdf: bool = False,
    backend: str = "ollama",
    base_url: str = "http://localhost:11434",
    timeout_sec: int = 180,
    evaluator_model: str | None = None,
    **kwargs: Any,
) -> FullWorkflowRunResult:
    """Краткая обёртка для типового полного запуска.

    По умолчанию ``end_date=None`` означает «последняя доступная неделя после
    загрузки данных», а не произвольная фиксированная дата в коде.
    """

    return run_full_workflow_from_sources(
        source=AnalysisSourceConfig(
            city=city,
            begin_year=begin_year,
            begin_week=begin_week,
            end_date=end_date,
            fetch_weather=fetch_weather,
        ),
        llm=WorkflowLLMConfig(
            model=model,
            backend=backend,
            base_url=base_url,
            timeout_sec=timeout_sec,
            evaluator_model=evaluator_model,
            reuse_narrator_for_evaluation=evaluator_model is None,
        ),
        settings=FullWorkflowSettings(
            run_editor=run_editor,
            run_evaluation=run_evaluation,
            render_pdf=render_pdf,
        ),
        output=FullWorkflowOutputConfig(output_dir=Path(output_dir)),
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Public table-based workflow
# ---------------------------------------------------------------------------


def run_full_workflow_from_tables(
    *,
    influenza_weekly: pd.DataFrame,
    weather_weekly: pd.DataFrame | None = None,
    hourly_weather: pd.DataFrame | None = None,
    age_group_frame: pd.DataFrame | None = None,
    llm: WorkflowLLMConfig,
    settings: FullWorkflowSettings | None = None,
    output: FullWorkflowOutputConfig | None = None,
    analysis_settings: AnalysisPipelineSettings | None = None,
    pipeline_settings: PipelineSettings | None = None,
    narrator_settings: NarratorSettings | None = None,
    editor_settings: EditorSettings | None = None,
    eval_config: EvalConfig | None = None,
    extra_analysis_kwargs: Mapping[str, Any] | None = None,
) -> FullWorkflowRunResult:
    """Выполнить полный workflow из уже подготовленных DataFrame."""

    tables = {
        "influenza_weekly": influenza_weekly,
        "weather_weekly": weather_weekly,
        "hourly_weather": hourly_weather,
        "age_group_frame": age_group_frame,
    }
    return _run_full_workflow(
        source=None,
        tables=tables,
        llm=llm,
        settings=settings,
        output=output,
        analysis_settings=analysis_settings,
        pipeline_settings=pipeline_settings,
        narrator_settings=narrator_settings,
        editor_settings=editor_settings,
        eval_config=eval_config,
        extra_analysis_kwargs=extra_analysis_kwargs,
    )


def run_full_workflow_from_table_files(
    *,
    influenza_weekly_path: str | Path,
    llm: WorkflowLLMConfig,
    weather_weekly_path: str | Path | None = None,
    hourly_weather_path: str | Path | None = None,
    age_group_frame_path: str | Path | None = None,
    settings: FullWorkflowSettings | None = None,
    output: FullWorkflowOutputConfig | None = None,
    **kwargs: Any,
) -> FullWorkflowRunResult:
    """Загрузить таблицы из файлов и выполнить полный workflow."""

    influenza_weekly = read_table(
        influenza_weekly_path,
        options=TableReadOptions(parse_dates=("datetime",)),
    )
    weather_weekly = (
        read_table(weather_weekly_path, options=TableReadOptions(parse_dates=("week_start",)))
        if weather_weekly_path is not None
        else None
    )
    hourly_weather = (
        read_table(hourly_weather_path, options=TableReadOptions(parse_dates=("time",)))
        if hourly_weather_path is not None
        else None
    )
    age_group_frame = (
        read_table(age_group_frame_path, options=TableReadOptions(parse_dates=("datetime",)))
        if age_group_frame_path is not None
        else None
    )
    return run_full_workflow_from_tables(
        influenza_weekly=influenza_weekly,
        weather_weekly=weather_weekly,
        hourly_weather=hourly_weather,
        age_group_frame=age_group_frame,
        llm=llm,
        settings=settings,
        output=output,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Internal orchestration
# ---------------------------------------------------------------------------


def _run_full_workflow(
    *,
    source: AnalysisSourceConfig | None,
    tables: Mapping[str, Any] | None,
    llm: WorkflowLLMConfig,
    settings: FullWorkflowSettings | None,
    output: FullWorkflowOutputConfig | None,
    analysis_settings: AnalysisPipelineSettings | None,
    pipeline_settings: PipelineSettings | None,
    narrator_settings: NarratorSettings | None,
    editor_settings: EditorSettings | None,
    eval_config: EvalConfig | None,
    extra_analysis_kwargs: Mapping[str, Any] | None,
) -> FullWorkflowRunResult:
    cfg = settings or FullWorkflowSettings()
    out = output or FullWorkflowOutputConfig()
    analysis_cfg = analysis_settings or AnalysisPipelineSettings()
    bulletin_cfg = pipeline_settings or PipelineSettings()
    started = time.perf_counter()

    analysis_result: AnalysisRunResult | None = None
    bulletin_result: PipelineRunResult | None = None
    editor_result: EditorRunResult | None = None
    evaluation_report: EvalReport | None = None
    rendered: RenderedBulletin | None = None
    pdf_result: PdfRenderResult | None = None
    base_pdf_result: PdfRenderResult | None = None
    edited_pdf_result: PdfRenderResult | None = None
    context: GlobalContext | None = None
    base_bulletin: Bulletin | None = None
    final_bulletin: Bulletin | None = None
    final_bulletin_kind: WorkflowBulletinKind = "none"
    effective_end_date: date | None = None
    artifacts: dict[str, str] = {}
    publication_assets: dict[str, Any] = {}
    warnings: list[str] = []
    error_message: str | None = None

    try:
        if cfg.run_analysis:
            analysis_result = _run_analysis_stage(
                source=source,
                tables=tables,
                settings=analysis_cfg,
                output=out.make_analysis_output(),
                extra_kwargs=dict(extra_analysis_kwargs or {}),
            )
            artifacts.update(_prefix_artifacts("analysis", analysis_result.artifacts))
            effective_end_date = _latest_week_date_from_analysis(analysis_result)
            context = analysis_result.context
            if analysis_result.status == "failed":
                raise WorkflowError(analysis_result.error_message or "Analysis stage failed.")
            if analysis_result.status == "partial":
                warnings.append("Analysis stage completed with warnings.")
        else:
            raise WorkflowConfigurationError("run_analysis=False пока не поддерживается: workflow требует GlobalContext из analysis stage.")

        if context is None or analysis_result is None or analysis_result.context_result is None:
            raise WorkflowError("Analysis stage did not produce GlobalContext.")

        if cfg.run_bulletin and (cfg.render_markdown_html or cfg.render_pdf):
            try:
                publication_assets = build_publication_figure_assets(
                    context=context,
                    analysis_artifacts=analysis_result.artifacts,
                    output_dir=out.rendering_dir / "figures",
                )
            except Exception as exc:
                publication_assets = {}
                warnings.append(f"Publication figure generation failed: {exc}")

        context_path = _context_path_from_analysis(analysis_result, out)

        if cfg.run_bulletin:
            bulletin_result = run_pipeline_from_files(
                context_path=context_path,
                model=llm.model,
                output=out.make_bulletin_output(),
                backend=llm.backend,
                base_url=llm.base_url,
                backend_timeout_sec=llm.timeout_sec,
                narrator_settings=narrator_settings,
                pipeline_settings=bulletin_cfg,
                assets=publication_assets,
            )
            artifacts.update(_collect_bulletin_artifacts(bulletin_result))
            if bulletin_result.status == "failed" or bulletin_result.bulletin is None:
                raise WorkflowError(bulletin_result.error_message or "Bulletin generation failed.")

            base_bulletin = bulletin_result.bulletin
            final_bulletin = base_bulletin
            final_bulletin_kind = "base"

            if cfg.run_editor:
                editor_result = edit_bulletin(
                    context=context,
                    bulletin=final_bulletin,
                    llm=llm.editor_backend(),
                    settings=editor_settings or EditorSettings(),
                )
                _save_editor_outputs(editor_result, out)
                artifacts["bulletin.editor_report"] = str(out.editor_report_path.resolve())
                artifacts["bulletin.edited_json"] = str(out.edited_bulletin_json_path.resolve())
                artifacts["bulletin.edited_markdown"] = str(out.edited_bulletin_markdown_path.resolve())

                if editor_result.status == "failed":
                    raise WorkflowError("Editor stage failed.")
                if editor_result.status == "partial":
                    warnings.append("Editor stage completed partially; see editor report.")

                final_bulletin = editor_result.bulletin
                final_bulletin_kind = "edited"

        if final_bulletin is None:
            raise WorkflowError("Workflow did not produce a Bulletin.")

        if cfg.run_evaluation:
            effective_eval_config = eval_config or EvalConfig(llm=llm.evaluator_backend())
            evaluation_report = evaluate_bulletin(context, final_bulletin, config=effective_eval_config)
            out.evaluation_dir.mkdir(parents=True, exist_ok=True)
            save_eval_report(evaluation_report, out.evaluation_report_path)
            artifacts["evaluation.report"] = str(out.evaluation_report_path.resolve())
            if evaluation_report.errors:
                message = "Evaluation found errors."
                if cfg.fail_on_evaluation_errors:
                    raise WorkflowError(message)
                warnings.append(message)

        if cfg.render_markdown_html:
            rendered = render_bulletin_to_files(
                final_bulletin,
                settings=make_default_render_settings(),
                output=out.make_render_output(),
            )
            artifacts.update(_prefix_artifacts("render", rendered.output_paths))

        if cfg.render_pdf:
            if base_bulletin is None:
                raise WorkflowError("PDF rendering requested, but base Bulletin is unavailable.")

            base_pdf_result = render_bulletin_pdf(base_bulletin, output=out.make_base_pdf_output())
            artifacts["pdf.base_file"] = str(base_pdf_result.pdf_path.resolve())
            if base_pdf_result.manifest_path is not None:
                artifacts["pdf.base_manifest"] = str(base_pdf_result.manifest_path.resolve())

            if final_bulletin_kind == "edited" and final_bulletin is not base_bulletin:
                edited_pdf_result = render_bulletin_pdf(final_bulletin, output=out.make_edited_pdf_output())
                artifacts["pdf.edited_file"] = str(edited_pdf_result.pdf_path.resolve())
                if edited_pdf_result.manifest_path is not None:
                    artifacts["pdf.edited_manifest"] = str(edited_pdf_result.manifest_path.resolve())
                pdf_result = edited_pdf_result
            else:
                pdf_result = base_pdf_result

            # Backward-compatible aliases: downstream code that expects a single
            # final PDF can keep reading pdf.file/pdf.manifest.
            artifacts["pdf.file"] = str(pdf_result.pdf_path.resolve())
            if pdf_result.manifest_path is not None:
                artifacts["pdf.manifest"] = str(pdf_result.manifest_path.resolve())

    except Exception as exc:
        error_message = str(exc)
        if cfg.raise_on_error:
            raise

    status = _infer_workflow_status(error_message=error_message, warnings=warnings)
    result = FullWorkflowRunResult(
        status=status,
        duration_sec=round(time.perf_counter() - started, 3),
        final_bulletin_kind=final_bulletin_kind,
        effective_end_date=effective_end_date,
        analysis_result=analysis_result,
        bulletin_result=bulletin_result,
        editor_result=editor_result,
        evaluation_report=evaluation_report,
        rendered=rendered,
        pdf_result=pdf_result,
        base_pdf_result=base_pdf_result,
        edited_pdf_result=edited_pdf_result,
        context=context,
        bulletin=final_bulletin,
        artifacts=artifacts,
        warnings=warnings,
        error_message=error_message,
    )

    if out.save_workflow_report:
        out.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(result.to_public_dict(include_heavy_objects=False), out.workflow_report_path)
        result.artifacts["workflow.report"] = str(out.workflow_report_path.resolve())

    return result


def _run_analysis_stage(
    *,
    source: AnalysisSourceConfig | None,
    tables: Mapping[str, Any] | None,
    settings: AnalysisPipelineSettings,
    output: AnalysisOutputConfig,
    extra_kwargs: Mapping[str, Any],
) -> AnalysisRunResult:
    if source is not None and tables is not None:
        raise WorkflowConfigurationError("Передайте либо source, либо tables, но не оба одновременно.")
    if source is not None:
        return run_analysis_pipeline_from_sources(
            source=source,
            settings=settings,
            output=output,
            **dict(extra_kwargs),
        )
    if tables is None:
        raise WorkflowConfigurationError("Для workflow требуется source или tables.")
    return run_analysis_pipeline(
        influenza_weekly=tables["influenza_weekly"],
        weather_weekly=tables.get("weather_weekly"),
        hourly_weather=tables.get("hourly_weather"),
        age_group_frame=tables.get("age_group_frame"),
        settings=settings,
        output=output,
        **dict(extra_kwargs),
    )


def _save_editor_outputs(editor_result: EditorRunResult, output: FullWorkflowOutputConfig) -> None:
    output.bulletin_dir.mkdir(parents=True, exist_ok=True)
    save_bulletin(editor_result.bulletin, output.edited_bulletin_json_path)
    save_bulletin_markdown(editor_result.bulletin, output.edited_bulletin_markdown_path, include_metadata=True)
    write_json(editor_result.to_public_dict(), output.editor_report_path)


def _context_path_from_analysis(result: AnalysisRunResult, output: FullWorkflowOutputConfig) -> Path:
    if result.context_result is not None and "context" in result.context_result.artifacts:
        return Path(result.context_result.artifacts["context"])
    if "context.context" in result.artifacts:
        return Path(result.artifacts["context.context"])
    return output.analysis_dir / "context_relevant.json"


def _latest_week_date_from_analysis(result: AnalysisRunResult) -> date | None:
    frame = result.merged_weekly
    if frame is None or "datetime" not in frame.columns or frame.empty:
        return None
    dates = pd.to_datetime(frame["datetime"], errors="coerce").dropna()
    if dates.empty:
        return None
    return dates.max().date()


def _collect_bulletin_artifacts(result: PipelineRunResult) -> dict[str, str]:
    paths: dict[str, str] = {}
    raw = result.output_paths.model_dump(mode="json", exclude_none=True)
    for name, path in raw.items():
        if path:
            paths[f"bulletin.{name}"] = str(path)
    return paths


def _prefix_artifacts(prefix: str, artifacts: Mapping[str, Any]) -> dict[str, str]:
    return {f"{prefix}.{key}": str(value) for key, value in artifacts.items()}


def _infer_workflow_status(*, error_message: str | None, warnings: list[str]) -> WorkflowStatus:
    if error_message:
        return "failed"
    if warnings:
        return "partial"
    return "ok"


__all__ = [
    "FullWorkflowOutputConfig",
    "FullWorkflowRunResult",
    "FullWorkflowSettings",
    "WorkflowBulletinKind",
    "WorkflowConfigurationError",
    "WorkflowError",
    "WorkflowLLMConfig",
    "WorkflowStatus",
    "run_full_workflow_from_source_params",
    "run_full_workflow_from_sources",
    "run_full_workflow_from_table_files",
    "run_full_workflow_from_tables",
]

