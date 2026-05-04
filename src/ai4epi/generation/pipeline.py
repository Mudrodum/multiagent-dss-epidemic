"""
End-to-end orchestration layer for the ai4epi bulletin pipeline.

The module coordinates the already separated layers:
- GlobalContext loading and validation;
- SectionRegistry validation;
- section-wise Narrator execution;
- base Bulletin assembly, corresponding to B_decomposed in the notebook;
- deterministic runtime checks for the base Bulletin;
- optional independent plain-language editor pass, corresponding to E_editor;
- deterministic runtime checks for the edited Bulletin, when configured;
- optional JSON/Markdown persistence.

The pipeline deliberately contains no epidemiological heuristics and no
LLM-specific prompt logic. Domain rules live in context, sections,
runtime_checks, narrator and editor modules. Evaluation is intentionally not
called here: it is a downstream diagnostic/comparison layer.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4epi.generation.bulletin import (
    Bulletin,
    BulletinBuilder,
    save_bulletin,
    save_bulletin_markdown,
)
from ai4epi.core.context import GlobalContext, load_global_context
from ai4epi.generation.editor import BulletinEditor, EditorRunResult, EditorSettings
from ai4epi.generation.narrator import (
    ChatBackend,
    Narrator,
    NarratorSettings,
    SectionNarrationResult,
    make_chat_backend,
)
from ai4epi.quality.runtime_checks import RuntimeChecker
from ai4epi.core.sections import SectionRegistry, default_section_registry, load_section_registry


JsonObject = dict[str, Any]
PipelineStatus = Literal["ok", "partial", "failed"]
FinalBulletinKind = Literal["base", "edited", "none"]


class PipelineConfigurationError(ValueError):
    """Invalid pipeline configuration."""


class PipelineExecutionError(RuntimeError):
    """Unrecoverable pipeline execution error."""


class StrictModel(BaseModel):
    """Base pydantic model with a closed contract."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        arbitrary_types_allowed=False,
    )


class PipelineSettings(StrictModel):
    """
    Execution settings for one pipeline run.

    Parameters
    ----------
    validate_registry_against_context:
        Validate that every configured context path can be resolved before
        making any LLM calls.
    fail_fast:
        Stop section generation after the first failed section. The pipeline
        still returns a structured PipelineRunResult instead of hiding partial
        state.
    require_complete_bulletin:
        Require every available required section to be successfully generated
        before the base Bulletin can be assembled.
    validate_section_payloads:
        Re-check every generated section against its output_schema during
        Bulletin assembly.
    run_runtime_checks:
        Run deterministic runtime checks after base Bulletin assembly.
    run_editor:
        Run independent E_editor pass after a successful base Bulletin.
    require_editor_success:
        Treat a partial editor pass as a failed pipeline run. If false, a
        partial editor pass yields a partial pipeline status and keeps the
        edited Bulletin as the final artefact.
    run_runtime_checks_after_edit:
        Run deterministic runtime checks on the edited Bulletin. This setting
        is enforced by the pipeline for the editor instance used in a run.
    """

    validate_registry_against_context: bool = True
    fail_fast: bool = False
    require_complete_bulletin: bool = True
    validate_section_payloads: bool = True
    run_runtime_checks: bool = True
    run_editor: bool = False
    require_editor_success: bool = False
    run_runtime_checks_after_edit: bool = True


class PipelineOutputConfig(StrictModel):
    """Persistence settings for generated pipeline artefacts."""

    output_dir: Path
    base_bulletin_json_filename: str = Field(default="bulletin_base.json", min_length=1)
    base_bulletin_markdown_filename: str = Field(default="bulletin_base.md", min_length=1)
    edited_bulletin_json_filename: str = Field(default="bulletin_edited.json", min_length=1)
    edited_bulletin_markdown_filename: str = Field(default="bulletin_edited.md", min_length=1)
    narration_results_filename: str = Field(default="narration_results.json", min_length=1)
    editor_run_filename: str = Field(default="editor_run.json", min_length=1)
    run_report_filename: str = Field(default="pipeline_run.json", min_length=1)
    save_base_bulletin_json: bool = True
    save_base_bulletin_markdown: bool = True
    save_edited_bulletin_json: bool = True
    save_edited_bulletin_markdown: bool = True
    save_narration_results: bool = True
    save_editor_run: bool = True
    save_run_report: bool = True
    markdown_include_metadata: bool = True

    @field_validator(
        "base_bulletin_json_filename",
        "base_bulletin_markdown_filename",
        "edited_bulletin_json_filename",
        "edited_bulletin_markdown_filename",
        "narration_results_filename",
        "editor_run_filename",
        "run_report_filename",
    )
    @classmethod
    def validate_relative_filename(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or len(path.parts) != 1:
            raise ValueError("Output filenames must be simple relative filenames, not paths.")
        return value

    @model_validator(mode="after")
    def validate_at_least_one_output(self) -> "PipelineOutputConfig":
        if not any(
            (
                self.save_base_bulletin_json,
                self.save_base_bulletin_markdown,
                self.save_edited_bulletin_json,
                self.save_edited_bulletin_markdown,
                self.save_narration_results,
                self.save_editor_run,
                self.save_run_report,
            )
        ):
            raise ValueError("At least one output artefact must be enabled.")
        return self


class PipelineOutputPaths(StrictModel):
    """Paths of artefacts written by one pipeline run."""

    base_bulletin_json: str | None = None
    base_bulletin_markdown: str | None = None
    edited_bulletin_json: str | None = None
    edited_bulletin_markdown: str | None = None
    narration_results_json: str | None = None
    editor_run_json: str | None = None
    run_report_json: str | None = None


class PipelineRunResult(StrictModel):
    """Structured result of one end-to-end pipeline execution."""

    status: PipelineStatus
    duration_sec: float = Field(ge=0.0)
    final_bulletin_kind: FinalBulletinKind = "none"
    generated_section_ids: list[str] = Field(default_factory=list)
    failed_section_ids: list[str] = Field(default_factory=list)
    skipped_section_ids: list[str] = Field(default_factory=list)
    narration_results: dict[str, JsonObject] = Field(default_factory=dict)
    base_bulletin: Bulletin | None = None
    edited_bulletin: Bulletin | None = None
    bulletin: Bulletin | None = None
    editor_result: JsonObject | None = None
    output_paths: PipelineOutputPaths = Field(default_factory=PipelineOutputPaths)
    error_message: str | None = None
    warnings: list[str] = Field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether the run is fully successful."""

        return self.status == "ok"

    @property
    def partial(self) -> bool:
        """Whether the run completed with non-fatal editor-stage issues."""

        return self.status == "partial"

    def to_public_dict(self, *, include_bulletins: bool = True) -> JsonObject:
        """Return a JSON-serializable public representation of the run."""

        data = self.model_dump(mode="json", exclude_none=False)
        if not include_bulletins:
            data["base_bulletin"] = None
            data["edited_bulletin"] = None
            data["bulletin"] = None
        return data

    def raise_for_failure(self) -> None:
        """Raise PipelineExecutionError if the run failed."""

        if self.status == "failed":
            raise PipelineExecutionError(self.error_message or "ai4epi pipeline failed.")


class Ai4EpiPipeline:
    """
    End-to-end executor for section generation, Bulletin assembly and optional editor-pass.

    The class is intentionally thin: it coordinates explicit contracts instead
    of embedding hidden defaults. New sections are added through SectionRegistry,
    not by changing this class.
    """

    def __init__(
        self,
        *,
        narrator: Narrator,
        registry: SectionRegistry | None = None,
        builder: BulletinBuilder | None = None,
        runtime_checker: RuntimeChecker | None = None,
        editor: BulletinEditor | None = None,
        settings: PipelineSettings | None = None,
    ) -> None:
        self.narrator = narrator
        self.registry = registry or default_section_registry()
        self.runtime_checker = runtime_checker or RuntimeChecker()
        self.builder = builder or BulletinBuilder(
            registry=self.registry,
            runtime_checker=self.runtime_checker,
        )
        self.editor = editor
        self.settings = settings or PipelineSettings()

    def run(
        self,
        context: GlobalContext | Mapping[str, Any],
        *,
        feedback_by_section: Mapping[str, str] | None = None,
        output: PipelineOutputConfig | None = None,
        assets: Mapping[str, Any] | None = None,
        metadata_extra: Mapping[str, Any] | None = None,
        editorial_notes: Sequence[str] | None = None,
    ) -> PipelineRunResult:
        """
        Execute the pipeline on an already loaded GlobalContext.

        Ordinary LLM, validation and runtime-check failures are reflected in
        PipelineRunResult. Partial artefacts are kept whenever they can be
        represented safely.
        """

        started = time.perf_counter()
        ctx: GlobalContext | None = None
        results: dict[str, SectionNarrationResult] = {}
        base_bulletin: Bulletin | None = None
        edited_bulletin: Bulletin | None = None
        editor_run: EditorRunResult | None = None
        error_message: str | None = None
        warnings: list[str] = []

        try:
            ctx = context if isinstance(context, GlobalContext) else GlobalContext.model_validate(context)

            if self.settings.validate_registry_against_context:
                self.registry.validate_against_context(ctx)

            results = self._generate_sections(ctx, feedback_by_section=feedback_by_section)

            base_bulletin = self.builder.build_from_narration_results(
                context=ctx,
                results=results,
                run_runtime_checks=self.settings.run_runtime_checks,
                require_complete=self.settings.require_complete_bulletin,
                validate_section_payloads=self.settings.validate_section_payloads,
                assets=assets,
                metadata_extra=metadata_extra,
                editorial_notes=editorial_notes,
            )

            if base_bulletin.status == "failed":
                error_message = _summarize_bulletin_failure(base_bulletin, prefix="Base Bulletin")
            elif self.settings.run_editor:
                if self.editor is None:
                    raise PipelineConfigurationError(
                        "PipelineSettings.run_editor=True requires a configured BulletinEditor."
                    )
                editor_run = self._run_editor(ctx, base_bulletin)
                edited_bulletin = editor_run.bulletin

                if editor_run.status == "failed":
                    error_message = _summarize_editor_failure(editor_run)
                elif editor_run.status == "partial":
                    message = _summarize_editor_failure(editor_run)
                    if self.settings.require_editor_success:
                        error_message = message
                    else:
                        warnings.append(message)

                if edited_bulletin.status == "failed":
                    message = _summarize_bulletin_failure(edited_bulletin, prefix="Edited Bulletin")
                    if error_message is None:
                        error_message = message

        except Exception as exc:
            error_message = str(exc)

        generated_section_ids = _generated_section_ids(results)
        failed_section_ids = _failed_section_ids(results)
        skipped_section_ids = self._skipped_section_ids(ctx, results) if ctx is not None else []
        final_bulletin = edited_bulletin if edited_bulletin is not None else base_bulletin
        final_bulletin_kind: FinalBulletinKind
        if edited_bulletin is not None:
            final_bulletin_kind = "edited"
        elif base_bulletin is not None:
            final_bulletin_kind = "base"
        else:
            final_bulletin_kind = "none"

        status = self._infer_status(
            error_message=error_message,
            base_bulletin=base_bulletin,
            editor_run=editor_run,
            edited_bulletin=edited_bulletin,
            warnings=warnings,
        )

        result = PipelineRunResult(
            status=status,
            duration_sec=round(time.perf_counter() - started, 3),
            final_bulletin_kind=final_bulletin_kind,
            generated_section_ids=generated_section_ids,
            failed_section_ids=failed_section_ids,
            skipped_section_ids=skipped_section_ids,
            narration_results={
                section_id: _narration_result_to_public_dict(result)
                for section_id, result in results.items()
            },
            base_bulletin=base_bulletin,
            edited_bulletin=edited_bulletin,
            bulletin=final_bulletin,
            editor_result=editor_run.to_public_dict() if editor_run is not None else None,
            error_message=error_message,
            warnings=warnings,
        )

        if output is not None:
            result.output_paths = self.save_outputs(result=result, output=output)

        return result

    def run_from_files(
        self,
        *,
        context_path: str | Path,
        output: PipelineOutputConfig | None = None,
        feedback_by_section: Mapping[str, str] | None = None,
        assets: Mapping[str, Any] | None = None,
        metadata_extra: Mapping[str, Any] | None = None,
        editorial_notes: Sequence[str] | None = None,
    ) -> PipelineRunResult:
        """Load GlobalContext from JSON and execute the pipeline."""

        context = load_global_context(context_path)
        return self.run(
            context,
            feedback_by_section=feedback_by_section,
            output=output,
            assets=assets,
            metadata_extra=metadata_extra,
            editorial_notes=editorial_notes,
        )

    def _generate_sections(
        self,
        context: GlobalContext,
        *,
        feedback_by_section: Mapping[str, str] | None = None,
    ) -> dict[str, SectionNarrationResult]:
        feedback_by_section = feedback_by_section or {}
        results: dict[str, SectionNarrationResult] = {}

        for section in self.registry.available_sections(context):
            result = self.narrator.generate_section(
                section,
                context,
                feedback=feedback_by_section.get(section.section_id),
            )
            results[section.section_id] = result
            if self.settings.fail_fast and result.status != "ok":
                break

        return results

    def _run_editor(self, context: GlobalContext, bulletin: Bulletin) -> EditorRunResult:
        assert self.editor is not None
        original_flag = self.editor.settings.run_runtime_checks_after_edit
        self.editor.settings.run_runtime_checks_after_edit = self.settings.run_runtime_checks_after_edit
        try:
            return self.editor.edit_bulletin(context=context, bulletin=bulletin)
        finally:
            self.editor.settings.run_runtime_checks_after_edit = original_flag

    def _skipped_section_ids(
        self,
        context: GlobalContext | None,
        results: Mapping[str, SectionNarrationResult],
    ) -> list[str]:
        if context is None:
            return []
        available_ids = [section.section_id for section in self.registry.available_sections(context)]
        return [section_id for section_id in available_ids if section_id not in results]

    def _infer_status(
        self,
        *,
        error_message: str | None,
        base_bulletin: Bulletin | None,
        editor_run: EditorRunResult | None,
        edited_bulletin: Bulletin | None,
        warnings: Sequence[str],
    ) -> PipelineStatus:
        if error_message is not None:
            return "failed"
        if base_bulletin is None:
            return "failed"
        if base_bulletin.status == "failed":
            return "failed"
        if self.settings.run_editor:
            if editor_run is None or edited_bulletin is None:
                return "failed"
            if editor_run.status == "failed" or edited_bulletin.status == "failed":
                return "failed"
            if editor_run.status == "partial" or warnings:
                return "partial"
        return "ok"

    def save_outputs(self, *, result: PipelineRunResult, output: PipelineOutputConfig) -> PipelineOutputPaths:
        """Persist available artefacts according to PipelineOutputConfig."""

        output.output_dir.mkdir(parents=True, exist_ok=True)
        paths = PipelineOutputPaths()

        if result.base_bulletin is not None:
            if output.save_base_bulletin_json:
                path = output.output_dir / output.base_bulletin_json_filename
                save_bulletin(result.base_bulletin, path)
                paths.base_bulletin_json = str(path)

            if output.save_base_bulletin_markdown:
                path = output.output_dir / output.base_bulletin_markdown_filename
                save_bulletin_markdown(
                    result.base_bulletin,
                    path,
                    include_metadata=output.markdown_include_metadata,
                )
                paths.base_bulletin_markdown = str(path)

        if result.edited_bulletin is not None:
            if output.save_edited_bulletin_json:
                path = output.output_dir / output.edited_bulletin_json_filename
                save_bulletin(result.edited_bulletin, path)
                paths.edited_bulletin_json = str(path)

            if output.save_edited_bulletin_markdown:
                path = output.output_dir / output.edited_bulletin_markdown_filename
                save_bulletin_markdown(
                    result.edited_bulletin,
                    path,
                    include_metadata=output.markdown_include_metadata,
                )
                paths.edited_bulletin_markdown = str(path)

        if output.save_narration_results and result.narration_results:
            path = output.output_dir / output.narration_results_filename
            _write_json(path, result.narration_results)
            paths.narration_results_json = str(path)

        if output.save_editor_run and result.editor_result is not None:
            path = output.output_dir / output.editor_run_filename
            _write_json(path, result.editor_result)
            paths.editor_run_json = str(path)

        if output.save_run_report:
            path = output.output_dir / output.run_report_filename
            report = result.to_public_dict(include_bulletins=False)
            report["output_paths"] = paths.model_dump(mode="json", exclude_none=False)
            _write_json(path, report)
            paths.run_report_json = str(path)

        return paths


class PipelineFactoryConfig(StrictModel):
    """Declarative configuration for constructing Ai4EpiPipeline."""

    backend: str = Field(default="ollama", min_length=1)
    model: str = Field(min_length=1)
    base_url: str = Field(default="http://localhost:11434", min_length=1)
    backend_timeout_sec: int = Field(default=180, gt=0)
    narrator_settings: NarratorSettings = Field(default_factory=NarratorSettings)
    pipeline_settings: PipelineSettings = Field(default_factory=PipelineSettings)
    editor_model: str | None = None
    editor_settings: EditorSettings = Field(default_factory=EditorSettings)



def create_pipeline(
    *,
    llm: ChatBackend | None = None,
    editor_llm: ChatBackend | None = None,
    backend: str = "ollama",
    model: str | None = None,
    base_url: str = "http://localhost:11434",
    backend_timeout_sec: int = 180,
    editor_backend: str | None = None,
    editor_model: str | None = None,
    editor_base_url: str | None = None,
    editor_backend_timeout_sec: int | None = None,
    narrator_settings: NarratorSettings | None = None,
    editor_settings: EditorSettings | None = None,
    pipeline_settings: PipelineSettings | None = None,
    registry: SectionRegistry | None = None,
    runtime_checker: RuntimeChecker | None = None,
) -> Ai4EpiPipeline:
    """
    Construct Ai4EpiPipeline from explicit ChatBackend objects or backend names.

    If the editor is enabled and ``editor_llm`` is not provided, the narrator
    backend is reused unless ``editor_model`` is specified.
    """

    pipeline_settings = pipeline_settings or PipelineSettings()
    narrator_settings = narrator_settings or NarratorSettings()
    editor_settings = editor_settings or EditorSettings()

    if llm is None:
        if model is None:
            raise PipelineConfigurationError("Either llm or model must be provided.")
        llm = make_chat_backend(
            backend,
            model=model,
            base_url=base_url,
            default_timeout=backend_timeout_sec,
        )

    editor: BulletinEditor | None = None
    if pipeline_settings.run_editor:
        if editor_llm is None:
            if editor_model is not None:
                editor_llm = make_chat_backend(
                    editor_backend or backend,
                    model=editor_model,
                    base_url=editor_base_url or base_url,
                    default_timeout=editor_backend_timeout_sec or backend_timeout_sec,
                )
            else:
                editor_llm = llm
        editor = BulletinEditor(
            editor_llm,
            registry=registry,
            settings=editor_settings,
            runtime_checker=runtime_checker,
        )

    narrator = Narrator(llm=llm, settings=narrator_settings)
    return Ai4EpiPipeline(
        narrator=narrator,
        registry=registry,
        runtime_checker=runtime_checker,
        editor=editor,
        settings=pipeline_settings,
    )



def create_pipeline_from_config(
    config: PipelineFactoryConfig,
    *,
    registry: SectionRegistry | None = None,
    runtime_checker: RuntimeChecker | None = None,
) -> Ai4EpiPipeline:
    """Construct Ai4EpiPipeline from PipelineFactoryConfig."""

    return create_pipeline(
        backend=config.backend,
        model=config.model,
        base_url=config.base_url,
        backend_timeout_sec=config.backend_timeout_sec,
        editor_model=config.editor_model,
        narrator_settings=config.narrator_settings,
        editor_settings=config.editor_settings,
        pipeline_settings=config.pipeline_settings,
        registry=registry,
        runtime_checker=runtime_checker,
    )



def load_pipeline_registry(path: str | Path | None) -> SectionRegistry:
    """Load a section registry from path, or return the default registry."""

    if path is None:
        return default_section_registry()
    return load_section_registry(path)



def run_pipeline_from_files(
    *,
    context_path: str | Path,
    model: str,
    registry_path: str | Path | None = None,
    output: PipelineOutputConfig | None = None,
    backend: str = "ollama",
    base_url: str = "http://localhost:11434",
    backend_timeout_sec: int = 180,
    editor_model: str | None = None,
    editor_backend: str | None = None,
    editor_base_url: str | None = None,
    editor_backend_timeout_sec: int | None = None,
    narrator_settings: NarratorSettings | None = None,
    editor_settings: EditorSettings | None = None,
    pipeline_settings: PipelineSettings | None = None,
    runtime_checker: RuntimeChecker | None = None,
    feedback_by_section: Mapping[str, str] | None = None,
    assets: Mapping[str, Any] | None = None,
    metadata_extra: Mapping[str, Any] | None = None,
    editorial_notes: Sequence[str] | None = None,
) -> PipelineRunResult:
    """Functional helper for the standard file-based pipeline run."""

    registry = load_pipeline_registry(registry_path)
    pipeline = create_pipeline(
        backend=backend,
        model=model,
        base_url=base_url,
        backend_timeout_sec=backend_timeout_sec,
        editor_backend=editor_backend,
        editor_model=editor_model,
        editor_base_url=editor_base_url,
        editor_backend_timeout_sec=editor_backend_timeout_sec,
        narrator_settings=narrator_settings,
        editor_settings=editor_settings,
        pipeline_settings=pipeline_settings,
        registry=registry,
        runtime_checker=runtime_checker,
    )
    return pipeline.run_from_files(
        context_path=context_path,
        output=output,
        feedback_by_section=feedback_by_section,
        assets=assets,
        metadata_extra=metadata_extra,
        editorial_notes=editorial_notes,
    )



def section_ids_from_registry(registry: SectionRegistry, context: GlobalContext) -> list[str]:
    """Return available section ids for a given context in execution order."""

    return [section.section_id for section in registry.available_sections(context)]



def _generated_section_ids(results: Mapping[str, SectionNarrationResult]) -> list[str]:
    return [
        section_id
        for section_id, result in results.items()
        if result.status == "ok" and result.draft is not None
    ]



def _failed_section_ids(results: Mapping[str, SectionNarrationResult]) -> list[str]:
    return [
        section_id
        for section_id, result in results.items()
        if result.status != "ok" or result.draft is None
    ]



def _narration_result_to_public_dict(result: SectionNarrationResult) -> JsonObject:
    return {
        "section_id": result.section_id,
        "status": result.status,
        "draft": result.draft,
        "trace": result.trace,
        "failure_reason": result.failure_reason,
    }



def _summarize_bulletin_failure(bulletin: Bulletin, *, prefix: str = "Bulletin") -> str:
    if bulletin.failures_by_section:
        failed = ", ".join(sorted(bulletin.failures_by_section))
        return f"{prefix} has failed sections: {failed}."
    if bulletin.runtime_check_report and bulletin.runtime_check_report.errors:
        errors = bulletin.runtime_check_report.errors[:5]
        messages = "; ".join(f"{issue.section_id}/{issue.check_name}: {issue.message}" for issue in errors)
        return f"{prefix} runtime checks failed: {messages}"
    return f"{prefix} status is failed."



def _summarize_editor_failure(editor_run: EditorRunResult) -> str:
    if not editor_run.errors:
        return f"Editor finished with status {editor_run.status}."
    shown = list(editor_run.errors.items())[:5]
    messages = "; ".join(f"{key}: {value}" for key, value in shown)
    return f"Editor finished with status {editor_run.status}: {messages}"



def _write_json(path: str | Path, data: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, default=_json_default)



def _json_default(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")


__all__ = [
    "Ai4EpiPipeline",
    "FinalBulletinKind",
    "JsonObject",
    "PipelineConfigurationError",
    "PipelineExecutionError",
    "PipelineFactoryConfig",
    "PipelineOutputConfig",
    "PipelineOutputPaths",
    "PipelineRunResult",
    "PipelineSettings",
    "PipelineStatus",
    "create_pipeline",
    "create_pipeline_from_config",
    "load_pipeline_registry",
    "run_pipeline_from_files",
    "section_ids_from_registry",
]

