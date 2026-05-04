"""
Командный интерфейс ai4epi.

CLI является тонким orchestration-слоем поверх уже реализованных модулей. Он
не содержит доменной логики, не меняет параметры моделей неявно и не дублирует
пайплайны. Каждая команда вызывает публичные функции соответствующего слоя:

* analysis_pipeline.py — численный pipeline до GlobalContext;
* config.py / pipeline.py — LLM/bulletin pipeline;
* evaluation.py — downstream evaluation;
* rendering.py — Markdown/HTML rendering;
* pdf.py — PDF rendering.

Модуль рассчитан на подключение через ``pyproject.toml`` entry point:
``ai4epi = ai4epi.cli:main``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable, Sequence

try:  # Пакетный импорт.
    from ai4epi.analysis.analysis_pipeline import (
        AnalysisOutputConfig,
        AnalysisPipelineSettings,
        AnalysisSourceConfig,
        run_analysis_pipeline,
        run_analysis_pipeline_from_sources,
    )
    from ai4epi.core.config import (
        default_config,
        load_ai4epi_config,
        run_pipeline_from_config,
        save_ai4epi_config,
    )
    from ai4epi.quality.evaluation import EvalConfig, evaluate_bulletin_from_files, print_eval_report, save_eval_report
    from ai4epi.generation.narrator import make_chat_backend
    from ai4epi.core.io import TableReadOptions, read_table, write_json
    from ai4epi.output.pdf import PdfFontConfig, PdfOutputConfig, PdfSettings, render_bulletin_pdf_file
    from ai4epi.output.rendering import RenderOutputConfig, make_default_render_settings, render_bulletin_file
    from ai4epi.orchestration.workflow import (
        FullWorkflowOutputConfig,
        FullWorkflowSettings,
        WorkflowLLMConfig,
        run_full_workflow_from_sources,
    )
except ImportError:  # pragma: no cover - поддержка запуска из плоского исследовательского каталога.
    from ai4epi.analysis.analysis_pipeline import (  # type: ignore[no-redef]
        AnalysisOutputConfig,
        AnalysisPipelineSettings,
        AnalysisSourceConfig,
        run_analysis_pipeline,
        run_analysis_pipeline_from_sources,
    )
    from ai4epi.core.config import (  # type: ignore[no-redef]
        default_config,
        load_ai4epi_config,
        run_pipeline_from_config,
        save_ai4epi_config,
    )
    from ai4epi.quality.evaluation import EvalConfig, evaluate_bulletin_from_files, print_eval_report, \
        save_eval_report  # type: ignore[no-redef]
    from ai4epi.core.io import TableReadOptions, read_table, write_json  # type: ignore[no-redef]
    from ai4epi.generation.narrator import make_chat_backend  # type: ignore[no-redef]
    from ai4epi.output.pdf import PdfFontConfig, PdfOutputConfig, PdfSettings, render_bulletin_pdf_file  # type: ignore[no-redef]
    from ai4epi.output.rendering import RenderOutputConfig, make_default_render_settings, render_bulletin_file  # type: ignore[no-redef]
    from ai4epi.orchestration.workflow import (  # type: ignore[no-redef]
        FullWorkflowOutputConfig,
        FullWorkflowSettings,
        WorkflowLLMConfig,
        run_full_workflow_from_sources,
    )


JsonObject = dict[str, Any]
CommandHandler = Callable[[argparse.Namespace], int]


EXIT_OK = 0
EXIT_FAILED = 2
EXIT_USAGE_ERROR = 64
DEFAULT_EVALUATOR_BACKEND = "ollama"
DEFAULT_EVALUATOR_MODEL = "gemma3:12b"
DEFAULT_EVALUATOR_TIMEOUT_SEC = 600


def main(argv: Sequence[str] | None = None) -> int:
    """Точка входа CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help(sys.stderr)
        return EXIT_USAGE_ERROR

    try:
        return int(handler(args))
    except KeyboardInterrupt:
        print("Выполнение прервано пользователем.", file=sys.stderr)
        return EXIT_FAILED
    except Exception as exc:  # CLI должен возвращать явный код и текст ошибки.
        print(f"Ошибка: {exc}", file=sys.stderr)
        if getattr(args, "debug", False):
            raise
        return EXIT_FAILED


def build_parser() -> argparse.ArgumentParser:
    """Построить argparse-парсер."""

    parser = argparse.ArgumentParser(
        prog="ai4epi",
        description="CLI для численного и LLM-пайплайнов ai4epi.",
    )
    parser.add_argument("--debug", action="store_true", help="Пробрасывать исключения вместо компактного сообщения.")

    subparsers = parser.add_subparsers(dest="command", required=True)

    _add_init_config_command(subparsers)
    _add_run_analysis_source_command(subparsers)
    _add_run_analysis_tables_command(subparsers)
    _add_run_all_command(subparsers)
    _add_generate_bulletin_command(subparsers)
    _add_evaluate_bulletin_command(subparsers)
    _add_render_command(subparsers)
    _add_render_pdf_command(subparsers)

    return parser


# ---------------------------------------------------------------------------
# Command registration
# ---------------------------------------------------------------------------


def _add_init_config_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "init-config",
        help="Создать минимальный JSON/YAML-конфиг для LLM/bulletin pipeline.",
    )
    parser.add_argument("--context", required=True, type=Path, help="Путь к context_relevant.json.")
    parser.add_argument("--model", required=True, help="Имя LLM-модели, например qwen3.5:9b.")
    parser.add_argument("--output-dir", default="outputs", type=Path, help="Каталог артефактов bulletin pipeline.")
    parser.add_argument("--registry", default=None, type=Path, help="Путь к пользовательскому реестру секций.")
    parser.add_argument("--run-editor", action="store_true", help="Включить editor-pass в создаваемом конфиге.")
    parser.add_argument("--config-out", required=True, type=Path, help="Куда сохранить конфиг (.json/.yaml/.yml).")
    parser.set_defaults(handler=_cmd_init_config)


def _add_run_analysis_source_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "run-analysis-source",
        help="Запустить численный pipeline с загрузкой influenza/weather из источников.",
    )
    parser.add_argument("--city", required=True, help="Город: например spb, moscow или русское название.")
    parser.add_argument("--begin-year", type=int, default=2011, help="Первый год загрузки influenza DB. По умолчанию: 2011.")
    parser.add_argument("--begin-week", type=int, default=1, help="Первая ISO-неделя загрузки influenza DB. По умолчанию: 1.")
    parser.add_argument("--end-date", default=None, help="Последняя дата периода, например 2025-04-01.")
    parser.add_argument("--output-dir", default="analysis_outputs", type=Path, help="Каталог численных артефактов.")
    parser.add_argument("--no-weather", action="store_true", help="Не загружать Open-Meteo погоду.")
    parser.add_argument("--weather-latitude", type=float, default=None, help="Явная широта погодной точки.")
    parser.add_argument("--weather-longitude", type=float, default=None, help="Явная долгота погодной точки.")
    parser.add_argument("--weather-timezone", default=None, help="Timezone погодной точки, например Europe/Moscow.")
    parser.add_argument("--no-context", action="store_true", help="Не собирать GlobalContext после численных этапов.")
    parser.add_argument("--no-explainability", action="store_true", help="Пропустить SHAP-анализ.")
    parser.add_argument("--no-epidemic-waves", action="store_true", help="Пропустить анализ эпидемических волн.")
    parser.add_argument("--no-age-group", action="store_true", help="Пропустить возрастной сезонный анализ.")
    parser.add_argument("--require-age-group", action="store_true", help="Считать отсутствие возрастного блока ошибкой.")
    parser.add_argument("--raise-on-error", action="store_true", help="Пробрасывать ошибку из analysis pipeline.")
    parser.set_defaults(handler=_cmd_run_analysis_source)


def _add_run_analysis_tables_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "run-analysis-tables",
        help="Запустить численный pipeline по готовым таблицам.",
    )
    parser.add_argument("--influenza-weekly", required=True, type=Path, help="Таблица weekly influenza с datetime/inc_per_10k.")
    parser.add_argument("--weather-weekly", default=None, type=Path, help="Таблица weekly weather с week_start/temp_mean/...")
    parser.add_argument("--hourly-weather", default=None, type=Path, help="Почасовая weather-таблица time/temp/rh.")
    parser.add_argument("--age-group-frame", default=None, type=Path, help="Таблица возрастных групп.")
    parser.add_argument("--output-dir", default="analysis_outputs", type=Path, help="Каталог численных артефактов.")
    parser.add_argument("--no-weather-required", action="store_true", help="Разрешить отсутствие погодной таблицы.")
    parser.add_argument("--no-context", action="store_true", help="Не собирать GlobalContext после численных этапов.")
    parser.add_argument("--no-explainability", action="store_true", help="Пропустить SHAP-анализ.")
    parser.add_argument("--no-epidemic-waves", action="store_true", help="Пропустить анализ эпидемических волн.")
    parser.add_argument("--no-age-group", action="store_true", help="Пропустить возрастной сезонный анализ.")
    parser.add_argument("--require-age-group", action="store_true", help="Считать отсутствие возрастного блока ошибкой.")
    parser.add_argument("--raise-on-error", action="store_true", help="Пробрасывать ошибку из analysis pipeline.")
    parser.set_defaults(handler=_cmd_run_analysis_tables)



def _add_run_all_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "run-all",
        help="Запустить полный workflow: данные → прогноз → SHAP → контекст → бюллетень → editor/evaluation/rendering.",
    )
    parser.add_argument("--city", required=True, help="Город: например spb, moscow или русское название.")
    parser.add_argument("--begin-year", type=int, default=2011, help="Первый год загрузки influenza DB. По умолчанию: 2011.")
    parser.add_argument("--begin-week", type=int, default=1, help="Первая ISO-неделя загрузки influenza DB. По умолчанию: 1.")
    parser.add_argument(
        "--end-date",
        default=None,
        help=(
            "Последняя дата периода. Если не задана, workflow использует последнюю "
            "фактически доступную неделю после загрузки данных."
        ),
    )
    parser.add_argument("--model", required=True, help="Имя LLM-модели, например qwen3.5:9b.")
    parser.add_argument(
        "--evaluator-model",
        default=os.getenv("EVAL_OLLAMA_MODEL", DEFAULT_EVALUATOR_MODEL),
        help="Отдельная LLM-модель для evaluation-layer. По умолчанию: %(default)s.",
    )
    parser.add_argument(
        "--backend",
        default=os.getenv("EVAL_LLM_BACKEND", DEFAULT_EVALUATOR_BACKEND),
        help="LLM backend. Сейчас поддерживается ollama.",
    )
    parser.add_argument("--base-url", default="http://localhost:11434", help="Base URL LLM backend.")
    parser.add_argument("--timeout-sec", type=int, default=180, help="HTTP timeout для LLM backend.")
    parser.add_argument("--output-dir", default="runs/ai4epi_run", type=Path, help="Корневой каталог полного запуска.")
    parser.add_argument("--no-weather", action="store_true", help="Не загружать Open-Meteo погоду.")
    parser.add_argument("--no-editor", action="store_true", help="Не запускать editor-pass.")
    parser.add_argument("--no-evaluation", action="store_true", help="Не запускать downstream evaluation.")
    parser.add_argument("--render-pdf", action="store_true", help="Дополнительно сформировать PDF.")
    parser.add_argument("--raise-on-error", action="store_true", help="Пробрасывать исключения из workflow.")
    parser.add_argument("--fail-on-evaluation-errors", action="store_true", help="Считать ошибки evaluation фатальными.")
    parser.set_defaults(handler=_cmd_run_all)

def _add_generate_bulletin_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "generate-bulletin",
        help="Запустить LLM/bulletin pipeline из JSON/YAML-конфига.",
    )
    parser.add_argument("--config", required=True, type=Path, help="Ai4EpiConfig JSON/YAML.")
    parser.argument_default = None
    parser.set_defaults(handler=_cmd_generate_bulletin)


def _add_evaluate_bulletin_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "evaluate-bulletin",
        help="Оценить уже готовый бюллетень как downstream diagnostic layer.",
    )
    parser.add_argument("--context", required=True, type=Path, help="Путь к context_relevant.json.")
    parser.add_argument("--bulletin", required=True, type=Path, help="Путь к bulletin JSON.")
    parser.add_argument("--config", default=None, type=Path, help="Опциональный Ai4EpiConfig для EvalConfig/LLM evaluator.")
    parser.add_argument(
        "--evaluator-model",
        default=None,
        help=(
            "LLM-модель для evaluator-ов. Если не задана, используется llm.evaluator/llm.narrator "
            "из --config, иначе EVAL_OLLAMA_MODEL или gemma3:12b."
        ),
    )
    parser.add_argument(
        "--backend",
        default=None,
        help="LLM backend для standalone evaluation. Если не задан, используется --config или EVAL_LLM_BACKEND/ollama.",
    )
    parser.add_argument("--base-url", default=None, help="Base URL LLM backend для standalone evaluation.")
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=None,
        help="HTTP timeout backend-а для standalone evaluation. По умолчанию: EVAL_REQUEST_TIMEOUT_SEC или 600.",
    )
    parser.add_argument(
        "--request-timeout-sec",
        type=int,
        default=None,
        help="Таймаут одного LLM-запроса внутри evaluator-а. По умолчанию совпадает с --timeout-sec.",
    )
    parser.add_argument("--no-llm", action="store_true", help="Явно отключить LLM-based evaluator-ы.")
    parser.add_argument("--output", default=None, type=Path, help="Куда сохранить evaluation_report.json.")
    parser.add_argument("--quiet", action="store_true", help="Не печатать человекочитаемую сводку.")
    parser.set_defaults(handler=_cmd_evaluate_bulletin)


def _add_render_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "render",
        help="Сформировать Markdown/HTML из Bulletin JSON.",
    )
    parser.add_argument("--bulletin", required=True, type=Path, help="Путь к bulletin JSON.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Каталог rendering-артефактов.")
    parser.add_argument("--no-markdown", action="store_true", help="Не сохранять Markdown.")
    parser.add_argument("--no-html", action="store_true", help="Не сохранять HTML.")
    parser.add_argument("--no-manifest", action="store_true", help="Не сохранять render_manifest.json.")
    parser.add_argument("--no-assets", action="store_true", help="Не включать assets в rendering settings.")
    parser.set_defaults(handler=_cmd_render)


def _add_render_pdf_command(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser(
        "render-pdf",
        help="Сформировать PDF из Bulletin JSON.",
    )
    parser.add_argument("--bulletin", required=True, type=Path, help="Путь к bulletin JSON.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Каталог PDF-артефактов.")
    parser.add_argument("--pdf-filename", default="bulletin.pdf", help="Имя PDF-файла.")
    parser.add_argument("--font-regular", default=None, type=Path, help="Путь к regular TTF-шрифту с кириллицей.")
    parser.add_argument("--font-bold", default=None, type=Path, help="Путь к bold TTF-шрифту с кириллицей.")
    parser.add_argument("--no-manifest", action="store_true", help="Не сохранять pdf_manifest.json.")
    parser.set_defaults(handler=_cmd_render_pdf)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


def _cmd_init_config(args: argparse.Namespace) -> int:
    config = default_config(
        context_path=args.context,
        model=args.model,
        output_dir=args.output_dir,
        registry_path=args.registry,
        run_editor=bool(args.run_editor),
    )
    save_ai4epi_config(config, args.config_out)
    _print_json({"status": "ok", "config_path": str(Path(args.config_out).resolve())})
    return EXIT_OK


def _cmd_run_analysis_source(args: argparse.Namespace) -> int:
    settings = AnalysisPipelineSettings(
        require_weather=not bool(args.no_weather),
        run_explainability=not bool(args.no_explainability),
        run_epidemic_waves=not bool(args.no_epidemic_waves),
        run_age_group_season=not bool(args.no_age_group),
        require_age_group_season=bool(args.require_age_group),
        build_context=not bool(args.no_context),
        raise_on_error=bool(args.raise_on_error),
    )
    source = AnalysisSourceConfig(
        city=args.city,
        begin_year=args.begin_year,
        begin_week=args.begin_week,
        end_date=args.end_date,
        fetch_weather=not bool(args.no_weather),
        weather_latitude=args.weather_latitude,
        weather_longitude=args.weather_longitude,
        weather_timezone=args.weather_timezone,
    )
    result = run_analysis_pipeline_from_sources(
        source=source,
        settings=settings,
        output=AnalysisOutputConfig(output_dir=args.output_dir),
    )
    _print_json(result.to_public_dict(include_heavy_objects=False))
    return EXIT_OK if result.status != "failed" else EXIT_FAILED


def _cmd_run_analysis_tables(args: argparse.Namespace) -> int:
    influenza_weekly = _read_table_with_dates(args.influenza_weekly, parse_dates=("datetime",))
    weather_weekly = _read_table_with_dates(args.weather_weekly, parse_dates=("week_start",)) if args.weather_weekly else None
    hourly_weather = _read_table_with_dates(args.hourly_weather, parse_dates=("time",)) if args.hourly_weather else None
    age_group_frame = _read_table_with_dates(args.age_group_frame, parse_dates=("datetime",)) if args.age_group_frame else None

    settings = AnalysisPipelineSettings(
        require_weather=not bool(args.no_weather_required),
        run_explainability=not bool(args.no_explainability),
        run_epidemic_waves=not bool(args.no_epidemic_waves),
        run_age_group_season=not bool(args.no_age_group),
        require_age_group_season=bool(args.require_age_group),
        build_context=not bool(args.no_context),
        raise_on_error=bool(args.raise_on_error),
    )
    result = run_analysis_pipeline(
        influenza_weekly=influenza_weekly,
        weather_weekly=weather_weekly,
        hourly_weather=hourly_weather,
        age_group_frame=age_group_frame,
        settings=settings,
        output=AnalysisOutputConfig(output_dir=args.output_dir),
    )
    _print_json(result.to_public_dict(include_heavy_objects=False))
    return EXIT_OK if result.status != "failed" else EXIT_FAILED



def _cmd_run_all(args: argparse.Namespace) -> int:
    source = AnalysisSourceConfig(
        city=args.city,
        begin_year=args.begin_year,
        begin_week=args.begin_week,
        end_date=args.end_date,
        fetch_weather=not bool(args.no_weather),
    )
    llm = WorkflowLLMConfig(
        model=args.model,
        backend=args.backend,
        base_url=args.base_url,
        timeout_sec=args.timeout_sec,
        evaluator_model=args.evaluator_model,
        reuse_narrator_for_evaluation=args.evaluator_model is None,
    )
    settings = FullWorkflowSettings(
        run_editor=not bool(args.no_editor),
        run_evaluation=not bool(args.no_evaluation),
        render_pdf=bool(args.render_pdf),
        raise_on_error=bool(args.raise_on_error),
        fail_on_evaluation_errors=bool(args.fail_on_evaluation_errors),
    )
    result = run_full_workflow_from_sources(
        source=source,
        llm=llm,
        settings=settings,
        output=FullWorkflowOutputConfig(output_dir=args.output_dir),
    )
    _print_json(result.to_public_dict(include_heavy_objects=False))
    return EXIT_OK if result.status != "failed" else EXIT_FAILED

def _cmd_generate_bulletin(args: argparse.Namespace) -> int:
    result = run_pipeline_from_config(args.config)
    _print_json(result.to_public_dict(include_bulletins=False))
    return EXIT_OK if result.status != "failed" else EXIT_FAILED


def _cmd_evaluate_bulletin(args: argparse.Namespace) -> int:
    eval_config = _make_cli_eval_config(args)

    report = evaluate_bulletin_from_files(
        context_path=args.context,
        bulletin_path=args.bulletin,
        config=eval_config,
    )
    if args.output is not None:
        save_eval_report(report, args.output)
    if not args.quiet:
        print_eval_report(report)
    _print_json({"status": "ok", "aggregate": report.aggregate, "errors": len(report.errors), "warnings": len(report.warnings)})
    return EXIT_OK


def _cmd_render(args: argparse.Namespace) -> int:
    output = RenderOutputConfig(
        output_dir=args.output_dir,
        save_markdown=not bool(args.no_markdown),
        save_html=not bool(args.no_html),
        save_manifest=not bool(args.no_manifest),
    )
    rendered = render_bulletin_file(
        args.bulletin,
        settings=make_default_render_settings(include_assets=not bool(args.no_assets)),
        output=output,
    )
    _print_json({"status": "ok", "output_paths": dict(rendered.output_paths)})
    return EXIT_OK


def _cmd_render_pdf(args: argparse.Namespace) -> int:
    font = PdfFontConfig(
        regular_path=args.font_regular,
        bold_path=args.font_bold,
    )
    settings = PdfSettings(font=font)
    result = render_bulletin_pdf_file(
        args.bulletin,
        output=PdfOutputConfig(
            output_dir=args.output_dir,
            pdf_filename=args.pdf_filename,
            save_manifest=not bool(args.no_manifest),
        ),
        pdf_settings=settings,
        render_settings=make_default_render_settings(),
    )
    _print_json(result.to_public_dict())
    return EXIT_OK


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_cli_eval_config(args: argparse.Namespace) -> EvalConfig | None:
    """Собрать EvalConfig для standalone evaluate-bulletin.

    Явный запуск evaluation не должен зависеть от флага evaluation.enabled в
    Ai4EpiConfig: этот флаг управляет автоматическим evaluation внутри pipeline,
    а не ручной CLI-командой.
    """

    if getattr(args, "no_llm", False):
        if args.config is not None:
            cfg = load_ai4epi_config(args.config)
            return cfg.evaluation.make_eval_config(llm=None)
        return EvalConfig(llm=None)

    if args.config is not None:
        cfg = load_ai4epi_config(args.config)
        eval_config = cfg.make_eval_config(require_enabled=False)

        if args.evaluator_model is None and args.backend is None and args.base_url is None and args.timeout_sec is None:
            if args.request_timeout_sec is not None:
                return eval_config.model_copy(update={"request_timeout_sec": args.request_timeout_sec})
            return eval_config

        base_backend = cfg.llm.evaluator or cfg.llm.narrator
        backend = args.backend or base_backend.backend
        model = args.evaluator_model or base_backend.model
        base_url = args.base_url or base_backend.base_url
        timeout_sec = args.timeout_sec or base_backend.timeout_sec
        request_timeout_sec = args.request_timeout_sec or eval_config.request_timeout_sec or timeout_sec

        llm = make_chat_backend(
            backend,
            model=model,
            base_url=base_url,
            default_timeout=timeout_sec,
        )
        return eval_config.model_copy(
            update={
                "llm": llm,
                "request_timeout_sec": request_timeout_sec,
            }
        )

    backend = args.backend or os.getenv("EVAL_LLM_BACKEND", DEFAULT_EVALUATOR_BACKEND)
    model = args.evaluator_model or os.getenv("EVAL_OLLAMA_MODEL", DEFAULT_EVALUATOR_MODEL)
    base_url = args.base_url or os.getenv("EVAL_OLLAMA_BASE_URL", "http://localhost:11434")
    timeout_sec = args.timeout_sec or int(os.getenv("EVAL_REQUEST_TIMEOUT_SEC", str(DEFAULT_EVALUATOR_TIMEOUT_SEC)))
    request_timeout_sec = args.request_timeout_sec or timeout_sec

    llm = make_chat_backend(
        backend,
        model=model,
        base_url=base_url,
        default_timeout=timeout_sec,
    )
    return EvalConfig(llm=llm, request_timeout_sec=request_timeout_sec)


def _read_table_with_dates(path: Path, *, parse_dates: tuple[str, ...]) -> Any:
    """Прочитать таблицу, пробуя parse_dates, но не скрывая ошибок структуры файла."""

    try:
        return read_table(path, options=TableReadOptions(parse_dates=parse_dates))
    except ValueError as exc:
        # pandas падает, если parse_dates содержит отсутствующую колонку. Для
        # пользовательских таблиц допустимо прочитать без parse_dates: нижележащие
        # доменные валидаторы всё равно приведут/проверят даты строго.
        if "Missing column provided to 'parse_dates'" not in str(exc):
            raise
        return read_table(path)


def _print_json(data: Any) -> None:
    print(json.dumps(_to_jsonable(data), ensure_ascii=False, indent=2))


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

