"""
Typed context contract for the ai4epi pipeline.

The module defines the stable, validated representation of the global context
used by section narrators. Standard context blocks are modelled explicitly,
while additional top-level blocks are preserved to support user-defined
sections without changing the core package.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class ContextPathError(KeyError):
    """Raised when a configured context path cannot be resolved."""


class StrictModel(BaseModel):
    """Base model for closed contracts."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
    )


class ExtensibleModel(BaseModel):
    """
    Base model for scientific bundles whose exact set of fields can evolve.

    Extra fields are intentionally preserved. This allows the forecasting,
    interpretation and epidemiological-analysis layers to add measured
    artefacts without breaking downstream repositories.
    """

    model_config = ConfigDict(
        extra="allow",
        populate_by_name=True,
        validate_assignment=True,
    )


class OriginContext(StrictModel):
    """Temporal anchor of a bulletin run."""

    origin_date: date
    iso_year: int
    iso_week: int = Field(ge=1, le=53)
    week_start: str
    week_end: str


class CurrentSituationContext(ExtensibleModel):
    """Observed epidemiological situation at the forecast origin."""

    current_week_date: date
    current_value: float
    previous_value: float
    direction_word: str
    change_pct: float
    trend_4w_values: list[float] = Field(min_length=1)
    trend_4w_label: str
    semantic: dict[str, Any] = Field(default_factory=dict)


class ForecastHorizon(ExtensibleModel):
    """Point and interval forecast for one forecast horizon."""

    horizon_weeks: int = Field(ge=1)
    target_date: date | None = None
    target_iso_week: int | None = Field(default=None, ge=1, le=53)
    point_forecast: float
    q_lo: float | None = None
    q_hi: float | None = None
    interval_width: float | None = None
    pi_width: float | None = None

    @model_validator(mode="after")
    def validate_interval_consistency(self) -> "ForecastHorizon":
        if self.q_lo is not None and self.q_hi is not None and self.q_hi < self.q_lo:
            raise ValueError("q_hi must be greater than or equal to q_lo.")
        return self

    @property
    def resolved_interval_width(self) -> float | None:
        """Return the canonical interval width, accepting the legacy pi_width name."""

        return self.interval_width if self.interval_width is not None else self.pi_width


class ForecastContext(ExtensibleModel):
    """Forecast summary used by risk and current-situation sections."""

    horizons: list[ForecastHorizon] = Field(min_length=1)
    trend_label: str
    dynamics_description: str
    slope_per_week: float | None = None
    uncertainty_label: str
    relative_uncertainty_pct: float | None = None
    max_interval_width: float | None = None
    semantic: dict[str, Any] = Field(default_factory=dict)


class ShapFeature(ExtensibleModel):
    """One SHAP-derived feature description in publication-facing Russian fields."""

    name: str = Field(alias="название")
    influence_strength: float = Field(alias="сила_влияния")
    direction: str = Field(alias="направление")
    direction_reliable: bool = Field(alias="направление_надёжное")


class ShapSummaryContext(ExtensibleModel):
    """Model-interpretation block grouped by forecast horizon."""

    by_horizon: dict[str, list[ShapFeature]] = Field(default_factory=dict)
    key_insight: str
    semantic: dict[str, Any] = Field(default_factory=dict)

    @field_validator("by_horizon", mode="before")
    @classmethod
    def normalize_horizon_keys(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        return {str(key): item for key, item in value.items()}


class MetricRecord(ExtensibleModel):
    """Forecast quality metrics for one horizon."""

    horizon_weeks: int = Field(ge=1)
    mae: float
    rmse: float
    r2: float
    n_test: int = Field(ge=0)


class ErrorStatsH1(ExtensibleModel):
    """Aggregated absolute-error statistics for the h=1 forecast."""

    median_error_h1: float
    p75_error_h1: float
    max_error_h1: float
    mean_error_h1: float


class ModelQualityContext(ExtensibleModel):
    """Quality and limitation evidence for the model-quality section."""

    metrics: list[MetricRecord] = Field(default_factory=list)
    error_stats_h1: ErrorStatsH1
    worst_case_pattern: str
    semantic: dict[str, Any] = Field(default_factory=dict)


class ModelInfoContext(ExtensibleModel):
    """Publication-safe model card."""

    family_ru: str | None = None
    strategy_ru: str | None = None
    task_type_ru: str | None = None
    forecast_target_ru: str | None = None
    horizons: list[int] = Field(default_factory=list)
    feature_groups: list[str] = Field(default_factory=list)
    feature_groups_ru: dict[str, str] = Field(default_factory=dict)
    calibration_start_year: int | None = None
    calibration_data_description: str | None = None
    semantic: dict[str, Any] = Field(default_factory=dict)


class WaveRecord(ExtensibleModel):
    """One epidemic-wave record in the comparison bundle."""

    season_label: str | None = None
    peak_week: int | None = Field(default=None, ge=1, le=53)
    peak_date: date | None = None
    peak_value: float | None = None
    wave_status: str | None = None
    fwhm_weeks: float | None = None
    fwhm_lower_bound_weeks: float | None = None
    asymmetry_ratio: float | None = None
    season_area: float | None = None
    secondary_peak_ratio: float | None = None


class EpidemicWaveComparisonContext(ExtensibleModel):
    """Structured evidence bundle for comparison of recent epidemic waves."""

    series_name: str | None = None
    series_label_ru: str | None = None
    season_definition: str | None = None
    width_definition: str | None = None
    smoothing: dict[str, Any] | None = None
    season_labels: list[str] = Field(default_factory=list)
    waves: list[WaveRecord] = Field(default_factory=list)
    latest_vs_previous: list[dict[str, Any]] = Field(default_factory=list)
    peak_ranking: list[Any] = Field(default_factory=list)
    width_ranking_complete: list[Any] = Field(default_factory=list)
    latest_wave_status: str | None = None
    allowed_claims: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)


class AgeGroupSeasonRow(ExtensibleModel):
    """One row of age-group seasonal burden analysis."""

    age_group_code: str
    age_group_label: str
    is_total_row: bool = False
    season_cases: float | int | None = None
    cumulative_incidence_pct: float
    peak_week: int | None = Field(default=None, ge=1, le=53)
    peak_date: date | None = None
    peak_season_position: float | None = None
    peak_inc_per_10k: float
    mean_weekly_inc_per_10k: float | None = None
    peak_width_weeks: float | None = None
    peak_width_defined: bool | None = None
    peak_width_reason: str | None = None
    width_level: float | None = None
    share_of_total_cases_pct: float


class AgeGroupSeasonContext(ExtensibleModel):
    """Structured evidence bundle for age-group seasonal overview."""

    season_label: str
    metric_label_ru: str
    width_definition: str
    rows: list[AgeGroupSeasonRow] = Field(default_factory=list)
    derived_findings: dict[str, Any] = Field(default_factory=dict)
    peak_width_undefined_note: str | None = None
    comparison_scope_note: str | None = None
    semantic: dict[str, Any] = Field(default_factory=dict)


class GlobalContext(ExtensibleModel):
    """
    Validated global context available to all section narrators.

    Standard blocks are explicit fields. Additional top-level blocks are allowed
    and preserved; user-defined sections can address them through configured
    context paths.
    """

    origin: OriginContext
    unit: str
    per_population: int | float
    current_situation: CurrentSituationContext
    epidemic_wave_comparison: EpidemicWaveComparisonContext
    forecast: ForecastContext
    shap_summary: ShapSummaryContext
    model_quality: ModelQualityContext
    model_info: ModelInfoContext
    age_group_season: AgeGroupSeasonContext | None = None

    @model_validator(mode="after")
    def validate_temporal_alignment(self) -> "GlobalContext":
        if self.origin.iso_year != self.origin.origin_date.isocalendar().year:
            raise ValueError("origin.iso_year does not match origin.origin_date ISO year.")
        if self.origin.iso_week != self.origin.origin_date.isocalendar().week:
            raise ValueError("origin.iso_week does not match origin.origin_date ISO week.")
        return self

    def to_public_dict(self, *, by_alias: bool = True) -> dict[str, Any]:
        """Return a JSON-serializable representation of the context."""

        return self.model_dump(mode="json", by_alias=by_alias, exclude_none=False)

    def block_names(self) -> list[str]:
        """Return names of all available top-level context blocks."""

        standard_names = list(type(self).model_fields)
        extra_names = list((self.model_extra or {}).keys())
        return standard_names + [name for name in extra_names if name not in standard_names]

    def get_block(self, name: str) -> Any:
        """Return one top-level block by name."""

        if name in type(self).model_fields:
            return getattr(self, name)
        if self.model_extra and name in self.model_extra:
            return self.model_extra[name]
        raise ContextPathError(f"Context block is absent: {name!r}")

    def resolve_path(self, path: str | Sequence[str | int]) -> Any:
        """
        Resolve a configured context path.

        A string path uses dot notation. Numeric path segments index lists:
        ``forecast.horizons.0.point_forecast``.
        """

        segments = _normalize_path(path)
        current: Any = self

        for segment in segments:
            current = _resolve_segment(current, segment, path)

        if isinstance(current, BaseModel):
            return current.model_dump(mode="json", by_alias=True, exclude_none=False)
        return current

    def project(self, mapping: Mapping[str, str | Sequence[str | int]]) -> dict[str, Any]:
        """
        Build a section-specific evidence packet from configured paths.

        The mapping key is the output field name in the packet; the mapping
        value is the path inside GlobalContext.
        """

        return {output_key: self.resolve_path(path) for output_key, path in mapping.items()}


def _normalize_path(path: str | Sequence[str | int]) -> list[str | int]:
    if isinstance(path, str):
        if not path.strip():
            raise ContextPathError("Context path must not be empty.")
        return [segment for segment in path.split(".") if segment]
    if isinstance(path, Sequence):
        if not path:
            raise ContextPathError("Context path must not be empty.")
        return list(path)
    raise TypeError("Context path must be a dot-separated string or a sequence of keys.")


def _resolve_segment(current: Any, segment: str | int, original_path: str | Sequence[str | int]) -> Any:
    if isinstance(current, BaseModel):
        if isinstance(segment, str):
            field_name = _field_name_from_segment(current, segment)
            if field_name is not None:
                return getattr(current, field_name)
            if current.model_extra and segment in current.model_extra:
                return current.model_extra[segment]
        raise ContextPathError(f"Segment {segment!r} is absent in context path {original_path!r}.")

    if isinstance(current, Mapping):
        if segment in current:
            return current[segment]
        raise ContextPathError(f"Segment {segment!r} is absent in context path {original_path!r}.")

    if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
        index = _coerce_list_index(segment, original_path)
        try:
            return current[index]
        except IndexError as exc:
            raise ContextPathError(f"List index {index} is out of range in context path {original_path!r}.") from exc

    raise ContextPathError(
        f"Cannot resolve segment {segment!r} inside object of type {type(current).__name__} "
        f"for context path {original_path!r}."
    )


def _field_name_from_segment(model: BaseModel, segment: str) -> str | None:
    fields = type(model).model_fields
    if segment in fields:
        return segment

    for field_name, field_info in fields.items():
        if field_info.alias == segment:
            return field_name

    return None


def _coerce_list_index(segment: str | int, original_path: str | Sequence[str | int]) -> int:
    if isinstance(segment, int):
        return segment
    if isinstance(segment, str) and segment.isdecimal():
        return int(segment)
    raise ContextPathError(f"List segment must be an integer in context path {original_path!r}; got {segment!r}.")


def load_global_context(path: str | Path) -> GlobalContext:
    """Load and validate GlobalContext from a JSON file."""

    context_path = Path(path)
    with context_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    return GlobalContext.model_validate(data)


def save_global_context(context: GlobalContext, path: str | Path, *, indent: int = 2) -> None:
    """Save GlobalContext as UTF-8 JSON."""

    context_path = Path(path)
    context_path.parent.mkdir(parents=True, exist_ok=True)
    with context_path.open("w", encoding="utf-8") as stream:
        json.dump(context.to_public_dict(), stream, ensure_ascii=False, indent=indent)


__all__ = [
    "AgeGroupSeasonContext",
    "AgeGroupSeasonRow",
    "ContextPathError",
    "CurrentSituationContext",
    "EpidemicWaveComparisonContext",
    "ErrorStatsH1",
    "ForecastContext",
    "ForecastHorizon",
    "GlobalContext",
    "MetricRecord",
    "ModelInfoContext",
    "ModelQualityContext",
    "OriginContext",
    "ShapFeature",
    "ShapSummaryContext",
    "WaveRecord",
    "load_global_context",
    "save_global_context",
]

