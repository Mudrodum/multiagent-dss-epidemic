"""
Источники погодных данных для ai4epi.

Модуль закрывает слой получения и нормализации часовых погодных наблюдений.
Он не выполняет feature engineering и не строит прогноз: его задача — получить
почасовые данные в устойчивом контракте ``time, temp, rh`` и, при необходимости,
сохранить почасовые и недельные погодные артефакты.

Фактический источник из исследовательского notebook — Open-Meteo Archive API:
``https://archive-api.open-meteo.com/v1/archive`` с hourly-полями
``temperature_2m`` и ``relative_humidity_2m``. Для городов с известными
координатами используется локальный реестр; для остальных городов доступен
геокодинг Open-Meteo.
"""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

try:  # requests нужен только для сетевых источников.
    import requests
except ImportError:  # pragma: no cover - проверяется при реальном fetch.
    requests = None  # type: ignore[assignment]

try:  # Поддержка пакетного импорта.
    from ai4epi.core.io import TableSchema, TableWriteOptions, write_json, write_table
    from ai4epi.analysis.preprocessing import WeatherAggregationConfig, aggregate_hourly_weather_to_weekly
except ImportError:  # pragma: no cover - поддержка запуска отдельных файлов в текущем исследовательском каталоге.
    from ai4epi.core.io import TableSchema, TableWriteOptions, write_json, write_table  # type: ignore[no-redef]
    from ai4epi.analysis.preprocessing import WeatherAggregationConfig, aggregate_hourly_weather_to_weekly  # type: ignore[no-redef]


JsonObject = dict[str, Any]


OPEN_METEO_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_HOURLY_FIELDS = "temperature_2m,relative_humidity_2m"


class WeatherSourceError(RuntimeError):
    """Базовая ошибка источника погодных данных."""


class WeatherLocationError(WeatherSourceError):
    """Ошибка разрешения города в координаты."""


class WeatherApiError(WeatherSourceError):
    """Ошибка сетевого запроса или структуры ответа Open-Meteo."""


class WeatherFrameError(WeatherSourceError):
    """Погодная таблица не соответствует ожидаемому контракту."""


class StrictModel(BaseModel):
    """Базовая pydantic-модель с закрытым контрактом."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


class WeatherLocation(StrictModel):
    """Координаты и timezone погодного источника."""

    city: str = Field(min_length=1)
    query: str = Field(min_length=1)
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    timezone: str = Field(min_length=1)
    source: str = Field(default="preset", min_length=1)


class WeatherApiConfig(StrictModel):
    """Настройки Open-Meteo API."""

    geocoding_url: str = Field(default=OPEN_METEO_GEOCODING_URL, min_length=1)
    archive_url: str = Field(default=OPEN_METEO_ARCHIVE_URL, min_length=1)
    hourly_fields: str = Field(default=OPEN_METEO_HOURLY_FIELDS, min_length=1)
    timeout_seconds: float = Field(default=60.0, gt=0)
    geocoding_language: str = Field(default="ru", min_length=1)
    geocoding_count: int = Field(default=1, ge=1, le=10)
    chunk_by_year: bool = True


class WeatherFetchRequest(StrictModel):
    """Параметры загрузки почасовой погоды."""

    city: str = Field(min_length=1)
    start_date: date
    end_date: date
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    timezone: str | None = Field(default=None, min_length=1)

    @field_validator("city")
    @classmethod
    def normalize_city(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("city не должен быть пустым.")
        return value

    @model_validator(mode="after")
    def validate_interval(self) -> "WeatherFetchRequest":
        if self.end_date < self.start_date:
            raise ValueError("end_date не может быть раньше start_date.")
        if any(value is not None for value in (self.latitude, self.longitude, self.timezone)):
            if self.latitude is None or self.longitude is None or self.timezone is None:
                raise ValueError("latitude, longitude и timezone должны задаваться совместно.")
        return self


class WeatherFrameBundle(StrictModel):
    """Результат получения и нормализации погодных данных."""

    request: WeatherFetchRequest
    location: WeatherLocation
    hourly: Any
    weekly: Any
    source_url: str

    @field_validator("hourly", "weekly")
    @classmethod
    def validate_dataframe(cls, value: Any) -> pd.DataFrame:
        if not isinstance(value, pd.DataFrame):
            raise TypeError("Ожидался pandas.DataFrame.")
        return value


class WeatherOutputConfig(StrictModel):
    """Настройки сохранения погодных артефактов."""

    output_dir: Path = Path("results_csv")
    hourly_filename: str = Field(default="weather_hourly.csv", min_length=1)
    weekly_filename: str = Field(default="weather_weekly.csv", min_length=1)
    location_filename: str = Field(default="weather_location.json", min_length=1)
    save_hourly: bool = True
    save_weekly: bool = True
    save_location: bool = True
    csv_encoding: str = Field(default="utf-8-sig", min_length=1)

    @field_validator("hourly_filename", "weekly_filename", "location_filename")
    @classmethod
    def validate_filename(cls, value: str) -> str:
        path = Path(value)
        if path.is_absolute() or len(path.parts) != 1:
            raise ValueError("Имена выходных файлов должны быть простыми относительными именами.")
        return value


WEATHER_HOURLY_SCHEMA = TableSchema(
    required_columns=["time", "temp", "rh"],
    column_types={"time": "datetime", "temp": "numeric", "rh": "numeric"},
    non_null_columns=["time", "temp", "rh"],
    unique_key=["time"],
)

WEATHER_WEEKLY_SCHEMA = TableSchema(
    required_columns=[
        "week_start",
        "temp_mean",
        "temp_max",
        "temp_min",
        "rh_mean",
        "rh_max",
        "rh_min",
        "n_hours",
    ],
    column_types={
        "week_start": "datetime",
        "temp_mean": "numeric",
        "temp_max": "numeric",
        "temp_min": "numeric",
        "rh_mean": "numeric",
        "rh_max": "numeric",
        "rh_min": "numeric",
        "n_hours": "integer",
    },
    non_null_columns=["week_start", "temp_mean", "temp_max", "temp_min", "rh_mean", "n_hours"],
    unique_key=["week_start"],
)


CITY_WEATHER_PRESETS: dict[str, WeatherLocation] = {
    "spb": WeatherLocation(
        city="spb",
        query="Saint Petersburg",
        latitude=59.9311,
        longitude=30.3609,
        timezone="Europe/Moscow",
    ),
    "moscow": WeatherLocation(
        city="moscow",
        query="Moscow",
        latitude=55.7558,
        longitude=37.6173,
        timezone="Europe/Moscow",
    ),
    "novosibirsk": WeatherLocation(
        city="novosibirsk",
        query="Novosibirsk",
        latitude=54.9833,
        longitude=82.8964,
        timezone="Asia/Novosibirsk",
    ),
    "yekaterinburg": WeatherLocation(
        city="yekaterinburg",
        query="Yekaterinburg",
        latitude=56.8389,
        longitude=60.6057,
        timezone="Asia/Yekaterinburg",
    ),
    "krasnodar": WeatherLocation(
        city="krasnodar",
        query="Krasnodar",
        latitude=45.0355,
        longitude=38.9753,
        timezone="Europe/Moscow",
    ),
}

_CITY_ALIASES: dict[str, str] = {
    "санкт-петербург": "spb",
    "санкт петербург": "spb",
    "петербург": "spb",
    "спб": "spb",
    "москва": "moscow",
    "новосибирск": "novosibirsk",
    "екатеринбург": "yekaterinburg",
    "краснодар": "krasnodar",
}


def normalize_weather_city_slug(city: str) -> str:
    """Нормализовать пользовательское имя города в slug."""

    raw = str(city).strip().lower()
    if not raw:
        raise WeatherLocationError("Пустое имя города.")
    if raw in _CITY_ALIASES:
        return _CITY_ALIASES[raw]
    raw = raw.replace("ё", "е")
    raw = re.sub(r"\s+", "_", raw)
    raw = re.sub(r"[^0-9a-zа-я_-]+", "_", raw)
    raw = re.sub(r"_+", "_", raw).strip("_")
    if not raw:
        raise WeatherLocationError("Не удалось построить slug города.")
    return raw


def resolve_weather_location(
    city: str,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
    timezone: str | None = None,
    config: WeatherApiConfig | None = None,
    session: Any | None = None,
) -> WeatherLocation:
    """Разрешить город в координаты и timezone."""

    slug = normalize_weather_city_slug(city)
    query = str(city).strip()

    if latitude is not None or longitude is not None or timezone is not None:
        if latitude is None or longitude is None or timezone is None:
            raise WeatherLocationError("latitude, longitude и timezone должны быть заданы совместно.")
        return WeatherLocation(
            city=slug,
            query=query,
            latitude=float(latitude),
            longitude=float(longitude),
            timezone=str(timezone),
            source="explicit",
        )

    if slug in CITY_WEATHER_PRESETS:
        return CITY_WEATHER_PRESETS[slug]

    return geocode_weather_location(query or slug, city_slug=slug, config=config, session=session)


def geocode_weather_location(
    query: str,
    *,
    city_slug: str | None = None,
    config: WeatherApiConfig | None = None,
    session: Any | None = None,
) -> WeatherLocation:
    """Получить координаты через Open-Meteo Geocoding API."""

    _require_requests()
    cfg = config or WeatherApiConfig()
    http = session or requests
    params = {
        "name": query,
        "count": cfg.geocoding_count,
        "language": cfg.geocoding_language,
        "format": "json",
    }
    response = http.get(cfg.geocoding_url, params=params, timeout=cfg.timeout_seconds)
    response.raise_for_status()
    data = response.json()
    results = data.get("results") or []
    if not results:
        raise WeatherLocationError(f"Не удалось определить координаты для города: {query!r}.")

    best = results[0]
    if "latitude" not in best or "longitude" not in best:
        raise WeatherLocationError(f"Ответ геокодинга не содержит координат для города: {query!r}.")

    return WeatherLocation(
        city=city_slug or normalize_weather_city_slug(query),
        query=query,
        latitude=float(best["latitude"]),
        longitude=float(best["longitude"]),
        timezone=str(best.get("timezone") or "UTC"),
        source="open_meteo_geocoding",
    )


def fetch_open_meteo_hourly_chunk(
    *,
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    timezone: str,
    config: WeatherApiConfig | None = None,
    session: Any | None = None,
) -> pd.DataFrame:
    """Загрузить один непрерывный chunk почасовой погоды из Open-Meteo Archive API."""

    _require_requests()
    cfg = config or WeatherApiConfig()
    if end_date < start_date:
        raise WeatherApiError("end_date не может быть раньше start_date.")

    params = {
        "latitude": float(latitude),
        "longitude": float(longitude),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "hourly": cfg.hourly_fields,
        "timezone": timezone,
    }
    http = session or requests
    response = http.get(cfg.archive_url, params=params, timeout=cfg.timeout_seconds)
    response.raise_for_status()
    payload = response.json()
    hourly = payload.get("hourly")
    if not isinstance(hourly, Mapping):
        raise WeatherApiError("Ответ Open-Meteo не содержит объекта hourly.")

    try:
        frame = pd.DataFrame(
            {
                "time": pd.to_datetime(hourly["time"]),
                "temp": hourly["temperature_2m"],
                "rh": hourly["relative_humidity_2m"],
            }
        )
    except KeyError as exc:
        raise WeatherApiError(f"Ответ Open-Meteo не содержит ожидаемого hourly-поля: {exc}.") from exc

    return normalize_hourly_weather_frame(frame)


def fetch_open_meteo_hourly(
    *,
    location: WeatherLocation,
    start_date: date,
    end_date: date,
    config: WeatherApiConfig | None = None,
    session: Any | None = None,
) -> pd.DataFrame:
    """Загрузить почасовую погоду, при необходимости разбивая запрос по годам."""

    cfg = config or WeatherApiConfig()
    if end_date < start_date:
        raise WeatherApiError("end_date не может быть раньше start_date.")

    if not cfg.chunk_by_year:
        return fetch_open_meteo_hourly_chunk(
            latitude=location.latitude,
            longitude=location.longitude,
            start_date=start_date,
            end_date=end_date,
            timezone=location.timezone,
            config=cfg,
            session=session,
        )

    parts: list[pd.DataFrame] = []
    for year in range(start_date.year, end_date.year + 1):
        chunk_start = max(start_date, date(year, 1, 1))
        chunk_end = min(end_date, date(year, 12, 31))
        if chunk_start > chunk_end:
            continue
        parts.append(
            fetch_open_meteo_hourly_chunk(
                latitude=location.latitude,
                longitude=location.longitude,
                start_date=chunk_start,
                end_date=chunk_end,
                timezone=location.timezone,
                config=cfg,
                session=session,
            )
        )

    if not parts:
        raise WeatherApiError("Не сформировано ни одного погодного chunk.")

    out = pd.concat(parts, ignore_index=True)
    out = out.drop_duplicates(subset=["time"]).sort_values("time").reset_index(drop=True)
    return normalize_hourly_weather_frame(out)


def load_weather_from_open_meteo(
    request: WeatherFetchRequest,
    *,
    api_config: WeatherApiConfig | None = None,
    aggregation_config: WeatherAggregationConfig | None = None,
    session: Any | None = None,
) -> WeatherFrameBundle:
    """Загрузить почасовую и недельную погоду из Open-Meteo."""

    cfg = api_config or WeatherApiConfig()
    location = resolve_weather_location(
        request.city,
        latitude=request.latitude,
        longitude=request.longitude,
        timezone=request.timezone,
        config=cfg,
        session=session,
    )
    hourly = fetch_open_meteo_hourly(
        location=location,
        start_date=request.start_date,
        end_date=request.end_date,
        config=cfg,
        session=session,
    )
    weekly = aggregate_hourly_weather_to_weekly(hourly, config=aggregation_config)
    WEATHER_WEEKLY_SCHEMA.validate_dataframe(weekly, name="weather_weekly")
    return WeatherFrameBundle(
        request=request,
        location=location,
        hourly=hourly,
        weekly=weekly,
        source_url=cfg.archive_url,
    )


def load_weather_until_date(
    city: str,
    *,
    start_date: date | datetime | pd.Timestamp | str,
    end_date: date | datetime | pd.Timestamp | str,
    latitude: float | None = None,
    longitude: float | None = None,
    timezone: str | None = None,
    api_config: WeatherApiConfig | None = None,
    aggregation_config: WeatherAggregationConfig | None = None,
    session: Any | None = None,
) -> WeatherFrameBundle:
    """Функциональная обёртка для типового сценария загрузки погоды."""

    request = WeatherFetchRequest(
        city=city,
        start_date=_coerce_date(start_date),
        end_date=_coerce_date(end_date),
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
    )
    return load_weather_from_open_meteo(
        request,
        api_config=api_config,
        aggregation_config=aggregation_config,
        session=session,
    )


def load_weather_aligned_to_influenza(
    city: str,
    influenza_weekly: pd.DataFrame,
    *,
    datetime_col: str = "datetime",
    extend_days_after_last_week: int = 6,
    latitude: float | None = None,
    longitude: float | None = None,
    timezone: str | None = None,
    api_config: WeatherApiConfig | None = None,
    aggregation_config: WeatherAggregationConfig | None = None,
    session: Any | None = None,
) -> WeatherFrameBundle:
    """Загрузить погоду на период, покрывающий недельный ряд заболеваемости."""

    if datetime_col not in influenza_weekly.columns:
        raise WeatherFrameError(f"В influenza_weekly отсутствует колонка {datetime_col!r}.")
    dates = pd.to_datetime(influenza_weekly[datetime_col], errors="coerce")
    if dates.isna().any():
        raise WeatherFrameError("В influenza_weekly есть некорректные даты.")
    start = dates.min().date()
    end = (dates.max() + pd.Timedelta(days=int(extend_days_after_last_week))).date()
    return load_weather_until_date(
        city,
        start_date=start,
        end_date=end,
        latitude=latitude,
        longitude=longitude,
        timezone=timezone,
        api_config=api_config,
        aggregation_config=aggregation_config,
        session=session,
    )


def normalize_hourly_weather_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Нормализовать почасовую погодную таблицу к контракту ``time, temp, rh``."""

    column_map = _infer_weather_column_map(frame.columns)
    if any(source is None for source in column_map.values()):
        raise WeatherFrameError("Не удалось определить обязательные погодные колонки time/temp/rh.")

    out = pd.DataFrame(
        {
            "time": pd.to_datetime(frame[column_map["time"]], errors="coerce"),
            "temp": pd.to_numeric(frame[column_map["temp"]], errors="coerce"),
            "rh": pd.to_numeric(frame[column_map["rh"]], errors="coerce"),
        }
    )
    if out["time"].isna().any():
        raise WeatherFrameError("В погодной таблице есть некорректные значения time.")
    if out[["temp", "rh"]].isna().any().any():
        bad = out.loc[out[["temp", "rh"]].isna().any(axis=1)].head(10)
        raise WeatherFrameError(f"В погодной таблице есть NaN в temp/rh:\n{bad}")
    if out["time"].dt.tz is not None:
        out["time"] = out["time"].dt.tz_localize(None)
    out = out.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    WEATHER_HOURLY_SCHEMA.validate_dataframe(out, name="weather_hourly")
    return out


def save_weather_bundle(bundle: WeatherFrameBundle, output: WeatherOutputConfig | None = None) -> dict[str, Path]:
    """Сохранить погодные артефакты и вернуть пути."""

    cfg = output or WeatherOutputConfig()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    table_options = TableWriteOptions(encoding=cfg.csv_encoding)

    if cfg.save_hourly:
        paths["hourly"] = write_table(bundle.hourly, cfg.output_dir / cfg.hourly_filename, options=table_options)
    if cfg.save_weekly:
        paths["weekly"] = write_table(bundle.weekly, cfg.output_dir / cfg.weekly_filename, options=table_options)
    if cfg.save_location:
        payload = {
            "request": bundle.request.model_dump(mode="json"),
            "location": bundle.location.model_dump(mode="json"),
            "source_url": bundle.source_url,
        }
        paths["location"] = write_json(payload, cfg.output_dir / cfg.location_filename)
    return paths


def _infer_weather_column_map(columns: Sequence[str]) -> dict[str, str | None]:
    normalized = {str(col).strip().lower(): str(col) for col in columns}
    aliases = {
        "time": ("time", "datetime", "date", "timestamp"),
        "temp": ("temp", "temperature", "temperature_2m", "t2m", "temp_c"),
        "rh": ("rh", "relative_humidity", "relative_humidity_2m", "humidity"),
    }
    result: dict[str, str | None] = {}
    for target, names in aliases.items():
        found = None
        for name in names:
            if name in normalized:
                found = normalized[name]
                break
        result[target] = found
    return result


def _coerce_date(value: date | datetime | pd.Timestamp | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def _require_requests() -> None:
    if requests is None:
        raise WeatherApiError("Для загрузки погоды из Open-Meteo требуется пакет requests.")

