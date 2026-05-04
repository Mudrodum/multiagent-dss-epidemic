"""
Источники эпидемиологических данных для ai4epi.

Модуль переносит в пакетную архитектуру контракт доступа к данным, который в
исходном notebook использовался через ``model_complex.InfluenzaData``. Внешний
источник данных — CSV-эндпоинт ``db.influenza.spb.ru`` с недельными данными по
ОРВИ, ПЦР-тестам на грипп и возрастным группам.

Модуль не выполняет preprocessing признаков, не строит прогнозы и не формирует
``GlobalContext``. Его задача — получить и нормализовать первичную таблицу до
устойчивого табличного контракта, пригодного для последующих этапов.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import importlib
from pathlib import Path
import re
from typing import Any, Callable, Mapping, Sequence, TextIO, cast
from urllib.parse import urlencode

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ai4epi.core.io import TableSchema, TableWriteOptions, write_table


def _text_stream(text: str) -> TextIO:
    """Возвращает текстовый поток из стандартной библиотеки.

    В пакете есть модуль ``ai4epi.io``; некоторые IDE из-за этого могут
    ошибочно разрешать ``from io import StringIO`` как обращение к локальному
    модулю. Динамический импорт делает зависимость от стандартной библиотеки
    однозначной и сохраняет совместимость при запуске из разных рабочих
    директорий.
    """

    stdlib_io = importlib.import_module("io")
    string_io_factory = cast(Callable[[str], TextIO], getattr(stdlib_io, "StringIO"))
    return string_io_factory(text)

try:  # requests нужен только для сетевого источника.
    import requests
except ImportError:  # pragma: no cover - проверяется при реальном fetch.
    requests = None  # type: ignore[assignment]


INFLUENZA_DB_BASE_URL = "https://db.influenza.spb.ru/scripts/report/rmancgi.exe"
INFLUENZA_DB_REPORT_NAME = "get_csv"
INFLUENZA_DB_REPORT_ID = "aripcr"

# Токен перенесён из исходного ``model_complex.epid_data.influenza_data``.
# Он является частью уже используемого контракта доступа к публичному CSV-
# отчёту. При публикации репозитория значение можно переопределить через
# ``InfluenzaDbConfig(auth_token=...)`` без изменения кода.
DEFAULT_INFLUENZA_DB_AUTH_TOKEN = (
    "7e283896cf78e49c321dc60fab2850745a25215b621f600f648424d242a78c4a"
)


RAW_INFLUENZA_COLUMNS: dict[str, str] = {
    "YEAR": "datetime",
    "REGION_NAME": "region_name",
    "DISTRICT_NAME": "district_name",
    "ARI_TOTAL": "sars_total_cases",
    "ARI_0_2": "sars_cases_age_group_0",
    "ARI_3_6": "sars_cases_age_group_1",
    "ARI_7_14": "sars_cases_age_group_2",
    "ARI_15_64": "sars_cases_age_group_4",
    "ARI_65": "sars_cases_age_group_5",
    "POP_TOTAL": "total_population",
    "POP_0_2": "population_age_group_0",
    "POP_3_6": "population_age_group_1",
    "POP_7_14": "population_age_group_2",
    "POP_15_64": "population_age_group_4",
    "POP_65": "population_age_group_5",
    "SWB_TOTAL": "tested_total",
    "A_TOTAL": "tested_strain_0",
    "PDM_TOTAL": "tested_strain_1",
    "H3_TOTAL": "tested_strain_2",
    "B_TOTAL": "tested_strain_3",
}

STRAIN_NAMES: dict[int, str] = {
    0: "A (субтип не определен)",
    1: "A(H1)pdm09",
    2: "A(H3)",
    3: "B",
}

MODEL_INFLUENZA_TOTAL_STRAIN_INDICES: tuple[int, ...] = (0, 1, 2, 3)


class CityInfo(BaseModel):
    """Стабильное описание города, поддерживаемого influenza API."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    slug: str = Field(min_length=1)
    api_id: int = Field(ge=0)
    name_ru: str = Field(min_length=1)


# CITY_REGISTRY: slug -> CityInfo.
# Источник — реестр из notebook и файл ``Города_в_API.xlsx``, использованный в
# исходном исследовательском пайплайне. ``russia`` добавлен как агрегатный код 0.
CITY_REGISTRY: dict[str, CityInfo] = {
    "russia": CityInfo(slug="russia", api_id=0, name_ru="Россия"),
    "birobidzhan": CityInfo(slug="birobidzhan", api_id=7, name_ru="Биробиджан"),
    "arkhangelsk": CityInfo(slug="arkhangelsk", api_id=9, name_ru="Архангельск"),
    "astrakhan": CityInfo(slug="astrakhan", api_id=10, name_ru="Астрахань"),
    "barnaul": CityInfo(slug="barnaul", api_id=11, name_ru="Барнаул"),
    "orenburg": CityInfo(slug="orenburg", api_id=12, name_ru="Оренбург"),
    "vladivostok": CityInfo(slug="vladivostok", api_id=13, name_ru="Владивосток"),
    "volgograd": CityInfo(slug="volgograd", api_id=14, name_ru="Волгоград"),
    "voronezh": CityInfo(slug="voronezh", api_id=15, name_ru="Воронеж"),
    "nizhny_novgorod": CityInfo(slug="nizhny_novgorod", api_id=16, name_ru="Нижний Новгород"),
    "irkutsk": CityInfo(slug="irkutsk", api_id=19, name_ru="Иркутск"),
    "kaliningrad": CityInfo(slug="kaliningrad", api_id=20, name_ru="Калининград"),
    "murmansk": CityInfo(slug="murmansk", api_id=21, name_ru="Мурманск"),
    "novosibirsk": CityInfo(slug="novosibirsk", api_id=22, name_ru="Новосибирск"),
    "saratov": CityInfo(slug="saratov", api_id=24, name_ru="Саратов"),
    "khabarovsk": CityInfo(slug="khabarovsk", api_id=26, name_ru="Хабаровск"),
    "moscow": CityInfo(slug="moscow", api_id=32, name_ru="Москва"),
    "tomsk": CityInfo(slug="tomsk", api_id=34, name_ru="Томск"),
    "vladimir": CityInfo(slug="vladimir", api_id=36, name_ru="Владимир"),
    "spb": CityInfo(slug="spb", api_id=38, name_ru="Санкт-Петербург"),
    "yaroslavl": CityInfo(slug="yaroslavl", api_id=40, name_ru="Ярославль"),
    "kazan": CityInfo(slug="kazan", api_id=41, name_ru="Казань"),
    "kemerovo": CityInfo(slug="kemerovo", api_id=43, name_ru="Кемерово"),
    "kirov": CityInfo(slug="kirov", api_id=44, name_ru="Киров"),
    "cheboksary": CityInfo(slug="cheboksary", api_id=45, name_ru="Чебоксары"),
    "magadan": CityInfo(slug="magadan", api_id=46, name_ru="Магадан"),
    "norilsk": CityInfo(slug="norilsk", api_id=47, name_ru="Норильск"),
    "vladikavkaz": CityInfo(slug="vladikavkaz", api_id=48, name_ru="Владикавказ"),
    "perm": CityInfo(slug="perm", api_id=49, name_ru="Пермь"),
    "petropavlovsk": CityInfo(slug="petropavlovsk", api_id=50, name_ru="Петропавловск"),
    "rostov_na_donu": CityInfo(slug="rostov_na_donu", api_id=51, name_ru="Ростов-на-Дону"),
    "smolensk": CityInfo(slug="smolensk", api_id=53, name_ru="Смоленск"),
    "stavropol": CityInfo(slug="stavropol", api_id=54, name_ru="Ставрополь"),
    "ulan_ude": CityInfo(slug="ulan_ude", api_id=55, name_ru="Улан-Удэ"),
    "ufa": CityInfo(slug="ufa", api_id=56, name_ru="Уфа"),
    "chelyabinsk": CityInfo(slug="chelyabinsk", api_id=57, name_ru="Челябинск"),
    "yakutsk": CityInfo(slug="yakutsk", api_id=58, name_ru="Якутск"),
    "chita": CityInfo(slug="chita", api_id=59, name_ru="Чита"),
    "yuzhno_sakhalinsk": CityInfo(slug="yuzhno_sakhalinsk", api_id=60, name_ru="Южно-Сахалинск"),
    "krasnodar": CityInfo(slug="krasnodar", api_id=61, name_ru="Краснодар"),
    "krasnoyarsk": CityInfo(slug="krasnoyarsk", api_id=62, name_ru="Красноярск"),
    "samara": CityInfo(slug="samara", api_id=63, name_ru="Самара"),
    "omsk": CityInfo(slug="omsk", api_id=64, name_ru="Омск"),
    "yekaterinburg": CityInfo(slug="yekaterinburg", api_id=68, name_ru="Екатеринбург"),
    "pskov": CityInfo(slug="pskov", api_id=69, name_ru="Псков"),
    "petrozavodsk": CityInfo(slug="petrozavodsk", api_id=70, name_ru="Петрозаводск"),
    "lipetsk": CityInfo(slug="lipetsk", api_id=71, name_ru="Липецк"),
    "izhevsk": CityInfo(slug="izhevsk", api_id=72, name_ru="Ижевск"),
    "tula": CityInfo(slug="tula", api_id=73, name_ru="Тула"),
    "ulyanovsk": CityInfo(slug="ulyanovsk", api_id=74, name_ru="Ульяновск"),
    "bryansk": CityInfo(slug="bryansk", api_id=75, name_ru="Брянск"),
    "vologda": CityInfo(slug="vologda", api_id=76, name_ru="Вологда"),
    "syktyvkar": CityInfo(slug="syktyvkar", api_id=77, name_ru="Сыктывкар"),
    "orel": CityInfo(slug="orel", api_id=78, name_ru="Орёл"),
    "ryazan": CityInfo(slug="ryazan", api_id=79, name_ru="Рязань"),
    "tver": CityInfo(slug="tver", api_id=80, name_ru="Тверь"),
    "belgorod": CityInfo(slug="belgorod", api_id=81, name_ru="Белгород"),
    "kursk": CityInfo(slug="kursk", api_id=82, name_ru="Курск"),
    "cherepovets": CityInfo(slug="cherepovets", api_id=83, name_ru="Череповец"),
    "penza": CityInfo(slug="penza", api_id=84, name_ru="Пенза"),
    "veliky_novgorod": CityInfo(slug="veliky_novgorod", api_id=85, name_ru="Великий Новгород"),
    "simferopol": CityInfo(slug="simferopol", api_id=91, name_ru="Симферополь"),
    "sevastopol": CityInfo(slug="sevastopol", api_id=92, name_ru="Севастополь"),
    "donetsk": CityInfo(slug="donetsk", api_id=102, name_ru="Донецк"),
    "lugansk": CityInfo(slug="lugansk", api_id=103, name_ru="Луганск"),
    "kherson": CityInfo(slug="kherson", api_id=104, name_ru="Херсон"),
    "zaporizhzhia": CityInfo(slug="zaporizhzhia", api_id=105, name_ru="Запорожье"),
}


_CITY_NAME_ALIASES: dict[str, str] = {
    "санкт-петербург": "spb",
    "санкт петербург": "spb",
    "петербург": "spb",
    "спб": "spb",
    "москва": "moscow",
    "екатеринбург": "yekaterinburg",
    "новосибирск": "novosibirsk",
    "нижний новгород": "nizhny_novgorod",
    "ростов-на-дону": "rostov_na_donu",
    "ростов на дону": "rostov_na_donu",
    "россия": "russia",
}


class InfluenzaDataSourceError(RuntimeError):
    """Ошибка доступа к источнику данных influenza API."""


class CityResolutionError(KeyError):
    """Город не найден в поддерживаемом реестре."""


class InfluenzaDbConfig(BaseModel):
    """Настройки подключения к CSV-отчёту influenza DB."""

    model_config = ConfigDict(extra="forbid")

    base_url: str = INFLUENZA_DB_BASE_URL
    report_name: str = INFLUENZA_DB_REPORT_NAME
    report_id: str = INFLUENZA_DB_REPORT_ID
    auth_token: str = DEFAULT_INFLUENZA_DB_AUTH_TOKEN
    timeout_seconds: float = Field(default=60.0, gt=0)
    encoding: str = "utf-8"
    separator: str = "|"

    @field_validator("base_url", "report_name", "report_id", "auth_token", "encoding", "separator")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Значение не должно быть пустым.")
        return value


class InfluenzaFetchRequest(BaseModel):
    """Параметры запроса к influenza DB."""

    model_config = ConfigDict(extra="forbid")

    city: str = Field(min_length=1)
    begin_year: int = Field(ge=1900, le=2100)
    begin_week: int = Field(ge=1, le=53)
    end_year: int = Field(ge=1900, le=2100)
    end_week: int = Field(ge=1, le=53)

    @field_validator("city")
    @classmethod
    def _normalize_city(cls, value: str) -> str:
        return normalize_city_slug(value)

    @model_validator(mode="after")
    def _valid_interval(self) -> "InfluenzaFetchRequest":
        start = iso_week_start_date(self.begin_year, self.begin_week)
        end = iso_week_start_date(self.end_year, self.end_week)
        if end < start:
            raise ValueError("Конец периода не может быть раньше начала периода.")
        return self


class InfluenzaFrameBundle(BaseModel):
    """Нормализованный результат загрузки данных influenza API."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    request: InfluenzaFetchRequest
    city_info: CityInfo
    raw_cases: Any
    cases: Any
    weekly: Any
    source_url: str

    @field_validator("raw_cases", "cases", "weekly")
    @classmethod
    def _must_be_dataframe(cls, value: Any) -> pd.DataFrame:
        if not isinstance(value, pd.DataFrame):
            raise TypeError("Ожидался pandas.DataFrame.")
        return value


class InfluenzaWeeklyOutput(BaseModel):
    """Описание сохранённого недельного ряда."""

    model_config = ConfigDict(extra="forbid")

    path: Path
    city: CityInfo
    start_date: date
    end_date: date
    row_count: int = Field(ge=0)


@dataclass(frozen=True)
class ModelComplexInfluenzaDataLike:
    """
    Минимальный структурный контракт объекта ``model_complex.InfluenzaData``.

    Класс не используется как runtime-зависимость. Он документирует, какие поля
    старого объекта реально нужны ai4epi при переносе notebook: ``cases_df`` и
    параметры периода/города.
    """

    cases_df: pd.DataFrame
    city: str
    begin_year: int
    begin_week: int
    end_year: int
    end_week: int


INFLUENZA_CASES_SCHEMA = TableSchema(
    required_columns=[
        "datetime",
        "region_name",
        "district_name",
        "sars_total_cases",
        "sars_cases_age_group_0",
        "sars_cases_age_group_1",
        "sars_cases_age_group_2",
        "sars_cases_age_group_3",
        "sars_cases_age_group_4",
        "sars_cases_age_group_5",
        "total_population",
        "population_age_group_0",
        "population_age_group_1",
        "population_age_group_2",
        "population_age_group_3",
        "population_age_group_4",
        "population_age_group_5",
        "rel_strain_0",
        "rel_strain_1",
        "rel_strain_2",
        "rel_strain_3",
        "real_cases_strain_0",
        "real_cases_strain_1",
        "real_cases_strain_2",
        "real_cases_strain_3",
    ],
    column_types={
        "datetime": "datetime",
        "sars_total_cases": "numeric",
        "total_population": "numeric",
    },
    non_null_columns=["datetime", "total_population"],
)

INFLUENZA_WEEKLY_SCHEMA = TableSchema(
    required_columns=[
        "datetime",
        "iso_year",
        "iso_week",
        "total_population",
        "total_cases_formula",
        "inc_per_10k",
    ],
    column_types={
        "datetime": "datetime",
        "iso_year": "integer",
        "iso_week": "integer",
        "total_population": "numeric",
        "total_cases_formula": "numeric",
        "inc_per_10k": "numeric",
    },
    non_null_columns=["datetime", "iso_year", "iso_week", "total_population", "inc_per_10k"],
    unique_key=["datetime"],
)


def safe_city_slug(city: str) -> str:
    """Возвращает безопасный slug для имени города или пользовательского ввода."""

    city = str(city).strip().lower()
    city = re.sub(r"\s+", "_", city)
    city = re.sub(r"[^0-9a-zA-Zа-яА-Я_-]+", "_", city)
    city = re.sub(r"_+", "_", city).strip("_")
    if not city:
        raise ValueError("Пустой slug города.")
    return city


def normalize_city_slug(city: str) -> str:
    """Нормализует пользовательский ввод к ключу ``CITY_REGISTRY``."""

    raw = str(city).strip().lower().replace("ё", "е")
    raw = re.sub(r"\s+", " ", raw)
    if raw in _CITY_NAME_ALIASES:
        return _CITY_NAME_ALIASES[raw]

    slug = safe_city_slug(raw)
    if slug in CITY_REGISTRY:
        return slug

    for info in CITY_REGISTRY.values():
        name = info.name_ru.lower().replace("ё", "е")
        if raw == name:
            return info.slug
    return slug


def resolve_city(city: str) -> CityInfo:
    """Возвращает описание города по slug, русскому имени или поддержанному alias."""

    slug = normalize_city_slug(city)
    try:
        return CITY_REGISTRY[slug]
    except KeyError as exc:
        available = ", ".join(sorted(CITY_REGISTRY))
        raise CityResolutionError(
            f"Неизвестный город {city!r}. Доступные slug: {available}."
        ) from exc


def list_supported_cities() -> list[CityInfo]:
    """Возвращает поддерживаемые города, отсортированные по русскому названию."""

    return sorted(CITY_REGISTRY.values(), key=lambda item: item.name_ru)


def iso_week_start_date(year: int, week: int) -> date:
    """Дата понедельника для ISO-года и ISO-недели."""

    try:
        return datetime.strptime(f"{year}-W{week}-1", "%G-W%V-%u").date()
    except ValueError as exc:
        raise ValueError(f"Некорректная ISO-неделя: {year} W{week:02d}.") from exc


def normalize_end_date(end_date: date | datetime | pd.Timestamp | str | None = None) -> date:
    """Нормализует дату окончания периода к ``datetime.date``."""

    if end_date is None:
        return date.today()
    if isinstance(end_date, pd.Timestamp):
        return end_date.date()
    if isinstance(end_date, datetime):
        return end_date.date()
    if isinstance(end_date, date):
        return end_date
    return pd.Timestamp(end_date).date()


def fetch_request_until_date(
    city: str,
    *,
    end_date: date | datetime | pd.Timestamp | str | None = None,
    begin_year: int = 2011,
    begin_week: int = 1,
) -> InfluenzaFetchRequest:
    """Строит запрос от фиксированной начальной недели до недели указанной даты."""

    end = normalize_end_date(end_date)
    iso = end.isocalendar()
    return InfluenzaFetchRequest(
        city=city,
        begin_year=begin_year,
        begin_week=begin_week,
        end_year=int(iso.year),
        end_week=int(iso.week),
    )


def build_influenza_db_url(
    request: InfluenzaFetchRequest,
    *,
    config: InfluenzaDbConfig | None = None,
) -> str:
    """Строит URL CSV-отчёта influenza DB."""

    cfg = config or InfluenzaDbConfig()
    city_info = resolve_city(request.city)
    query = urlencode(
        {
            "reportname": cfg.report_name,
            "id": cfg.report_id,
            "byear": request.begin_year,
            "bweek": request.begin_week,
            "eyear": request.end_year,
            "eweek": request.end_week,
            "district": city_info.api_id,
            "auth": cfg.auth_token,
        }
    )
    return f"{cfg.base_url}?{query}"


def fetch_influenza_csv_text(
    request: InfluenzaFetchRequest,
    *,
    config: InfluenzaDbConfig | None = None,
    session: Any | None = None,
) -> str:
    """Загружает исходный CSV-текст из influenza DB."""

    if requests is None and session is None:
        raise InfluenzaDataSourceError(
            "Для сетевой загрузки требуется пакет requests или пользовательский session."
        )

    cfg = config or InfluenzaDbConfig()
    url = build_influenza_db_url(request, config=cfg)
    http = session or requests
    response = http.get(url, timeout=cfg.timeout_seconds)
    if hasattr(response, "raise_for_status"):
        response.raise_for_status()

    content = getattr(response, "content", None)
    if content is not None:
        return content.decode(cfg.encoding)

    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text

    raise InfluenzaDataSourceError("HTTP-ответ не содержит ни content, ни text.")


def _date_from_api_row(row: Mapping[str, Any]) -> datetime:
    return datetime.strptime(f"{int(row['YEAR'])}-W{int(row['WEEK'])}-1", "%G-W%V-%u")


def parse_influenza_api_csv(
    csv_text: str,
    *,
    separator: str = "|",
) -> pd.DataFrame:
    """
    Преобразует CSV-ответ influenza DB в нормализованный ``cases`` DataFrame.

    Поведение соответствует классу ``model_complex.InfluenzaData``: дата
    создаётся из ``YEAR``/``WEEK``, оставляются только известные столбцы,
    добавляется группа ``15+`` и рассчитываются доли/оценки случаев по штаммам.
    """

    if not csv_text.strip():
        raise ValueError("Пустой CSV-ответ influenza DB.")

    raw = pd.read_csv(_text_stream(csv_text), sep=separator)
    return normalize_influenza_api_frame(raw)


def normalize_influenza_api_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Нормализует сырой DataFrame influenza DB к контракту ``cases``."""

    missing = [col for col in [*RAW_INFLUENZA_COLUMNS, "WEEK"] if col not in raw.columns]
    if missing:
        raise ValueError(f"В ответе influenza DB отсутствуют столбцы: {missing}.")

    df = raw.copy()
    df["YEAR"] = df.apply(_date_from_api_row, axis=1)
    df = df.loc[:, list(RAW_INFLUENZA_COLUMNS.keys())]
    df = df.rename(columns=RAW_INFLUENZA_COLUMNS)
    df["datetime"] = pd.to_datetime(df["datetime"])

    numeric_columns = [
        column
        for column in df.columns
        if column not in {"datetime", "region_name", "district_name"}
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["sars_cases_age_group_3"] = (
        df["sars_cases_age_group_4"] + df["sars_cases_age_group_5"]
    )
    df["population_age_group_3"] = (
        df["population_age_group_4"] + df["population_age_group_5"]
    )

    for strain_index in STRAIN_NAMES:
        tested_strain_col = f"tested_strain_{strain_index}"
        rel_col = f"rel_strain_{strain_index}"
        real_col = f"real_cases_strain_{strain_index}"
        df[rel_col] = df[tested_strain_col] / df["tested_total"]
        df[real_col] = (df[rel_col] * df["sars_total_cases"]).round()

    df = df.drop(
        columns=[
            "tested_total",
            "tested_strain_0",
            "tested_strain_1",
            "tested_strain_2",
            "tested_strain_3",
        ]
    )
    df = df.sort_values("datetime").reset_index(drop=True)
    INFLUENZA_CASES_SCHEMA.validate_dataframe(df, name="influenza_cases")
    return df


def weekly_incidence_from_cases(
    cases: pd.DataFrame,
    *,
    strain_indices: Sequence[int] = MODEL_INFLUENZA_TOTAL_STRAIN_INDICES,
) -> pd.DataFrame:
    """
    Строит недельный ряд заболеваемости на 10 000 населения.

    Формула перенесена из notebook: ``total_cases_formula`` — сумма оценённых
    случаев по штаммам ``A(H1)pdm09``, ``A(H3)`` и ``B``. Несубтипированный
    ``A`` по умолчанию не входит в целевую переменную, что сохраняет текущий
    контракт модели.
    """

    df = cases.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    strain_cols = [f"real_cases_strain_{idx}" for idx in strain_indices]
    missing = [col for col in strain_cols if col not in df.columns]
    if missing:
        raise ValueError(f"В cases DataFrame отсутствуют столбцы штаммов: {missing}.")

    df["total_cases_formula"] = df.fillna(0)[strain_cols].sum(axis=1)
    df["inc_per_10k"] = df["total_cases_formula"] / df["total_population"] * 10_000
    iso = df["datetime"].dt.isocalendar()
    df["iso_year"] = iso.year.astype(int)
    df["iso_week"] = iso.week.astype(int)

    weekly = df[
        [
            "datetime",
            "iso_year",
            "iso_week",
            "total_population",
            "total_cases_formula",
            "inc_per_10k",
        ]
    ].dropna(subset=["inc_per_10k", "total_population"])
    weekly = weekly.sort_values("datetime").reset_index(drop=True)
    INFLUENZA_WEEKLY_SCHEMA.validate_dataframe(weekly, name="influenza_weekly")
    return weekly


class InfluenzaDbClient:
    """Клиент для загрузки данных из CSV-эндпоинта influenza DB."""

    def __init__(
        self,
        config: InfluenzaDbConfig | None = None,
        *,
        session: Any | None = None,
    ) -> None:
        self.config = config or InfluenzaDbConfig()
        self.session = session

    def fetch(self, request: InfluenzaFetchRequest) -> InfluenzaFrameBundle:
        """Загружает и нормализует данные по точному недельному интервалу."""

        city_info = resolve_city(request.city)
        source_url = build_influenza_db_url(request, config=self.config)
        csv_text = fetch_influenza_csv_text(
            request,
            config=self.config,
            session=self.session,
        )
        raw_df = pd.read_csv(_text_stream(csv_text), sep=self.config.separator)
        cases = normalize_influenza_api_frame(raw_df)
        weekly = weekly_incidence_from_cases(cases)
        return InfluenzaFrameBundle(
            request=request,
            city_info=city_info,
            raw_cases=raw_df,
            cases=cases,
            weekly=weekly,
            source_url=source_url,
        )

    def fetch_until_date(
        self,
        city: str,
        *,
        end_date: date | datetime | pd.Timestamp | str | None = None,
        begin_year: int = 2011,
        begin_week: int = 1,
    ) -> InfluenzaFrameBundle:
        """Загружает ряд от начальной недели до ISO-недели указанной даты."""

        request = fetch_request_until_date(
            city,
            end_date=end_date,
            begin_year=begin_year,
            begin_week=begin_week,
        )
        return self.fetch(request)


def load_influenza_weekly_from_api(
    city: str,
    *,
    end_date: date | datetime | pd.Timestamp | str | None = None,
    begin_year: int = 2011,
    begin_week: int = 1,
    config: InfluenzaDbConfig | None = None,
    session: Any | None = None,
) -> InfluenzaFrameBundle:
    """Функциональная обёртка над ``InfluenzaDbClient.fetch_until_date``."""

    return InfluenzaDbClient(config=config, session=session).fetch_until_date(
        city,
        end_date=end_date,
        begin_year=begin_year,
        begin_week=begin_week,
    )


def load_influenza_from_model_complex_object(
    obj: Any,
    *,
    city: str | None = None,
    begin_year: int | None = None,
    begin_week: int | None = None,
    end_year: int | None = None,
    end_week: int | None = None,
) -> InfluenzaFrameBundle:
    """
    Адаптирует уже созданный объект ``model_complex.InfluenzaData``.

    Это нужно для воспроизводимого переноса notebook: если старый объект уже
    создан внешним кодом, ai4epi может принять его ``cases_df`` без повторного
    сетевого запроса.
    """

    if not hasattr(obj, "cases_df"):
        raise TypeError("Объект должен иметь атрибут cases_df.")
    cases = obj.cases_df.copy()
    if "datetime" not in cases.columns and "YEAR" in cases.columns:
        cases = cases.rename(columns={"YEAR": "datetime"})
    cases["datetime"] = pd.to_datetime(cases["datetime"])
    cases = cases.sort_values("datetime").reset_index(drop=True)
    INFLUENZA_CASES_SCHEMA.validate_dataframe(cases, name="influenza_cases")

    resolved_city = city or getattr(obj, "city", None)
    if resolved_city is None:
        raise ValueError("Не удалось определить город: передайте city явно.")
    city_info = resolve_city(str(resolved_city))

    request = InfluenzaFetchRequest(
        city=city_info.slug,
        begin_year=int(begin_year if begin_year is not None else getattr(obj, "begin_year")),
        begin_week=int(begin_week if begin_week is not None else getattr(obj, "begin_week")),
        end_year=int(end_year if end_year is not None else getattr(obj, "end_year")),
        end_week=int(end_week if end_week is not None else getattr(obj, "end_week")),
    )
    weekly = weekly_incidence_from_cases(cases)
    return InfluenzaFrameBundle(
        request=request,
        city_info=city_info,
        raw_cases=cases.copy(),
        cases=cases,
        weekly=weekly,
        source_url="model_complex.InfluenzaData",
    )


def default_weekly_output_path(
    city: str | CityInfo,
    *,
    output_dir: str | Path = ".",
    begin_year: int = 2011,
) -> Path:
    """Имя CSV-файла в формате, совместимом с notebook."""

    info = city if isinstance(city, CityInfo) else resolve_city(city)
    return (
        Path(output_dir)
        / f"influenza_{safe_city_slug(info.slug)}_weekly_{begin_year}_to_latest_formula.csv"
    )


def save_weekly_incidence_csv(
    weekly: pd.DataFrame,
    path: str | Path,
    *,
    city: str | CityInfo,
) -> InfluenzaWeeklyOutput:
    """Сохраняет недельный ряд и возвращает описание артефакта."""

    INFLUENZA_WEEKLY_SCHEMA.validate_dataframe(weekly, name="influenza_weekly")
    out_path = write_table(weekly, path, options=TableWriteOptions(index=False), schema=INFLUENZA_WEEKLY_SCHEMA)
    info = city if isinstance(city, CityInfo) else resolve_city(city)
    frame = weekly.copy()
    frame["datetime"] = pd.to_datetime(frame["datetime"])
    return InfluenzaWeeklyOutput(
        path=out_path,
        city=info,
        start_date=frame["datetime"].iloc[0].date(),
        end_date=frame["datetime"].iloc[-1].date(),
        row_count=int(len(frame)),
    )


def save_influenza_bundle(
    bundle: InfluenzaFrameBundle,
    *,
    output_dir: str | Path = ".",
    weekly_filename: str | None = None,
    save_raw: bool = False,
    save_cases: bool = False,
) -> dict[str, Path]:
    """Сохраняет таблицы из ``InfluenzaFrameBundle``."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    weekly_path = output / (
        weekly_filename
        or default_weekly_output_path(
            bundle.city_info,
            output_dir="../../..",
            begin_year=bundle.request.begin_year,
        ).name
    )
    paths: dict[str, Path] = {"weekly": write_table(bundle.weekly, weekly_path, options=TableWriteOptions(index=False), schema=INFLUENZA_WEEKLY_SCHEMA)}

    if save_raw:
        paths["raw_cases"] = write_table(
            bundle.raw_cases,
            output / f"influenza_{bundle.city_info.slug}_raw_cases.csv",
            options=TableWriteOptions(index=False),
        )
    if save_cases:
        paths["cases"] = write_table(
            bundle.cases,
            output / f"influenza_{bundle.city_info.slug}_cases_normalized.csv",
            options=TableWriteOptions(index=False),
            schema=INFLUENZA_CASES_SCHEMA,
        )
    return paths


__all__ = [
    "CITY_REGISTRY",
    "DEFAULT_INFLUENZA_DB_AUTH_TOKEN",
    "INFLUENZA_CASES_SCHEMA",
    "INFLUENZA_WEEKLY_SCHEMA",
    "InfluenzaDataSourceError",
    "CityResolutionError",
    "CityInfo",
    "InfluenzaDbConfig",
    "InfluenzaFetchRequest",
    "InfluenzaFrameBundle",
    "InfluenzaWeeklyOutput",
    "InfluenzaDbClient",
    "ModelComplexInfluenzaDataLike",
    "safe_city_slug",
    "normalize_city_slug",
    "resolve_city",
    "list_supported_cities",
    "iso_week_start_date",
    "normalize_end_date",
    "fetch_request_until_date",
    "build_influenza_db_url",
    "fetch_influenza_csv_text",
    "parse_influenza_api_csv",
    "normalize_influenza_api_frame",
    "weekly_incidence_from_cases",
    "load_influenza_weekly_from_api",
    "load_influenza_from_model_complex_object",
    "default_weekly_output_path",
    "save_weekly_incidence_csv",
    "save_influenza_bundle",
]

