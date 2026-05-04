"""
Унифицированный слой ввода-вывода для ai4epi.

Модуль отвечает за воспроизводимое чтение и запись файловых артефактов:
табличных данных, JSON/JSONL-структур и манифестов результатов. Он не содержит
эпидемиологических правил, не строит прогнозы и не формирует ``GlobalContext``;
эти задачи принадлежат последующим слоям ``preprocessing.py``,
``forecasting.py`` и ``context_builders.py``.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_integer_dtype,
    is_numeric_dtype,
    is_string_dtype,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


JsonObject = dict[str, Any]
JsonArray = list[Any]
JsonRoot = JsonObject | JsonArray
TableFormat = Literal["csv", "json", "jsonl", "parquet", "excel"]
ColumnType = Literal["numeric", "integer", "string", "datetime", "boolean"]
ManifestStatus = Literal["planned", "written", "missing", "failed"]


class IOErrorContractError(ValueError):
    """Базовая ошибка контракта файлового ввода-вывода ai4epi."""


class FileFormatError(IOErrorContractError):
    """Ошибка определения или поддержки формата файла."""


class TableSchemaError(IOErrorContractError):
    """Ошибка соответствия DataFrame заявленной табличной схеме."""


class StrictModel(BaseModel):
    """Базовая pydantic-модель с закрытым контрактом."""

    model_config = ConfigDict(
        extra="forbid",
        populate_by_name=True,
        validate_assignment=True,
        arbitrary_types_allowed=True,
    )


class TableReadOptions(StrictModel):
    """Параметры чтения табличного файла."""

    format: TableFormat | None = None
    encoding: str = Field(default="utf-8-sig", min_length=1)
    sep: str = Field(default=",", min_length=1)
    decimal: str = Field(default=".", min_length=1)
    sheet_name: str | int | None = 0
    parse_dates: tuple[str, ...] = Field(default_factory=tuple)
    dtype: dict[str, str] | None = None
    usecols: tuple[str, ...] | None = None
    pandas_kwargs: JsonObject = Field(default_factory=dict)


class TableWriteOptions(StrictModel):
    """Параметры записи табличного файла."""

    format: TableFormat | None = None
    encoding: str = Field(default="utf-8-sig", min_length=1)
    sep: str = Field(default=",", min_length=1)
    decimal: str = Field(default=".", min_length=1)
    index: bool = False
    sheet_name: str = Field(default="Sheet1", min_length=1)
    float_format: str | None = None
    pandas_kwargs: JsonObject = Field(default_factory=dict)


class TableSchema(StrictModel):
    """
    Проверяемый контракт табличного набора данных.

    ``required_columns`` задаёт обязательные столбцы. ``optional_columns`` вместе
    с ``allow_extra_columns=False`` позволяет зафиксировать точный набор
    допустимых столбцов. Типы в ``column_types`` проверяются через публичные
    pandas-предикаты и являются строгими категориями, а не неявным приведением.
    """

    required_columns: tuple[str, ...] = Field(default_factory=tuple)
    optional_columns: tuple[str, ...] = Field(default_factory=tuple)
    allow_extra_columns: bool = True
    min_rows: int = Field(default=0, ge=0)
    non_null_columns: tuple[str, ...] = Field(default_factory=tuple)
    unique_key: tuple[str, ...] = Field(default_factory=tuple)
    column_types: dict[str, ColumnType] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_schema_self_consistency(self) -> "TableSchema":
        duplicates = _duplicates([*self.required_columns, *self.optional_columns])
        if duplicates:
            raise ValueError(f"Columns cannot be both required/optional more than once: {duplicates!r}.")

        known_columns = set(self.required_columns) | set(self.optional_columns)
        if not self.allow_extra_columns:
            for column in [*self.non_null_columns, *self.unique_key, *self.column_types.keys()]:
                if column not in known_columns:
                    raise ValueError(
                        f"Column {column!r} is constrained but is absent from required/optional columns."
                    )
        return self

    def validate_dataframe(self, df: pd.DataFrame, *, name: str = "table") -> None:
        """Проверить DataFrame и выбросить TableSchemaError при нарушении."""

        validate_table_schema(df, self, name=name)


class TableBundleItem(StrictModel):
    """Описание одного табличного файла в наборе артефактов."""

    name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_\-]*$")
    path: Path
    read_options: TableReadOptions = Field(default_factory=TableReadOptions)
    write_options: TableWriteOptions = Field(default_factory=TableWriteOptions)
    validation_schema: TableSchema | None = Field(default=None, alias="schema")
    required: bool = True


class TableBundleSpec(StrictModel):
    """Описание набора таблиц, расположенных относительно одного каталога."""

    base_dir: Path
    items: tuple[TableBundleItem, ...]

    @field_validator("items")
    @classmethod
    def validate_unique_item_names(cls, value: tuple[TableBundleItem, ...]) -> tuple[TableBundleItem, ...]:
        names = [item.name for item in value]
        duplicates = _duplicates(names)
        if duplicates:
            raise ValueError(f"Table bundle item names must be unique: {duplicates!r}.")
        return value

    def resolve_path(self, item: TableBundleItem) -> Path:
        """Вернуть абсолютный путь элемента bundle."""

        return resolve_path(item.path, base_dir=self.base_dir)


class TableBundle(StrictModel):
    """Загруженный набор таблиц."""

    base_dir: Path
    tables: dict[str, pd.DataFrame]
    missing_optional: tuple[str, ...] = Field(default_factory=tuple)

    def require(self, name: str) -> pd.DataFrame:
        """Вернуть таблицу по имени или выбросить KeyError."""

        if name not in self.tables:
            raise KeyError(f"Table {name!r} is absent from the loaded bundle.")
        return self.tables[name]


class ArtifactRecord(StrictModel):
    """Запись о файловом артефакте одного запуска."""

    name: str = Field(min_length=1)
    path: Path
    kind: str = Field(min_length=1)
    status: ManifestStatus = "planned"
    size_bytes: int | None = Field(default=None, ge=0)
    sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: JsonObject = Field(default_factory=dict)

    @classmethod
    def from_file(
        cls,
        *,
        name: str,
        path: str | Path,
        kind: str,
        status: ManifestStatus | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "ArtifactRecord":
        """Создать запись манифеста по фактическому состоянию файла."""

        file_path = Path(path)
        exists = file_path.exists() and file_path.is_file()
        return cls(
            name=name,
            path=file_path,
            kind=kind,
            status=status or ("written" if exists else "missing"),
            size_bytes=file_path.stat().st_size if exists else None,
            sha256=sha256_file(file_path) if exists else None,
            metadata=dict(metadata or {}),
        )


class ArtifactManifest(StrictModel):
    """Манифест файловых артефактов одного запуска."""

    run_id: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    records: list[ArtifactRecord] = Field(default_factory=list)
    metadata: JsonObject = Field(default_factory=dict)

    @field_validator("records")
    @classmethod
    def validate_unique_records(cls, value: list[ArtifactRecord]) -> list[ArtifactRecord]:
        names = [record.name for record in value]
        duplicates = _duplicates(names)
        if duplicates:
            raise ValueError(f"Artifact record names must be unique: {duplicates!r}.")
        return value

    def add_file(
        self,
        *,
        name: str,
        path: str | Path,
        kind: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> ArtifactRecord:
        """Добавить или заменить запись о файле и вернуть её."""

        record = ArtifactRecord.from_file(name=name, path=path, kind=kind, metadata=metadata)
        self.records = [old for old in self.records if old.name != name]
        self.records.append(record)
        return record

    def to_public_dict(self) -> JsonObject:
        """Вернуть JSON-сериализуемое представление манифеста."""

        return self.model_dump(mode="json")


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def ensure_directory(path: str | Path) -> Path:
    """Создать каталог, если он отсутствует, и вернуть абсолютный Path."""

    directory = Path(path).expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    return directory.resolve()


def ensure_parent_dir(path: str | Path) -> Path:
    """Создать родительский каталог файла и вернуть нормализованный путь файла."""

    file_path = Path(path).expanduser()
    file_path.parent.mkdir(parents=True, exist_ok=True)
    return file_path


def resolve_path(path: str | Path, *, base_dir: str | Path | None = None) -> Path:
    """Разрешить путь относительно base_dir или текущего процесса."""

    value = Path(path).expanduser()
    if value.is_absolute():
        return value.resolve()
    base = Path(base_dir).expanduser() if base_dir is not None else Path.cwd()
    return (base / value).resolve()


def require_existing_file(path: str | Path) -> Path:
    """Проверить существование обычного файла."""

    file_path = Path(path).expanduser().resolve()
    if not file_path.exists():
        raise FileNotFoundError(f"File does not exist: {file_path}")
    if not file_path.is_file():
        raise IOErrorContractError(f"Path is not a regular file: {file_path}")
    return file_path


def require_existing_directory(path: str | Path) -> Path:
    """Проверить существование каталога."""

    directory = Path(path).expanduser().resolve()
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    if not directory.is_dir():
        raise IOErrorContractError(f"Path is not a directory: {directory}")
    return directory


def list_files(
    directory: str | Path,
    *,
    pattern: str = "*",
    recursive: bool = False,
    require_non_empty: bool = False,
) -> list[Path]:
    """Вернуть отсортированный список файлов по glob-шаблону."""

    root = require_existing_directory(directory)
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    files = sorted(path for path in iterator if path.is_file())
    if require_non_empty and not files:
        raise FileNotFoundError(f"No files match pattern {pattern!r} in {root}.")
    return files


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def read_json(path: str | Path) -> JsonRoot:
    """Прочитать JSON-файл с корнем object или array."""

    file_path = require_existing_file(path)
    with file_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, (dict, list)):
        raise IOErrorContractError("JSON root must be an object or an array.")
    return data


def read_json_object(path: str | Path) -> JsonObject:
    """Прочитать JSON-файл, требуя объект в корне."""

    data = read_json(path)
    if not isinstance(data, dict):
        raise IOErrorContractError("JSON root must be an object.")
    return data


def read_json_array(path: str | Path) -> JsonArray:
    """Прочитать JSON-файл, требуя массив в корне."""

    data = read_json(path)
    if not isinstance(data, list):
        raise IOErrorContractError("JSON root must be an array.")
    return data


def write_json(
    data: Any,
    path: str | Path,
    *,
    indent: int = 2,
    ensure_ascii: bool = False,
    sort_keys: bool = False,
) -> Path:
    """Записать JSON-совместимые данные в файл."""

    output_path = ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(to_jsonable(data), stream, ensure_ascii=ensure_ascii, indent=indent, sort_keys=sort_keys)
    return output_path.resolve()


def read_jsonl(path: str | Path) -> list[JsonObject]:
    """Прочитать JSON Lines файл как список объектов."""

    file_path = require_existing_file(path)
    records: list[JsonObject] = []
    with file_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if not isinstance(record, dict):
                raise IOErrorContractError(f"JSONL line {line_number} must be an object.")
            records.append(record)
    return records


def write_jsonl(records: Sequence[Mapping[str, Any]], path: str | Path) -> Path:
    """Записать последовательность JSON-объектов в JSON Lines файл."""

    output_path = ensure_parent_dir(path)
    with output_path.open("w", encoding="utf-8") as stream:
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise IOErrorContractError(f"JSONL record at index {index} must be a mapping.")
            stream.write(json.dumps(to_jsonable(dict(record)), ensure_ascii=False))
            stream.write("\n")
    return output_path.resolve()


# ---------------------------------------------------------------------------
# Table helpers
# ---------------------------------------------------------------------------


def infer_table_format(path: str | Path) -> TableFormat:
    """Определить табличный формат по расширению файла."""

    suffix = Path(path).suffix.lower()
    if suffix == ".csv":
        return "csv"
    if suffix == ".json":
        return "json"
    if suffix == ".jsonl":
        return "jsonl"
    if suffix == ".parquet":
        return "parquet"
    if suffix in {".xlsx", ".xls"}:
        return "excel"
    raise FileFormatError(f"Unsupported table file extension: {suffix!r}.")


def read_table(
    path: str | Path,
    *,
    options: TableReadOptions | None = None,
    schema: TableSchema | None = None,
    name: str | None = None,
) -> pd.DataFrame:
    """Прочитать таблицу в pandas DataFrame и при необходимости проверить схему."""

    file_path = require_existing_file(path)
    read_options = options or TableReadOptions()
    table_format = read_options.format or infer_table_format(file_path)
    pandas_kwargs = dict(read_options.pandas_kwargs)

    if table_format == "csv":
        df = pd.read_csv(
            file_path,
            encoding=read_options.encoding,
            sep=read_options.sep,
            decimal=read_options.decimal,
            parse_dates=list(read_options.parse_dates) or None,
            dtype=read_options.dtype,
            usecols=list(read_options.usecols) if read_options.usecols is not None else None,
            **pandas_kwargs,
        )
    elif table_format == "json":
        df = pd.read_json(file_path, encoding=read_options.encoding, **pandas_kwargs)
    elif table_format == "jsonl":
        df = pd.read_json(file_path, encoding=read_options.encoding, lines=True, **pandas_kwargs)
    elif table_format == "parquet":
        df = pd.read_parquet(file_path, **pandas_kwargs)
    elif table_format == "excel":
        df = pd.read_excel(
            file_path,
            sheet_name=read_options.sheet_name,
            dtype=read_options.dtype,
            usecols=list(read_options.usecols) if read_options.usecols is not None else None,
            **pandas_kwargs,
        )
    else:  # pragma: no cover: закрыто Literal и infer_table_format
        raise FileFormatError(f"Unsupported table format: {table_format!r}.")

    if not isinstance(df, pd.DataFrame):
        raise IOErrorContractError(
            "Table reader returned a non-DataFrame object. For Excel files, specify one sheet_name."
        )

    if schema is not None:
        validate_table_schema(df, schema, name=name or str(file_path))
    return df


def write_table(
    df: pd.DataFrame,
    path: str | Path,
    *,
    options: TableWriteOptions | None = None,
    schema: TableSchema | None = None,
    name: str | None = None,
) -> Path:
    """Записать DataFrame в файл и при необходимости предварительно проверить схему."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    write_options = options or TableWriteOptions()
    output_path = ensure_parent_dir(path)
    table_format = write_options.format or infer_table_format(output_path)

    if schema is not None:
        validate_table_schema(df, schema, name=name or str(output_path))

    pandas_kwargs = dict(write_options.pandas_kwargs)
    if table_format == "csv":
        df.to_csv(
            output_path,
            encoding=write_options.encoding,
            sep=write_options.sep,
            decimal=write_options.decimal,
            index=write_options.index,
            float_format=write_options.float_format,
            **pandas_kwargs,
        )
    elif table_format == "json":
        df.to_json(output_path, force_ascii=False, orient=pandas_kwargs.pop("orient", "records"), **pandas_kwargs)
    elif table_format == "jsonl":
        df.to_json(output_path, force_ascii=False, orient="records", lines=True, **pandas_kwargs)
    elif table_format == "parquet":
        df.to_parquet(output_path, index=write_options.index, **pandas_kwargs)
    elif table_format == "excel":
        df.to_excel(output_path, index=write_options.index, sheet_name=write_options.sheet_name, **pandas_kwargs)
    else:  # pragma: no cover
        raise FileFormatError(f"Unsupported table format: {table_format!r}.")

    return output_path.resolve()


def validate_table_schema(df: pd.DataFrame, schema: TableSchema, *, name: str = "table") -> None:
    """Проверить DataFrame по явной схеме."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("df must be a pandas DataFrame.")

    columns = list(map(str, df.columns))
    duplicate_columns = _duplicates(columns)
    if duplicate_columns:
        raise TableSchemaError(f"{name}: duplicate columns: {duplicate_columns!r}.")

    missing = [column for column in schema.required_columns if column not in df.columns]
    if missing:
        raise TableSchemaError(f"{name}: missing required columns: {missing!r}.")

    if not schema.allow_extra_columns:
        allowed = set(schema.required_columns) | set(schema.optional_columns)
        extra = [column for column in df.columns if column not in allowed]
        if extra:
            raise TableSchemaError(f"{name}: unexpected columns: {extra!r}.")

    if len(df) < schema.min_rows:
        raise TableSchemaError(f"{name}: expected at least {schema.min_rows} rows, got {len(df)}.")

    for column in schema.non_null_columns:
        _require_column(df, column, name=name)
        if df[column].isna().any():
            raise TableSchemaError(f"{name}: column {column!r} contains null values.")

    if schema.unique_key:
        for column in schema.unique_key:
            _require_column(df, column, name=name)
        duplicated = df.duplicated(subset=list(schema.unique_key), keep=False)
        if bool(duplicated.any()):
            count = int(duplicated.sum())
            raise TableSchemaError(f"{name}: unique key {schema.unique_key!r} has {count} duplicated rows.")

    for column, expected_type in schema.column_types.items():
        _require_column(df, column, name=name)
        if not _series_matches_type(df[column], expected_type):
            actual = str(df[column].dtype)
            raise TableSchemaError(
                f"{name}: column {column!r} must be {expected_type!r}, got dtype {actual!r}."
            )


def read_table_bundle(spec: TableBundleSpec) -> TableBundle:
    """Загрузить набор таблиц по декларативной спецификации."""

    tables: dict[str, pd.DataFrame] = {}
    missing_optional: list[str] = []

    for item in spec.items:
        path = spec.resolve_path(item)
        if not path.exists():
            if item.required:
                raise FileNotFoundError(f"Required table {item.name!r} does not exist: {path}")
            missing_optional.append(item.name)
            continue
        tables[item.name] = read_table(
            path,
            options=item.read_options,
            schema=item.validation_schema,
            name=item.name,
        )

    return TableBundle(base_dir=resolve_path(spec.base_dir), tables=tables, missing_optional=tuple(missing_optional))


def write_table_bundle(bundle: TableBundle, spec: TableBundleSpec) -> dict[str, Path]:
    """Записать набор таблиц согласно спецификации."""

    written: dict[str, Path] = {}
    for item in spec.items:
        if item.name not in bundle.tables:
            if item.required:
                raise KeyError(f"Required table {item.name!r} is absent from bundle.")
            continue
        path = spec.resolve_path(item)
        written[item.name] = write_table(
            bundle.tables[item.name],
            path,
            options=item.write_options,
            schema=item.validation_schema,
            name=item.name,
        )
    return written


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------


def save_artifact_manifest(manifest: ArtifactManifest, path: str | Path) -> Path:
    """Сохранить манифест артефактов в JSON."""

    return write_json(manifest.to_public_dict(), path)


def load_artifact_manifest(path: str | Path) -> ArtifactManifest:
    """Загрузить манифест артефактов из JSON."""

    return ArtifactManifest.model_validate(read_json_object(path))


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Вычислить SHA-256 для существующего файла."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")
    file_path = require_existing_file(path)
    digest = hashlib.sha256()
    with file_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------


def to_jsonable(value: Any) -> Any:
    """Преобразовать распространённые Python/pandas/pydantic-объекты к JSON."""

    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, TypeError, AttributeError):
            pass
    return value


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _duplicates(values: Sequence[Any]) -> list[Any]:
    seen: set[Any] = set()
    duplicate_values: list[Any] = []
    for value in values:
        if value in seen and value not in duplicate_values:
            duplicate_values.append(value)
        seen.add(value)
    return duplicate_values


def _require_column(df: pd.DataFrame, column: str, *, name: str) -> None:
    if column not in df.columns:
        raise TableSchemaError(f"{name}: constrained column {column!r} is absent.")


def _series_matches_type(series: pd.Series, expected_type: ColumnType) -> bool:
    if expected_type == "numeric":
        return bool(is_numeric_dtype(series))
    if expected_type == "integer":
        return bool(is_integer_dtype(series))
    if expected_type == "string":
        return bool(is_string_dtype(series) or series.dtype == object)
    if expected_type == "datetime":
        return bool(is_datetime64_any_dtype(series))
    if expected_type == "boolean":
        return bool(is_bool_dtype(series))
    raise FileFormatError(f"Unsupported column type: {expected_type!r}.")


__all__ = [
    "ArtifactManifest",
    "ArtifactRecord",
    "ColumnType",
    "FileFormatError",
    "IOErrorContractError",
    "JsonArray",
    "JsonObject",
    "JsonRoot",
    "ManifestStatus",
    "StrictModel",
    "TableBundle",
    "TableBundleItem",
    "TableBundleSpec",
    "TableFormat",
    "TableReadOptions",
    "TableSchema",
    "TableSchemaError",
    "TableWriteOptions",
    "ensure_directory",
    "ensure_parent_dir",
    "infer_table_format",
    "list_files",
    "load_artifact_manifest",
    "read_json",
    "read_json_array",
    "read_json_object",
    "read_jsonl",
    "read_table",
    "read_table_bundle",
    "require_existing_directory",
    "require_existing_file",
    "resolve_path",
    "save_artifact_manifest",
    "sha256_file",
    "to_jsonable",
    "validate_table_schema",
    "write_json",
    "write_jsonl",
    "write_table",
    "write_table_bundle",
]

