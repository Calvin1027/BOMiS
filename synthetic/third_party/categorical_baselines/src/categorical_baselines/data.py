from dataclasses import dataclass

import numpy as np
import pandas as pd

from .contracts import AdapterFailure, DatasetSchema


def categorical_schema(train: pd.DataFrame, target: str = "target") -> DatasetSchema:
    return DatasetSchema.from_outer_train(
        train,
        target=target,
        positive_label=1,
        numerical_columns=(),
        categorical_columns=tuple(c for c in train if c != target),
    )


def require_categorical(schema: DatasetSchema) -> None:
    if schema.numerical_columns or schema.feature_columns != schema.categorical_columns:
        raise AdapterFailure(
            "configuration", "CATEGORICAL_ONLY", "all features must be categorical"
        )


def read_train(path, target="target"):
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if target not in frame or not frame[target].isin(["0", "1"]).all():
        raise ValueError("training target must contain only literal 0/1 labels")
    frame[target] = frame[target].astype(int)
    return frame


@dataclass
class JointCodec:
    """Train-only integer symbols, including the target as a categorical column."""

    columns: list[str]
    domains: list[list[str]]
    target: str

    @classmethod
    def fit(cls, train, schema):
        require_categorical(schema)
        columns = [*schema.feature_columns, schema.target]
        domains = [list(map(str, schema.category_domains[c])) for c in schema.feature_columns]
        return cls(columns, domains + [["0", "1"]], schema.target)

    @property
    def cardinalities(self):
        return [len(d) for d in self.domains]

    def encode(self, frame):
        result = np.empty((len(frame), len(self.columns)), dtype=np.int64)
        for j, (column, levels) in enumerate(zip(self.columns, self.domains, strict=True)):
            mapped = frame[column].astype(str).map({v: i for i, v in enumerate(levels)})
            if mapped.isna().any():
                raise ValueError(f"unknown training category in {column}")
            result[:, j] = mapped.to_numpy(dtype=np.int64)
        return result

    def decode(self, codes):
        codes = np.asarray(codes)
        if codes.ndim != 2 or codes.shape[1] != len(self.columns):
            raise AdapterFailure("sample", "DECODE_SHAPE", str(codes.shape))
        if not np.isfinite(codes).all() or not np.equal(codes, np.floor(codes)).all():
            raise AdapterFailure(
                "sample", "INVALID_CATEGORY_CODE", "non-integer categorical output"
            )
        output = {}
        for j, (column, levels) in enumerate(zip(self.columns, self.domains, strict=True)):
            indices = codes[:, j].astype(np.int64)
            if np.any(indices < 0) or np.any(indices >= len(levels)):
                raise AdapterFailure("sample", "INVALID_CATEGORY_CODE", column)
            output[column] = np.asarray(levels, dtype=object)[indices]
        frame = pd.DataFrame(output)
        frame[self.target] = frame[self.target].astype(int)
        return frame
