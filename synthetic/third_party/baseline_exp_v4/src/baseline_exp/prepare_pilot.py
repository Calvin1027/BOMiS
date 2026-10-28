from __future__ import annotations

import argparse
import io
import json
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .provenance import sha256_file, write_json_new


@dataclass(frozen=True)
class Spec:
    name: str
    uci_id: int
    numerical: tuple[str, ...]
    categorical: tuple[str, ...]
    target: str
    positive_rule: str
    rename: tuple[tuple[str, str], ...] = ()


BANK = Spec(
    "bank_marketing", 222,
    ("age", "balance", "day", "campaign", "pdays", "previous"),
    ("job", "marital", "education", "default", "housing", "loan", "contact", "month", "poutcome"),
    "y", "yes", (("day_of_week", "day"),),
)
SHOPPERS = Spec(
    "online_shoppers", 468,
    ("administrative", "administrative_duration", "informational", "informational_duration", "productrelated", "productrelated_duration", "bouncerates", "exitrates", "pagevalues", "specialday"),
    ("month", "operating_systems", "browser", "region", "traffic_type", "visitor_type", "weekend"),
    "revenue", "true",
    (("operatingsystems", "operating_systems"), ("traffictype", "traffic_type"), ("visitortype", "visitor_type")),
)
CENSUS = Spec(
    "census_income_kdd", 117,
    ("age", "wage_per_hour", "capital_gains", "capital_losses", "dividends_from_stocks", "num_persons_worked_for_employer", "weeks_worked_in_year"),
    ("class_of_worker", "detailed_industry_recode", "detailed_occupation_recode", "education", "enroll_in_edu_inst_last_wk", "marital_stat", "major_industry_code", "major_occupation_code", "race", "hispanic_origin", "sex", "member_of_a_labor_union", "reason_for_unemployment", "full_or_part_time_employment_stat", "tax_filer_stat", "region_of_previous_residence", "state_of_previous_residence", "detailed_household_and_family_stat", "detailed_household_summary_in_household", "migration_code_change_in_msa", "migration_code_change_in_reg", "migration_code_move_within_reg", "live_in_this_house_1_year_ago", "migration_prev_res_in_sunbelt", "family_members_under_18", "country_of_birth_father", "country_of_birth_mother", "country_of_birth_self", "citizenship", "own_business_or_self_employed", "fill_inc_questionnaire_for_veteran_s_admin", "veterans_benefits", "year"),
    "income", "census_high_income",
    (("aage", "age"), ("aclswkr", "class_of_worker"), ("adtink", "detailed_industry_recode"), ("adtocc", "detailed_occupation_recode"), ("ahga", "education"), ("ahrspay", "wage_per_hour"), ("ahscol", "enroll_in_edu_inst_last_wk"), ("amaritl", "marital_stat"), ("amjind", "major_industry_code"), ("amjocc", "major_occupation_code"), ("arace", "race"), ("areorgn", "hispanic_origin"), ("asex", "sex"), ("aunmem", "member_of_a_labor_union"), ("auntype", "reason_for_unemployment"), ("awkstat", "full_or_part_time_employment_stat"), ("capgain", "capital_gains"), ("gaploss", "capital_losses"), ("divval", "dividends_from_stocks"), ("filestat", "tax_filer_stat"), ("grinreg", "region_of_previous_residence"), ("grinst", "state_of_previous_residence"), ("hhdfmx", "detailed_household_and_family_stat"), ("hhdrel", "detailed_household_summary_in_household"), ("migmtr1", "migration_code_change_in_msa"), ("migmtr3", "migration_code_change_in_reg"), ("migmtr4", "migration_code_move_within_reg"), ("migsame", "live_in_this_house_1_year_ago"), ("migsun", "migration_prev_res_in_sunbelt"), ("noemp", "num_persons_worked_for_employer"), ("parent", "family_members_under_18"), ("pefntvty", "country_of_birth_father"), ("pemntvty", "country_of_birth_mother"), ("penatvty", "country_of_birth_self"), ("prcitshp", "citizenship"), ("seotr", "own_business_or_self_employed"), ("vetqva", "fill_inc_questionnaire_for_veteran_s_admin"), ("vetyn", "veterans_benefits"), ("wkswork", "weeks_worked_in_year")),
)


def canonical(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def download_source(spec: Spec) -> bytes:
    url = f"https://archive.ics.uci.edu/static/public/{spec.uci_id}/data.csv"
    with urllib.request.urlopen(url, timeout=180) as response:
        return response.read()


def prepare(
    spec: Spec,
    root: Path,
    seed: int,
    *,
    purpose: str = "full-size engineering pilot; not the scientific final run",
    schema_version: str = "dataset.pilot.v1",
    source_bytes: bytes | None = None,
) -> Path:
    destination = root / spec.name / f"split-{seed}"
    destination.mkdir(parents=True, exist_ok=False)
    url = f"https://archive.ics.uci.edu/static/public/{spec.uci_id}/data.csv"
    raw_path = destination / "uci_data.csv"
    payload = download_source(spec) if source_bytes is None else source_bytes
    with raw_path.open("xb") as handle:
        handle.write(payload)
    frame = pd.read_csv(io.BytesIO(raw_path.read_bytes()))
    frame.columns = [canonical(column) for column in frame.columns]
    frame = frame.rename(columns=dict(spec.rename))
    required = [*spec.numerical, *spec.categorical, spec.target]
    missing = sorted(set(required) - set(frame.columns))
    if missing or len(set(frame.columns)) != len(frame.columns):
        raise ValueError(f"{spec.name}: invalid source schema; missing={missing}")
    clean = frame.loc[:, required].copy()
    target = clean.pop(spec.target).astype("string").str.strip().str.lower().str.replace(" ", "", regex=False)
    if spec.positive_rule == "yes":
        y = target.eq("yes")
    elif spec.positive_rule == "true":
        y = target.isin({"true", "1", "yes"})
    else:
        y = target.str.contains("50000+", regex=False)
    for column in spec.numerical:
        clean[column] = pd.to_numeric(clean[column], errors="coerce")
    for column in spec.categorical:
        clean[column] = clean[column].astype("string").str.strip().fillna("__MISSING__").replace({"?": "__MISSING__"}).astype(str)
    clean["target"] = y.astype(np.int8)
    train_valid, test = train_test_split(clean, test_size=0.15, random_state=seed, stratify=clean["target"])
    train, validation = train_test_split(train_valid, test_size=0.15 / 0.85, random_state=seed + 1, stratify=train_valid["target"])
    medians = {column: float(train[column].median()) for column in spec.numerical}
    if not all(np.isfinite(value) for value in medians.values()):
        raise ValueError(f"{spec.name}: nonfinite outer-train median")
    splits = {}
    for role, part in (("outer_train", train), ("validation", validation), ("test", test)):
        part = part.copy()
        for column, median in medians.items():
            part[column] = part[column].fillna(median)
        part = part.loc[:, [*spec.numerical, *spec.categorical, "target"]].reset_index(drop=True)
        path = destination / f"{role}.csv.gz"
        with path.open("xb") as handle:
            part.to_csv(handle, index=False, compression="gzip")
        splits[role] = {"path": str(path.resolve()), "sha256": sha256_file(path), "rows": len(part), "minority_rows": int(part["target"].sum())}
    majority = int((train["target"] == 0).sum())
    manifest = {
        "schema_version": schema_version, "status": "FROZEN_FOR_EXECUTION",
        "purpose": purpose,
        "name": spec.name, "uci_id": spec.uci_id, "source_url": url,
        "source_sha256": sha256_file(raw_path), "split_seed": seed,
        "target": "target", "positive_label": 1,
        "numerical_columns": list(spec.numerical), "categorical_columns": list(spec.categorical),
        "M_d": majority, "minority_bank_rule": "match outer-train majority count",
        "imputation": {"fit_on": "outer_train_only", "numerical_medians": medians, "categorical_missing_token": "__MISSING__"},
        "splits": splits,
    }
    write_json_new(destination / "dataset.json", manifest)
    return destination / "dataset.json"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260904)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=False)
    for item in (BANK, SHOPPERS, CENSUS):
        print(prepare(item, args.output_root, args.seed))
