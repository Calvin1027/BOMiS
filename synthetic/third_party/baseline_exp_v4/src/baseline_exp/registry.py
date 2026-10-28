from __future__ import annotations

from pathlib import Path
from typing import Any

from .adapters import (
    ADASYNOHEAdapter,
    BorderlineSMOTEOHEAdapter,
    CTGANAdapter,
    GaussianCopulaAdapter,
    IndependentMarginalAdapter,
    RandomOverSamplerAdapter,
    SMOTEENCAdapter,
    SMOTENCAdapter,
    TVAEAdapter,
)
from .adapters.ctabgan_plus import CTABGANPlusAdapter
from .adapters.forest_flow import ForestFlowAdapter
from .adapters.tabddpm import TabDDPMAdapter
from .adapters.tabsyn_family import TabbyFlowAdapter, TabSynAdapter
from .contracts import AdapterFailure, GeneratorAdapter


def build_adapter(name: str, spec: dict[str, Any], *, project_root: Path) -> GeneratorAdapter:
    params = dict(spec)
    source = params.pop("source_dir", None)
    if params.pop("enabled", True) is not True:
        raise AdapterFailure("configuration", "MODEL_DISABLED", str(spec.get("reason", name)))
    if name == "random_over_sampler":
        return RandomOverSamplerAdapter()
    if name == "independent_marginal":
        return IndependentMarginalAdapter()
    if name == "gaussian_copula":
        params.pop("max_raw_multiplier", None)
        params.pop("raw_batch_size", None)
        return GaussianCopulaAdapter()
    if name == "smotenc":
        return SMOTENCAdapter(**params)
    if name == "smote_enc":
        return SMOTEENCAdapter(**params)
    if name == "borderline_smote":
        return BorderlineSMOTEOHEAdapter(**params)
    if name == "adasyn":
        return ADASYNOHEAdapter(**params)
    if name == "ctgan":
        return CTGANAdapter(**params)
    if name == "tvae":
        return TVAEAdapter(**params)
    if source is None:
        raise AdapterFailure("configuration", "SOURCE_PATH_MISSING", name)
    source_path = (project_root / source).resolve() if not Path(source).is_absolute() else Path(source)
    if "patch_file" in params:
        patch = Path(params["patch_file"])
        params["patch_file"] = (project_root / patch).resolve() if not patch.is_absolute() else patch
    if name == "ctabgan_plus":
        return CTABGANPlusAdapter(source_path, **params)
    if name == "tabddpm":
        return TabDDPMAdapter(source_path, **params)
    if name == "tabsyn":
        return TabSynAdapter(source_path, **params)
    if name == "tabbyflow":
        return TabbyFlowAdapter(source_path, **params)
    if name == "forest_flow":
        return ForestFlowAdapter(source_path, **params)
    if name == "tabdiff":
        raise AdapterFailure(
            "configuration",
            "TABDIFF_PATCH_NOT_AUDITED",
            "official TabDiff lacks independent fit/sample seed and artifact-output hooks",
        )
    raise AdapterFailure("configuration", "UNKNOWN_MODEL", name)
