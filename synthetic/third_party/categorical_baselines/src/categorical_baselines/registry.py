MODELS = (
    "smote",
    "smoten",
    "smote_enc",
    "ctgan",
    "tvae",
    "tabsyn",
    "tabnat",
    "arf",
    "stasy",
    "tabdiff",
)


def make_model(name, **options):
    if name not in MODELS:
        raise ValueError(f"unknown model {name}; choose from {MODELS}")
    if name in {"smote", "smoten"}:
        from .adapters.resampling import NominalResampler

        return NominalResampler(name, **options)
    if name == "smote_enc":
        from .adapters.smote_enc import SMOTEENCAdapter

        return SMOTEENCAdapter(**options)
    if name in {"ctgan", "tvae"}:
        from .adapters.inprocess import CTGANAdapter, TVAEAdapter

        return (CTGANAdapter if name == "ctgan" else TVAEAdapter)(**options)
    from importlib import import_module

    classes = {
        "tabsyn": "TabSynAdapter",
        "tabnat": "TabNATAdapter",
        "arf": "ARFAdapter",
        "stasy": "STaSyAdapter",
        "tabdiff": "TabDiffAdapter",
    }
    module = import_module(f".adapters.{name}", __package__)
    return getattr(module, classes[name])(**options)
