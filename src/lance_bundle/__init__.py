from .runtime import load, load_dataset

__all__ = ["save", "ExportDataset", "load", "load_dataset"]

def __getattr__(name: str):
    if name == "save":
        from .export import save as _save
        return _save
    if name == "ExportDataset":
        from .export import ExportDataset as _ExportDataset
        return _ExportDataset

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return __all__
