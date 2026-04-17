from __future__ import annotations

import warnings


_PYNVML_DEPRECATION_PATTERN = r"The pynvml package is deprecated\..*"


def suppress_optional_dependency_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=_PYNVML_DEPRECATION_PATTERN,
        category=FutureWarning,
    )


def configure_runtime_warning_filters() -> None:
    suppress_optional_dependency_warnings()
    warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
    warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
    warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
