# Copyright 2026 The xLLM Authors. Licensed under the Apache License, Version 2.0.
"""Load xLLM's source-built BangC operators into ``torch.ops.xllm_mlu``."""

from pathlib import Path

import torch
import torch_mlu  # Registers the MLU device and stream backend before loading.

__all__ = ["load_library"]
_LOADED = False


def load_library() -> None:
    """Load the packaged library without compiling or resolving external kernels."""
    global _LOADED
    if _LOADED:
        return
    library = Path(__file__).resolve().parents[4] / "libxllm_mlu_ops.so"
    if not library.is_file():
        raise ImportError(
            f"Missing xLLM BangC library: {library}. Build and install xLLM with `python setup.py build --device mlu`."
        )
    torch.ops.load_library(str(library))
    _LOADED = True


load_library()
