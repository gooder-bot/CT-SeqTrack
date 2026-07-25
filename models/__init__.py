"""Model registry.

Models are imported lazily so geometry/config utilities can be tested without
initializing PyTorch Lightning, nuScenes, or CUDA extensions.
"""

import importlib


def get_model(name):
    module = importlib.import_module(f"models.{name.lower()}")
    return getattr(module, name.upper())
