import os
import sys
import torch

# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

HMR2_ROOT = os.path.join(
    PROJECT_ROOT,
    "third_party",
    "4D-Humans"
)

sys.path.insert(0, HMR2_ROOT)

# ---------------------------------------------------------
# HMR2 imports
# ---------------------------------------------------------

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import load_hmr2, download_models, DEFAULT_CHECKPOINT


# ---------------------------------------------------------
# Load HMR2
# ---------------------------------------------------------

print("Loading HMR2...")

download_models(CACHE_DIR_4DHUMANS)

model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)

print("HMR2 loaded.")


# ---------------------------------------------------------
# Inspect SMPL joint information
# ---------------------------------------------------------

print("\nSMPL model:")
print(type(model.smpl))

print("\nSMPL attributes related to joints:")

for name in dir(model.smpl):

    name_lower = name.lower()

    if (
        "joint" in name_lower
        or "vertex" in name_lower
        or "regressor" in name_lower
    ):
        print(" -", name)


# ---------------------------------------------------------
# Inspect joint-related buffers/parameters
# ---------------------------------------------------------

print("\nPossible joint-related tensors:")

for name, value in model.smpl.named_buffers():

    if (
        "joint" in name.lower()
        or "regressor" in name.lower()
    ):
        print(
            name,
            tuple(value.shape)
        )