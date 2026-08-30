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
# HMR2
# ---------------------------------------------------------

from hmr2.configs import CACHE_DIR_4DHUMANS
from hmr2.models import load_hmr2, download_models, DEFAULT_CHECKPOINT


# ---------------------------------------------------------
# OpenPose BODY_25 names
# ---------------------------------------------------------

BODY_25_NAMES = [
    "Nose",
    "Neck",
    "RightShoulder",
    "RightElbow",
    "RightWrist",
    "LeftShoulder",
    "LeftElbow",
    "LeftWrist",
    "MidHip",
    "RightHip",
    "RightKnee",
    "RightAnkle",
    "LeftHip",
    "LeftKnee",
    "LeftAnkle",
    "RightEye",
    "LeftEye",
    "RightEar",
    "LeftEar",
    "LeftBigToe",
    "LeftSmallToe",
    "LeftHeel",
    "RightBigToe",
    "RightSmallToe",
    "RightHeel",
]


# ---------------------------------------------------------
# Mapping used by 4D-Humans
# ---------------------------------------------------------

SMPL_TO_OPENPOSE = [
    24, 12, 17, 19, 21,
    16, 18, 20, 0, 2, 5, 8,
    1, 4, 7,
    25, 26, 27, 28, 29, 30, 31, 32, 33, 34
]


def main():

    print("Loading HMR2...")

    download_models(CACHE_DIR_4DHUMANS)

    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)

    print("HMR2 loaded.\n")

    # -----------------------------------------------------
    # Print the 25 HMR2/OpenPose joints
    # -----------------------------------------------------

    print("==============================================")
    print("HMR2 25-JOINT MAPPING")
    print("==============================================")

    print(
        f"{'HMR2 index':<12}"
        f"{'SMPL index':<12}"
        f"{'Joint name'}"
    )

    for hmr_index, smpl_index in enumerate(SMPL_TO_OPENPOSE):

        print(
            f"{hmr_index:<12}"
            f"{smpl_index:<12}"
            f"{BODY_25_NAMES[hmr_index]}"
        )

    # -----------------------------------------------------
    # Extra joints
    # -----------------------------------------------------

    print("\n==============================================")
    print("19 EXTRA JOINTS")
    print("==============================================")

    print(
        "These are generated using joint_regressor_extra "
        "from SMPL_to_J19.pkl."
    )

    print(
        "The current HMR2 source does not provide anatomical "
        "names for these 19 regressors."
    )

    print()

    extra = model.smpl.joint_regressor_extra

    print("Extra joint regressor shape:", tuple(extra.shape))

    for i in range(extra.shape[0]):

        hmr_index = 25 + i

        print(
            f"HMR2 index {hmr_index:2d} -> "
            f"extra_joint_{i:02d}"
        )


if __name__ == "__main__":
    main()