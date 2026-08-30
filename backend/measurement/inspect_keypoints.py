import os
import numpy as np

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../")
)

OUTPUT_DIR = os.path.join(
    PROJECT_ROOT,
    "backend",
    "measurement",
    "hmr2_output"
)


def inspect(name):

    path = os.path.join(OUTPUT_DIR, name)

    keypoints = np.load(path)

    print("\n===================================")
    print(name)
    print("===================================")

    print("Shape:", keypoints.shape)

    for i, point in enumerate(keypoints):
        print(
            f"{i:2d}: "
            f"x={point[0]: .4f}, "
            f"y={point[1]: .4f}, "
            f"z={point[2]: .4f}"
        )


inspect("person_001_front_keypoints_3d.npy")
inspect("person_001_side_keypoints_3d.npy")