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


def inspect_mesh(name):

    path = os.path.join(OUTPUT_DIR, name)

    vertices = np.load(path)

    print("\n================================")
    print(name)
    print("================================")

    print("Shape:", vertices.shape)

    x = vertices[:, 0]
    y = vertices[:, 1]
    z = vertices[:, 2]

    print("\nX range:")
    print("min:", x.min())
    print("max:", x.max())
    print("size:", x.max() - x.min())

    print("\nY range:")
    print("min:", y.min())
    print("max:", y.max())
    print("size:", y.max() - y.min())

    print("\nZ range:")
    print("min:", z.min())
    print("max:", z.max())
    print("size:", z.max() - z.min())


inspect_mesh("person_001_front_vertices.npy")
inspect_mesh("person_001_side_vertices.npy")