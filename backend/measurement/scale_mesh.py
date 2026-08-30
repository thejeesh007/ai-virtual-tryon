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

ACTUAL_HEIGHT_CM = 172.0


def scale_mesh(input_file, output_file):

    input_path = os.path.join(OUTPUT_DIR, input_file)
    output_path = os.path.join(OUTPUT_DIR, output_file)

    vertices = np.load(input_path)

    # Find reconstructed height along Y axis
    y_min = vertices[:, 1].min()
    y_max = vertices[:, 1].max()

    reconstructed_height = y_max - y_min

    # Convert HMR2 coordinates to centimeters
    scale = ACTUAL_HEIGHT_CM / reconstructed_height

    scaled_vertices = vertices * scale

    np.save(output_path, scaled_vertices)

    print("\n================================")
    print(input_file)
    print("================================")

    print("Reconstructed height:",
          reconstructed_height)

    print("Actual height:",
          ACTUAL_HEIGHT_CM, "cm")

    print("Scale:",
          scale)

    print("Saved:")
    print(output_path)

    # Verify scaled height
    scaled_y_min = scaled_vertices[:, 1].min()
    scaled_y_max = scaled_vertices[:, 1].max()

    print("Scaled height:",
          scaled_y_max - scaled_y_min,
          "cm")


# Front
scale_mesh(
    "person_001_front_vertices.npy",
    "person_001_front_scaled.npy"
)

# Side
scale_mesh(
    "person_001_side_vertices.npy",
    "person_001_side_scaled.npy"
)