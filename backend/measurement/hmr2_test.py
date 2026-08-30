import os
import sys

import cv2
import numpy as np
import torch
import mediapipe as mp

# ---------------------------------------------------------
# Add 4D-Humans to Python path
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
from hmr2.utils import recursive_to
from hmr2.datasets.vitdet_dataset import ViTDetDataset
from hmr2.utils.renderer import cam_crop_to_full


# ---------------------------------------------------------
# MediaPipe
# ---------------------------------------------------------
mp_pose = mp.solutions.pose


def get_mediapipe_bbox(image):
    """
    Run MediaPipe Pose and calculate a person bounding box.

    Returns:
        bbox: numpy array [[x1, y1, x2, y2]]
        landmarks: MediaPipe landmarks
    """

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        min_detection_confidence=0.5
    ) as pose:

        result = pose.process(image_rgb)

    if not result.pose_landmarks:
        raise RuntimeError("MediaPipe could not detect a person.")

    h, w = image.shape[:2]

    points = []

    for landmark in result.pose_landmarks.landmark:

        # Ignore landmarks with very low visibility
        if landmark.visibility < 0.3:
            continue

        x = landmark.x * w
        y = landmark.y * h

        points.append((x, y))

    if len(points) < 5:
        raise RuntimeError("Not enough visible MediaPipe landmarks.")

    points = np.array(points)

    x_min = points[:, 0].min()
    y_min = points[:, 1].min()
    x_max = points[:, 0].max()
    y_max = points[:, 1].max()

    # -----------------------------------------------------
    # Add padding around MediaPipe landmarks
    # -----------------------------------------------------

    width = x_max - x_min
    height = y_max - y_min

    # Horizontal padding
    x_padding = width * 0.20

    # Vertical padding
    y_padding = height * 0.15

    x1 = x_min - x_padding
    y1 = y_min - y_padding

    x2 = x_max + x_padding
    y2 = y_max + y_padding

    # Keep bbox inside image
    x1 = max(0, x1)
    y1 = max(0, y1)

    x2 = min(w - 1, x2)
    y2 = min(h - 1, y2)

    bbox = np.array(
        [[x1, y1, x2, y2]],
        dtype=np.float32
    )

    return bbox, result.pose_landmarks


# ---------------------------------------------------------
# Main HMR2 test
# ---------------------------------------------------------
def main():

    # -----------------------------------------------------
    # CHANGE THIS IMAGE PATH
    # -----------------------------------------------------

    image_path = os.path.join(
    PROJECT_ROOT,
    "data",
    "dataset",
    "person_001",
    "side.png"
)

    print("Image:")
    print(image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found:\n{image_path}"
        )

    # -----------------------------------------------------
    # Load image
    # -----------------------------------------------------

    img_cv2 = cv2.imread(image_path)

    if img_cv2 is None:
        raise RuntimeError("Could not read image.")

    print("Image shape:", img_cv2.shape)

    # -----------------------------------------------------
    # MediaPipe
    # -----------------------------------------------------

    print("\nRunning MediaPipe...")

    boxes, landmarks = get_mediapipe_bbox(img_cv2)

    print("MediaPipe detected person.")

    print("\nBounding box:")
    print(boxes)

    # -----------------------------------------------------
    # Device
    # -----------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nDevice:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # -----------------------------------------------------
    # Download/load HMR2 pretrained model
    # -----------------------------------------------------

    print("\nLoading HMR2...")

    download_models(CACHE_DIR_4DHUMANS)

    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)

    model = model.to(device)
    model.eval()

    print("HMR2 loaded.")

    # -----------------------------------------------------
    # Create HMR2 dataset
    # -----------------------------------------------------

    print("\nCreating HMR2 dataset...")

    dataset = ViTDetDataset(
        model_cfg,
        img_cv2,
        boxes
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    # -----------------------------------------------------
    # Run HMR2
    # -----------------------------------------------------

    print("\nRunning HMR2 inference...")

    with torch.no_grad():

        for batch in dataloader:

            batch = recursive_to(batch, device)

            output = model(batch)

    print("HMR2 inference completed.")

    # -----------------------------------------------------
    # Inspect output
    # -----------------------------------------------------

    print("\nHMR2 output keys:")

    for key in output.keys():
        print(" -", key)

    # -----------------------------------------------------
    # Vertices
    # -----------------------------------------------------
    # -----------------------------------------------------
# Save HMR2 3D keypoints
# -----------------------------------------------------

    if "pred_keypoints_3d" in output:

        keypoints_3d = output["pred_keypoints_3d"][0]

        keypoints_3d_cpu = (
            keypoints_3d.detach()
            .cpu()
            .numpy()
        )

        keypoints_path = os.path.join(
            PROJECT_ROOT,
            "backend",
            "measurement",
            "hmr2_output",
            "person_001_side_keypoints_3d.npy"
        )

        np.save(
            keypoints_path,
            keypoints_3d_cpu
        )

        print("\nSaved 3D keypoints:")
        print(keypoints_path)

        print("Keypoint shape:")
        print(keypoints_3d_cpu.shape)

        print("First 5 keypoints:")
        print(keypoints_3d_cpu[:5])
    if "pred_vertices" in output:

        vertices = output["pred_vertices"][0]

        print("\nSMPL vertices:")
        print("Shape:", vertices.shape)

        vertices_cpu = vertices.detach().cpu().numpy()

        # -------------------------------------------------
        # Output directory
        # -------------------------------------------------

        output_dir = os.path.join(
            PROJECT_ROOT,
            "backend",
            "measurement",
            "hmr2_output"
        )

        os.makedirs(output_dir, exist_ok=True)

        # -------------------------------------------------
        # Save vertices
        # -------------------------------------------------

        vertices_path = os.path.join(
            output_dir,
            "person_001_side_vertices.npy"
        )

        np.save(vertices_path, vertices_cpu)

        print("Saved vertices:")
        print(vertices_path)

        # -------------------------------------------------
        # Save SMPL mesh as OBJ
        # -------------------------------------------------

        faces = model.smpl.faces

        obj_path = os.path.join(
            output_dir,
            "person_001_side.obj"
        )

        with open(obj_path, "w") as f:

            # Vertices
            for vertex in vertices_cpu:
                x, y, z = vertex
                f.write(f"v {x} {y} {z}\n")

            # Faces
            for face in faces:
                v1, v2, v3 = face
                f.write(
                    f"f {v1 + 1} {v2 + 1} {v3 + 1}\n"
                )

        print("\nSaved 3D mesh:")
        print(obj_path)
    # -----------------------------------------------------
    # Camera parameters
    # -----------------------------------------------------

    if "pred_cam" in output:

        print("\nPredicted camera:")
        print(
            output["pred_cam"][0]
            .detach()
            .cpu()
            .numpy()
        )

    print("\n===================================")
    print("HMR2 TEST COMPLETED")
    print("===================================")


if __name__ == "__main__":
    main()