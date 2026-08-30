import pandas as pd
import numpy as np
import os

INPUT_CSV = "backend/measurement/output/landmarks.csv"
OUTPUT_CSV = "backend/measurement/output/features.csv"


# ----------------------------------------------------
# Euclidean Distance
# ----------------------------------------------------
def distance(row, p1, p2):

    x1 = row[f"x_{p1}"]
    y1 = row[f"y_{p1}"]

    x2 = row[f"x_{p2}"]
    y2 = row[f"y_{p2}"]

    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# ----------------------------------------------------
# Body Part Length
# ----------------------------------------------------
def chain_length(row, points):

    total = 0

    for i in range(len(points) - 1):
        total += distance(row, points[i], points[i + 1])

    return total


# ----------------------------------------------------
# Read landmarks
# ----------------------------------------------------
df = pd.read_csv(INPUT_CSV)

features = []

# ----------------------------------------------------
# Process each row
# ----------------------------------------------------
for _, row in df.iterrows():

    f = {}

    f["person_id"] = row["person_id"]
    f["image_type"] = row["image_type"]

    # -----------------------------------------
    # Widths
    # -----------------------------------------
    f["shoulder_width"] = distance(row, 11, 12)

    f["hip_width"] = distance(row, 23, 24)

    # -----------------------------------------
    # Torso
    # -----------------------------------------
    shoulder_center_x = (row["x_11"] + row["x_12"]) / 2
    shoulder_center_y = (row["y_11"] + row["y_12"]) / 2

    hip_center_x = (row["x_23"] + row["x_24"]) / 2
    hip_center_y = (row["y_23"] + row["y_24"]) / 2

    f["torso_length"] = np.sqrt(
        (hip_center_x - shoulder_center_x) ** 2
        +
        (hip_center_y - shoulder_center_y) ** 2
    )

    # -----------------------------------------
    # Arms
    # -----------------------------------------
    f["left_arm_length"] = chain_length(row, [11, 13, 15])

    f["right_arm_length"] = chain_length(row, [12, 14, 16])

    # -----------------------------------------
    # Legs
    # -----------------------------------------
    f["left_leg_length"] = chain_length(row, [23, 25, 27])

    f["right_leg_length"] = chain_length(row, [24, 26, 28])

    # -----------------------------------------
    # Ratios
    # -----------------------------------------
    f["shoulder_hip_ratio"] = (
        f["shoulder_width"] /
        (f["hip_width"] + 1e-6)
    )

    f["torso_leg_ratio"] = (
        f["torso_length"] /
        ((f["left_leg_length"] + f["right_leg_length"]) / 2 + 1e-6)
    )

    f["arm_leg_ratio"] = (
        ((f["left_arm_length"] + f["right_arm_length"]) / 2)
        /
        ((f["left_leg_length"] + f["right_leg_length"]) / 2 + 1e-6)
    )

    features.append(f)

# ----------------------------------------------------
# Save
# ----------------------------------------------------
feature_df = pd.DataFrame(features)

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

feature_df.to_csv(OUTPUT_CSV, index=False)

print(feature_df.head())

print("\nSaved:", OUTPUT_CSV)