import os
import cv2
import mediapipe as mp
import pandas as pd

# -----------------------------
# CONFIG
# -----------------------------
DATASET_DIR = "data/dataset"
OUTPUT_FILE = "backend/measurement/output/landmarks.csv"

# -----------------------------
# MediaPipe
# -----------------------------
mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    min_detection_confidence=0.5
)

rows = []

# -----------------------------
# Iterate through every person
# -----------------------------
for person in sorted(os.listdir(DATASET_DIR)):

    person_path = os.path.join(DATASET_DIR, person)

    if not os.path.isdir(person_path):
        continue

    for image_type in ["front", "side"]:

        image_path = os.path.join(person_path, f"{image_type}.png")

        if not os.path.exists(image_path):
            print(f"Missing: {image_path}")
            continue

        image = cv2.imread(image_path)

        if image is None:
            print(f"Cannot read: {image_path}")
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = pose.process(rgb)

        if results.pose_landmarks is None:
            print(f"No landmarks: {image_path}")
            continue

        row = {
            "person_id": person,
            "image_type": image_type
        }

        for idx, lm in enumerate(results.pose_landmarks.landmark):

            row[f"x_{idx}"] = lm.x
            row[f"y_{idx}"] = lm.y
            row[f"z_{idx}"] = lm.z
            row[f"visibility_{idx}"] = lm.visibility

        rows.append(row)

        print(f"Processed: {person} ({image_type})")

# -----------------------------
# Save CSV
# -----------------------------
df = pd.DataFrame(rows)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

df.to_csv(OUTPUT_FILE, index=False)

print("\nDone!")
print(df.head())