import os
import cv2
import mediapipe as mp

# -----------------------------
# Configuration
# -----------------------------
DATASET_PATH = "data/dataset"
OUTPUT_PATH = "backend/measurement/output"

# -----------------------------
# MediaPipe Setup
# -----------------------------
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.5
)


def process_image(image_path, output_image_path):

    image = cv2.imread(image_path)

    if image is None:
        print(f"❌ Cannot read {image_path}")
        return False

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = pose.process(rgb)

    if results.pose_landmarks is None:
        print(f"❌ No landmarks detected -> {image_path}")
        return False

    print(f"✅ {os.path.basename(image_path)}")

    print(f"Detected {len(results.pose_landmarks.landmark)} landmarks")

    for idx, landmark in enumerate(results.pose_landmarks.landmark):

        print(
            idx,
            round(landmark.x,4),
            round(landmark.y,4),
            round(landmark.z,4),
            round(landmark.visibility,4)
        )

    mp_draw.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS
    )

    cv2.imwrite(output_image_path, image)

    return True


def main():

    people = sorted(os.listdir(DATASET_PATH))

    for person in people:

        person_folder = os.path.join(DATASET_PATH, person)

        if not os.path.isdir(person_folder):
            continue

        print("="*60)
        print(person)
        print("="*60)

        output_person = os.path.join(OUTPUT_PATH, person)
        os.makedirs(output_person, exist_ok=True)

        front = os.path.join(person_folder, "front.png")
        side = os.path.join(person_folder, "side.png")

        process_image(
            front,
            os.path.join(output_person, "front_landmarks.png")
        )

        process_image(
            side,
            os.path.join(output_person, "side_landmarks.png")
        )


if __name__ == "__main__":
    main()