"""
Runner for the v2 measurement engine (body_measurement.py).

Validates predicted measurements against known ground truth per person.
Add new entries to PEOPLE as more people are added to data/dataset and
run through scale_mesh.py.
"""

import os

import body_measurement as bm


PEOPLE = {
    "person_001": {
        "height_cm": 172.0,
        "ground_truth": {
            "shoulder": 101.0,
            "chest": 98.0,
            "waist": 84.0,
            "hip": 105.0,
        },
    },
    # "person_002": {
    #     "height_cm": ...,
    #     "ground_truth": {"shoulder": ..., "chest": ..., "waist": ..., "hip": ...},
    # },
}


def run():
    model = bm.load_model()

    all_rows = []

    for person_id, info in PEOPLE.items():

        mesh_path = os.path.join(
            bm.OUTPUT_DIR, f"{person_id}_front_scaled.npy"
        )

        if not os.path.exists(mesh_path):
            print(f"\nSkipping {person_id}: no scaled mesh at {mesh_path}")
            continue

        vertices = bm.load_scaled_mesh(person_id, view="front")

        print("\n" + "=" * 70)
        print(person_id)
        print("=" * 70)

        result = bm.measure_person(model, vertices)

        print("\nAnatomical levels (cm):")
        for k, v in result["levels"].items():
            print(f"  {k:18s}: {v:8.2f}")

        ground_truth = info.get("ground_truth")

        print(
            f"\n{'Measurement':<12}"
            f"{'Predicted':>12}"
            f"{'Actual':>10}"
            f"{'Error':>10}"
            f"{'Error %':>10}"
        )

        row = {"person": person_id, "height_cm": info["height_cm"]}

        for name in ["shoulder", "chest", "waist", "hip"]:

            r = result[name]

            if r is None:
                print(f"{name:<12} -- could not extract contour --")
                continue

            predicted = r["circumference"]

            if ground_truth and name in ground_truth:
                actual = ground_truth[name]
                error = predicted - actual
                error_pct = abs(error) / actual * 100

                print(
                    f"{name:<12}"
                    f"{predicted:>12.2f}"
                    f"{actual:>10.2f}"
                    f"{error:>10.2f}"
                    f"{error_pct:>9.2f}%"
                )

                row[f"{name}_pred"] = predicted
                row[f"{name}_actual"] = actual
                row[f"{name}_error_pct"] = error_pct
            else:
                print(f"{name:<12}{predicted:>12.2f}{'--':>10}{'--':>10}{'--':>10}")
                row[f"{name}_pred"] = predicted

        all_rows.append(row)

    print("\n" + "=" * 70)
    print("SUMMARY (all people)")
    print("=" * 70)

    header = f"{'Person':<12}{'Height':>8}"
    for name in ["shoulder", "chest", "waist", "hip"]:
        header += f"{name.capitalize():>10}"
    print(header)

    for row in all_rows:
        line = f"{row['person']:<12}{row['height_cm']:>8.1f}"
        for name in ["shoulder", "chest", "waist", "hip"]:
            key = f"{name}_error_pct"
            if key in row:
                line += f"{row[key]:>9.2f}%"
            else:
                line += f"{'--':>10}"
        print(line)


if __name__ == "__main__":
    run()
