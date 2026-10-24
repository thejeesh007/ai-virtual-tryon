# DressFit: AI Virtual Try-On

DressFit is a research prototype for estimating body measurements from two photos, building a personalized 3D avatar, and previewing garments on that avatar. The repository also contains the data preparation, model training, evaluation, and geometry tools used to develop the prototype.

## What the project does

- Accepts front and side photos plus a user's height, segments the body silhouette, and estimates 13 measurements: chest, waist, hip, shoulder breadth, arm length, ankle, bicep, calf, forearm, leg length, shoulder-to-crotch, thigh, and wrist.
- Fits an SMPL body shape to the estimated measurements and returns a 3D avatar mesh for the browser viewer.
- Supports sign-up and login, and stores a user's latest scan and avatar shape in MySQL.
- Supports a phone capture page and QR/session handoff so a scan can be captured on a phone and viewed on a desktop.
- Provides garment catalog, single-garment and outfit draping, size recommendations, movement/pose fit checks, and fit heatmaps with fabric-specific ease settings.
- Includes BodyM dataset processing, baseline and CNN model training/evaluation, real-photo validation, SMPL fitting and mesh measurement, garment export/drape helpers, and fabric/ease reference data.

The app is a prototype. Measurement quality depends on the input photos and trained model. Garment draping depends on the separately supplied GarmentCode assets and patterns.

## Project approach and contributions

DressFit evaluates fit from a user's body shape and the garment's sewing pattern rather than by placing a garment image over a person. The end-to-end flow is:

1. Capture front and side photos and enter height.
2. Segment and frame the silhouettes, then predict 13 measurements. The project compares Random Forest and HistGradientBoosting models built on silhouette width profiles with a dual-branch CNN that processes the front and side masks.
3. Fit the predicted measurements to an SMPL body shape and pose the avatar to match the garment patterns' assumptions.
4. Generate or load a size-specific sewing pattern and drape it on the user's avatar.
5. Return a 3D viewer result, a vertex-color heatmap showing local tightness/looseness, and a separate Snug / Good fit / Loose verdict.

Key design decisions from the project are to calculate achieved ease against the individual user's predicted measurements, apply fabric-specific expected ease, and keep the local drape heatmap separate from the across-size fit verdict. An optional multi-pose check compares the verdict across small pose changes instead of presenting the result from one pose as certainty.

## Evaluation results reported in the project report

The measurement models were evaluated on BodyM's `testA` split. The dual-branch CNN had the lowest chest mean absolute error (3.27 cm). HistGradientBoosting had the lowest waist (3.67 cm) and hip (3.60 cm) errors. The CNN reached 62.6% accuracy and a 63.9% macro F1 on the report's derived size-classification task. No single model was best for every measurement; the current web inference path uses the HGB baseline. These are dataset evaluation results, not a guarantee of an individual scan's accuracy.

The report's fit-verdict case study used a 112.5 cm chest and a fabric expected ease of 4.0 cm: size S had -13.44 cm achieved ease and was classified Snug; size XXL had +6.88 cm achieved ease and was classified Good fit. This illustrates why garment-to-body distance is useful for the visual heatmap while expected-versus-achieved ease is used for the overall size verdict.

## Catalogue and known limits

The report describes three real calibrated products (Nobero Classic Polo, Nobero Oversized Sweatshirt, and Nobero cargo pants), plus 19 generic fabric garments (9 tops and 10 bottoms). Real garment dimensions come from published size guides; generic items use the project's fabric reference data.

The report records primary measurement errors around 3-5 cm, which can be close to the difference between size buckets. Only a small number of real products have calibrated size guides, and phone-specific camera behavior, including face blurring and mirroring correction, still needs broader validation on physical devices. Pose agreement is a heuristic based on sampled poses and does not include measurement uncertainty. Warp support is environment-dependent and a garment that has not been visually validated may need additional tuning.

## Repository layout

```text
backend/
  api/             FastAPI app and HTTP endpoints
  auth/            Account and saved-scan services
  avatar/          Measurement-to-avatar and garment/fit processing
  measurement/     BodyM data, model, SMPL fit, and mesh tools
frontend/public/   Browser app and phone capture page
data/              Local datasets and sample inputs
design_params.yaml Garment design parameters
real_garments_package.zip  Additional garment package files
```

## Requirements and files not included in Git

- Python with the packages in `requirements.txt`. Python 3.10 or 3.11 is a reasonable starting point; some computer-vision and 3D packages may require platform-specific setup.
- A MySQL server for account and saved-scan features. The API initializes its account schema on startup, so MySQL must be reachable even if you only want to try the scan flow.
- The BodyM dataset at `data/bodym/` and a trained HGB baseline at `backend/measurement/bodym_output/baseline_hgb.joblib` for photo measurement inference. Both are local/generated files and are not tracked.
- The licensed SMPL neutral model at `backend/measurement/smpl_models/smpl/SMPL_NEUTRAL.pkl`. Obtain it under the SMPL model license; it is not included in Git.
- The GarmentCode checkout and garment assets/patterns expected under `third_party/GarmentCode/` for garment draping and catalog features. `third_party/` is gitignored, so it must be supplied locally. `real_garments_package.zip` is tracked and contains the project-specific real garment scripts, example outputs, and an `INSTRUCTIONS.md` describing how to install them into a GarmentCode checkout.

BodyM is available from the public AWS Open Data registry under CC BY-NC 4.0. Review its [dataset page and license](https://registry.opendata.aws/bodym/) before use. The SMPL model has its own license and download process.

## Setup

Run commands from the repository root.

### 1. Create a Python environment

Windows PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

PyTorch installation may need a platform-specific command for your CPU/GPU. Use the [official PyTorch install selector](https://pytorch.org/get-started/locally/) if the default `pip install -r requirements.txt` installation does not fit your system.

### 2. Configure MySQL

Create a MySQL database named `dressfit`, then copy the example settings and edit the credentials:

```powershell
Copy-Item .env.example .env
```

Set `MYSQL_HOST`, `MYSQL_PORT`, `MYSQL_USER`, `MYSQL_PASSWORD`, and `MYSQL_DATABASE` in `.env`. `.env` is ignored by Git and should contain only local credentials.

### 3. Add the data and model assets

Download BodyM and install AWS CLI if needed:

```bash
pip install awscli
aws s3 cp --no-sign-request s3://amazon-bodym/ ./data/bodym/ --recursive
```

Check that `data/bodym/` contains `train/`, `testA/`, and `testB/`, each with the CSV metadata and mask folders. Then build the inference baseline:

```bash
cd backend/measurement
python bodym_data.py
python bodym_baseline.py --model hgb
cd ../..
```

Place your licensed `SMPL_NEUTRAL.pkl` at `backend/measurement/smpl_models/smpl/SMPL_NEUTRAL.pkl`.

For garment features, make the expected GarmentCode checkout available at `third_party/GarmentCode/`, including its assets and garment patterns. Extract the tracked `real_garments_package.zip` and follow its `INSTRUCTIONS.md` to copy the project-specific scripts and patch into that checkout. The API imports garment catalog modules from this checkout when it starts; a missing checkout prevents the API from loading. The external checkout itself is not tracked in this repository.

## Run the web app

After setup, with the virtual environment active, MySQL running, and required model/data files in place, start the API from the repository root:

```bash
uvicorn backend.api.main:app --reload --port 8000
```

Open [http://localhost:8000/](http://localhost:8000/) for the desktop app and [http://localhost:8000/capture](http://localhost:8000/capture) for the phone capture page. The first startup loads the measurement framing data and SMPL model, so it can take a little while. Use two clear, full-body photos (front and side) and enter height in centimeters.

Interactive API documentation is available at [http://localhost:8000/docs](http://localhost:8000/docs).

## Garment draping with NVIDIA Warp

The garment pipeline supports three engines: `simple` (the default in the web/API drape routes), `pbd` (this project's position-based dynamics solver, validated for tops), and `warp` (GarmentCode's NVIDIA Warp GPU cloth simulator). To use Warp, select `engine="warp"` when calling the garment-drape API. The UI's ordinary drape action currently uses `simple`; the pose robustness action uses `warp`.

In Warp mode, the app first generates or loads the garment's sewing-pattern specification and design parameters, and exports the user's fitted SMPL body as a floor-aligned OBJ in meters. The Windows backend then launches `warp_drape_runner.py` through WSL2. That runner uses the full GarmentCode checkout and its compiled NVIDIA Warp fork to simulate the garment panels and seams against the exported body, including cloth behavior and body/self collision handling. The simulation output is returned to the backend, which computes garment-to-body distances and writes a colorized mesh for the viewer and fit heatmap. Fabric material settings feed the simulator; garment ease is built into the pattern dimensions during size calibration.

Warp is a separate environment from the main Python virtual environment. On Windows, it requires WSL2, a GPU-capable Linux setup, the GarmentCode-compatible NVIDIA Warp fork and its dependencies built inside the WSL environment, and the full GarmentCode checkout there. The current bridge in `third_party/GarmentCode/dress_dataset/warp_drape.py` contains machine-specific WSL distro, conda Python, and checkout paths; update `WSL_DISTRO`, `WSL_CONDA_PYTHON`, and `WSL_GARMENTCODE_CLONE` for your machine before using it. The Linux checkout must include GarmentCode's simulation module, simulation assets, and `system.json` support. A generic `pip install warp-lang` alone does not provide this configured GarmentCode simulator. Without the WSL/GPU setup, use the `simple` engine; it is a geometric wrap-and-relax approximation rather than a Warp cloth simulation.

## Multi-pose fit robustness

The fit robustness check reuses the same person's SMPL shape and garment size, creates a newly posed body for each variation, and runs a separate drape simulation for each pose. It compares the per-pose fit labels and reports how many agree plus a High/Medium/Low agreement confidence. Each pose has its own avatar and draped garment mesh so the results can be inspected individually. This is a robustness check across several static poses, not one continuous animation or a motion simulation.

- Tops: 3 poses total: the default arm angle and two small arm-angle variations.
- Pants: 5 poses total: the default stance, two independent knee-bend variants, and two independent leg-spread variants. Knee bend and leg spread are tested separately.

This check is opt-in because each additional Warp pose requires a full simulation and can take roughly 20–25 seconds on the configured GPU. The API exposes it at `/api/drape_pose_check` and `/api/session/{session_id}/drape_pose_check`; the web app calls it only when the fit robustness action is selected. It requires the Warp/WSL setup described above.

## Measurement pipeline and development commands

These commands are optional if you only want to run the web app with a prepared `baseline_hgb.joblib` model. Run them from `backend/measurement/` with the virtual environment active:

```bash
python bodym_data.py
python bodym_baseline.py --model hgb
python bodym_cnn.py --epochs 25 --batch-size 32
python bodym_evaluate.py
```

The baseline command creates `bodym_output/baseline_hgb.joblib`, which the web app uses. CNN training is an additional experiment; it is not required to run the current web inference path. Training time depends on your hardware.

To validate predictions on your own photos, place front/side images under `data/dataset/person_XXX/` and run:

```bash
python bodym_validate_real.py --heights person_XXX=170 --people person_XXX
```

Add a `real_ground_truth.csv` with measured values to compare predictions against tape measurements using `--ground-truth-csv real_ground_truth.csv`.

## Data and privacy notes

Datasets, personal photos, generated model outputs, model checkpoints, SMPL files, third-party checkouts, local `.env` credentials, and per-machine settings are gitignored. Each developer must provide or generate the local files required for the features they use. Do not commit private photos, credentials, or license-restricted model/data files.
