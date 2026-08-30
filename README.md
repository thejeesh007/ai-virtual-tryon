"# DressFit" 
"testing branch" 

## Body measurement pipeline setup

The BodyM dataset, real people's photos, trained model checkpoints, and the
`third_party/4D-Humans` clone are all gitignored (privacy + size — see
`.gitignore`), so a fresh clone needs these set up locally before anything
in `backend/measurement/` will run.

### 1. Python environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt
```

### 2. BodyM dataset (for `bodym_*.py` — the current pipeline)

No AWS account needed (`--no-sign-request`):

```bash
pip install awscli
aws s3 cp --no-sign-request s3://amazon-bodym/ ./data/bodym --recursive
```

This should produce `data/bodym/{train,testA,testB}/`, each with a `mask/`,
`mask_left/`, `hwg_metadata.csv`, `measurements.csv`, and
`subject_to_photo_map.csv`. License: CC BY-NC 4.0 — see
[registry.opendata.aws/bodym](https://registry.opendata.aws/bodym/).

### 3. 4D-Humans (only if working on the older geometric pipeline in `run_pipeline.py`, `mesh_utils.py`, etc.)

```bash
git clone https://github.com/shubham-goel/4D-Humans third_party/4D-Humans
```

### 4. Your own validation photos (optional)

Drop real front/side photos into `data/dataset/person_XXX/front.*` +
`side.*` (png/jpg/jpeg all work) to test the trained models on real people.
These are gitignored — each teammate keeps their own locally.

### 5. Running the pipeline

From `backend/measurement/`, in order:

```bash
python bodym_data.py          # merge BodyM CSVs, sanity-check, visual check
python bodym_baseline.py      # train the RandomForest baseline (~6 min)
python bodym_cnn.py --epochs 25 --batch-size 32   # train the CNN (~25 min on GPU)
python bodym_evaluate.py      # score both models on testA/testB
```

Trained models, mask caches, and logs land in `bodym_output/` (gitignored —
regenerate locally rather than expecting it after a fresh clone).

To test on your own photos:

```bash
python bodym_validate_real.py --heights person_XXX=<height_cm> --people person_XXX
```

If you have real tape measurements for someone, add a row to
`real_ground_truth.csv` (`person,chest,waist,hip,shoulder-breadth,arm-length`)
and pass `--ground-truth-csv real_ground_truth.csv` to get actual error
instead of just predictions.
