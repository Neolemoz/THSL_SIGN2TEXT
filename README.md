# THSL_SIGN2TEXT

## Quickstart (Windows PowerShell)
1. Create venv: `python -m venv .venv`
2. Activate: `.\.venv\Scripts\Activate.ps1`
3. Install: `python -m pip install -U pip ; python -m pip install -r requirements.txt`
4. Run doctor: `python tools/doctor.py`

Next step: run `ml/preprocess/extract_keypoints.py` on your videos.

Example (video file):
`python ml/preprocess/extract_keypoints.py --input path\to\video.mp4 --out reports\sample_keypoints.npz`

ThaiSignVis manifest conversion:
`python tools/convert_thaisignvis_to_manifest.py --root data\raw\thaisignvis --out data\manifest\manifest.jsonl --qc_out reports\thaisignvis_qc.json`

## Run This (Smoke)
Batch extract (small limit):
`python -m ml.preprocess.batch_extract --manifest data\manifest\manifest.jsonl --out_dir data\processed\keypoints --limit 10`

QC report:
`python -m ml.preprocess.qc_keypoints --manifest data\manifest\manifest.jsonl --kp_dir data\processed\keypoints --limit 20 --out reports\qc_keypoints_smoke.json`
Note: if QC limit > extracted samples, you will see `missing_npz` in the summary (use `--skip_missing_npz` or match limits).

Train + eval (short smoke run):
`python -m ml.train_seq2seq --manifest data\manifest\manifest.jsonl --kp_dir data\processed\keypoints --out_dir reports\smoke_train --epochs 2 --limit 32 --batch_size 8 --lr 3e-4`
`python -m ml.eval_seq2seq --manifest data\manifest\manifest.jsonl --kp_dir data\processed\keypoints --checkpoint reports\smoke_train\best.pt --out reports\smoke_eval --split val --limit 20`

## Limitations
- Hand-only 126D keypoints (left+right hands). No pose/face yet.
- Model collapse is not fully solved; this PR improves observability and pipeline stability first.
