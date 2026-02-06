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

Batch keypoint extraction (quick test):
`python ml/preprocess/batch_extract.py --limit 10`

Baseline training (quick run):
`python ml/train.py --manifest data/manifest/manifest.jsonl --kp_dir data/processed/keypoints --out_dir reports/run1 --epochs 3`

Seq2seq baseline (overfit test):
`python -m ml.train_seq2seq --manifest data/manifest/manifest.jsonl --kp_dir data/processed/keypoints --out_dir reports/overfit_seq2seq --epochs 30 --limit 16`
`python -m ml.eval_seq2seq --manifest data/manifest/manifest.jsonl --kp_dir data/processed/keypoints --checkpoint reports/overfit_seq2seq/best.pt --out reports/overfit_seq2seq --split train --limit 16`
