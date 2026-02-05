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
