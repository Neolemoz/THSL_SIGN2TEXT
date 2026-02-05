import argparse
import json
import sys

import cv2
import mediapipe as mp
import numpy as np


FEATURE_DIM = 21 * 3 * 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MediaPipe Holistic hand keypoints to .npz."
    )
    parser.add_argument("--input", required=True, help="Video path or webcam index.")
    parser.add_argument("--out", required=True, help="Output .npz path.")
    parser.add_argument("--max_frames", type=int, default=None, help="Optional frame cap.")
    parser.add_argument(
        "--resize_width",
        type=int,
        default=None,
        help="Optional resize width (keeps aspect ratio).",
    )
    return parser.parse_args()


def _open_capture(input_arg: str) -> tuple[cv2.VideoCapture, str]:
    if input_arg.isdigit():
        return cv2.VideoCapture(int(input_arg)), "webcam"
    return cv2.VideoCapture(input_arg), "video"


def _resize_frame(frame: np.ndarray, resize_width: int | None) -> np.ndarray:
    if not resize_width:
        return frame
    height, width = frame.shape[:2]
    if width == resize_width:
        return frame
    scale = resize_width / float(width)
    new_height = int(round(height * scale))
    return cv2.resize(frame, (resize_width, new_height), interpolation=cv2.INTER_AREA)


def _hand_landmarks_to_vec(hand_landmarks) -> np.ndarray:
    if hand_landmarks is None:
        return np.zeros((21, 3), dtype=np.float32)
    coords = np.array(
        [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32
    )
    if coords.shape != (21, 3):
        pad = np.zeros((21, 3), dtype=np.float32)
        pad[: coords.shape[0], : coords.shape[1]] = coords
        return pad
    return coords


def main() -> int:
    args = _parse_args()
    cap, input_mode = _open_capture(args.input)
    if not cap.isOpened():
        print(f"ERROR: Unable to open input: {args.input}", file=sys.stderr)
        return 1

    fps_value = cap.get(cv2.CAP_PROP_FPS)
    fps = float(fps_value) if fps_value and fps_value > 0 else None

    keypoints = []
    missing_left = 0
    missing_right = 0

    mp_holistic = mp.solutions.holistic
    with mp_holistic.Holistic(
        model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
    ) as holistic:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = _resize_frame(frame, args.resize_width)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            left_vec = _hand_landmarks_to_vec(results.left_hand_landmarks)
            right_vec = _hand_landmarks_to_vec(results.right_hand_landmarks)
            if results.left_hand_landmarks is None:
                missing_left += 1
            if results.right_hand_landmarks is None:
                missing_right += 1

            feature = np.concatenate([left_vec.flatten(), right_vec.flatten()]).astype(
                np.float32
            )
            keypoints.append(feature)

            frame_index += 1
            if args.max_frames and frame_index >= args.max_frames:
                break

            if input_mode == "webcam":
                cv2.imshow("THSL_SIGN2TEXT - Webcam", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if input_mode == "webcam":
        cv2.destroyAllWindows()

    keypoints_arr = (
        np.stack(keypoints).astype(np.float32)
        if keypoints
        else np.zeros((0, FEATURE_DIM), dtype=np.float32)
    )
    frame_count = int(keypoints_arr.shape[0])
    missing_left_rate = float(missing_left / frame_count) if frame_count else 0.0
    missing_right_rate = float(missing_right / frame_count) if frame_count else 0.0

    meta = {
        "fps": fps,
        "frame_count": frame_count,
        "feature_dim": FEATURE_DIM,
        "missing_left_rate": missing_left_rate,
        "missing_right_rate": missing_right_rate,
        "input_path": args.input,
        "input_mode": input_mode,
        "resize_width": args.resize_width,
    }

    np.savez(
        args.out,
        keypoints=keypoints_arr,
        meta=meta,
        meta_json=json.dumps(meta),
    )

    print(
        "Summary:",
        f"mode={input_mode}",
        f"fps={fps}",
        f"frames={frame_count}",
        f"D={FEATURE_DIM}",
        f"missing_left={missing_left_rate:.3f}",
        f"missing_right={missing_right_rate:.3f}",
        f"out={args.out}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
