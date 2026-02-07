# tools/overlay_mediapipe.py
from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import mediapipe as mp


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Overlay MediaPipe Holistic hand landmarks on video/webcam and save mp4.")
    p.add_argument("--input", required=True, help="Video path OR webcam index like 0")
    p.add_argument("--out", required=True, help="Output mp4 path")
    p.add_argument("--max_frames", type=int, default=0, help="0 = no limit")
    p.add_argument("--resize_width", type=int, default=0, help="0 = keep original")
    p.add_argument("--show", action="store_true", help="Show live preview window")
    p.add_argument("--model_complexity", type=int, default=1, choices=[0, 1, 2])
    p.add_argument("--min_det_conf", type=float, default=0.5)
    p.add_argument("--min_track_conf", type=float, default=0.5)
    return p.parse_args()


def _is_webcam(s: str) -> bool:
    return s.strip().isdigit()


def main() -> int:
    args = _parse_args()
    inp = args.input
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if _is_webcam(inp):
        cap = cv2.VideoCapture(int(inp))
        mode = f"webcam:{inp}"
    else:
        cap = cv2.VideoCapture(str(inp))
        mode = f"video:{inp}"

    if not cap.isOpened():
        print(f"ERROR: cannot open input ({mode})")
        return 2

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 1e-6:
        fps = 30.0  # webcam/unknown fallback

    # Read one frame to determine size
    ok, frame = cap.read()
    if not ok or frame is None:
        print("ERROR: cannot read first frame")
        return 3

    if args.resize_width and args.resize_width > 0:
        h, w = frame.shape[:2]
        new_w = args.resize_width
        new_h = int(h * (new_w / w))
        frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    H, W = frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (W, H))
    if not writer.isOpened():
        print(f"ERROR: cannot open VideoWriter for {out_path}")
        return 4

    mp_holistic = mp.solutions.holistic
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    print(f"INFO: mode={mode} fps={fps:.2f} size={W}x{H} out={out_path}")
    max_frames = args.max_frames if args.max_frames and args.max_frames > 0 else 10**18

    frame_idx = 0
    missing_l = 0
    missing_r = 0

    with mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=args.model_complexity,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
        min_detection_confidence=float(args.min_det_conf),
        min_tracking_confidence=float(args.min_track_conf),
    ) as holistic:
        # process first frame + loop
        while True:
            if frame_idx == 0:
                cur = frame
            else:
                ok, cur = cap.read()
                if not ok or cur is None:
                    break
                if args.resize_width and args.resize_width > 0:
                    cur = cv2.resize(cur, (W, H), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(cur, cv2.COLOR_BGR2RGB)
            res = holistic.process(rgb)

            has_l = res.left_hand_landmarks is not None
            has_r = res.right_hand_landmarks is not None
            if not has_l:
                missing_l += 1
            if not has_r:
                missing_r += 1

            # Draw hands only (ชัดสุดสำหรับงานเรา)
            if res.left_hand_landmarks:
                mp_draw.draw_landmarks(
                    cur,
                    res.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )
            if res.right_hand_landmarks:
                mp_draw.draw_landmarks(
                    cur,
                    res.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style(),
                )

            # overlay text
            txt = f"frame={frame_idx} L={'Y' if has_l else 'N'} R={'Y' if has_r else 'N'}"
            cv2.putText(cur, txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            writer.write(cur)

            if args.show:
                cv2.imshow("MediaPipe Overlay (press q to quit)", cur)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if frame_idx >= max_frames:
                break

    cap.release()
    writer.release()
    if args.show:
        cv2.destroyAllWindows()

    if frame_idx == 0:
        print("ERROR: no frames processed")
        return 5

    miss_l_rate = missing_l / frame_idx
    miss_r_rate = missing_r / frame_idx
    print(
        f"DONE: frames={frame_idx} missing_left={miss_l_rate:.3f} missing_right={miss_r_rate:.3f} saved={out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
