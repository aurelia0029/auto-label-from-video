"""
extract_frames.py
─────────────────
從影片檔解碼並抽取 frames，儲存為圖片檔。

執行方式：
  python src/extract_frames.py               # 使用預設 config.yaml
  python src/extract_frames.py --config path/to/config.yaml
"""

import argparse
from pathlib import Path

import cv2
import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def extract_frames(video_path: str, output_dir: str,
                   frame_interval: int, max_frames: int) -> list[str]:
    """
    從影片檔解碼並按 frame_interval 間隔儲存圖片。
    回傳已儲存的圖片路徑清單。
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片：{video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    print(f"影片資訊：總幀數 {total_frames}，FPS {fps:.1f}")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    saved = []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            filename = f"{frame_idx:06d}.jpg"
            dst = out_path / filename
            cv2.imwrite(str(dst), frame)
            saved.append(str(dst))
            print(f"  儲存 {filename}  ({len(saved)}/{max_frames if max_frames > 0 else '∞'})")

            if max_frames > 0 and len(saved) >= max_frames:
                break

        frame_idx += 1

    cap.release()
    return saved


def main(config_path: str = "config.yaml"):
    cfg = load_config(config_path)

    video_path  = cfg["source"]["path"]
    output_dir  = cfg["extraction"]["output_dir"]
    interval    = cfg["extraction"]["frame_interval"]
    max_frames  = cfg["extraction"]["max_frames"]

    print(f"影片路徑  : {video_path}")
    print(f"輸出目錄  : {output_dir}")
    print(f"抽樣間隔  : 每 {interval} 幀抽 1 張")
    print(f"上限      : {'無限制' if max_frames == 0 else f'{max_frames} 張'}")
    print()

    saved = extract_frames(video_path, output_dir, interval, max_frames)

    print()
    print(f"完成！共抽取 {len(saved)} 張 frame 到 {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="從影片解碼並抽取 frames")
    parser.add_argument("--config", default="config.yaml", help="設定檔路徑")
    args = parser.parse_args()
    main(args.config)
