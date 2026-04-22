"""
auto_label.py
─────────────
用 CLIP 將 extracted_frames/ 裡的圖片分類成兩個資料夾。

流程：
  1. 讀取 config.yaml 的分類設定
  2. 對每張圖片計算 positive_label vs negative_label 的相似度
  3. positive 機率 > confidence_threshold → 複製到 positive_folder
     否則 → 複製到 negative_folder

執行方式：
  python src/auto_label.py               # 使用預設 config.yaml
  python src/auto_label.py --config path/to/config.yaml
  python src/auto_label.py --frames_dir data/extracted_frames   # 指定 frames 目錄
"""

import argparse
import shutil
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import CLIPModel, CLIPProcessor


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_clip(model_name: str = "openai/clip-vit-base-patch32"):
    """載入 CLIP 模型（第一次執行會自動下載約 600MB）"""
    print(f"載入 CLIP 模型：{model_name} ...")
    model = CLIPModel.from_pretrained(model_name)
    processor = CLIPProcessor.from_pretrained(model_name)
    model.eval()
    return model, processor


def classify_image_binary(image_path: str,
                         model: CLIPModel,
                         processor: CLIPProcessor,
                         positive_label: str,
                         negative_labels: list[str],
                         threshold: float) -> tuple[bool, float]:
    """
    二元分類：回傳 (is_positive, positive_probability)

    positive_label 排在第一位，softmax 跨所有標籤計算。
    negative_labels 可填多個干擾項，幫助模型做更精確的排除。
    """
    image = Image.open(image_path).convert("RGB")
    labels = [positive_label] + negative_labels

    inputs = processor(text=labels, images=image,
                       return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1)
    pos_prob = probs[0][0].item()

    return pos_prob >= threshold, pos_prob


def classify_image_multiclass(image_path: str,
                              model: CLIPModel,
                              processor: CLIPProcessor,
                              labels: list[str]) -> tuple[int, float]:
    """
    多類分類：回傳 (class_index, confidence)

    使用 argmax 選擇概率最高的類別
    """
    image = Image.open(image_path).convert("RGB")

    inputs = processor(text=labels, images=image,
                       return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1)[0]
    max_idx = probs.argmax().item()
    confidence = probs[max_idx].item()

    return max_idx, confidence


def main(config_path: str = "config.yaml", frames_dir: str | None = None):
    cfg = load_config(config_path)

    # ── 讀取設定 ──
    label_cfg    = cfg["classification"]
    ternary_mode = label_cfg.get("ternary_mode", False)
    output_dir   = Path(label_cfg["output_dir"])

    # frames 來源：優先用命令列參數，否則用 extraction 設定
    src_dir = Path(frames_dir) if frames_dir else Path(cfg["extraction"]["output_dir"])

    if ternary_mode:
        # 三元或多類分類
        labels  = label_cfg["ternary_labels"]
        folders = label_cfg["ternary_folders"]

        # ── 建立輸出資料夾 ──
        folder_paths = {folder: output_dir / folder for folder in folders}
        for folder_path in folder_paths.values():
            folder_path.mkdir(parents=True, exist_ok=True)

        # ── 蒐集圖片 ──
        images = sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.png"))
        if not images:
            print(f"在 {src_dir} 找不到任何圖片，請先執行 extract_frames.py")
            return

        print(f"\n設定摘要（三元分類模式）")
        for i, label in enumerate(labels):
            print(f"  類別 {i}: {label} → {folders[i]}")
        print(f"  圖片來源        : {src_dir}  ({len(images)} 張)")
        print(f"  輸出目錄        : {output_dir}")
        print()

        # ── 載入 CLIP ──
        model, processor = build_clip()

        # ── 逐張分類 ──
        stats = {folder: 0 for folder in folders}

        for i, img_path in enumerate(images, 1):
            class_idx, confidence = classify_image_multiclass(
                str(img_path), model, processor, labels
            )

            result_folder = folders[class_idx]
            dst_folder = folder_paths[result_folder]
            shutil.copy2(img_path, dst_folder / img_path.name)
            stats[result_folder] += 1

            print(f"[{i:>4}/{len(images)}] {img_path.name}  "
                  f"class={result_folder} confidence={confidence:.2%}")

        # ── 統計 ──
        print()
        print("=" * 50)
        print(f"分類完成！")
        for folder in folders:
            print(f"  {folder:>12} : {stats[folder]} 張")
        print(f"  輸出位置 : {output_dir}")
        print("=" * 50)

    else:
        # 二元分類
        pos_label    = label_cfg["positive_label"]
        neg_labels   = label_cfg["negative_labels"]
        pos_folder   = label_cfg["positive_folder"]
        neg_folder   = label_cfg["negative_folder"]
        threshold    = label_cfg["confidence_threshold"]

        # ── 建立輸出資料夾 ──
        pos_out = output_dir / pos_folder
        neg_out = output_dir / neg_folder
        pos_out.mkdir(parents=True, exist_ok=True)
        neg_out.mkdir(parents=True, exist_ok=True)

        # ── 蒐集圖片 ──
        images = sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.png"))
        if not images:
            print(f"在 {src_dir} 找不到任何圖片，請先執行 extract_frames.py")
            return

        print(f"\n設定摘要（二元分類模式）")
        print(f"  Positive label  : {pos_label}")
        for i, nl in enumerate(neg_labels):
            prefix = "  Negative labels :" if i == 0 else "                   "
            print(f"{prefix} {nl}")
        print(f"  門檻值          : {threshold}")
        print(f"  圖片來源        : {src_dir}  ({len(images)} 張)")
        print(f"  輸出目錄        : {output_dir}")
        print()

        # ── 載入 CLIP ──
        model, processor = build_clip()

        # ── 逐張分類 ──
        stats = {pos_folder: 0, neg_folder: 0}

        for i, img_path in enumerate(images, 1):
            is_pos, prob = classify_image_binary(
                str(img_path), model, processor,
                pos_label, neg_labels, threshold
            )

            dst_folder = pos_out if is_pos else neg_out
            shutil.copy2(img_path, dst_folder / img_path.name)

            result = pos_folder if is_pos else neg_folder
            stats[result] += 1

            print(f"[{i:>4}/{len(images)}] {img_path.name}  "
                  f"positive={prob:.2%}  → {result}")

        # ── 統計 ──
        print()
        print("=" * 50)
        print(f"分類完成！")
        print(f"  {pos_folder:>12} : {stats[pos_folder]} 張")
        print(f"  {neg_folder:>12} : {stats[neg_folder]} 張")
        print(f"  輸出位置 : {output_dir}")
        print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="用 CLIP 將 frames 自動分成兩類")
    parser.add_argument("--config",     default="config.yaml", help="設定檔路徑")
    parser.add_argument("--frames_dir", default=None,          help="覆蓋 frame 來源目錄")
    args = parser.parse_args()
    main(args.config, args.frames_dir)
