# auto-label-from-video

使用 OpenAI CLIP 模型，從影片自動抽取 frames 並分類成兩個資料夾的工具。

## 專案架構

```
auto-label-from-video/
├── config.yaml             # 設定檔（影片路徑、標籤、門檻值）
├── src/
│   ├── extract_frames.py  # Step 1：從影片解碼並抽取 frames
│   ├── auto_label.py      # Step 2：CLIP 分類，輸出到兩個資料夾
│   ├── label.py           # 單張圖片紅衣偵測（原型程式）
│   └── view_PETA.py       # PETA 資料集處理工具
├── data/
│   ├── extracted_frames/  # extract_frames.py 的輸出
│   └── labeled_output/    # auto_label.py 的輸出
│       ├── red/
│       └── non_red/
├── CCTV_01/               # 影片資料集
└── pyproject.toml
```

## 安裝

```bash
uv sync
```

或使用 pip：

```bash
pip install opencv-python pillow pyyaml torch torchvision transformers
```

> 首次執行會自動下載 CLIP 模型（約 600MB）

## 使用方式

### Step 1 — 設定 `config.yaml`

```yaml
source:
  path: "CCTV_01/VIRAT_S_000002.mp4"   # 影片檔路徑

extraction:
  output_dir: "data/extracted_frames"
  frame_interval: 10    # 每隔幾幀抽一張
  max_frames: 200       # 上限，0 = 不限制

classification:
  output_dir: "data/labeled_output"
  positive_label: "a person wearing red clothes"   # 想找的類別
  negative_labels:                                 # 干擾項（可填多個）
    - "a person not wearing red clothes"
    - "a red object or vehicle"
    - "a street background"
  positive_folder: "red"
  negative_folder: "non_red"
  confidence_threshold: 0.55
```

### Step 2 — 抽取 frames

```bash
python src/extract_frames.py
# 結果輸出到 data/extracted_frames/
```

### Step 3 — 自動分類

```bash
python src/auto_label.py
# 結果輸出到 data/labeled_output/red/ 和 data/labeled_output/non_red/
```

## 設定說明

| 欄位 | 說明 |
|------|------|
| `source.path` | 影片檔路徑（支援 `.mp4` / `.avi` / `.mov` 等） |
| `extraction.frame_interval` | 每幾幀抽一張（`1` = 每幀，`30` ≈ 每秒 1 張） |
| `extraction.max_frames` | 最多抽幾張（`0` = 不限制） |
| `classification.positive_label` | CLIP 正類描述，即你想找的目標 |
| `classification.negative_labels` | CLIP 干擾項列表，愈多干擾項排除效果愈精準 |
| `classification.confidence_threshold` | 正類機率門檻，建議範圍 `0.5` ~ `0.7` |

### 多干擾項的原理

CLIP 以 softmax 計算圖片與所有標籤的相似度，干擾項愈具體，正類機率的估計就愈精準。例如偵測紅衣人時加入 `"a red object or vehicle"` 可避免把紅色車輛誤判為正類。

換成其他偵測目標（例如「戴安全帽」）只需修改 `config.yaml` 的標籤文字，不需動程式碼。

## 技術細節

- **模型**：OpenAI CLIP (`clip-vit-base-patch32`)
- **框架**：PyTorch + Transformers + OpenCV
- **分類邏輯**：`positive_label` 機率 ≥ `confidence_threshold` → positive 資料夾，否則 → negative 資料夾
