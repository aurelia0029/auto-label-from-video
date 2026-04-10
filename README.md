# auto-label-from-video

使用 OpenAI CLIP 模型，從影片自動抽取 frames 並分類成兩個資料夾的工具。支援網頁介面進行人工校正。

## 專案架構

```
auto-label-from-video/
├── config.yaml              # 命令列使用的設定檔
├── src/
│   ├── app.py               # FastAPI 網頁伺服器
│   ├── static/
│   │   └── index.html       # 網頁介面
│   ├── extract_frames.py    # 命令列：從影片解碼抽取 frames
│   ├── auto_label.py        # 命令列：CLIP 分類到兩個資料夾
│   ├── label.py             # 單張圖片紅衣偵測（原型）
│   └── view_PETA.py         # PETA 資料集處理工具
├── data/
│   ├── uploads/             # 上傳的影片暫存
│   ├── extracted_frames/    # 抽取的 frames
│   └── labeled_output/      # 分類結果
│       ├── red/             # positive 圖片
│       └── non_red/         # negative 圖片
├── CCTV_01/                 # 影片資料集
└── pyproject.toml
```

## 安裝

```bash
uv sync
```

或使用 pip：

```bash
pip install fastapi uvicorn[standard] python-multipart opencv-python pillow pyyaml torch torchvision transformers
```

> 首次執行會自動下載 CLIP 模型（約 600MB）

---

## 網頁介面（推薦）

```bash
uvicorn src.app:app --reload
```

開啟瀏覽器：`http://localhost:8000`

### 操作流程

**1. 上傳影片**
拖曳或點擊上傳 `.mp4` / `.avi` / `.mov` 等格式的影片。

**2. 設定標籤**
- **Positive label**：想找的目標，例如 `a person wearing red clothes`
- **Negative labels**：干擾項，可新增多個，幫助 CLIP 做更精準的排除

**3. 調整參數**

| 參數 | 說明 |
|------|------|
| Frame 抽樣間隔 | 每幾幀抽一張（建議 10～30） |
| 最多抽取張數 | 上限，`0` = 不限 |
| 信心門檻 | Positive 機率須超過此值才分類為正類 |
| 資料夾名稱 | 自訂兩個輸出資料夾的名稱 |

**4. 執行分類**
點擊「執行分類」，頂部進度條即時顯示抽取和分類進度。

**5. 人工校正**

結果以圖片卡片顯示在左右兩欄（Positive / Negative）。

**縮放**
每個欄位標題的 **−** / **＋** 按鈕可獨立調整該欄的圖片大小（80px ～ 280px）。

**顯示信心值**
頂部勾選「顯示信心值」，每張圖片下方會出現 CLIP 對 positive label 的機率百分比。

**查看圖片細節（Lightbox）**

| 信心值顯示 | 點擊行為 |
|-----------|---------|
| OFF | 點擊卡片 = 選取 |
| ON  | 點擊左上角 checkbox 區域 = 選取；點擊圖片其他區域 = 開啟 Lightbox |

Lightbox 顯示完整圖片、檔名、所屬分類、信心值，並可：
- 點擊 ← / → 或按鍵盤方向鍵切換同類別圖片
- 按 `Esc` 或點擊背景關閉
- 直接在 Lightbox 內移動或刪除圖片（操作後自動顯示下一張）

**選取多張圖片**

| 操作 | 方式 |
|------|------|
| 選取單張 | 點擊圖片卡片 |
| 選取範圍 | Shift + 點擊 |
| 框選多張 | 在空白區域拖曳（rubber-band） |
| 全選 | 點擊欄位標題的「全選」按鈕 |

選取後，欄位上方出現工具列，可對所有選取圖片批次**移動**或**刪除**。

**6. 下載結果**
分類完成後點擊「⬇ 下載 ZIP」，ZIP 內保留資料夾結構：

```
labeled_output.zip
├── red/
│   ├── 000010.jpg
│   └── ...
└── non_red/
    ├── 000020.jpg
    └── ...
```

---

## 命令列使用

設定 `config.yaml` 後依序執行：

```bash
# Step 1：從影片解碼並抽取 frames
python src/extract_frames.py

# Step 2：CLIP 自動分類
python src/auto_label.py
```

### `config.yaml` 說明

```yaml
source:
  path: "CCTV_01/VIRAT_S_000002.mp4"   # 影片檔路徑

extraction:
  output_dir: "data/extracted_frames"
  frame_interval: 10    # 每幾幀抽一張
  max_frames: 200       # 上限，0 = 不限

classification:
  output_dir: "data/labeled_output"
  positive_label: "a person wearing red clothes"
  negative_labels:
    - "a person not wearing red clothes"
    - "a red object or vehicle"
    - "a street background"
  positive_folder: "red"
  negative_folder: "non_red"
  confidence_threshold: 0.55
```

---

## 多干擾項原理

CLIP 以 softmax 計算圖片與所有標籤的相似度，干擾項愈具體，正類機率的估計就愈精準。
例如偵測紅衣人時加入 `"a red object or vehicle"` 可避免把紅色車輛誤判為正類。

換成其他偵測目標只需修改標籤文字，不需動程式碼。

---

## 技術細節

- **模型**：OpenAI CLIP (`clip-vit-base-patch32`)
- **後端**：FastAPI + OpenCV + PyTorch
- **分類邏輯**：`positive_label` 機率 ≥ `confidence_threshold` → positive，否則 → negative
- **分類結果位置**：`data/labeled_output/{positive_folder}/` 和 `data/labeled_output/{negative_folder}/`
