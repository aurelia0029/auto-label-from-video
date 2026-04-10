# Label Red - 紅色衣物偵測系統

使用 OpenAI CLIP 模型進行紅色衣物偵測的電腦視覺專案。

## 專案簡介

本專案提供兩個主要功能：
1. **紅衣偵測** - 使用 CLIP 模型偵測圖片中是否有穿紅色衣服的人
2. **PETA 資料集處理** - 從 PETA 行人屬性資料集中提取紅色/非紅色樣本

## 專案架構

```
label_red/
├── src/                    # 主要程式碼
│   ├── label.py           # CLIP 紅衣偵測程式
│   └── view_PETA.py       # PETA 資料集處理程式
├── data/                   # 資料目錄
│   ├── test_images/       # 測試用圖片
│   └── PETA/              # PETA 資料集 (需自行下載)
├── pyproject.toml          # Python 專案設定
└── README.md
```

## 安裝方式

### 1. 安裝依賴套件

使用 uv (推薦):
```bash
uv sync
```

或使用 pip:
```bash
pip install pillow scipy torch torchvision transformers
```

### 2. 下載 PETA 資料集 (選用)

如果需要使用 PETA 資料集處理功能，請：

1. 從 [Dropbox 下載 PETA 資料夾](https://www.dropbox.com/scl/fo/boipdmufnsnsvmfdle5um/AMbwWDNnlBWnVbnxxv4VcFM?rlkey=ftwdjxqo5l1dfnshy97ej8kka&e=1&st=ropvnxnj&dl=0)
2. 將下載的 `PETA` 資料夾放到 `data/` 目錄下
3. 確保目錄結構如下：
   ```
   data/PETA/
   ├── PETA.mat
   └── images/  (或其他包含圖片的資料夾)
   ```

資料來源：[OpenPAR GitHub Repository](https://github.com/Event-AHU/OpenPAR)

## 使用方式

### 紅衣偵測

```python
from src.label import detect_red_clothes

# 偵測圖片中的紅衣人物
detect_red_clothes("data/test_images/test_red.jpg")
```

輸出範例：
```
--- 檢測結果 ---
圖片路徑: data/test_images/test_red.jpg
偵測到紅衣人: Yes
信心指數: 78.45%
```

### PETA 資料集處理

```python
from src.view_PETA import process_peta_multiclass

# 處理 PETA 資料集，分類紅色/非紅色樣本
process_peta_multiclass()
```

## 技術細節

- **模型**: OpenAI CLIP (clip-vit-base-patch32)
- **框架**: PyTorch + Transformers
- **信心門檻**: 55% (可在 `src/label.py` 中調整)

## 注意事項

- 首次執行會自動下載 CLIP 模型 (約 600MB)
- PETA 資料集不包含在此 repository 中，需自行下載
- 建議使用 Python 3.14+

## License

本專案僅供學習研究使用。
