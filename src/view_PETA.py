import scipy.io
import os
import shutil

# 設定路徑
MAT_PATH = '../data/PETA/PETA.mat'
IMG_ROOT = '../data/PETA/images'  # 指向包含各個子資料夾（如 VIPeR, PRID）的根目錄
OUTPUT_DIR = '../data/PETA/peta_red_dataset'

def process_peta_multiclass():
    mat = scipy.io.loadmat(MAT_PATH)
    peta_data = mat['peta'][0, 0]
    data_items = peta_data[0]
    
    # 建立輸出目錄
    os.makedirs(os.path.join(OUTPUT_DIR, 'red'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'non_red'), exist_ok=True)

    stats = {'red': 0, 'non_red': 0}

    # 根據官方說明，顏色是最後四個多分類屬性
    # 通常在 PETA.mat 中，labels 矩陣的最後四列就是這四個多分類
    # 1: footwear, 2: hair, 3: lowerbody, 4: upperbody
    # 顏色的順序中，Red 索引通常是 8 (對應顏色列表中的第 9 個)

    for i in range(len(data_items)):
        img_rel_path = str(data_items[i][0][0])
        # labels 是一個 1x65 的向量 (61 個二元 + 4 個多分類)
        labels = data_items[i][1].flatten()
        
        # 取得最後兩個多分類：lowerbody (index 63) 與 upperbody (index 64)
        # 注意：具體索引可能因 mat 檔封裝方式微調，建議先 print(labels.shape)
        lower_color = labels[63]
        upper_color = labels[64]
        
        # 判斷是否為紅色 (假設 8 代表 Red)
        is_red = (upper_color == 8 or lower_color == 8)
        
        label_key = 'red' if is_red else 'non_red'
        src = os.path.join(IMG_ROOT, img_rel_path)
        
        if os.path.exists(src):
            # 轉換路徑分隔符號以避免存檔錯誤
            safe_name = img_rel_path.replace('/', '_').replace('\\', '_')
            dst = os.path.join(OUTPUT_DIR, label_key, safe_name)
            shutil.copy(src, dst)
            stats[label_key] += 1

    print(f"\n✅ 處理完成！")
    print(f"🔴 紅色樣本: {stats['red']}")
    print(f"⚪ 非紅色樣本: {stats['non_red']}")

if __name__ == "__main__":
    process_peta_multiclass()
