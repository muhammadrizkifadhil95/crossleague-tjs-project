import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from ViT_v2 import BinaryClassificationViTModel
from Prepfortrain import preprocess, data_split

# Membaca ulang data untuk mendapatkan daftar fitur
data_temp = pd.read_csv('../data_extraction/data/final_df.csv')
feature_names = [c for c in data_temp.columns if c not in [
    'target', 'pitcher', 'new_before_tj',
    'player_name', 'league_level', 'game_date'
]]

# 1. Persiapan Data dan Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
X_full, y_full = preprocess()

# Menggunakan Seed 700 sebagai referensi utama
_, _, X_test, _, _, y_test = data_split(X_full, y_full, random_state=700)

model = BinaryClassificationViTModel(0.35, torch.tensor(10)).to(device)
model.load_state_dict(torch.load('Run14_Seed700_Window60.pth'))
model.eval()

def get_padded_tensor(X_tensor):
    import torch.nn.functional as F
    return F.pad(X_tensor, (0, 224 - X_tensor.shape[2], 0, 224 - X_tensor.shape[1]), "constant", -5)

# 2. Ekstraksi Attention untuk Kasus "Cedera"
# Memfilter data dengan label cedera (Target = 1)
injured_indices = np.where(y_test.flatten() == 1)[0]
sample_idx = injured_indices[0]  # Mengambil satu sampel cedera
single_sample = get_padded_tensor(X_test[sample_idx:sample_idx+1]).to(device)

with torch.no_grad():
    # Memanggil model dengan return_attention=True
    logits, mean_attention, max_attention = model(single_sample, return_attention=True)

# 3. Pemrosesan Tensor Attention
# Bentuk tensor: (batch_size, num_heads, num_patches, num_patches)
# Visualisasi hubungan antara token CLS dan patch input
# Grid patch: 14x14 (224/16 = 14)
attn_map = mean_attention[0].mean(dim=0)  # Rata-rata antar head
cls_attn = attn_map[0, 1:].reshape(14, 14).cpu().numpy()  # Attention CLS ke semua patch

# 4. Visualisasi Heatmap (Gaya Gambar 8)
plt.figure(figsize=(10, 8))
sns.heatmap(cls_attn, annot=False, cmap='magma')
plt.title("Peta Attention ViT: Identifikasi Pola Cedera (Seed 700)")
plt.xlabel("Patch Metrik Lemparan")
plt.ylabel("Patch Deret Waktu (Hari sebelum Cedera)")
plt.savefig("vit_attention_map_gambar8.png")
plt.show()

# --- PERSIAPAN LANGKAH 2: Identifikasi Metrik pada Patch 2 ---
# Mengambil fitur berdasarkan lebar patch (16 piksel)
patch_2_metrics = feature_names[32:48]

print("\n" + "="*50)
print("METRIK DALAM AREA ATTENTION TINGGI (PATCH 2)")
print("="*50)
for i, name in enumerate(patch_2_metrics):
    index = 32 + i
    print(f"Indeks {index}: {name}")
print("="*50)