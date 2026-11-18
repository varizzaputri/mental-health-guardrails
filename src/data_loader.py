import os
import json
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """Membaca CSV, JSON, atau XLSX dan mengembalikan DataFrame terstandar."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File dataset tidak ditemukan: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext == '.csv':
        df = pd.read_csv(file_path)
    elif ext == '.json':
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data)
    elif ext == '.jsonl':
        df = pd.read_json(file_path, lines=True)
    elif ext in ['.xlsx', '.xls']:
        df = pd.read_excel(file_path)
    else:
        raise ValueError(f"Format {ext} tidak didukung. Gunakan CSV, JSON, atau XLSX.")

    # Deteksi otomatis nama kolom teks dan label
    col_text = next((c for c in df.columns if str(c).lower() in ['text', 'kalimat', 'prompt', 'utterance', 'input', 'kalimat pengguna (sintetis)']), None)
    col_label = next((c for c in df.columns if str(c).lower() in ['ground_truth', 'label', 'kelas', 'target', 'kategori', 'label risiko']), None)
    col_id = next((c for c in df.columns if str(c).lower() in ['id', 'sample_id', 'id_sampel', 'no', 'id sampel']), None)

    if not col_text or not col_label:
        raise KeyError(f"Dataset wajib memiliki kolom teks dan label ground truth. Kolom yang terbaca: {list(df.columns)}")

    clean_df = pd.DataFrame()
    clean_df['id'] = df[col_id].astype(str) if col_id else [f"S-{i+1:03d}" for i in range(len(df))]
    clean_df['text'] = df[col_text].astype(str)
    clean_df['ground_truth'] = df[col_label].astype(str).str.strip().str.upper()

    return clean_df
