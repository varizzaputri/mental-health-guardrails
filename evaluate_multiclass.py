import argparse
import pandas as pd
import requests
import json
import time
import os
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from prompts import get_multiclass_prompt, get_system_prompt

ITERATIONS = 3
TARGET_CLASSES = ["RENDAH", "SEDANG", "TINGGI"]
DATASET_PATH = "../Dataset_Evaluasi_LLM_Classifier_Teta (1).xlsx"
TETA_URL = "https://u7720-3qst-e5002966.singapore-a.gpuhub.com:8443/v1/chat/completions"

def parse_json_output(content):
    match = re.search(r'\{.*\}', content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            pred = data.get("kelas", data.get("label", "ERROR")).upper()
            return pred, data.get("confidence", 0.0)
        except:
            pass
    return "ERROR", 0.0

def run_evaluation(model_name):
    df = pd.read_excel(DATASET_PATH)
    hasil_akhir = []
    
    is_api = model_name.lower() == "teta-v3.2"
    model, tokenizer = None, None

    print(f"\n{'='*50}\nMemulai Evaluasi: {model_name}\n{'='*50}")

    # Load HF Model sekali saja di awal agar tidak boros RAM/VRAM
    if not is_api:
        print("[+] Memuat model ke GPU via Transformers...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        print("[+] Model berhasil dimuat!")

    for iterasi in range(1, ITERATIONS + 1):
        print(f"\n--- Menjalankan Iterasi {iterasi}/{ITERATIONS} ---")
        y_true, y_pred = [], []
        
        for index, row in df.iterrows():
            teks = row['Kalimat Pengguna (Sintetis)']
            gt = str(row['Label Risiko']).strip().upper()
            
            sys_prompt = get_system_prompt()
            user_prompt = get_multiclass_prompt(teks)
            
            lat = 0.0
            content = ""

            if is_api:
                # Mode API untuk TETA v3.2
                payload = {
                    "model": "teta-v3.2",
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "temperature": 0.0
                }
                try:
                    start_time = time.time()
                    response = requests.post(TETA_URL, json=payload).json()
                    lat = (time.time() - start_time) * 1000
                    content = response['choices'][0]['message']['content']
                except Exception as e:
                    print(f"[ERROR API] {e}")
            else:
                # Mode Local Transformers untuk Qwen, Gemma, dll
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": user_prompt}
                ]
                text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = tokenizer(text, return_tensors="pt").to(model.device)
                
                start_time = time.time()
                with torch.no_grad():
                    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
                lat = (time.time() - start_time) * 1000
                
                input_length = inputs.input_ids.shape[1]
                content = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)

            pred_kelas, conf = parse_json_output(content)
            if pred_kelas not in TARGET_CLASSES: pred_kelas = "ERROR"
            
            y_true.append(gt)
            y_pred.append(pred_kelas)
            print(f"[{index+1}/{len(df)}] GT: {gt} | Pred: {pred_kelas} | Conf: {conf} | Lat: {lat:.1f}ms")

        # Kalkulasi Metrik (Hanya ambil data yang valid/bukan ERROR)
        valid_idx = [i for i, p in enumerate(y_pred) if p != "ERROR"]
        yt_val = [y_true[i] for i in valid_idx]
        yp_val = [y_pred[i] for i in valid_idx]

        if len(yt_val) > 0:
            p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(yt_val, yp_val, labels=TARGET_CLASSES, average=None, zero_division=0)
            p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(yt_val, yp_val, labels=TARGET_CLASSES, average='macro', zero_division=0)
            p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(yt_val, yp_val, labels=TARGET_CLASSES, average='micro', zero_division=0)
            
            print(f"\n--- HASIL ITERASI {iterasi} [{model_name}] ---")
            print(f"Macro -> Precision: {p_mac:.4f}, Recall: {r_mac:.4f}, F1: {f1_mac:.4f}")
            print(f"Micro -> Precision: {p_mic:.4f}, Recall: {r_mic:.4f}, F1: {f1_mic:.4f}")

            hasil_akhir.append({
                "Model": model_name, "Iterasi": iterasi,
                "Macro_F1": f1_mac, "Macro_Precision": p_mac, "Macro_Recall": r_mac,
                "Micro_F1": f1_mic, "Micro_Precision": p_mic, "Micro_Recall": r_mic,
                "F1_RENDAH": f1_bin[0], "F1_SEDANG": f1_bin[1], "F1_TINGGI": f1_bin[2]
            })

            cm = confusion_matrix(yt_val, yp_val, labels=TARGET_CLASSES)
            plt.figure(figsize=(8,6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=TARGET_CLASSES, yticklabels=TARGET_CLASSES)
            plt.title(f'Confusion Matrix: {model_name} (Iterasi {iterasi})')
            plt.ylabel('Ground Truth')
            plt.xlabel('Prediksi Model')
            
            safe_model_name = model_name.replace(':', '_').replace('/', '_')
            plot_name = f"CM_{safe_model_name}_Iter{iterasi}.png"
            plt.savefig(plot_name)
            plt.close()

    df_hasil = pd.DataFrame(hasil_akhir)
    safe_model_name = model_name.replace(':', '_').replace('/', '_')
    excel_name = f"Rekap_MultiClass_{safe_model_name}.xlsx"
    df_hasil.to_excel(excel_name, index=False)
    print(f"\n[+] SELESAI! Rekap disimpan di: {excel_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Contoh: Qwen/Qwen3Guard-Gen-8B atau teta-v3.2")
    args = parser.parse_args()
    run_evaluation(args.model)
