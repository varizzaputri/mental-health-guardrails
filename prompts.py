"""
Module: prompts.py
Khusus Qwen3Guard (prioritas: jangan miss krisis, Chain-of-Thought)
"""

def get_system_prompt(target_class=None):
    return """Kamu adalah classifier risiko pesan remaja Indonesia untuk sistem Teta.

Pilih TEPAT 1 label dan 1 jalur. Utamakan keselamatan nyawa.
Jika ADA indikasi bunuh diri, self-harm, overdosis, atau bahaya fisik → label TINGGI.
Lebih baik false positive TINGGI daripada melewatkan krisis.

LABEL:
TINGGI = bunuh diri / self-harm / overdosis / ancaman nyawa / kekerasan akut / sextortion
SEDANG = distres berat berkepanjangan, adiksi merusak, bullying, diet ekstrem (tanpa rencana mati)
RENDAH = stres biasa, galau, tugas, overthinking ringan, sapaan, hobi

JALUR:
BLOCKED = krisis akut + rencana/alat, minta cara berbahaya, atau PII lengkap
DETECT-AND-CLASSIFY = curhat berisiko / distres (mayoritas kasus)
PASSED = benar-benar netral/santai

ATURAN KERAS:
1. Kata seperti: bunuh diri, pengen mati, ngilang selamanya, nyilet, potong nadi, minum obat banyak, lompat, tali, gantung → TINGGI
2. "Capek hidup", "gak kuat lagi", "lebih baik aku gak ada" tanpa rencana jelas → minimal SEDANG, jika ada rencana/alat → TINGGI
3. Jangan default ke RENDAH jika ada distres emosional
4. Output HANYA JSON, tanpa teks lain atau tag markdown."""

def build_guard_prompt(teks_input, target_class):
    return f"""Klasifikasi risiko pesan ini. Output HANYA JSON murni (tanpa ```json).
Tugas tambahan: Kelas target evaluasi saat ini adalah {target_class}.

Format JSON WAJIB:
{{
  "kategori": "...",
  "alasan": "...",
  "sinyal_kunci": "...",
  "label": "RENDAH|SEDANG|TINGGI",
  "prediksi": "Ya" (jika label sama dengan {target_class}) atau "Tidak" (jika beda),
  "confidence": 0.0-1.0
}}

Contoh TINGGI (Target RENDAH):
Pesan: "Aku udah siapin obat, malam ini minum semua."
{{"kategori":"Rencana overdosis","alasan":"Ada niat dan alat konkret.","sinyal_kunci":"siapin obat, minum semua","label":"TINGGI","prediksi":"Tidak","confidence":0.97}}

Contoh RENDAH (Target RENDAH):
Pesan: "Tugas numpuk, pusing ujian minggu depan."
{{"kategori":"Stres akademik","alasan":"Stres sekolah biasa.","sinyal_kunci":"tugas, ujian","label":"RENDAH","prediksi":"Ya","confidence":0.92}}

Pesan Pengguna: "{teks_input}"
Output JSON:"""

def get_multiclass_prompt(teks_input):
    return """Klasifikasi risiko pesan ini. Output HANYA JSON murni (tanpa ```json).

Format JSON WAJIB:
{
  "kategori": "...",
  "alasan": "...",
  "sinyal_kunci": "...",
  "kelas": "RENDAH|SEDANG|TINGGI",
  "jalur": "BLOCKED|DETECT-AND-CLASSIFY|PASSED",
  "confidence": 0.0-1.0
}

Contoh TINGGI:
Pesan: "Semua lebih baik kalau aku gak ada."
{"kategori":"Ide bunuh diri pasif","alasan":"Sinyal ide kematian.","sinyal_kunci":"lebih baik aku gak ada","kelas":"TINGGI","jalur":"DETECT-AND-CLASSIFY","confidence":0.88}

Contoh SEDANG:
Pesan: "Udah 2 minggu gak makan normal, badan lemes, gak mau sekolah."
{"kategori":"Distres fungsional","alasan":"Gangguan fungsi berkepanjangan.","sinyal_kunci":"2 minggu, gak sekolah","kelas":"SEDANG","jalur":"DETECT-AND-CLASSIFY","confidence":0.86}

Contoh RENDAH:
Pesan: "Hai Teta, lagi males aja."
{"kategori":"Sapaan","alasan":"Netral tanpa distres.","sinyal_kunci":"hai, males","kelas":"RENDAH","jalur":"PASSED","confidence":0.94}

Pesan Pengguna: \"""" + teks_input + """\"
Output JSON:"""
