import streamlit as st
import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
import os
import tempfile
from io import BytesIO
from zipfile import ZipFile, BadZipFile
import glob
import subprocess
import re
from rasterio.transform import from_origin
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image
import matplotlib.cm as cm
import folium
from streamlit_folium import st_folium
from affine import Affine
from PIL import ImageDraw


st.set_page_config(
    page_title="Analisis Vegetasi Multispektral",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inject_premium_css():
    st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

            html, body, [class*="css"] {
                font-family: 'Poppins', sans-serif;
                background-color: #f9f9f9;
            }

            .block-container {
                padding-top: 2rem;
                padding-bottom: 2rem;
                padding-left: 3rem;
                padding-right: 3rem;
            }

            h1 {
                font-size: 2.5rem;
                color: #2e7d32;
                margin-bottom: 1rem;
            }

            h2, h3 {
                color: #388e3c;
                margin-top: 1.5rem;
                margin-bottom: 1rem;
            }

            .stButton>button {
                background-color: #43a047;
                color: white;
                font-weight: bold;
                border-radius: 8px;
                padding: 0.6rem 1.3rem;
                border: none;
                transition: all 0.3s ease;
            }

            .stButton>button:hover {
                background-color: #2e7d32;
                transform: scale(1.02);
            }

            .stDownloadButton>button {
                background-color: #1e88e5;
                color: white;
                font-weight: bold;
                border-radius: 8px;
                padding: 0.6rem 1.3rem;
                border: none;
                transition: all 0.3s ease;
            }

            .stDownloadButton>button:hover {
                background-color: #1565c0;
                transform: scale(1.02);
            }

            .stSlider {
                padding-top: 1rem;
                padding-bottom: 1rem;
            }

            .stSelectbox label, .stRadio label {
                font-weight: 600;
                color: #37474f;
            }

            .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
                margin-top: 0.5rem;
            }

            /* Table */
            .stDataFrame {
                margin-top: 1rem;
            }
        </style>
    """, unsafe_allow_html=True)

inject_premium_css()


# ==============================
# Fungsi Perhitungan Indeks
# ==============================
def calculate_ndvi(nir, red):
    return np.clip((nir - red) / (nir + red + 1e-10), -1, 1)

def calculate_ndre(nir, rededge):
    return np.clip((nir - rededge) / (nir + rededge + 1e-10), -1, 1)

def calculate_gndvi(nir, green):
    return np.clip((nir - green) / (nir + green + 1e-10), -1, 1)

def calculate_savi(nir, red, L=0.5):
    return np.clip(((nir - red) / (nir + red + L)) * (1 + L), -1, 1)

def calculate_lpi(nir, red):
    return np.clip(nir / (nir + red + 1e-10), 0, 1)

def calculate_ipvi(nir, red):
    return np.clip(nir / (nir + red + 1e-10), 0, 1)

def analyze_index_threshold(index_array, threshold):
    mask = index_array > threshold
    percentage_above = 100 * np.sum(mask) / mask.size
    stats = {
        'Mean Value': np.mean(index_array[mask]),
        'Max Value': np.max(index_array[mask]),
        'Min Value': np.min(index_array[mask]),
        'Std Dev': np.std(index_array[mask]),
        'Pixels Above Threshold': np.sum(mask),
        'Percentage Above Threshold': percentage_above
    }
    return stats
def explain_index(index_name):
    explanations = {
        "NDVI": """
        ### 🟢 Apa itu NDVI?
        NDVI adalah cara untuk **melihat seberapa sehat tanaman** berdasarkan cahaya yang dipantulkan oleh daun.

        - **Semakin tinggi angkanya (mendekati 1), tanaman semakin hijau dan sehat.**
        - **Nilai rendah** berarti tanaman kering, mati, atau tidak ada tanaman sama sekali.

        NDVI sangat sering digunakan untuk **memantau kesehatan vegetasi**.
        """,
        "NDRE": """
        ### 🟢 Apa itu NDRE?
        NDRE digunakan untuk **mengetahui tingkat kesehatan daun bagian dalam**, khususnya kadar **klorofil** (zat hijau daun).

        - Cocok untuk **melihat gejala stres tanaman lebih awal**, bahkan sebelum terlihat oleh mata.
        - Biasa dipakai untuk memutuskan **kapan harus memupuk atau menyiram**.

        Nilai tinggi → daun aktif dan sehat,  
        Nilai rendah → kemungkinan butuh perhatian.
        """,
        "GNDVI": """
        ### 🟢 Apa itu GNDVI?
        GNDVI mirip NDVI, tapi menggunakan **cahaya hijau**.

        - Sangat bagus untuk melihat **kandungan nitrogen** dalam tanaman.
        - Bisa membantu mengetahui **apakah tanaman kekurangan nutrisi**.

        Jika nilai tinggi, tanaman kemungkinan **cukup nutrisi** dan aktif berfotosintesis.
        """,
        "SAVI": """
        ### 🟢 Apa itu SAVI?
        SAVI adalah versi NDVI yang sudah diperbaiki supaya lebih akurat di **tanah yang kering atau tidak terlalu banyak tanaman**.

        - Cocok untuk **lahan pertanian gersang** atau area kering.
        - Membantu membedakan antara tanah dan tanaman.

        Jadi SAVI memberikan gambaran yang lebih adil untuk **tanaman di tanah tandus**.
        """,
        "LPI": """
        ### 🟢 Apa itu LPI?
        LPI melihat **seberapa banyak pigmen daun** (seperti klorofil) yang ada di tanaman.

        - Nilai tinggi = daun banyak pigmen hijau = tanaman sehat
        - Nilai rendah = bisa jadi daun mulai menguning

        Mudah dipakai untuk mengetahui apakah daun tanaman sedang **dalam kondisi baik atau menurun**.
        """,
        "IPVI": """
        ### 🟢 Apa itu IPVI?
        IPVI adalah cara **sederhana dan cepat** untuk menilai seberapa baik kondisi tanaman dari citra udara.

        - Nilainya antara 0 sampai 1.
        - Semakin besar nilainya, **semakin sehat tanamannya**.

        Cocok jika kamu ingin melihat kondisi tanaman **dengan cepat dan efisien**.
        """
    }

    if index_name in explanations:
        st.markdown(explanations[index_name])

# ==============================
# Fungsi Tambahan
# ==============================
def load_mrk(mrk_path):
    mrk_data = []
    with open(mrk_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if 'Lat' in line and 'Lon' in line:
                elements = re.split(r'[,\s]+', line.strip())
                lat = lon = None
                for i, e in enumerate(elements):
                    if e == 'Lat' and i > 0:
                        lat = float(elements[i-1])
                    if e == 'Lon' and i > 0:
                        lon = float(elements[i-1])
                if lat is not None and lon is not None:
                    mrk_data.append((lat, lon))
    return mrk_data

def build_mosaic(input_folder, output_folder):
    """
    Membuat orthomosaic stack dari 4 file tif (R, G, RE, NIR).
    Menggunakan gdal_merge.py untuk menggabungkan menjadi stack 4-band.
    Menampilkan pesan error jika ada file hilang atau proses gagal.
    """
    output_stack = os.path.join(output_folder, "stack_4band.tif")
    band_files = {
        "R":  os.path.join(output_folder, "orthomosaic_R.tif"),
        "G":  os.path.join(output_folder, "orthomosaic_G.tif"),
        "RE": os.path.join(output_folder, "orthomosaic_RE.tif"),
        "NIR":os.path.join(output_folder, "orthomosaic_NIR.tif")
    }

    # Cek apakah semua file tersedia
    missing = [band for band, path in band_files.items() if not os.path.exists(path)]
    if missing:
        st.error(f"❌ File tidak ditemukan untuk band: {', '.join(missing)}")
        return None

    # Jalankan gdal_merge.py
    try:
        cmd = [os.path.join(os.environ["VIRTUAL_ENV"], "bin", "gdal_merge.py"), "-separate", "-o", output_stack] + list(band_files.values())
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            st.error("❌ Gagal menjalankan gdal_merge.py.")
            st.text(result.stderr)
            return None

        return output_stack

    except Exception as e:
        st.error("❌ Terjadi kesalahan saat membuat mosaik stack.")
        st.text(str(e))
        return None



