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
import gdown
from streamlit_image_coordinates import streamlit_image_coordinates
from PIL import Image
import matplotlib.cm as cm
import folium
from streamlit_folium import st_folium
from affine import Affine
from streamlit_folium import st_folium
from PIL import ImageDraw


st.set_page_config(
    page_title="🌿 Analisis Vegetasi Multispektral",
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

def embed_coordinates(tiff_folder, mrk_data, pixel_size=0.001905):
    tif_files = sorted(glob.glob(os.path.join(tiff_folder, '*.tif')))
    idx_mapping = {}
    idx = 0
    for i in range(len(mrk_data)):
        idx_mapping[i] = tif_files[idx:idx+5]
        idx += 5
    for i, files in idx_mapping.items():
        lat, lon = mrk_data[i]
        for file_path in files:
            with rasterio.open(file_path) as src:
                image = src.read(1)
                profile = src.profile
            width = profile['width']
            height = profile['height']
            transform = from_origin(
                lon - (width / 2) * pixel_size,
                lat + (height / 2) * pixel_size,
                pixel_size, pixel_size
            )
            profile.update({
                'driver': 'GTiff',
                'transform': transform,
                'crs': 'EPSG:4326',
                'count': 1
            })
            with rasterio.open(file_path, 'w', **profile) as dst:
                dst.write(image, 1)

def safe_extract_zip(zip_path, extract_to):
    try:
        with ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except BadZipFile:
        st.error("❌ File ZIP tidak valid atau rusak.")
        return False

def classify_ndvi(value):
    if value > 0.66:
        return "Sangat Sehat"
    elif value > 0.33:
        return "Sehat"
    elif value > 0.1:
        return "Kurang Sehat"
    elif value >= 0.0:
        return "Tidak Sehat"
    else:
        return "Bukan Tanaman"

def classify_ndre(value):
    if value > 0.45:
        return "Sangat Sehat"
    elif value > 0.3:
        return "Sehat"
    elif value > 0.15:
        return "Kurang Sehat"
    elif value >= 0.0:
        return "Tidak Sehat"
    else:
        return "Bukan Tanaman"
        
def classify_gndvi(value):
    if value > 0.6:
        return "Sangat Sehat"
    elif value > 0.4:
        return "Sehat"
    elif value > 0.2:
        return "Kurang Sehat"
    elif value >= 0.0:
        return "Tidak Sehat"
    else:
        return "Bukan Tanaman"

def classify_savi(value):
    if value > 0.5:
        return "Sangat Sehat"
    elif value > 0.3:
        return "Sehat"
    elif value > 0.15:
        return "Kurang Sehat"
    elif value >= 0.0:
        return "Tidak Sehat"
    else:
        return "Bukan Tanaman"

def analyze_classification(index_array, classify_func, pixel_area=0.45):
    flat = index_array.flatten()
    flat = flat[~np.isnan(flat)]  # hindari nilai NaN

    classes = [classify_func(val) for val in flat]
    df = pd.DataFrame({'Class': classes})
    summary = df['Class'].value_counts().reset_index()
    summary.columns = ['Kondisi Tanaman', 'Jumlah Pixel']
    
    desired_order = ["Sangat Sehat", "Sehat", "Kurang Sehat", "Tidak Sehat", "Bukan Tanaman"]
    summary['Kondisi Tanaman'] = pd.Categorical(summary['Kondisi Tanaman'], categories=desired_order, ordered=True)
    summary = summary.sort_values('Kondisi Tanaman')

    summary['Percentase (%)'] = (summary['Jumlah Pixel'] / len(flat) * 100).round(2)
    summary['Estimasi Area (m²)'] = (summary['Jumlah Pixel'] * pixel_area).round(2)
    return summary


def render_index_visualization(index_array, index_name, profile):
    st.subheader(f"{index_name} Map")

    # Konversi array ke RGB untuk ditampilkan sebagai gambar
    norm = plt.Normalize(vmin=index_array.min(), vmax=index_array.max())
    cmap = cm.get_cmap('RdYlGn')
    rgba = cmap(norm(index_array))
    rgb = (rgba[:, :, :3] * 255).astype('uint8')
    img = Image.fromarray(rgb)

    # Hitung ukuran tampilan proporsional
    orig_w, orig_h = img.size
    disp_w = 600
    disp_h = int(orig_h * disp_w / orig_w)
    scale_x = orig_w / disp_w
    scale_y = orig_h / disp_h

    # Gambar interaktif
    coords = streamlit_image_coordinates(img, key=f"coord_{index_name}", width=disp_w)

    # Tampilkan informasi jika diklik
    if coords:
        raw_x = coords["x"] * scale_x
        raw_y = coords["y"] * scale_y
        col = int(raw_x)
        row = int(raw_y)
        if 0 <= row < index_array.shape[0] and 0 <= col < index_array.shape[1]:
            t = profile["transform"]
            lon = t.c + col * t.a
            lat = t.f + row * t.e
            val = float(index_array[row, col])

            if index_name == "NDVI":
                kondisi = classify_ndvi(val)
            elif index_name == "NDRE":
                kondisi = classify_ndre(val)
            elif index_name == "GNDVI":
                kondisi = classify_gndvi(val)
            elif index_name == "SAVI":
                kondisi = classify_savi(val)
            else:
                kondisi = "-"

            st.markdown(
                f"📍 **Lon:** `{lon:.6f}`, **Lat:** `{lat:.6f}`, **{index_name}:** `{val:.4f}` → 🌿 **{kondisi}**"
            )
        else:
            st.warning("Klik di luar area citra.")
    else:
    st.markdown("ℹ️ Klik pada gambar untuk melihat nilai **koordinat**, **indeks**, dan **kondisi tanaman**.")

    # === Filter rentang indeks
    st.subheader("Filter Berdasarkan Nilai Indeks")
    mn, mx = float(index_array.min()), float(index_array.max())
    lo, hi = st.slider(f"Rentang {index_name}", mn, mx, (mn, mx), step=0.01)
    filtered = np.where((index_array >= lo) & (index_array <= hi), index_array, np.nan)

    # === Filter berdasarkan klasifikasi tanaman
    st.subheader("Filter Berdasarkan Klasifikasi Tanaman")
    class_option = st.selectbox(
        f"Pilih Klasifikasi {index_name} yang Ditampilkan",
        options=["Semua", "Sangat Sehat", "Sehat", "Kurang Sehat", "Tidak Sehat", "Bukan Tanaman"]
    )

    if class_option != "Semua":
        if index_name == "NDVI":
            mask = np.vectorize(classify_ndvi)(index_array) == class_option
        elif index_name == "NDRE":
            mask = np.vectorize(classify_ndre)(index_array) == class_option
        elif index_name == "GNDVI":
            mask = np.vectorize(classify_gndvi)(index_array) == class_option
        elif index_name == "SAVI":
            mask = np.vectorize(classify_savi)(index_array) == class_option
        else:
            mask = np.ones_like(index_array, dtype=bool)
        filtered = np.where(mask, filtered, np.nan)

    # Tampilkan hasil filter
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    im2 = ax2.imshow(filtered, cmap='RdYlGn', vmin=-1, vmax=1)
    ax2.axis('off')
    fig2.colorbar(im2, ax=ax2, label=index_name)
    st.pyplot(fig2)

    # Analisis klasifikasi
    st.subheader(f"Analisis Klasifikasi Kesehatan Tanaman ({index_name})")
    try:
        pixel_area = profile["transform"].a * abs(profile["transform"].e)
        if pixel_area == 0:
            pixel_area = 0.25
    except:
        pixel_area = 0.25

    if index_name == "NDVI":
        summary = analyze_classification(index_array, classify_ndvi, pixel_area)
    elif index_name == "NDRE":
        summary = analyze_classification(index_array, classify_ndre, pixel_area)
    elif index_name == "GNDVI":
        summary = analyze_classification(index_array, classify_gndvi, pixel_area)
    elif index_name == "SAVI":
        summary = analyze_classification(index_array, classify_savi, pixel_area)

    st.dataframe(summary.rename(columns={
        "Jumlah Pixel": "Jumlah Piksel",
        "Percentase (%)": "Persentase (%)",
        "Estimasi Area (m²)": "Luas (m²)"
    }), hide_index=True)

    # Download GeoTIFF
    st.subheader(f"⬇️ Download {index_name} GeoTIFF")
    profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=None)
    with BytesIO() as mem:
        with rasterio.open(mem, 'w', **profile) as dst:
            dst.write(index_array.astype(rasterio.float32), 1)
        mem.seek(0)
        st.download_button(
            f"Download {index_name}.tif",
            data=mem,
            file_name=f"{index_name.lower()}.tif",
            mime="image/tiff"
        )

    # Download RGB PNG
    st.subheader(f"⬇️ Download {index_name} RGB (Berwarna)")
    with BytesIO() as rgb_buffer:
        img.save(rgb_buffer, format="PNG")
        st.download_button(
            f"Download {index_name} RGB.png",
            data=rgb_buffer.getvalue(),
            file_name=f"{index_name.lower()}_rgb.png",
            mime="image/png"
        )


# 1) Fungsi builder Folium Map dengan caching
# ====================================================
@st.cache_data(show_spinner=False)
def make_folium_map(index_array, transform):
    # 1. Konversi indeks ke RGB
    norm = plt.Normalize(vmin=index_array.min(), vmax=index_array.max())
    cmap = cm.get_cmap("RdYlGn")
    rgba = cmap(norm(index_array))  # (H, W, 4)
    rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    # 2. Simpan sebagai file PNG sementara
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmpfile:
        Image.fromarray(rgb).save(tmpfile.name)
        image_path = tmpfile.name

    # 3. Hitung koordinat geospasial
    h, w = index_array.shape
    west  = transform.c
    north = transform.f
    east  = west + w * transform.a
    south = north + h * transform.e

    # 4. Bangun peta dengan OpenStreetMap
    m = folium.Map(
        location=[(north + south) / 2, (west + east) / 2],
        zoom_start=17,
        tiles='OpenStreetMap'
    )

    # 5. Tampilkan image overlay
    folium.raster_layers.ImageOverlay(
        name='Vegetation Index',
        image=image_path,
        bounds=[[south, west], [north, east]],
        opacity=0.6
    ).add_to(m)

    folium.LayerControl().add_to(m)
    return m
                        
def render_index_on_google_map(index_array, index_name, profile):
    st.subheader(f"{index_name} di Google Map")

    with st.spinner("🔄 Membangun peta berbasis OpenStreetMap..."):
        m = make_folium_map(index_array, profile["transform"])
        data = st_folium(m, width=700, height=500)

    # Klik koordinat
    clicked = data.get("last_clicked")
    if clicked:
        lat, lon = clicked["lat"], clicked["lng"]
        t = profile["transform"]
        col = int((lon - t.c) / t.a)
        row = int((lat - t.f) / t.e)
        if 0 <= row < index_array.shape[0] and 0 <= col < index_array.shape[1]:
            val = float(index_array[row, col])
            if index_name == "NDVI":
                kondisi = classify_ndvi(val)
            elif index_name == "NDRE":
                kondisi = classify_ndre(val)
            elif index_name == "GNDVI":
                kondisi = classify_gndvi(val)
            elif index_name == "SAVI":
                kondisi = classify_savi(val)
            else:
                kondisi = "-"
            
            st.write(f"📍 Lon: **{lon:.6f}**, Lat: **{lat:.6f}**, {index_name}: **{val:.4f}** → 🌿 **{kondisi}**")
        else:
            st.warning("Klik di luar area citra.")

    # Filter rentang indeks
    st.subheader("Filter Index Range")
    mn, mx = float(index_array.min()), float(index_array.max())
    lo, hi = st.slider(f"Rentang {index_name}", mn, mx, (mn, mx), step=0.01)
    filtered = np.where((index_array >= lo) & (index_array <= hi), index_array, np.nan)
    fig2, ax2 = plt.subplots(figsize=(8,6))
    im2 = ax2.imshow(filtered, cmap='RdYlGn', vmin=-1, vmax=1)
    ax2.axis('off')
    fig2.colorbar(im2, ax=ax2, label=index_name)
    st.pyplot(fig2)

    # Analisis threshold
    st.subheader(f"Analisis Klasifikasi Kesehatan Tanaman ({index_name})")
    try:
        pixel_area = profile["transform"].a * abs(profile["transform"].e)
        if pixel_area == 0:
            pixel_area = 0.25  # fallback
    except:
        pixel_area = 0.25


    if index_name == "NDVI":
        summary = analyze_classification(index_array, classify_ndvi, pixel_area)
    elif index_name == "NDRE":
        summary = analyze_classification(index_array, classify_ndre, pixel_area)
    elif index_name == "GNDVI":
        summary = analyze_classification(index_array, classify_gndvi, pixel_area)
    elif index_name == "SAVI":
        summary = analyze_classification(index_array, classify_savi, pixel_area)
    
    st.dataframe(
        summary.rename(columns={
            "Jumlah Pixel": "Jumlah Piksel",
            "Percentase (%)": "Persentase (%)",
            "Estimasi Area (m²)": "Luas (m²)"
        }),
        hide_index=True
    )

    # Download GeoTIFF
    st.subheader(f"Download {index_name} GeoTIFF")
    profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=None)
    with BytesIO() as mem:
        with rasterio.open(mem, 'w', **profile) as dst:
            dst.write(index_array.astype(rasterio.float32), 1)
        mem.seek(0)
        st.download_button(
            f"Download {index_name}.tif",
            data=mem,
            file_name=f"{index_name.lower()}.tif",
            mime="image/tiff"
        )
        
    #Download Citra RGB Berwarna
    st.subheader(f"Download {index_name} RGB (Berwarna)")
    norm = plt.Normalize(vmin=index_array.min(), vmax=index_array.max())
    cmap = cm.get_cmap('RdYlGn')
    rgba = cmap(norm(index_array))[:, :, :3]
    rgb = (rgba * 255).astype('uint8')
    rgb_img = Image.fromarray(rgb)

    with BytesIO() as rgb_buffer:
        rgb_img.save(rgb_buffer, format="PNG")
        st.download_button(
            f"Download {index_name} RGB.png",
            data=rgb_buffer.getvalue(),
            file_name=f"{index_name.lower()}_rgb.png",
            mime="image/png"
        )


# ==============================
# Streamlit App
# ==============================
st.title("Analisis Indeks Citra Multispektral")
with st.expander("📌 Petunjuk Penggunaan", expanded=True):
    st.markdown("""
    Selamat datang di website **Analisis Indeks Citra Multispektral**.

    ### 🛠️ Langkah-langkah:
    1. Pilih mode input di sidebar:
        - **Manual**: Upload file `.tif` satu per satu (Red, NIR, dll).
        - **Upload Folder ZIP**: Upload file ZIP berisi semua citra dan file `.mrk`.
        - **Google Drive ZIP**: Tempelkan link file ZIP dari Google Drive.
    2. Pastikan file yang dibutuhkan tersedia:
        - Untuk **NDVI, SAVI, LPI, IPVI**: Red + NIR
        - Untuk **NDRE**: RedEdge + NIR
        - Untuk **GNDVI**: Green + NIR
    3. Pilih indeks vegetasi yang ingin dianalisis.
    4. Pilih tampilan visualisasi:
        - **Statik (matplotlib)**
        - **Interaktif (Google Map/OSM)**
    5. Klik pada citra untuk melihat:
        - Koordinat (Lat/Lon)
        - Nilai indeks
        - Kondisi kesehatan tanaman

    ### 📁 Format File yang Dibutuhkan:
    - **.tif** untuk citra multispektral
    - **.mrk** untuk data koordinat GPS drone (jika tersedia)
    - Folder ZIP berisi minimal 20–25 file `.tif` + 1 file `.mrk`

    ---
    🟢 *Silakan pilih mode input dan mulai upload file Anda di sidebar.*    
    """)

mode = st.sidebar.radio("Pilih Mode:", ("Manual", "Upload Folder ZIP", "Google Drive ZIP"))

if mode == "Manual":
    red_file   = st.sidebar.file_uploader("Red Band (R.tif)",   type=['tif'])
    nir_file   = st.sidebar.file_uploader("NIR Band (NIR.tif)", type=['tif'])
    rededge_file = st.sidebar.file_uploader("RedEdge Band (RE.tif) [opsional]", type=['tif'])
    green_file = st.sidebar.file_uploader("Green Band (G.tif) [opsional]", type=['tif'])

    index_choice = st.sidebar.selectbox("Pilih Indeks:", ("NDVI","NDRE","GNDVI","SAVI","LPI","IPVI"))

    if red_file and nir_file:
        # Baca RED
        with rasterio.open(red_file) as src:
            red = src.read(1).astype('float64')
            profile = src.profile.copy()
            t = profile["transform"]
            if t.e > 0:  # Koreksi arah sumbu-y jika perlu
                profile["transform"] = Affine(t.a, t.b, t.c, t.d, -t.e, t.f)

        # Baca NIR
        with rasterio.open(nir_file) as src:
            nir = src.read(1).astype('float64')

        # Hitung indeks yang dipilih
        index_array = None
        if   index_choice=="NDVI":  index_array = calculate_ndvi(nir, red)
        elif index_choice=="SAVI":  index_array = calculate_savi(nir, red)
        elif index_choice=="LPI":   index_array = calculate_lpi(nir, red)
        elif index_choice=="IPVI":  index_array = calculate_ipvi(nir, red)
        elif index_choice=="NDRE" and rededge_file:
            with rasterio.open(rededge_file) as src:
                rededge = src.read(1).astype('float64')
            index_array = calculate_ndre(nir, rededge)
        elif index_choice=="GNDVI" and green_file:
            with rasterio.open(green_file) as src:
                green = src.read(1).astype('float64')
            index_array = calculate_gndvi(nir, green)

        # Tampilkan hasil
        if index_array is not None:
            use_map = st.checkbox("Tampilkan di Google Map", value=False)
            if use_map:
                render_index_on_google_map(index_array, index_choice, profile)
            else:
                render_index_visualization(index_array, index_choice, profile)
        else:
            st.warning("Spektrum pendukung belum diupload.")


elif mode in ("Upload Folder ZIP","Google Drive ZIP"):
    extract_path = None

    if mode == "Upload Folder ZIP":
        zip_file = st.sidebar.file_uploader("Upload ZIP", type=['zip'])
        if zip_file:
            tmp = tempfile.TemporaryDirectory(); extract_path = tmp.name
            with ZipFile(zip_file, 'r') as z: z.extractall(extract_path)

    else:
        gdrive_url = st.text_input("Link ZIP Google Drive")
        if gdrive_url and st.sidebar.button("Download"):
            tmp = tempfile.TemporaryDirectory(); extract_path = tmp.name
            zip_out = os.path.join(extract_path, "in.zip")
            gdown.download(gdrive_url, zip_out, quiet=False)
            safe_extract_zip(zip_out, extract_path)

    if extract_path:
        mrk_file = next((os.path.join(extract_path,f) for f in os.listdir(extract_path) if f.lower().endswith(".mrk")), None)
        if mrk_file:
            mrk_data = load_mrk(mrk_file)
            embed_coordinates(extract_path, mrk_data)

        output_folder = os.path.join(extract_path, "OUTPUT")
        os.makedirs(output_folder, exist_ok=True)
        ortho = {}
        for band, patt in {'R':'_MS_R.TIF','G':'_MS_G.TIF','RE':'_MS_RE.TIF','NIR':'_MS_NIR.TIF'}.items():
            files = glob.glob(os.path.join(extract_path, f"*{patt}"))
            vrt = os.path.join(output_folder, f"{band}.vrt")
            out = os.path.join(output_folder, f"orthomosaic_{band}.tif")
            subprocess.run(["gdalbuildvrt", vrt] + files, stdout=subprocess.DEVNULL)
            subprocess.run(["gdal_translate", vrt, out], stdout=subprocess.DEVNULL)
            ortho[band] = out

        stack_tif = build_mosaic(extract_path, output_folder)
        with rasterio.open(stack_tif) as src:
            red   = src.read(1).astype('float32')
            green = src.read(2).astype('float32')
            rededge=src.read(3).astype('float32')
            nir   = src.read(4).astype('float32')
            profile = src.meta.copy()

        index_maps = {
            "NDVI":  calculate_ndvi(nir, red),
            "NDRE":  calculate_ndre(nir, rededge),
            "GNDVI": calculate_gndvi(nir, green),
            "SAVI":  calculate_savi(nir, red),
            "LPI":   calculate_lpi(nir, red),
            "IPVI":  calculate_ipvi(nir, red)
        }
        choice = st.selectbox("Pilih indeks untuk ditampilkan", list(index_maps.keys()))
        # Beri pilihan tampilkan di Streamlit biasa atau di Google Maps
        use_map = st.checkbox("Tampilkan di Google Map", value=False)
        if use_map:
            render_index_on_google_map(index_maps[choice], choice, profile)
        else:
            render_index_visualization(index_maps[choice], choice, profile)

