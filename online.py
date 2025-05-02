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

def render_index_visualization(index_array, index_name, profile):
    st.subheader(f"{index_name} Map")
    # 1) Tampilkan peta statis dengan matplotlib
    fig, ax = plt.subplots(figsize=(8,6))
    im = ax.imshow(index_array, cmap='RdYlGn', vmin=-1, vmax=1)
    ax.axis('off')
    fig.colorbar(im, ax=ax, label=index_name)
    st.pyplot(fig)

    # 2) Buat gambar RGB dari array untuk interaksi
    norm = plt.Normalize(vmin=index_array.min(), vmax=index_array.max())
    cmap = cm.get_cmap('RdYlGn')
    rgba = cmap(norm(index_array))         # Bentuk (baris, kolom, 4)
    rgb = (rgba[:, :, :3] * 255).astype('uint8')
    img = Image.fromarray(rgb)

    # 3) Interaksi: klik/hover untuk dapat (x,y)
    st.subheader("Klik atau arahkan kursor untuk koordinat & nilai")
    orig_w, orig_h = img.size
    disp_w = 600
    disp_h = int(orig_h * disp_w / orig_w)
    scale_x = orig_w / disp_w
    scale_y = orig_h / disp_h

    coords = streamlit_image_coordinates(
        img,
        key=f"coord_{index_name}",
        width=disp_w
    )
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
            st.write(f"📍 Lon: **{lon:.6f}**, Lat: **{lat:.6f}**, {index_name}: **{val:.4f}**")
        else:
            st.warning("Klik di luar area citra.")

    # 4) Filter range seperti sebelumnya
    st.subheader("Filter Index Range")
    mn, mx = float(index_array.min()), float(index_array.max())
    lo, hi = st.slider(f"Rentang {index_name}", mn, mx, (mn, mx), step=0.01)
    filtered = np.where((index_array >= lo) & (index_array <= hi), index_array, np.nan)
    fig2, ax2 = plt.subplots(figsize=(8,6))
    im2 = ax2.imshow(filtered, cmap='RdYlGn', vmin=-1, vmax=1)
    ax2.axis('off')
    fig2.colorbar(im2, ax=ax2, label=index_name)
    st.pyplot(fig2)

    # 5) Statistik threshold
    st.subheader(f"{index_name} Threshold Analysis")
    stats = {thr: analyze_index_threshold(index_array, thr) for thr in (0.1, 0.3, 0.5)}
    st.dataframe(pd.DataFrame(stats).round(3))

    # 6) Download GeoTIFF
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
    # pilih downsample via slider (opsional)
    ds = 4

    with st.spinner("🔄 Membangun peta Google Satellite…"):
        m = make_folium_map(index_array, profile["transform"])
        data = st_folium(m, width=700, height=500)

    # ambil klik terakhir
    clicked = data.get("last_clicked")
    if clicked:
        lat, lon = clicked["lat"], clicked["lng"]
        # hitung baris/kolom di array full-res
        t = profile["transform"]
        col = int((lon - t.c) / t.a)
        row = int((lat - t.f) / t.e)
        if 0 <= row < index_array.shape[0] and 0 <= col < index_array.shape[1]:
            val = float(index_array[row, col])
            st.write(f"📍 Lat: **{lat:.6f}**, Lon: **{lon:.6f}** → {index_name}: **{val:.4f}**")
        else:
            st.warning("Klik di luar area citra.")

    # 4) Filter range seperti sebelumnya
    st.subheader("Filter Index Range")
    mn, mx = float(index_array.min()), float(index_array.max())
    lo, hi = st.slider(f"Rentang {index_name}", mn, mx, (mn, mx), step=0.01)
    filtered = np.where((index_array >= lo) & (index_array <= hi), index_array, np.nan)
    fig2, ax2 = plt.subplots(figsize=(8,6))
    im2 = ax2.imshow(filtered, cmap='RdYlGn', vmin=-1, vmax=1)
    ax2.axis('off')
    fig2.colorbar(im2, ax=ax2, label=index_name)
    st.pyplot(fig2)

    # 5) Statistik threshold
    st.subheader(f"{index_name} Threshold Analysis")
    stats = {thr: analyze_index_threshold(index_array, thr) for thr in (0.1, 0.3, 0.5)}
    st.dataframe(pd.DataFrame(stats).round(3))

    # 6) Download GeoTIFF
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


# ==============================
# Streamlit App
# ==============================
st.title("Analisis dan Mosaic Citra Drone Multispektral")
mode = st.sidebar.radio("Pilih Mode:", ("Manual", "Upload Folder ZIP", "Google Drive ZIP"))

if mode == "Manual":
    red_file   = st.sidebar.file_uploader("Red Band (R.tif)",   type=['tif'])
    nir_file   = st.sidebar.file_uploader("NIR Band (NIR.tif)",  type=['tif'])
    rededge_file = st.sidebar.file_uploader("RedEdge Band (RE.tif) [opsional]", type=['tif'])
    green_file = st.sidebar.file_uploader("Green Band (G.tif) [opsional]",     type=['tif'])

    index_choice = st.sidebar.selectbox("Pilih Indeks:", ("NDVI","NDRE","GNDVI","SAVI","LPI","IPVI"))

    if red_file and nir_file:
        with rasterio.open(red_file) as src: red = src.read(1).astype('float64'); profile = src.profile
        with rasterio.open(nir_file) as src: nir = src.read(1).astype('float64')

        index_array = None
        if   index_choice=="NDVI":  index_array = calculate_ndvi(nir, red)
        elif index_choice=="SAVI":  index_array = calculate_savi(nir, red)
        elif index_choice=="LPI":   index_array = calculate_lpi(nir, red)
        elif index_choice=="IPVI":  index_array = calculate_ipvi(nir, red)
        elif index_choice=="NDRE" and rededge_file:
            with rasterio.open(rededge_file) as src: rededge = src.read(1).astype('float64')
            index_array = calculate_ndre(nir, rededge)
        elif index_choice=="GNDVI" and green_file:
            with rasterio.open(green_file) as src: green = src.read(1).astype('float64')
            index_array = calculate_gndvi(nir, green)

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

