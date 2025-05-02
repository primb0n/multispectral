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
                elements = re.split('[,\s]+', line.strip())
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
    fig, ax = plt.subplots(figsize=(8,6))
    cax = ax.imshow(index_array, cmap='RdYlGn', vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, label=index_name)
    st.pyplot(fig)

    st.subheader("Filter Index Range")
    min_val, max_val = float(np.min(index_array)), float(np.max(index_array))
    range_values = st.slider(f'Select Range for {index_name}', min_value=min_val, max_value=max_val, value=(min_val, max_val), step=0.01)
    filtered_array = np.where((index_array >= range_values[0]) & (index_array <= range_values[1]), index_array, np.nan)
    fig2, ax2 = plt.subplots(figsize=(8,6))
    cax2 = ax2.imshow(filtered_array, cmap='RdYlGn', vmin=-1, vmax=1)
    fig2.colorbar(cax2, ax=ax2, label=index_name)
    st.pyplot(fig2)

    st.subheader(f"{index_name} Threshold Analysis")
    thresholds = [0.1, 0.3, 0.5]
    all_stats = {thr: analyze_index_threshold(index_array, thr) for thr in thresholds}
    df_stats = pd.DataFrame(all_stats).round(3)
    st.dataframe(df_stats)

    st.subheader(f"Download {index_name} GeoTIFF")
    profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=None)
    with BytesIO() as memfile:
        with rasterio.open(memfile, 'w', **profile) as dst:
            dst.write(index_array.astype(rasterio.float32), 1)
        memfile.seek(0)
        st.download_button(
            label=f"Download {index_name} TIFF",
            data=memfile,
            file_name=f'{index_name.lower()}_result.tif',
            mime='image/tiff'
        )

# ==============================
# Streamlit App
# ==============================
st.title("Analisis dan Mosaic Citra Drone Multispektral")
mode = st.sidebar.radio("Pilih Mode:", ("Manual", "Upload Folder ZIP", "Google Drive ZIP"))

if mode == "Manual":
    red_file = st.sidebar.file_uploader("Red Band (R.tif)", type=['tif'])
    nir_file = st.sidebar.file_uploader("NIR Band (NIR.tif)", type=['tif'])
    rededge_file = st.sidebar.file_uploader("RedEdge Band (RE.tif)", type=['tif'])
    green_file = st.sidebar.file_uploader("Green Band (G.tif)", type=['tif'])

    index_choice = st.sidebar.selectbox("Pilih Indeks untuk Ditampilkan", ("NDVI", "NDRE", "GNDVI", "SAVI", "LPI", "IPVI"))

    if red_file and nir_file:
        with rasterio.open(red_file) as src: red = src.read(1).astype('float64'); profile = src.profile
        with rasterio.open(nir_file) as src: nir = src.read(1).astype('float64')

        index_array = None
        if index_choice == "NDVI":
            index_array = calculate_ndvi(nir, red)
        elif index_choice == "SAVI":
            index_array = calculate_savi(nir, red)
        elif index_choice == "LPI":
            index_array = calculate_lpi(nir, red)
        elif index_choice == "IPVI":
            index_array = calculate_ipvi(nir, red)
        elif index_choice == "NDRE" and rededge_file:
            with rasterio.open(rededge_file) as src: rededge = src.read(1).astype('float64')
            index_array = calculate_ndre(nir, rededge)
        elif index_choice == "GNDVI" and green_file:
            with rasterio.open(green_file) as src: green = src.read(1).astype('float64')
            index_array = calculate_gndvi(nir, green)

        if index_array is not None:
            render_index_visualization(index_array, index_choice, profile)
        else:
            st.warning("Spektrum pendukung belum diupload.")

elif mode in ("Upload Folder ZIP", "Google Drive ZIP"):
    extract_path = None
    if mode == "Upload Folder ZIP":
        zip_file = st.sidebar.file_uploader("Upload ZIP", type=['zip'])
        if zip_file:
            temp_dir = tempfile.TemporaryDirectory()
            extract_path = temp_dir.name
            with ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
    else:
        gdrive_url = st.text_input("Link ZIP dari Google Drive")
        if gdrive_url and st.button("Download"):
            temp_dir = tempfile.TemporaryDirectory()
            extract_path = temp_dir.name
            zip_output = os.path.join(extract_path, "input.zip")
            gdown.download(gdrive_url, zip_output, quiet=False)
            safe_extract_zip(zip_output, extract_path)

    if extract_path:
        mrk_file = next((os.path.join(extract_path, f) for f in os.listdir(extract_path) if f.lower().endswith(".mrk")), None)
        if mrk_file:
            mrk_data = load_mrk(mrk_file)
            embed_coordinates(extract_path, mrk_data)

        bands = {
            'R': '_MS_R.TIF',
            'G': '_MS_G.TIF',
            'RE': '_MS_RE.TIF',
            'NIR': '_MS_NIR.TIF'
        }

        output_folder = os.path.join(extract_path, "OUTPUT")
        os.makedirs(output_folder, exist_ok=True)
        ortho_files = {}
        for band, pattern in bands.items():
            files = glob.glob(os.path.join(extract_path, f"*{pattern}"))
            vrt_path = os.path.join(output_folder, f"{band}.vrt")
            ortho_path = os.path.join(output_folder, f"orthomosaic_{band}.tif")
            subprocess.run(["gdalbuildvrt", vrt_path] + files)
            subprocess.run(["gdal_translate", vrt_path, ortho_path])
            ortho_files[band] = ortho_path

        stack_vrt = os.path.join(output_folder, "multispectral_stack.vrt")
        stack_tif = os.path.join(output_folder, "multispectral_stack.tif")
        subprocess.run(["gdalbuildvrt", "-separate", stack_vrt,
                        ortho_files['R'], ortho_files['G'], ortho_files['RE'], ortho_files['NIR']])
        subprocess.run(["gdal_translate", stack_vrt, stack_tif])

        with rasterio.open(stack_tif) as src:
            red = src.read(1).astype('float32')
            green = src.read(2).astype('float32')
            rededge = src.read(3).astype('float32')
            nir = src.read(4).astype('float32')
            meta = src.meta.copy()

        index_maps = {
            "NDVI": calculate_ndvi(nir, red),
            "NDRE": calculate_ndre(nir, rededge),
            "GNDVI": calculate_gndvi(nir, green),
            "SAVI": calculate_savi(nir, red),
            "LPI": calculate_lpi(nir, red),
            "IPVI": calculate_ipvi(nir, red)
        }

        index_choice = st.selectbox("Pilih indeks yang ingin ditampilkan", list(index_maps.keys()))
        render_index_visualization(index_maps[index_choice], index_choice, meta)
