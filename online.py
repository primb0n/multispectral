import streamlit as st
import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
import os
import tempfile
from io import BytesIO
from zipfile import ZipFile
import glob
import subprocess
import re
from rasterio.transform import from_origin
import gdown

# Fungsi NDVI dan lain-lain
def calculate_ndvi(nir_band, red_band):
    return np.clip((nir_band - red_band) / (nir_band + red_band + 1e-10), -1, 1)

def calculate_ndre(nir_band, rededge_band):
    return np.clip((nir_band - rededge_band) / (nir_band + rededge_band + 1e-10), -1, 1)

def calculate_gndvi(nir_band, green_band):
    return np.clip((nir_band - green_band) / (nir_band + green_band + 1e-10), -1, 1)

def calculate_savi(nir_band, red_band, L=0.5):
    return np.clip(((nir_band - red_band) / (nir_band + red_band + L)) * (1 + L), -1, 1)

def calculate_lpi(nir_band, red_band):
    return np.clip(nir_band / (nir_band + red_band + 1e-10), 0, 1)

def calculate_ipvi(nir_band, red_band):
    return np.clip(nir_band / (nir_band + red_band + 1e-10), 0, 1)

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

# Streamlit interface
st.title("Analisis dan Mosaic Citra Drone Multispektral")
mode = st.sidebar.radio("Pilih Mode:", ("Manual Per Spektrum", "Upload Folder ZIP", "Ambil dari Google Drive"))

if mode == "Manual Per Spektrum":
    red_file = st.sidebar.file_uploader("Red Band (R.tif)", type=['tif'])
    nir_file = st.sidebar.file_uploader("NIR Band (NIR.tif)", type=['tif'])
    rededge_file = st.sidebar.file_uploader("RedEdge Band (RE.tif) [opsional]", type=['tif'])
    green_file = st.sidebar.file_uploader("Green Band (G.tif) [opsional]", type=['tif'])

    if red_file and nir_file:
        with rasterio.open(red_file) as src:
            red = src.read(1).astype('float64')
            profile = src.profile
        with rasterio.open(nir_file) as src:
            nir = src.read(1).astype('float64')

        ndvi = calculate_ndvi(nir, red)
        savi = calculate_savi(nir, red)
        lpi = calculate_lpi(nir, red)
        ipvi = calculate_ipvi(nir, red)

        st.subheader("NDVI Manual Upload")
        fig, ax = plt.subplots()
        im = ax.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax)
        st.pyplot(fig)

else:
    if mode == "Upload Folder ZIP":
        zip_file = st.sidebar.file_uploader("Upload Folder Drone (ZIP)", type=['zip'])
        if zip_file:
            temp_dir = tempfile.TemporaryDirectory()
            extract_path = temp_dir.name
            with ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
    else:
        gdrive_url = st.text_input("Masukkan link file ZIP Google Drive (shareable link)")
        if gdrive_url and st.button("Unduh dan Proses"):
            temp_dir = tempfile.TemporaryDirectory()
            extract_path = temp_dir.name
            zip_output = os.path.join(extract_path, "input.zip")
            gdown.download(gdrive_url, zip_output, quiet=False)
            with ZipFile(zip_output, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

    if 'extract_path' in locals():
        mrk_file = None
        for f in os.listdir(extract_path):
            if f.lower().endswith(".mrk"):
                mrk_file = os.path.join(extract_path, f)
                break

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

            ndvi = calculate_ndvi(nir, red)
            ndre = calculate_ndre(nir, rededge)
            gndvi = calculate_gndvi(nir, green)
            savi = calculate_savi(nir, red)
            lpi = calculate_lpi(nir, red)
            ipvi = calculate_ipvi(nir, red)

            meta.update(dtype=rasterio.float32, count=1)

            index_maps = {
                "NDVI": ndvi,
                "NDRE": ndre,
                "GNDVI": gndvi,
                "SAVI": savi,
                "LPI": lpi,
                "IPVI": ipvi
            }

            for name, arr in index_maps.items():
                out_path = os.path.join(output_folder, f"{name}.tif")
                with rasterio.open(out_path, 'w', **meta) as dst:
                    dst.write(arr, 1)

            st.subheader("Tampilan Mosaic Indeks Vegetasi")
            index_option = st.selectbox("Pilih indeks untuk ditampilkan", list(index_maps.keys()))
            path = os.path.join(output_folder, f"{index_option}.tif")
            with rasterio.open(path) as src:
                arr = src.read(1)
            fig, ax = plt.subplots()
            im = ax.imshow(arr, cmap='RdYlGn', vmin=-1, vmax=1)
            plt.colorbar(im, ax=ax)
            st.pyplot(fig)

            with open(path, "rb") as f:
                st.download_button("Download Mosaic GeoTIFF", f, file_name=f"{index_option}.tif")
