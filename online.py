import streamlit as st
import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_image_coordinates import streamlit_image_coordinates
from io import BytesIO
from PIL import Image

# Fungsi untuk menghitung berbagai indeks vegetasi
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

# Fungsi untuk analisis indeks berdasarkan threshold
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

# Tampilan web
st.title("Analisis Index Vegetasi Interaktif - Filtered Map Only")

st.sidebar.header("Upload Files")
red_file = st.sidebar.file_uploader("Upload Red Band (R.tif)", type=['tif'])
nir_file = st.sidebar.file_uploader("Upload NIR Band (NIR.tif)", type=['tif'])
rededge_file = st.sidebar.file_uploader("Upload RedEdge Band (RE.tif) [Optional for NDRE]", type=['tif'])
green_file = st.sidebar.file_uploader("Upload Green Band (G.tif) [Optional for GNDVI]", type=['tif'])

index_choice = st.sidebar.selectbox("Select Index to Analyze", ("NDVI", "NDRE", "GNDVI", "SAVI", "LPI", "IPVI"))

required_files = red_file and nir_file
optional_files_ok = True
if index_choice == "NDRE" and not rededge_file:
    optional_files_ok = False
if index_choice == "GNDVI" and not green_file:
    optional_files_ok = False

if required_files and optional_files_ok:
    with rasterio.open(red_file) as red_src:
        red = red_src.read(1).astype('float64')
        profile = red_src.profile

    with rasterio.open(nir_file) as nir_src:
        nir = nir_src.read(1).astype('float64')

    if index_choice == "NDVI":
        index_array = calculate_ndvi(nir, red)
    elif index_choice == "NDRE":
        with rasterio.open(rededge_file) as re_src:
            rededge = re_src.read(1).astype('float64')
        index_array = calculate_ndre(nir, rededge)
    elif index_choice == "GNDVI":
        with rasterio.open(green_file) as green_src:
            green = green_src.read(1).astype('float64')
        index_array = calculate_gndvi(nir, green)
    elif index_choice == "SAVI":
        index_array = calculate_savi(nir, red)
    elif index_choice == "LPI":
        index_array = calculate_lpi(nir, red)
    elif index_choice == "IPVI":
        index_array = calculate_ipvi(nir, red)

    # Slider untuk filter
    st.subheader("Filter Index Range")
    min_val, max_val = float(np.min(index_array)), float(np.max(index_array))
    range_values = st.slider('Select Range', min_value=min_val, max_value=max_val, value=(min_val, max_val), step=0.01)

    filtered_array = np.where((index_array >= range_values[0]) & (index_array <= range_values[1]), index_array, np.nan)

    # Normalize filtered map
    ndvi_norm_filtered = (filtered_array - np.nanmin(filtered_array)) / (np.nanmax(filtered_array) - np.nanmin(filtered_array))
    ndvi_norm_filtered = np.clip(ndvi_norm_filtered, 0, 1)
    
    # Convert filtered map to RGB image
    colormap = plt.get_cmap('RdYlGn')
    ndvi_rgb_filtered = (colormap(ndvi_norm_filtered)[:, :, :3] * 255).astype(np.uint8)
    ndvi_pil_filtered = Image.fromarray(ndvi_rgb_filtered)

    st.subheader("Filtered Map (Click to Inspect)")
    coords = streamlit_image_coordinates(ndvi_pil_filtered, key="click_filtered")
    st.image(ndvi_pil_filtered, caption=f"Filtered {index_choice} Map", use_container_width=False, width=ndvi_pil_filtered.width)

    # Kalau klik di filtered map
    if coords is not None:
        x_pix = int(coords["x"])
        y_pix = int(coords["y"])

        if 0 <= x_pix < filtered_array.shape[1] and 0 <= y_pix < filtered_array.shape[0]:
            index_value = filtered_array[y_pix, x_pix]
            st.success(f"Clicked Pixel (x={x_pix}, y={y_pix})")
            st.info(f"{index_choice} Value: {index_value:.3f}")

    # Statistik
    st.subheader(f"{index_choice} Threshold Analysis")
    thresholds = [0.1, 0.3, 0.5]
    all_stats = {}

    for threshold in thresholds:
        stats = analyze_index_threshold(index_array, threshold)
        all_stats[threshold] = stats

    df_stats = pd.DataFrame(all_stats).round(3)
    st.dataframe(df_stats)

    # Download GeoTIFF dari hasil index (bukan filtered)
    st.subheader(f"Download {index_choice} GeoTIFF")
    profile.update(dtype=rasterio.float32, count=1, compress='lzw', nodata=None)
    with BytesIO() as memfile:
        with rasterio.open(memfile, 'w', **profile) as dst:
            dst.write(index_array.astype(rasterio.float32), 1)
        memfile.seek(0)
        st.download_button(
            label=f"Download {index_choice} TIFF",
            data=memfile,
            file_name=f'{index_choice.lower()}_result.tif',
            mime='image/tiff'
        )
else:
    st.info("Upload file yang dibutuhkan!")
