import streamlit as st
import numpy as np
import rasterio
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

def calculate_ndvi(nir_band, red_band):
    ndvi = (nir_band - red_band) / (nir_band + red_band + 1e-10)
    ndvi = np.clip(ndvi, -1, 1)
    return ndvi

def calculate_ndre(nir_band, rededge_band):
    ndre = (nir_band - rededge_band) / (nir_band + rededge_band + 1e-10)
    ndre = np.clip(ndre, -1, 1)
    return ndre

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
st.title("Analisis Index Vegetasi")

st.sidebar.header("Upload Files")
red_file = st.sidebar.file_uploader("Upload Red Band (R.tif)", type=['tif'])
nir_file = st.sidebar.file_uploader("Upload NIR Band (NIR.tif)", type=['tif'])
rededge_file = st.sidebar.file_uploader("Upload RedEdge Band (RE.tif) [Optional for NDRE]", type=['tif'])

index_choice = st.sidebar.selectbox("Select Index to Analyze", ("NDVI", "NDRE"))

if red_file and nir_file and (index_choice == "NDVI" or (index_choice == "NDRE" and rededge_file)):
    with rasterio.open(red_file) as red_src:
        red = red_src.read(1).astype('float64')
        profile = red_src.profile

    with rasterio.open(nir_file) as nir_src:
        nir = nir_src.read(1).astype('float64')

    if index_choice == "NDVI":
        index_array = calculate_ndvi(nir, red)
    else:  # NDRE
        with rasterio.open(rededge_file) as re_src:
            rededge = re_src.read(1).astype('float64')
        index_array = calculate_ndre(nir, rededge)

    st.subheader(f"{index_choice} Map")
    fig, ax = plt.subplots(figsize=(8,6))
    cax = ax.imshow(index_array, cmap='RdYlGn', vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax, label=index_choice)
    st.pyplot(fig)

    # Slider untuk filter
    st.subheader("Filter Index Range")
    min_val, max_val = float(np.min(index_array)), float(np.max(index_array))
    range_values = st.slider('Select Range', min_value=min_val, max_value=max_val, value=(min_val, max_val), step=0.01)

    filtered_array = np.where((index_array >= range_values[0]) & (index_array <= range_values[1]), index_array, np.nan)

    st.subheader("Filtered Map")
    fig2, ax2 = plt.subplots(figsize=(8,6))
    cax2 = ax2.imshow(filtered_array, cmap='RdYlGn', vmin=-1, vmax=1)
    fig2.colorbar(cax2, ax=ax2, label=index_choice)
    st.pyplot(fig2)

    # Statistik
    st.subheader(f"{index_choice} Threshold Analysis")
    thresholds = [0.1, 0.3, 0.5]
    all_stats = {}

    for threshold in thresholds:
        stats = analyze_index_threshold(index_array, threshold)
        all_stats[threshold] = stats

    df_stats = pd.DataFrame(all_stats).round(3)
    st.dataframe(df_stats)

    # Download GeoTIFF
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
    st.info("Please upload the necessary band files to start.")
