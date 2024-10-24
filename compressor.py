import numpy as np
import cv2
import time
import streamlit as st
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image

# Streamlit setup
st.title("Image Compression with Wavelet Transform")

# Haar wavelet filter coefficients
haar_low = [1 / np.sqrt(2), 1 / np.sqrt(2)]
haar_high = [1 / np.sqrt(2), -1 / np.sqrt(2)]

db2_low = [0.48296, 0.83652, 0.22414, -0.12941]
db2_high = [-0.12941, -0.22414, 0.83652, -0.48296]

bior13_decomp_low = [0.3536, 0.7071, 0.3536]
bior13_decomp_high = [0.1768, -0.3536, 0.1768]
bior13_recon_low = [0.1768, 0.3536, 0.1768]
bior13_recon_high = [-0.3536, 0.7071, -0.3536]

def convolve(signal, filter):
    filter_len = len(filter)
    signal_len = len(signal)
    result = np.zeros(signal_len)

    for i in range(signal_len):
        for k in range(filter_len):
            if i + k < signal_len:
                result[i] += filter[k] * signal[i + k]
    return result
# Wavelet transform functions (Haar, Daubechies, Biorthogonal)
def haar_wavelet_transform(img):
    rows, cols = img.shape
    transformed = np.zeros((rows, cols))

    # Row-wise transformation
    for i in range(rows):
        approx = convolve(img[i, :], haar_low)[::2]
        detail = convolve(img[i, :], haar_high)[::2]
        transformed[i, :cols // 2] = approx
        transformed[i, cols // 2:] = detail

    # Column-wise transformation
    for j in range(cols):
        approx = convolve(transformed[:, j], haar_low)[::2]
        detail = convolve(transformed[:, j], haar_high)[::2]
        transformed[:rows // 2, j] = approx
        transformed[rows // 2:, j] = detail

    return transformed

def daubechies_wavelet_transform(img):
    rows, cols = img.shape
    transformed = np.zeros((rows, cols))

    # Row-wise transformation
    for i in range(rows):
        approx = convolve(img[i, :], db2_low)[::2]
        detail = convolve(img[i, :], db2_high)[::2]
        transformed[i, :cols // 2] = approx
        transformed[i, cols // 2:] = detail

    # Column-wise transformation
    for j in range(cols):
        approx = convolve(transformed[:, j], db2_low)[::2]
        detail = convolve(transformed[:, j], db2_high)[::2]
        transformed[:rows // 2, j] = approx
        transformed[rows // 2:, j] = detail

    return transformed

def biorthogonal_wavelet_transform(img):
    rows, cols = img.shape
    transformed = np.zeros((rows, cols))

    # Row-wise transformation
    for i in range(rows):
        approx = convolve(img[i, :], bior13_decomp_low)[::2]
        detail = convolve(img[i, :], bior13_decomp_high)[::2]
        transformed[i, :cols // 2] = approx
        transformed[i, cols // 2:] = detail

    # Column-wise transformation
    for j in range(cols):
        approx = convolve(transformed[:, j], bior13_decomp_low)[::2]
        detail = convolve(transformed[:, j], bior13_decomp_high)[::2]
        transformed[:rows // 2, j] = approx
        transformed[rows // 2:, j] = detail

    return transformed

# Inverse wavelet transform functions
def inverse_haar_wavelet_transform(transformed):
    rows, cols = transformed.shape
    img = np.zeros((rows, cols))

    # Inverse column-wise transformation first
    for j in range(cols):
        approx = transformed[:rows // 2, j]
        detail = transformed[rows // 2:, j]

        # Upsample the approximation and detail coefficients
        approx_upsampled = np.zeros(rows)
        detail_upsampled = np.zeros(rows)

        approx_upsampled[::2] = approx
        detail_upsampled[::2] = detail

        # Reconstruct the column
        img[:, j] = convolve(approx_upsampled, haar_low) + convolve(detail_upsampled, haar_high)

    # Inverse row-wise transformation
    for i in range(rows):
        approx = img[i, :cols // 2]
        detail = img[i, cols // 2:]

        # Upsample the approximation and detail coefficients
        approx_upsampled = np.zeros(cols)
        detail_upsampled = np.zeros(cols)

        approx_upsampled[::2] = approx
        detail_upsampled[::2] = detail

        # Reconstruct the row
        img[i, :] = convolve(approx_upsampled, haar_low) + convolve(detail_upsampled, haar_high)

    return img

def inverse_daubechies_wavelet_transform(transformed):
    rows, cols = transformed.shape
    img = np.zeros((rows, cols))

    for j in range(cols):
        approx = convolve(transformed[:rows // 2, j], db2_low)
        detail = convolve(transformed[rows // 2:, j], db2_high)
        img[:, j] = approx + detail

    for i in range(rows):
        approx = convolve(img[i, :cols // 2], db2_low)
        detail = convolve(img[i, cols // 2:], db2_high)
        img[i, :] = approx + detail

    return img

def inverse_biorthogonal_wavelet_transform(transformed):
    rows, cols = transformed.shape
    img = np.zeros((rows, cols))

    for j in range(cols):
        approx = convolve(transformed[:rows // 2, j], bior13_recon_low)
        detail = convolve(transformed[rows // 2:, j], bior13_recon_high)

        approx_upsampled = np.zeros(rows)
        detail_upsampled = np.zeros(rows)
        approx_upsampled[::2] = approx
        detail_upsampled[::2] = detail

        img[:, j] = approx_upsampled + detail_upsampled

    for i in range(rows):
        approx = convolve(img[i, :cols // 2], bior13_recon_low)
        detail = convolve(img[i, cols // 2:], bior13_recon_high)

        approx_upsampled = np.zeros(cols)
        detail_upsampled = np.zeros(cols)
        approx_upsampled[::2] = approx
        detail_upsampled[::2] = detail

        img[i, :] = approx_upsampled + detail_upsampled

    return img

# Thresholding function
def apply_threshold(transformed, threshold):
    return np.where(np.abs(transformed) > threshold, transformed, 0)

# RLE Compression
def rle_encode(data):
    encoding = []
    prev_value = data[0]
    count = 1
    for value in data[1:]:
        if value == prev_value:
            count += 1
        else:
            encoding.append((prev_value, count))
            prev_value = value
            count = 1
    encoding.append((prev_value, count))
    return encoding

# RLE Decoding
def rle_decode(encoded_data):
    decoded_data = []
    for value, count in encoded_data:
        decoded_data.extend([value] * count)
    return decoded_data

# Performance Evaluation
def calculate_psnr(original, compressed):
    if original.shape != compressed.shape:
        compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))
    return psnr(original, compressed)

# Compression ratio calculation
def calculate_compression_ratio(original_size, compressed_size):
    return original_size / compressed_size

# Image Compression and Decompression Function
def compress_and_decompress_streamlit(image_path, wavelet_type='haar', threshold=30):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Initial size of the image
    original_size = image.size

    # Start timing compression
    start_time = time.time()

    # Wavelet transform based on the selected type
    if wavelet_type == 'haar':
        transformed = haar_wavelet_transform(image)
    elif wavelet_type == 'db2':
        transformed = daubechies_wavelet_transform(image)
    elif wavelet_type == 'bior1.3':
        transformed = biorthogonal_wavelet_transform(image)

    # Apply threshold
    thresholded = apply_threshold(transformed, threshold)

    # Flatten data for RLE encoding
    flattened = thresholded.flatten()

    # RLE encoding
    rle_encoded = rle_encode(flattened)

    # Compressed size
    compressed_size = len(rle_encoded)

    # Calculate compression ratio
    compression_ratio = calculate_compression_ratio(original_size, compressed_size)

    # Inverse wavelet transform
    if wavelet_type == 'haar':
        decompressed = inverse_haar_wavelet_transform(thresholded)
    elif wavelet_type == 'db2':
        decompressed = inverse_daubechies_wavelet_transform(thresholded)
    elif wavelet_type == 'bior1.3':
        decompressed = inverse_biorthogonal_wavelet_transform(thresholded)

    # Stop timing
    end_time = time.time()

    # Calculate PSNR
    psnr_value = calculate_psnr(image, decompressed)

    # Return results including the images for visualization
    return {
        "original_image": image,
        "decompressed_image": decompressed,
        "compression_ratio": compression_ratio,
        "psnr": psnr_value,
        "time": end_time - start_time
    }

# Streamlit UI
st.title("Wavelet Image Compression and Decompression")

# File upload
uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

# Select wavelet type
wavelet_type = st.selectbox("Select wavelet type", ['haar', 'db2', 'bior1.3'])

# Set threshold slider
threshold = st.slider("Threshold", min_value=0, max_value=100, value=30)

if uploaded_file:
    # Save uploaded image temporarily
    with open("temp_image.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Perform compression and decompression
    results = compress_and_decompress_streamlit("temp_image.jpg", wavelet_type=wavelet_type, threshold=threshold)

    # Show original and decompressed images side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(results['original_image'], caption="Original Image", use_column_width=True)
    
    with col2:
        # Normalize the decompressed image to [0, 1] for display
        decompressed_normalized = results['decompressed_image'] / np.max(results['decompressed_image'])
        st.image(decompressed_normalized, caption="Decompressed Image", use_column_width=True)

    # Display compression details and PSNR
    st.write("Compression Results:")
    st.write(f"Original Size: {results['original_image'].size}")
    st.write(f"Compressed Size: {len(rle_encode(results['decompressed_image'].flatten()))}")
    st.write(f"Compression Ratio: {results['compression_ratio']:.2f}")
    st.write(f"PSNR: {results['psnr']:.2f} dB")
    st.write(f"Compression Time: {results['time']:.2f} seconds")