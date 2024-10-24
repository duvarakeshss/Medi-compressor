import streamlit as st
import numpy as np
import pywt
import cv2
from skimage.metrics import peak_signal_noise_ratio as psnr
import time
from PIL import Image

# Load image from user input
def load_image_from_upload(image_file):
    image = Image.open(image_file).convert('L')
    return np.array(image)

# Wavelet Decomposition and Thresholding
def wavelet_decompose(img, wavelet='haar', level=2, threshold=30):
    coeffs = pywt.wavedec2(img, wavelet, level=level)
    coeffs_thresh = []
    for level_coeffs in coeffs:
        if isinstance(level_coeffs, tuple):
            coeffs_thresh.append(tuple(pywt.threshold(c, threshold, mode='hard') for c in level_coeffs))
        else:
            coeffs_thresh.append(pywt.threshold(level_coeffs, threshold, mode='hard'))
    return coeffs_thresh, coeffs

# RLE Compression (simple implementation)
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

# Flatten the wavelet coefficients
def flatten_coeffs(coeffs):
    flattened = []
    shapes = []
    for level in coeffs:
        if isinstance(level, tuple):
            flattened.extend([c.flatten() for c in level])
            shapes.append([c.shape for c in level])
        else:
            flattened.append(level.flatten())
            shapes.append(level.shape)
    return np.concatenate(flattened), shapes

# Rebuild the coefficients from flattened data
def rebuild_coeffs(flattened, shapes):
    index = 0
    coeffs = []
    for shape in shapes:
        if isinstance(shape, list):
            level_coeffs = []
            for subshape in shape:
                size = np.prod(subshape)
                level_coeffs.append(flattened[index:index + size].reshape(subshape))
                index += size
            coeffs.append(tuple(level_coeffs))
        else:
            size = np.prod(shape)
            coeffs.append(flattened[index:index + size].reshape(shape))
            index += size
    return coeffs

# Inverse Wavelet Transform
def inverse_wavelet_transform(coeffs, wavelet='haar'):
    return pywt.waverec2(coeffs, wavelet)

# Performance Evaluation (PSNR)
# Performance Evaluation (PSNR)
def calculate_psnr(original, compressed):
    # Ensure both images are of type uint8 and in range [0, 255]
    original = original.astype(np.uint8)
    compressed = np.clip(compressed, 0, 255).astype(np.uint8)

    if original.shape != compressed.shape:
        compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))  # Resizing to original shape if needed
    
    return psnr(original, compressed)

# Compression ratio calculation
def calculate_compression_ratio(original_size, compressed_size):
    return compressed_size / original_size

# Compress Image
def compress_image(image, wavelet='haar', level=2, threshold=30):
    original_size = image.size
    start_time = time.time()
    coeffs_thresh, coeffs = wavelet_decompose(image, wavelet=wavelet, level=level, threshold=threshold)
    flattened_coeffs, shapes = flatten_coeffs(coeffs_thresh)
    rle_encoded = rle_encode(flattened_coeffs)
    compressed_size = len(rle_encoded)
    compression_ratio = calculate_compression_ratio(original_size, compressed_size)
    compression_time = time.time() - start_time
    return rle_encoded, shapes, coeffs, compression_ratio, compression_time

# Decompress Image
def decompress_image(rle_encoded, shapes, wavelet='haar'):
    decoded_coeffs = np.array(rle_decode(rle_encoded))
    rebuilt_coeffs = rebuild_coeffs(decoded_coeffs, shapes)
    reconstructed_image = inverse_wavelet_transform(rebuilt_coeffs, wavelet)
    return reconstructed_image

# Compress and Decompress with performance metrics
def compress_and_decompress(image, wavelet='haar', level=2, threshold=30):
    rle_encoded, shapes, coeffs, compression_ratio, compression_time = compress_image(image, wavelet, level, threshold)
    start_time = time.time()
    decompressed_image = decompress_image(rle_encoded, shapes, wavelet)
    decompression_time = time.time() - start_time
    
    # Calculate PSNR between original and decompressed images
    psnr_value = calculate_psnr(image, decompressed_image)
    
    # Normalize the decompressed image to be in the range [0, 255]
    decompressed_image = np.clip(decompressed_image, 0, 255)  # Clip values to be within [0, 255]
    decompressed_image = decompressed_image.astype(np.uint8)   # Convert to uint8

    return decompressed_image, compression_ratio, compression_time, decompression_time, psnr_value

# Streamlit App
st.title("Wavelet-based Image Compression and Decompression")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Load Image
    image = load_image_from_upload(uploaded_file)
    st.image(image, caption="Original Image", use_column_width=True)

    wavelets = ['haar', 'db2', 'bior1.3']

    # Compress and Decompress with each wavelet
    cols = st.columns(3)  # Create three columns for displaying images
    for i, wavelet in enumerate(wavelets):
        decompressed_image, compression_ratio, compression_time, decompression_time, psnr_value = compress_and_decompress(image, wavelet)
        
        # Display each image in its column
        with cols[i]:
            st.image(decompressed_image, caption=f"Reconstructed Image - {wavelet}", use_column_width=True)
            st.write(f"Compression Ratio: {compression_ratio:.4f}")
            st.write(f"Compression Time: {compression_time:.4f} seconds")
            st.write(f"Decompression Time: {decompression_time:.4f} seconds")
            st.write(f"PSNR: {psnr_value:.4f} dB")
