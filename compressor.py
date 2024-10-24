import pywt
import numpy as np
import cv2
import time
from skimage.metrics import peak_signal_noise_ratio as psnr

# Load image (grayscale)
def load_image(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img

# Save the decompressed image
def save_image(path, img):
    cv2.imwrite(path, img)

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
def calculate_psnr(original, compressed):
    if original.shape != compressed.shape:
        compressed = cv2.resize(compressed, (original.shape[1], original.shape[0]))  # Resizing to original shape if needed
    return psnr(original, compressed)

# Compression ratio calculation
def calculate_compression_ratio(original_size, compressed_size):
    return compressed_size / original_size

# Main Process Flow
def compress_image(image_path, wavelet='haar', level=2, threshold=30):
    image = load_image(image_path)
    original_size = image.size
    start_time = time.time()
    coeffs_thresh, coeffs = wavelet_decompose(image, wavelet=wavelet, level=level, threshold=threshold)
    flattened_coeffs, shapes = flatten_coeffs(coeffs_thresh)
    rle_encoded = rle_encode(flattened_coeffs)
    compressed_size = len(rle_encoded)
    compression_ratio = calculate_compression_ratio(original_size, compressed_size)
    compression_time = time.time() - start_time
    return rle_encoded, shapes, coeffs, compression_ratio, compression_time

def decompress_image(rle_encoded, shapes, wavelet='haar'):
    decoded_coeffs = np.array(rle_decode(rle_encoded))
    rebuilt_coeffs = rebuild_coeffs(decoded_coeffs, shapes)
    reconstructed_image = inverse_wavelet_transform(rebuilt_coeffs, wavelet=wavelet)
    return reconstructed_image

# Full compression and decompression test with performance metrics
def compress_and_decompress(image_path, wavelet='haar', level=2, threshold=30):
    rle_encoded, shapes, coeffs, compression_ratio, compression_time = compress_image(image_path, wavelet, level, threshold)
    start_time = time.time()
    decompressed_image = decompress_image(rle_encoded, shapes, wavelet)
    decompression_time = time.time() - start_time
    original_image = load_image(image_path)
    psnr_value = calculate_psnr(original_image, decompressed_image)
    return compression_ratio, compression_time, decompression_time, psnr_value

# Compare wavelets
def compare_wavelets(image_path, output_image_path, wavelets, level=2, threshold=30):
    results = {}
    for wavelet in wavelets:
        print(f"Processing with wavelet: {wavelet}")
        compression_ratio, compression_time, decompression_time, psnr_value = compress_and_decompress(image_path, wavelet, level, threshold)
        results[wavelet] = {
            "Compression Ratio": compression_ratio,
            "Compression Time": compression_time,
            "Decompression Time": decompression_time,
            "PSNR": psnr_value
        }
        print(f"Wavelet: {wavelet}")
        print(f"Compression Ratio: {compression_ratio}")
        print(f"Compression Time: {compression_time:.4f} seconds")
        print(f"Decompression Time: {decompression_time:.4f} seconds")
        print(f"PSNR: {psnr_value:.4f} dB")
    return results

# Example Usage
image_path = 'C:\\Users\\Duvar\\Downloads\\test2.jpg'
output_image_path = 'decompressed_image.jpg'
wavelets = ['haar', 'db2', 'bior1.3']  # Haar, Daubechies (db2), Biorthogonal (bior1.3)
compare_wavelets(image_path, output_image_path, wavelets, level=2, threshold=30)
