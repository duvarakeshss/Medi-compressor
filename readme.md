# Image Compression with Wavelet Transform

This project implements image compression using wavelet transforms, specifically the Haar, Daubechies, and Biorthogonal wavelets. It provides an interactive web application built with Streamlit, allowing users to upload images, select wavelet types, and set threshold values for compression.

## Features

- **Wavelet Transforms**: Supports Haar, Daubechies (db2), and Biorthogonal (bior1.3) wavelet transforms for image compression.
- **Thresholding**: Apply a threshold to control the amount of detail retained after compression.
- **RLE Compression**: Utilizes Run-Length Encoding (RLE) to further compress the wavelet coefficients.
- **Performance Metrics**: Displays PSNR (Peak Signal-to-Noise Ratio) and compression ratios to evaluate the quality of the compressed images.

## Requirements

- Python
- Streamlit
- NumPy
- OpenCV
- scikit-image
- Pillow

##### You can install the required packages using pip:

```bash
pip install streamlit numpy opencv-python scikit-image Pillow
```

### How to Run the Application

```bash
git clone https://github.com/duvarakeshss/WaveletWhiz
cd image-compression-wavelet
streamlit run app.py
```
