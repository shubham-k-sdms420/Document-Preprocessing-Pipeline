# Document Preprocessor

A modular, plug-and-play document preprocessing pipeline for any document type with AI-powered quality enhancement.

## 🌟 Features

- **Modular Architecture**: Clean separation of concerns - use only what you need
- **Automatic Quality Detection**: Intelligently detects whether a document is good or bad quality
- **Multi-Method Processing**: Supports OpenCV, Pillow, and Scikit-Image based enhancements
- **Watermark Reduction**: Advanced algorithms to reduce watermarks and stamps
- **PDF Support**: Optional PDF to image conversion (requires pdf2image)
- **Easy Integration**: Simple API for integration into any system
- **Configurable**: All parameters are customizable

## 📦 Installation

### Basic Installation (Core Functionality)

```bash
pip install opencv-python numpy Pillow scikit-image scipy
```

### With PDF Support

```bash
pip install -r requirements.txt
# Also install poppler-utils:
# Ubuntu/Debian: sudo apt-get install poppler-utils
# macOS: brew install poppler
# Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases
```

### With Web Interface (Optional)

```bash
pip install -r requirements.txt Flask werkzeug
```

## 🚀 Quick Start

### As a Library (Plug-and-Play)

```python
from document_preprocessor import ImageEnhancer
import cv2

# Initialize enhancer
enhancer = ImageEnhancer()

# Load image
image = cv2.imread('document.png')

# Process with auto method
result = enhancer.process(image, method='auto')

# Access results
enhanced_image = result['enhanced_image']
quality_info = result['quality_info']

# Save result
cv2.imwrite('enhanced.png', enhanced_image)

# Print quality metrics
print(f"Quality: {quality_info['quality_class']}")
print(f"Blur Score: {quality_info['blur_score']:.2f}")
```

### Using Individual Modules

```python
# Quality assessment only
from document_preprocessor import DocumentQualityAssessor
assessor = DocumentQualityAssessor()
quality = assessor.assess_quality(image)

# Watermark reduction only
from document_preprocessor import WatermarkReducer
reducer = WatermarkReducer()
reduced = reducer.reduce_bilateral(image)

# PDF processing only
from document_preprocessor import PDFProcessor
processor = PDFProcessor()
images = processor.pdf_to_images('document.pdf')
```

## 📚 API Reference

### ImageEnhancer

Main preprocessing class with quality-aware enhancement.

```python
enhancer = ImageEnhancer(
    bad_brightness=16,      # Brightness for bad quality docs
    bad_contrast=25,         # Contrast for bad quality docs
    good_brightness=13,      # Brightness for good quality docs
    good_contrast=-4         # Contrast for good quality docs
)

result = enhancer.process(
    image,                   # numpy array (BGR or grayscale)
    method='auto',           # 'auto', 'opencv', or 'scikit'
    apply_watermark_reduction=True,  # For bad quality
    apply_denoising=True,            # For bad quality
    apply_sharpening=True            # For both
)
```

**Returns:**
```python
{
    'enhanced_image': np.ndarray,    # Processed image
    'quality_info': {                # Quality metrics
        'blur_score': float,
        'contrast': float,
        'brightness': float,
        'noise_level': float,
        'is_good_quality': bool,
        'quality_class': 'GOOD' | 'BAD'
    },
    'method_used': str               # Method that was used
}
```

### DocumentQualityAssessor

Assesses document quality to determine preprocessing strategy.

```python
assessor = DocumentQualityAssessor(
    blur_threshold=100,      # Minimum blur score for good quality
    contrast_threshold=50,   # Minimum contrast for good quality
    brightness_threshold=127 # Average brightness threshold
)

quality = assessor.assess_quality(image)
```

### WatermarkReducer

Reduces watermarks and stamps using various techniques.

```python
reducer = WatermarkReducer()

# FFT-based reduction
reduced = reducer.reduce_fft(image)

# Morphological operations
reduced = reducer.reduce_morphological(image)

# Bilateral filtering (recommended)
reduced = reducer.reduce_bilateral(image, d=9, sigma_color=75, sigma_space=75)
```

### PDFProcessor

Converts PDF documents to images.

```python
processor = PDFProcessor()

# Convert all pages
images = processor.pdf_to_images('document.pdf', dpi=300)

# Convert specific pages
images = processor.pdf_to_images('document.pdf', first_page=1, last_page=5)

# Check if PDF support is available
if processor.is_pdf_supported():
    images = processor.pdf_to_images('document.pdf')
```

## 🎨 Processing Methods

### Bad Quality Documents
For documents with poor scan quality, watermarks, or low contrast:
- Brightness: +16 (configurable)
- Contrast: +25 (configurable)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Bilateral filtering for watermark reduction
- Curve adjustment for tone enhancement
- Denoising
- Sharpening

### Good Quality Documents
For already decent documents that need light enhancement:
- Brightness: +13 (configurable)
- Contrast: -4 (configurable)
- Slight sharpening
- Adaptive thresholding for OCR optimization

## 🔧 Integration Examples

### Django Integration

```python
from document_preprocessor import ImageEnhancer
import cv2

def process_document_view(request):
    uploaded_file = request.FILES['document']
    
    # Read image
    image_array = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # Process
    enhancer = ImageEnhancer()
    result = enhancer.process(image)
    
    # Save or return result
    return result
```

### FastAPI Integration

```python
from fastapi import FastAPI, UploadFile, File
from document_preprocessor import ImageEnhancer
import cv2
import numpy as np

app = FastAPI()
enhancer = ImageEnhancer()

@app.post("/process")
async def process_document(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    result = enhancer.process(image)
    return result['quality_info']
```

### Batch Processing Script

```python
from document_preprocessor import ImageEnhancer
import cv2
from pathlib import Path

enhancer = ImageEnhancer()

for image_path in Path('input').glob('*.png'):
    image = cv2.imread(str(image_path))
    result = enhancer.process(image)
    
    output_path = Path('output') / f"enhanced_{image_path.name}"
    cv2.imwrite(str(output_path), result['enhanced_image'])
```

## 🌐 Web Application (Optional)

If you want to use the included Flask web interface:

```bash
python app.py
```

Then open `http://localhost:5000` in your browser.

## 📊 Batch Processing (Optional)

Process multiple documents at once:

```bash
python batch_process.py -i input_folder -o output_folder -m auto
```

Options:
- `-i, --input`: Input directory with documents
- `-o, --output`: Output directory for enhanced documents
- `-m, --method`: Processing method ('auto', 'opencv', 'scikit')
- `--no-metrics`: Don't save quality metrics JSON

## 🗂️ Project Structure

```
.
├── document_preprocessor/       # Main package
│   ├── __init__.py              # Package exports
│   ├── quality.py               # Quality assessment module
│   ├── watermark.py             # Watermark reduction module
│   ├── enhancer.py              # Main enhancement module
│   └── pdf.py                   # PDF processing module
├── app.py                       # Flask web app (optional)
├── batch_process.py             # Batch processor (optional)
├── templates/
│   └── index.html               # Web UI template
├── requirements.txt             # Dependencies
└── README.md                    # This file
```

## ⚙️ Configuration

### Customizing Enhancement Parameters

```python
# Custom brightness/contrast values
enhancer = ImageEnhancer(
    bad_brightness=20,    # Increase brightness more
    bad_contrast=30,      # Increase contrast more
    good_brightness=10,   # Less brightness for good docs
    good_contrast=-2       # Less contrast reduction
)
```

### Customizing Quality Thresholds

```python
# Stricter quality detection
assessor = DocumentQualityAssessor(
    blur_threshold=150,      # Higher threshold
    contrast_threshold=60,    # Higher threshold
    brightness_threshold=127
)
```

### Custom Curve Points

Modify curve points in `enhance_bad_quality()` method:

```python
curve_points = [
    (0, 0),       # Black point
    (64, 70),     # Shadows
    (128, 140),   # Midtones
    (192, 210),   # Highlights
    (255, 255)    # White point
]
```

## 🎯 Watermark Reduction Techniques

The system uses three approaches:

1. **FFT-based**: Removes periodic watermark patterns in frequency domain
2. **Morphological**: Uses erosion/dilation to separate text from stamps
3. **Bilateral Filter**: Preserves edges while smoothing watermarks (recommended)

## 🐛 Troubleshooting

### PDF conversion fails
Make sure poppler-utils is installed:
```bash
# Test poppler
pdftoppm -h
```

### Out of memory errors
Reduce image DPI in PDF processing:
```python
images = processor.pdf_to_images('document.pdf', dpi=150)  # Reduce from 300
```

### Slow processing
- Use 'opencv' method instead of 'auto' for faster processing
- Reduce image resolution before processing
- Disable denoising: `enhancer.process(image, apply_denoising=False)`

## 📝 License

This project is for internal use at Stark Digital Media Services Private Limited.

## 👥 Credits

Developed for Stark Digital Media Services Pvt Ltd.
A general-purpose document preprocessing pipeline for any document type.

## 🔄 Version History

- **v1.0.0**: Initial release with modular architecture and plug-and-play design
