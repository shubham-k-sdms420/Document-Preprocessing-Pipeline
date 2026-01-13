#!/usr/bin/env python3
"""
Example usage of Document Preprocessor
Demonstrates plug-and-play integration
"""

import cv2
from document_preprocessor import ImageEnhancer, DocumentQualityAssessor, WatermarkReducer, PDFProcessor

def example_basic_usage():
    """Basic usage example"""
    print("=" * 60)
    print("Example 1: Basic Usage")
    print("=" * 60)
    
    # Initialize enhancer
    enhancer = ImageEnhancer()
    
    # Load image (replace with your image path)
    # image = cv2.imread('your_document.png')
    # 
    # # Process with auto method
    # result = enhancer.process(image, method='auto')
    # 
    # # Access results
    # enhanced_image = result['enhanced_image']
    # quality_info = result['quality_info']
    # 
    # # Save result
    # cv2.imwrite('enhanced.png', enhanced_image)
    # 
    # print(f"Quality: {quality_info['quality_class']}")
    # print(f"Blur Score: {quality_info['blur_score']:.2f}")
    
    print("Uncomment the code above and provide an image path to run this example")


def example_quality_assessment_only():
    """Example using only quality assessment"""
    print("\n" + "=" * 60)
    print("Example 2: Quality Assessment Only")
    print("=" * 60)
    
    assessor = DocumentQualityAssessor(
        blur_threshold=100,
        contrast_threshold=50
    )
    
    # image = cv2.imread('your_document.png')
    # quality = assessor.assess_quality(image)
    # print(f"Quality Class: {quality['quality_class']}")
    
    print("Uncomment the code above and provide an image path to run this example")


def example_watermark_reduction_only():
    """Example using only watermark reduction"""
    print("\n" + "=" * 60)
    print("Example 3: Watermark Reduction Only")
    print("=" * 60)
    
    reducer = WatermarkReducer()
    
    # image = cv2.imread('your_document.png')
    # reduced = reducer.reduce_bilateral(image)
    # cv2.imwrite('watermark_reduced.png', reduced)
    
    print("Uncomment the code above and provide an image path to run this example")


def example_custom_parameters():
    """Example with custom parameters"""
    print("\n" + "=" * 60)
    print("Example 4: Custom Parameters")
    print("=" * 60)
    
    # Custom brightness/contrast values
    enhancer = ImageEnhancer(
        bad_brightness=20,    # Increase brightness more
        bad_contrast=30,      # Increase contrast more
        good_brightness=10,   # Less brightness for good docs
        good_contrast=-2      # Less contrast reduction
    )
    
    # image = cv2.imread('your_document.png')
    # result = enhancer.process(image, method='auto')
    
    print("Uncomment the code above and provide an image path to run this example")


def example_pdf_processing():
    """Example with PDF processing"""
    print("\n" + "=" * 60)
    print("Example 5: PDF Processing")
    print("=" * 60)
    
    processor = PDFProcessor()
    
    if processor.is_pdf_supported():
        # images = processor.pdf_to_images('document.pdf', dpi=300)
        # print(f"Extracted {len(images)} pages")
        print("PDF processing is available")
        print("Uncomment the code above and provide a PDF path to run this example")
    else:
        print("PDF processing not available. Install pdf2image and poppler-utils.")


def example_integration_django():
    """Example Django integration"""
    print("\n" + "=" * 60)
    print("Example 6: Django Integration")
    print("=" * 60)
    
    code = '''
# In your Django view:
from document_preprocessor import ImageEnhancer
import cv2
import numpy as np

def process_document_view(request):
    uploaded_file = request.FILES['document']
    
    # Read image
    image_array = np.frombuffer(uploaded_file.read(), np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    # Process
    enhancer = ImageEnhancer()
    result = enhancer.process(image)
    
    # Return quality info as JSON
    return JsonResponse(result['quality_info'])
'''
    print(code)


def example_integration_fastapi():
    """Example FastAPI integration"""
    print("\n" + "=" * 60)
    print("Example 7: FastAPI Integration")
    print("=" * 60)
    
    code = '''
# In your FastAPI app:
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
'''
    print(code)


if __name__ == '__main__':
    print("\n" + "╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "Document Preprocessor Usage Examples" + " " * 10 + "║")
    print("╚" + "=" * 58 + "╝")
    
    example_basic_usage()
    example_quality_assessment_only()
    example_watermark_reduction_only()
    example_custom_parameters()
    example_pdf_processing()
    example_integration_django()
    example_integration_fastapi()
    
    print("\n" + "=" * 60)
    print("For more examples, see README.md")
    print("=" * 60)
