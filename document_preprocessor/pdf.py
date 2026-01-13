"""
PDF Processing Module
Converts PDF documents to images for processing
"""

import cv2
import numpy as np
from PIL import Image
import io
from typing import List, Optional

try:
    import pdf2image
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


class PDFProcessor:
    """Convert PDF to images and process them"""
    
    @staticmethod
    def pdf_to_images(pdf_path: str, dpi: int = 300, first_page: Optional[int] = None, last_page: Optional[int] = None) -> List[np.ndarray]:
        """
        Convert PDF pages to images
        
        Args:
            pdf_path: Path to PDF file
            dpi: Resolution for conversion (default: 300)
            first_page: First page to convert (1-indexed, None for all)
            last_page: Last page to convert (1-indexed, None for all)
            
        Returns:
            List of numpy arrays (BGR format for OpenCV)
            
        Raises:
            ImportError: If pdf2image is not installed
        """
        if not PDF_SUPPORT:
            raise ImportError(
                "pdf2image is required for PDF processing. "
                "Install it with: pip install pdf2image\n"
                "Also install poppler-utils:\n"
                "  Ubuntu/Debian: sudo apt-get install poppler-utils\n"
                "  macOS: brew install poppler\n"
                "  Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases"
            )
        
        # Convert PDF to PIL images
        if first_page is not None or last_page is not None:
            first_page = first_page or 1
            last_page = last_page or first_page
            images = pdf2image.convert_from_path(
                pdf_path, 
                dpi=dpi,
                first_page=first_page,
                last_page=last_page
            )
        else:
            images = pdf2image.convert_from_path(pdf_path, dpi=dpi)
        
        # Convert PIL images to numpy arrays (BGR for OpenCV)
        np_images = []
        for img in images:
            np_img = np.array(img)
            # Convert RGB to BGR for OpenCV
            if len(np_img.shape) == 3:
                np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)
            np_images.append(np_img)
        
        return np_images
    
    @staticmethod
    def image_to_bytes(image: np.ndarray, format: str = 'PNG') -> bytes:
        """
        Convert numpy array to image bytes
        
        Args:
            image: Image as numpy array (BGR or grayscale)
            format: Image format ('PNG', 'JPEG', etc.)
            
        Returns:
            Image bytes
        """
        if len(image.shape) == 2:
            pil_image = Image.fromarray(image)
        else:
            # Convert BGR to RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_image)
        
        byte_io = io.BytesIO()
        pil_image.save(byte_io, format=format)
        byte_io.seek(0)
        return byte_io.read()
    
    @staticmethod
    def is_pdf_supported() -> bool:
        """
        Check if PDF processing is supported
        
        Returns:
            True if pdf2image is installed
        """
        return PDF_SUPPORT
