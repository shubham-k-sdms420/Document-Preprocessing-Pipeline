"""
Document Preprocessor
A modular document preprocessing pipeline for any document type with AI-powered quality enhancement.

Usage:
    from document_preprocessor import ImageEnhancer
    
    enhancer = ImageEnhancer()
    result = enhancer.process(image, method='auto')
"""

from .quality import DocumentQualityAssessor
from .watermark import WatermarkReducer
from .enhancer import ImageEnhancer
from .pdf import PDFProcessor

__version__ = '1.0.0'
__all__ = [
    'DocumentQualityAssessor',
    'WatermarkReducer',
    'ImageEnhancer',
    'PDFProcessor'
]
