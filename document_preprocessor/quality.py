"""
Document Quality Assessment Module
Assesses document quality to determine preprocessing strategy
"""

import cv2
import numpy as np
from typing import Dict, Optional

try:
    from .config import QUALITY_CONFIG
except ImportError:
    # Fallback if config not available
    QUALITY_CONFIG = {
        'blur_threshold': 100.0,
        'contrast_threshold': 50.0,
        'brightness_threshold': 127.0,
    }


class DocumentQualityAssessor:
    """Assess document quality to determine preprocessing strategy"""
    
    def __init__(self, 
                 blur_threshold: Optional[float] = None, 
                 contrast_threshold: Optional[float] = None, 
                 brightness_threshold: Optional[float] = None):
        """
        Initialize quality assessor with configurable thresholds
        Values can be set via environment variables or passed directly
        
        Args:
            blur_threshold: Minimum Laplacian variance for good quality (default: from env or 100)
            contrast_threshold: Minimum contrast for good quality (default: from env or 50)
            brightness_threshold: Average brightness threshold (default: from env or 127)
        """
        self.thresholds = {
            'blur_threshold': blur_threshold if blur_threshold is not None else QUALITY_CONFIG['blur_threshold'],
            'contrast_threshold': contrast_threshold if contrast_threshold is not None else QUALITY_CONFIG['contrast_threshold'],
            'brightness_threshold': brightness_threshold if brightness_threshold is not None else QUALITY_CONFIG['brightness_threshold']
        }
    
    def assess_quality(self, image: np.ndarray) -> Dict:
        """
        Assess image quality and return metrics
        
        Args:
            image: Input image as numpy array (BGR or grayscale)
            
        Returns:
            dict with quality metrics and classification:
            - blur_score: Laplacian variance (higher = sharper)
            - contrast: Standard deviation of pixel values
            - brightness: Mean pixel value
            - noise_level: Estimated noise level
            - is_good_quality: Boolean quality classification
            - quality_class: 'GOOD' or 'BAD'
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Blur detection using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Contrast detection
        contrast = gray.std()
        
        # Brightness detection
        brightness = gray.mean()
        
        # Noise detection using local standard deviation
        noise_level = self._estimate_noise(gray)
        
        # Classify quality
        is_good_quality = (
            laplacian_var > self.thresholds['blur_threshold'] and
            contrast > self.thresholds['contrast_threshold']
        )
        
        # Convert to native Python types for JSON serialization
        return {
            'blur_score': float(laplacian_var),
            'contrast': float(contrast),
            'brightness': float(brightness),
            'noise_level': float(noise_level),
            'is_good_quality': bool(is_good_quality),  # Convert numpy bool to Python bool
            'quality_class': 'GOOD' if is_good_quality else 'BAD'
        }
    
    def _estimate_noise(self, gray: np.ndarray) -> float:
        """
        Estimate noise level using median absolute deviation
        
        Args:
            gray: Grayscale image
            
        Returns:
            Estimated noise level
        """
        H, W = gray.shape
        M = [[1, -2, 1],
             [-2, 4, -2],
             [1, -2, 1]]
        sigma = np.sum(np.sum(np.absolute(cv2.filter2D(gray, -1, np.array(M)))))
        sigma = sigma * np.sqrt(0.5 * np.pi) / (6 * (W-2) * (H-2))
        return float(sigma)
