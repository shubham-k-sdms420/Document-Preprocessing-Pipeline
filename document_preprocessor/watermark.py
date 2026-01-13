"""
Watermark Reduction Module
Reduces watermarks and stamps using various techniques
"""

import cv2
import numpy as np
from typing import Optional


class WatermarkReducer:
    """Reduce watermarks and stamps using various techniques"""
    
    @staticmethod
    def reduce_fft(image: np.ndarray) -> np.ndarray:
        """
        Reduce watermarks using FFT frequency domain filtering
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Processed image with reduced watermark patterns
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Apply FFT
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # Create mask for high-pass filter (removes watermark patterns)
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Create a mask to suppress periodic watermark patterns
        mask = np.ones((rows, cols, 2), np.uint8)
        r = 30
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]
        mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r
        mask[mask_area] = 1
        
        # Apply mask and inverse FFT
        fshift = dft_shift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
        
        # Normalize
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        return img_back.astype(np.uint8)
    
    @staticmethod
    def reduce_morphological(image: np.ndarray) -> np.ndarray:
        """
        Reduce stamps and watermarks using morphological operations
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Processed image with reduced stamps
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Morphological operations to separate text from stamps
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        
        # Opening to remove small noise/stamps
        opening = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Top-hat to enhance text
        tophat = cv2.morphologyEx(opening, cv2.MORPH_TOPHAT, kernel)
        
        return opening
    
    @staticmethod
    def reduce_bilateral(image: np.ndarray, d: int = 9, sigma_color: float = 75, sigma_space: float = 75) -> np.ndarray:
        """
        Use bilateral filtering to reduce watermarks while preserving edges
        
        Args:
            image: Input image (BGR or grayscale)
            d: Diameter of pixel neighborhood (default: 9)
            sigma_color: Filter sigma in color space (default: 75)
            sigma_space: Filter sigma in coordinate space (default: 75)
            
        Returns:
            Filtered image with reduced watermarks
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Bilateral filter preserves edges while smoothing
        filtered = cv2.bilateralFilter(gray, d, sigma_color, sigma_space)
        return filtered
