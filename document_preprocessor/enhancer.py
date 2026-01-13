"""
Image Enhancement Module
Main preprocessing pipeline with quality-aware enhancement
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance
from skimage import exposure, filters, restoration
from skimage.util import img_as_ubyte
from typing import Dict, List, Tuple, Optional

from .quality import DocumentQualityAssessor
from .watermark import WatermarkReducer

try:
    from .config import ENHANCEMENT_CONFIG, PROCESSING_CONFIG
except ImportError:
    # Fallback if config not available
    ENHANCEMENT_CONFIG = {
        'bad_brightness': 16,
        'bad_contrast': 25,
        'good_brightness': 13,
        'good_contrast': -4,
    }
    PROCESSING_CONFIG = {
        'max_image_dimension': 2000,
        'pdf_dpi': 200,
        'clahe_clip_limit_bad': 2.5,
        'clahe_clip_limit_good': 1.8,
        'denoise_h': 3.0,
        'apply_watermark_reduction': True,
        'apply_denoising': True,
        'apply_sharpening': True,
    }


class ImageEnhancer:
    """Enhanced image preprocessing using OpenCV, Pillow, and scikit-image"""
    
    def __init__(self, 
                 bad_brightness: Optional[int] = None,
                 bad_contrast: Optional[int] = None,
                 good_brightness: Optional[int] = None,
                 good_contrast: Optional[int] = None):
        """
        Initialize image enhancer with configurable parameters
        Values can be set via environment variables or passed directly
        
        Args:
            bad_brightness: Brightness adjustment for bad quality docs (default: from env or +16)
            bad_contrast: Contrast adjustment for bad quality docs (default: from env or +25)
            good_brightness: Brightness adjustment for good quality docs (default: from env or +13)
            good_contrast: Contrast adjustment for good quality docs (default: from env or -4)
        """
        self.quality_assessor = DocumentQualityAssessor()
        self.watermark_reducer = WatermarkReducer()
        self.bad_brightness = bad_brightness if bad_brightness is not None else ENHANCEMENT_CONFIG['bad_brightness']
        self.bad_contrast = bad_contrast if bad_contrast is not None else ENHANCEMENT_CONFIG['bad_contrast']
        self.good_brightness = good_brightness if good_brightness is not None else ENHANCEMENT_CONFIG['good_brightness']
        self.good_contrast = good_contrast if good_contrast is not None else ENHANCEMENT_CONFIG['good_contrast']
    
    def apply_curves(self, image: np.ndarray, points: List[Tuple[int, int]]) -> np.ndarray:
        """
        Apply curves adjustment similar to Photoshop/GIMP
        
        Args:
            image: Input image (grayscale or BGR)
            points: List of (x, y) tuples representing curve points
            
        Returns:
            Image with curves applied
        """
        # Create lookup table from curve points
        lut = np.zeros(256, dtype=np.uint8)
        
        # Interpolate between points
        x_points = [p[0] for p in points]
        y_points = [p[1] for p in points]
        
        # Create smooth curve using numpy interpolation
        for i in range(256):
            lut[i] = np.clip(np.interp(i, x_points, y_points), 0, 255)
        
        # Apply LUT to each channel
        if len(image.shape) == 3:
            enhanced = cv2.LUT(image, lut)
        else:
            enhanced = cv2.LUT(image, lut)
        
        return enhanced
    
    def enhance_bad_quality(self, image: np.ndarray, 
                           apply_watermark_reduction: bool = True,
                           apply_denoising: bool = True,
                           apply_sharpening: bool = True) -> np.ndarray:
        """
        Enhance bad quality documents with focus on preserving quality while improving visibility
        Based on graphic designer's approach: Brightness +16, Contrast +25
        
        Args:
            image: Input image
            apply_watermark_reduction: Whether to apply watermark reduction
            apply_denoising: Whether to apply denoising
            apply_sharpening: Whether to apply sharpening
            
        Returns:
            Enhanced image with clear text and reduced watermarks
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 1: Gentle watermark/stamp reduction (preserve text quality)
        if apply_watermark_reduction:
            # Use configurable bilateral filtering parameters
            d = PROCESSING_CONFIG.get('watermark_bilateral_d_bad', 7)
            sigma_color = PROCESSING_CONFIG.get('watermark_sigma_color_bad', 50.0)
            sigma_space = PROCESSING_CONFIG.get('watermark_sigma_space_bad', 50.0)
            blend_ratio = PROCESSING_CONFIG.get('watermark_blend_bad', 0.7)
            reduced = self.watermark_reducer.reduce_bilateral(gray, d=int(d), sigma_color=sigma_color, sigma_space=sigma_space)
            # Blend with original using configurable ratio
            reduced = cv2.addWeighted(reduced, blend_ratio, gray, 1.0 - blend_ratio, 0)
        else:
            reduced = gray
        
        # Step 2: Apply CLAHE for better contrast (moderate to preserve quality)
        clahe_clip_limit = PROCESSING_CONFIG.get('clahe_clip_limit_bad', 2.5)
        tile_size = PROCESSING_CONFIG.get('clahe_tile_size', 8)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(reduced)
        
        # Step 3: Apply brightness adjustment (+16)
        enhanced = cv2.convertScaleAbs(enhanced, alpha=1.0, beta=self.bad_brightness)
        
        # Step 4: Apply contrast adjustment (+25) - moderate approach
        # Use Pillow for smoother contrast adjustment
        pil_image = Image.fromarray(enhanced)
        contrast_factor = 1.0 + (self.bad_contrast / 100)  # 1.25
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast_factor)
        enhanced = np.array(pil_image, dtype=np.uint8)
        
        # Step 5: Light denoising (preserve edges and text)
        if apply_denoising:
            # Use configurable denoising parameters
            denoise_h = PROCESSING_CONFIG.get('denoise_h', 3.0)
            template_window = PROCESSING_CONFIG.get('denoise_template_window', 7)
            search_window = PROCESSING_CONFIG.get('denoise_search_window', 21)
            denoised = cv2.fastNlMeansDenoising(enhanced, None, h=denoise_h, 
                                                templateWindowSize=template_window, 
                                                searchWindowSize=search_window)
        else:
            denoised = enhanced
        
        # Step 6: Gentle tone curve for better text visibility (preserve quality)
        # Parse curve points from config (format: "x1,y1,x2,y2,...")
        curve_str = PROCESSING_CONFIG.get('curve_points_bad', '0,0,64,70,128,145,192,215,255,255')
        curve_values = [int(x) for x in curve_str.split(',')]
        curve_points = [(curve_values[i], curve_values[i+1]) for i in range(0, len(curve_values), 2)]
        curved = self.apply_curves(denoised, curve_points)
        curved = np.clip(curved, 0, 255).astype(np.uint8)
        
        # Step 7: Moderate sharpening for text clarity (avoid over-sharpening)
        if apply_sharpening:
            # Unsharp mask with configurable strength and sigma from env
            sharpen_strength = PROCESSING_CONFIG.get('sharpen_strength', 0.65)
            sharpen_sigma = PROCESSING_CONFIG.get('sharpen_sigma_bad', 1.0)
            # Convert strength (0-1) to addWeighted parameters
            alpha = 1.0 + sharpen_strength
            beta = -sharpen_strength
            blurred = cv2.GaussianBlur(curved, (0, 0), sharpen_sigma)
            sharpened = cv2.addWeighted(curved, alpha, blurred, beta, 0)
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        else:
            sharpened = curved
        
        return sharpened
    
    def enhance_good_quality(self, image: np.ndarray,
                            apply_watermark_reduction: bool = True,
                            apply_sharpening: bool = True,
                            apply_thresholding: bool = False) -> np.ndarray:
        """
        Enhance good quality documents - preserve quality while improving visibility
        Based on graphic designer's approach: Brightness +13, Contrast -4
        
        Args:
            image: Input image
            apply_sharpening: Whether to apply sharpening
            apply_thresholding: Whether to apply adaptive thresholding
            
        Returns:
            Enhanced image with preserved quality and improved visibility
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Step 1: Very light watermark reduction (preserve quality)
        if apply_watermark_reduction:
            # Use configurable bilateral filtering parameters
            d = PROCESSING_CONFIG.get('watermark_bilateral_d_good', 5)
            sigma_color = PROCESSING_CONFIG.get('watermark_sigma_color_good', 40.0)
            sigma_space = PROCESSING_CONFIG.get('watermark_sigma_space_good', 40.0)
            blend_ratio = PROCESSING_CONFIG.get('watermark_blend_good', 0.8)
            bilateral = self.watermark_reducer.reduce_bilateral(gray, d=int(d), sigma_color=sigma_color, sigma_space=sigma_space)
            reduced = cv2.addWeighted(gray, blend_ratio, bilateral, 1.0 - blend_ratio, 0)
        else:
            reduced = gray
        
        # Step 2: Apply brightness (+13) using Pillow for smoother adjustment
        pil_image = Image.fromarray(reduced)
        brightness_factor = 1.0 + (self.good_brightness / 255)
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(brightness_factor)
        enhanced = np.array(pil_image, dtype=np.uint8)
        
        # Step 3: Gentle contrast adjustment (-4%) using Pillow
        contrast_factor = 1.0 - (abs(self.good_contrast) / 100)  # 0.96
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast_factor)
        enhanced = np.array(pil_image, dtype=np.uint8)
        
        # Step 4: Light CLAHE for subtle contrast improvement
        clahe_clip_limit = PROCESSING_CONFIG.get('clahe_clip_limit_good', 1.8)
        tile_size = PROCESSING_CONFIG.get('clahe_tile_size', 8)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(tile_size, tile_size))
        enhanced = clahe.apply(enhanced)
        
        # Step 5: Gentle sharpening (preserve quality)
        if apply_sharpening:
            # Use configurable sharpening strength and sigma from env
            sharpen_strength = PROCESSING_CONFIG.get('sharpen_strength', 0.65)
            sharpen_sigma = PROCESSING_CONFIG.get('sharpen_sigma_good', 0.8)
            # Convert strength (0-1) to addWeighted parameters
            alpha = 1.0 + sharpen_strength
            beta = -sharpen_strength
            blurred = cv2.GaussianBlur(enhanced, (0, 0), sharpen_sigma)
            sharpened = cv2.addWeighted(enhanced, alpha, blurred, beta, 0)
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        else:
            sharpened = enhanced
        
        # Step 6: Adaptive thresholding for better OCR (optional)
        # Check if adaptive thresholding should be applied (from env or explicit flag)
        should_threshold = apply_thresholding or PROCESSING_CONFIG.get('apply_adaptive_threshold', False)
        if should_threshold:
            # Use configurable threshold parameters from env
            block_size = PROCESSING_CONFIG.get('threshold_block_size', 31)
            offset = PROCESSING_CONFIG.get('threshold_offset', 9)
            # Ensure block_size is odd (required by OpenCV)
            if block_size % 2 == 0:
                block_size += 1
            binary = cv2.adaptiveThreshold(
                sharpened, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, block_size, offset
            )
            return binary
        
        return sharpened
    
    def enhance_scikit(self, image: np.ndarray, quality_class: str) -> np.ndarray:
        """
        Alternative enhancement using scikit-image - focus on clarity and readability
        Pipeline steps:
        1. Brightness/Contrast Adjustment (PIL)
        2. Light Denoising (before CLAHE to prevent noise amplification) - uses SCIKIT_DENOISE_H
        3. CLAHE (clip_limit = 2.3) - uses SCIKIT_CLAHE_CLIP_LIMIT
        4. Background Noise Cleanup (light median filter) - uses SCIKIT_BACKGROUND_MEDIAN_KERNEL
        5. Watermark Reduction (bilateral filter, light blend)
        6. Morphological Enhancement (close operation)
        7. Sharpening (UnsharpMask with radius=0.5, strength=96%) - uses SCIKIT_SHARPEN_SIGMA
        
        Args:
            image: Input image
            quality_class: 'GOOD' or 'BAD'
            
        Returns:
            Enhanced image with improved clarity and readability
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Ensure image is uint8
        if gray.dtype != np.uint8:
            gray = gray.astype(np.uint8)
        
        # Start with original to preserve sharpness
        result = gray.copy()
        
        # Step 1: Apply brightness and contrast FIRST (like good_quality method)
        # This preserves sharpness better than doing it after other processing
        pil_image = Image.fromarray(result)
        
        if quality_class == 'BAD':
            # Use config values for brightness and contrast
            brightness_factor = 1.0 + (self.bad_brightness / 255)
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(brightness_factor)
            contrast_factor = 1.0 + (self.bad_contrast / 100)
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(contrast_factor)
        else:
            # Use config values for brightness and contrast
            brightness_factor = 1.0 + (self.good_brightness / 255)
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(brightness_factor)
            contrast_factor = 1.0 - (abs(self.good_contrast) / 100)
            enhancer = ImageEnhance.Contrast(pil_image)
            pil_image = enhancer.enhance(contrast_factor)
        
        result = np.array(pil_image, dtype=np.uint8)
        
        # Step 2: Light denoising BEFORE CLAHE to reduce noise amplification
        # CLAHE amplifies noise, so reduce it first
        apply_denoising = PROCESSING_CONFIG.get('scikit_apply_denoising', True)
        if apply_denoising:
            denoise_h = PROCESSING_CONFIG.get('scikit_denoise_h', 1.5)
            template_window = PROCESSING_CONFIG.get('scikit_denoise_template_window', 7)
            search_window = PROCESSING_CONFIG.get('scikit_denoise_search_window', 21)
            result = cv2.fastNlMeansDenoising(result, None, h=denoise_h,
                                              templateWindowSize=template_window,
                                              searchWindowSize=search_window)
        
        # Step 3: CLAHE with scikit-specific clip limit (2.3)
        clahe_clip_limit = PROCESSING_CONFIG.get('scikit_clahe_clip_limit', 2.3)
        tile_size = PROCESSING_CONFIG.get('clahe_tile_size', 8)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(tile_size, tile_size))
        result = clahe.apply(result)
        
        # Step 4: Background noise cleanup (light median filter for uniform background)
        apply_background_cleanup = PROCESSING_CONFIG.get('scikit_apply_background_cleanup', True)
        if apply_background_cleanup:
            median_kernel = PROCESSING_CONFIG.get('scikit_background_median_kernel', 3)
            blend_ratio = PROCESSING_CONFIG.get('scikit_background_blend_ratio', 0.7)
            # Light median filter to smooth background noise while preserving text edges
            background_smoothed = cv2.medianBlur(result, median_kernel)
            # Blend with original to preserve text sharpness (configurable ratio)
            result = cv2.addWeighted(result, blend_ratio, background_smoothed, 1.0 - blend_ratio, 0)
        
        # Step 5: Watermark reduction (bilateral filter, light blend)
        apply_watermark = PROCESSING_CONFIG.get('apply_watermark_reduction', True)
        if apply_watermark:
            # Use lighter parameters and higher blend ratio to preserve sharpness
            if quality_class == 'BAD':
                d = PROCESSING_CONFIG.get('watermark_bilateral_d_bad', 7)
                sigma_color = PROCESSING_CONFIG.get('watermark_sigma_color_bad', 50.0)
                sigma_space = PROCESSING_CONFIG.get('watermark_sigma_space_bad', 50.0)
                # Use scikit-specific blend ratio from config
                blend_ratio = PROCESSING_CONFIG.get('scikit_watermark_blend_bad', 0.85)
            else:
                d = PROCESSING_CONFIG.get('watermark_bilateral_d_good', 5)
                sigma_color = PROCESSING_CONFIG.get('watermark_sigma_color_good', 40.0)
                sigma_space = PROCESSING_CONFIG.get('watermark_sigma_space_good', 40.0)
                # Use scikit-specific blend ratio from config
                blend_ratio = PROCESSING_CONFIG.get('scikit_watermark_blend_good', 0.9)
            bilateral = self.watermark_reducer.reduce_bilateral(result, d=int(d), 
                                                                sigma_color=sigma_color, 
                                                                sigma_space=sigma_space)
            result = cv2.addWeighted(result, blend_ratio, bilateral, 1.0 - blend_ratio, 0)
        
        # Step 6: Morphological Enhancement (close operation)
        # Closing operation helps connect text strokes and reduce background noise
        kernel_size = PROCESSING_CONFIG.get('scikit_morphological_kernel_size', 3)
        iterations = PROCESSING_CONFIG.get('scikit_morphological_iterations', 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
        # Step 7: Sharpening with scikit-specific parameters
        apply_sharpening = PROCESSING_CONFIG.get('apply_sharpening', True)
        if apply_sharpening:
            # Use scikit-specific sharpen strength from config
            sharpen_strength = PROCESSING_CONFIG.get('scikit_sharpen_strength', 0.96)
            
            # Use scikit-specific sharpen sigma from config
            sharpen_sigma = PROCESSING_CONFIG.get('scikit_sharpen_sigma', 0.5)
            
            # Unsharp mask for clarity
            alpha = 1.0 + sharpen_strength
            beta = -sharpen_strength
            blurred = cv2.GaussianBlur(result, (0, 0), sharpen_sigma)
            sharpened = cv2.addWeighted(result, alpha, blurred, beta, 0)
            result = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return result
    
    def process(self, image: np.ndarray, method: str = 'auto', **kwargs) -> Dict:
        """
        Main processing pipeline
        
        Args:
            image: Input image as numpy array
            method: Processing method ('auto', 'opencv', 'scikit')
            **kwargs: Additional parameters for enhancement methods
            
        Returns:
            dict with:
            - enhanced_image: Processed image
            - quality_info: Quality assessment metrics
            - method_used: Method that was actually used
        """
        # Get processing flags from config (can be overridden by kwargs)
        apply_watermark = kwargs.get('apply_watermark_reduction', PROCESSING_CONFIG.get('apply_watermark_reduction', True))
        apply_denoise = kwargs.get('apply_denoising', PROCESSING_CONFIG.get('apply_denoising', True))
        apply_sharpen = kwargs.get('apply_sharpening', PROCESSING_CONFIG.get('apply_sharpening', True))
        
        # Assess quality
        quality_info = self.quality_assessor.assess_quality(image)
        
        # Apply appropriate enhancement
        if method == 'auto':
            if quality_info['is_good_quality']:
                enhanced = self.enhance_good_quality(image, 
                                                   apply_watermark_reduction=apply_watermark,
                                                   apply_sharpening=apply_sharpen)
            else:
                enhanced = self.enhance_bad_quality(image,
                                                   apply_watermark_reduction=apply_watermark,
                                                   apply_denoising=apply_denoise,
                                                   apply_sharpening=apply_sharpen)
        elif method == 'opencv':
            if quality_info['is_good_quality']:
                enhanced = self.enhance_good_quality(image,
                                                   apply_watermark_reduction=apply_watermark,
                                                   apply_sharpening=apply_sharpen)
            else:
                enhanced = self.enhance_bad_quality(image,
                                                     apply_watermark_reduction=apply_watermark,
                                                     apply_denoising=apply_denoise,
                                                     apply_sharpening=apply_sharpen)
        elif method == 'scikit':
            enhanced = self.enhance_scikit(image, quality_info['quality_class'])
        else:
            enhanced = image
        
        # Ensure enhanced image is valid and visible
        if enhanced is None or enhanced.size == 0:
            # Fallback to original if enhancement failed
            enhanced = image.copy()
        
        # Ensure image is uint8 and in valid range
        if enhanced.dtype != np.uint8:
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        else:
            enhanced = np.clip(enhanced, 0, 255)
        
        # Final validation: Ensure image is visible but preserve quality
        img_mean = np.mean(enhanced)
        img_min = np.min(enhanced)
        img_max = np.max(enhanced)
        
        # Check if image is too dark or has no contrast (using configurable thresholds)
        dark_threshold = PROCESSING_CONFIG.get('fallback_dark_threshold', 30.0)
        contrast_threshold = PROCESSING_CONFIG.get('fallback_contrast_threshold', 20.0)
        if np.all(enhanced == 0) or img_mean < dark_threshold or (img_max - img_min) < contrast_threshold:
            print(f"Warning: Image too dark (mean={img_mean:.2f}, range={img_max-img_min}), applying gentle fallback")
            # Fallback: gentle enhancement to preserve quality
            if len(image.shape) == 3:
                enhanced = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                enhanced = image.copy()
            
            # Ensure it's uint8
            if enhanced.dtype != np.uint8:
                enhanced = enhanced.astype(np.uint8)
            
            # Moderate CLAHE with configurable parameters
            fallback_clahe = PROCESSING_CONFIG.get('fallback_clahe_clip_limit', 2.5)
            tile_size = PROCESSING_CONFIG.get('clahe_tile_size', 8)
            clahe = cv2.createCLAHE(clipLimit=fallback_clahe, tileGridSize=(tile_size, tile_size))
            enhanced = clahe.apply(enhanced)
            
            # Gentle brightness boost using Pillow with configurable factor
            brightness_factor_1 = PROCESSING_CONFIG.get('fallback_brightness_1', 1.15)
            pil_image = Image.fromarray(enhanced)
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(brightness_factor_1)
            enhanced = np.array(pil_image, dtype=np.uint8)
        
        # Final brightness check - only if really needed (using configurable threshold)
        brightness_threshold = PROCESSING_CONFIG.get('fallback_brightness_threshold', 80.0)
        if img_mean < brightness_threshold:
            # Use Pillow for smoother brightness adjustment with configurable factor
            brightness_factor_2 = PROCESSING_CONFIG.get('fallback_brightness_2', 1.1)
            pil_image = Image.fromarray(enhanced)
            enhancer = ImageEnhance.Brightness(pil_image)
            pil_image = enhancer.enhance(brightness_factor_2)
            enhanced = np.array(pil_image, dtype=np.uint8)
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        
        return {
            'enhanced_image': enhanced,
            'quality_info': quality_info,
            'method_used': method
        }
