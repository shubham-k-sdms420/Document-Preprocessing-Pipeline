"""
Configuration Module
Loads configuration from environment variables with sensible defaults
"""

import os
from typing import Optional

# Try to load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # If dotenv not installed, just use environment variables


def get_env_float(key: str, default: float) -> float:
    """Get float value from environment variable"""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def get_env_int(key: str, default: int) -> int:
    """Get int value from environment variable"""
    value = os.getenv(key)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def get_env_bool(key: str, default: bool) -> bool:
    """Get boolean value from environment variable"""
    value = os.getenv(key)
    if value is None:
        return default
    return value.lower() in ('true', '1', 'yes', 'on')


# Quality Assessment Configuration
QUALITY_CONFIG = {
    'blur_threshold': get_env_float('BLUR_THRESHOLD', 100.0),
    'contrast_threshold': get_env_float('CONTRAST_THRESHOLD', 50.0),
    'brightness_threshold': get_env_float('BRIGHTNESS_THRESHOLD', 127.0),
}

# Enhancement Configuration
ENHANCEMENT_CONFIG = {
    'bad_brightness': get_env_int('BAD_BRIGHTNESS', 16),
    'bad_contrast': get_env_int('BAD_CONTRAST', 25),
    'good_brightness': get_env_int('GOOD_BRIGHTNESS', 13),
    'good_contrast': get_env_int('GOOD_CONTRAST', -4),
}

# Processing Configuration
PROCESSING_CONFIG = {
    'max_image_dimension': get_env_int('MAX_IMAGE_DIMENSION', 2000),
    'pdf_dpi': get_env_int('PDF_DPI', 200),
    'clahe_clip_limit_bad': get_env_float('CLAHE_CLIP_LIMIT_BAD', 2.5),
    'clahe_clip_limit_good': get_env_float('CLAHE_CLIP_LIMIT_GOOD', 1.8),
    'clahe_tile_size': get_env_int('CLAHE_TILE_SIZE', 8),
    'denoise_h': get_env_float('DENOISE_H', 3.0),
    'denoise_template_window': get_env_int('DENOISE_TEMPLATE_WINDOW', 7),
    'denoise_search_window': get_env_int('DENOISE_SEARCH_WINDOW', 21),
    'apply_watermark_reduction': get_env_bool('APPLY_WATERMARK_REDUCTION', True),
    'apply_denoising': get_env_bool('APPLY_DENOISING', True),
    'apply_sharpening': get_env_bool('APPLY_SHARPENING', True),
    'sharpen_strength': get_env_float('SHARPEN_STRENGTH', 0.65),
    'sharpen_sigma_bad': get_env_float('SHARPEN_SIGMA_BAD', 1.0),
    'sharpen_sigma_good': get_env_float('SHARPEN_SIGMA_GOOD', 0.8),
    'apply_adaptive_threshold': get_env_bool('APPLY_ADAPTIVE_THRESHOLD', False),
    'threshold_block_size': get_env_int('THRESHOLD_BLOCK_SIZE', 31),
    'threshold_offset': get_env_int('THRESHOLD_OFFSET', 9),
    # Watermark reduction parameters
    'watermark_bilateral_d_bad': get_env_int('WATERMARK_BILATERAL_D_BAD', 7),
    'watermark_sigma_color_bad': get_env_float('WATERMARK_SIGMA_COLOR_BAD', 50.0),
    'watermark_sigma_space_bad': get_env_float('WATERMARK_SIGMA_SPACE_BAD', 50.0),
    'watermark_blend_bad': get_env_float('WATERMARK_BLEND_BAD', 0.7),
    'watermark_bilateral_d_good': get_env_int('WATERMARK_BILATERAL_D_GOOD', 5),
    'watermark_sigma_color_good': get_env_float('WATERMARK_SIGMA_COLOR_GOOD', 40.0),
    'watermark_sigma_space_good': get_env_float('WATERMARK_SIGMA_SPACE_GOOD', 40.0),
    'watermark_blend_good': get_env_float('WATERMARK_BLEND_GOOD', 0.8),
    # Scikit method watermark blend ratios (for scikit-specific processing)
    'scikit_watermark_blend_bad': get_env_float('SCIKIT_WATERMARK_BLEND_BAD', 0.85),
    'scikit_watermark_blend_good': get_env_float('SCIKIT_WATERMARK_BLEND_GOOD', 0.9),
    # Curve points for bad quality (comma-separated pairs: x1,y1,x2,y2,...)
    'curve_points_bad': os.getenv('CURVE_POINTS_BAD', '0,0,64,70,128,145,192,215,255,255'),
    # Fallback parameters
    'fallback_clahe_clip_limit': get_env_float('FALLBACK_CLAHE_CLIP_LIMIT', 2.5),
    'fallback_brightness_1': get_env_float('FALLBACK_BRIGHTNESS_1', 1.15),
    'fallback_brightness_2': get_env_float('FALLBACK_BRIGHTNESS_2', 1.1),
    'fallback_dark_threshold': get_env_float('FALLBACK_DARK_THRESHOLD', 30.0),
    'fallback_contrast_threshold': get_env_float('FALLBACK_CONTRAST_THRESHOLD', 20.0),
    'fallback_brightness_threshold': get_env_float('FALLBACK_BRIGHTNESS_THRESHOLD', 80.0),
}
