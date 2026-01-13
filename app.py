"""
Flask Web Application for Document Preprocessing
Optional web interface - not required for core functionality
"""

from flask import Flask, render_template, request, jsonify, send_file
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from document_preprocessor import ImageEnhancer, PDFProcessor
from document_preprocessor.config import PROCESSING_CONFIG

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'outputs'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'pdf'}

# Create necessary directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Initialize processors
enhancer = ImageEnhancer()
pdf_processor = PDFProcessor()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def image_to_base64(image: np.ndarray) -> str:
    """Convert numpy image to base64 string"""
    # Convert to RGB if needed
    if len(image.shape) == 2:
        # Grayscale - convert to RGB mode for better browser compatibility
        pil_img = Image.fromarray(image, mode='L')
        # Convert grayscale to RGB (3 channels) for consistent display
        pil_img = pil_img.convert('RGB')
    else:
        # BGR to RGB
        rgb_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_img)
    
    buffered = BytesIO()
    pil_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"

def convert_to_json_serializable(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    return obj

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, PDF'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Process based on file type
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        # Get preprocessing method from request
        method = request.form.get('method', 'auto')
        
        if file_ext == 'pdf':
            if not pdf_processor.is_pdf_supported():
                return jsonify({'error': 'PDF processing not available. Install pdf2image and poppler-utils.'}), 400
            
            # Convert PDF to images (all pages)
            pdf_dpi = PROCESSING_CONFIG.get('pdf_dpi', 200)
            images = pdf_processor.pdf_to_images(filepath, dpi=pdf_dpi)
            if len(images) == 0:
                return jsonify({'error': 'Failed to extract images from PDF'}), 400
            
            # Process all pages
            processed_pages = []
            output_filenames = []
            
            for page_idx, original_image in enumerate(images):
                # Resize large images
                h, w = original_image.shape[:2]
                max_dimension = PROCESSING_CONFIG.get('max_image_dimension', 2000)
                if max(h, w) > max_dimension:
                    scale = max_dimension / max(h, w)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    original_image = cv2.resize(original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Process image
                result = enhancer.process(original_image, method=method)
                enhanced_image = result['enhanced_image']
                
                # Validate and fix if needed
                if enhanced_image is None or enhanced_image.size == 0:
                    continue
                
                if enhanced_image.dtype != np.uint8:
                    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
                
                img_mean = float(np.mean(enhanced_image))
                if np.all(enhanced_image == 0) or img_mean < 10:
                    if len(original_image.shape) == 3:
                        enhanced_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                    else:
                        enhanced_image = original_image.copy()
                    enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=1.1, beta=15)
                    enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
                
                # Save output
                base_name = filename.replace('.pdf', '')
                output_filename = f"enhanced_{base_name}_page{page_idx + 1}.png"
                output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                cv2.imwrite(output_path, enhanced_image)
                
                processed_pages.append({
                    'page_number': page_idx + 1,
                    'original_image': image_to_base64(original_image),
                    'enhanced_image': image_to_base64(enhanced_image),
                    'output_filename': output_filename,
                    'quality_info': convert_to_json_serializable(result['quality_info'])
                })
                output_filenames.append(output_filename)
            
            if len(processed_pages) == 0:
                return jsonify({'error': 'Failed to process any pages'}), 500
            
            # Prepare response for multi-page PDF
            response = {
                'success': True,
                'is_multi_page': True,
                'total_pages': len(processed_pages),
                'pages': processed_pages,
                'method_used': method,
                'original_filename': filename,
                'output_filenames': output_filenames
            }
            
            return jsonify(response)
        else:
            # Load single image
            original_image = cv2.imread(filepath)
            if original_image is None:
                return jsonify({'error': 'Failed to load image'}), 400
            
            # Resize large images
            h, w = original_image.shape[:2]
            max_dimension = PROCESSING_CONFIG.get('max_image_dimension', 2000)
            if max(h, w) > max_dimension:
                scale = max_dimension / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                original_image = cv2.resize(original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
            # Process image
            result = enhancer.process(original_image, method=method)
            enhanced_image = result['enhanced_image']
            
            # Validate image before saving
            if enhanced_image is None or enhanced_image.size == 0:
                return jsonify({'error': 'Image processing failed - got empty result'}), 500
            
            # Ensure image is uint8
            if enhanced_image.dtype != np.uint8:
                enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
            
            # Debug: Check image statistics
            img_mean = float(np.mean(enhanced_image))
            img_min = int(np.min(enhanced_image))
            img_max = int(np.max(enhanced_image))
            
            # Check if image is all black or too dark
            if np.all(enhanced_image == 0) or img_mean < 10:
                # Try fallback: use original with simple enhancement
                if len(original_image.shape) == 3:
                    enhanced_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                else:
                    enhanced_image = original_image.copy()
                
                # Apply simple enhancement
                enhanced_image = cv2.convertScaleAbs(enhanced_image, alpha=1.1, beta=15)
                enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
                
                print(f"Warning: Processing resulted in dark image (mean={img_mean}), using fallback enhancement")
            
            # Save output
            output_filename = f"enhanced_{filename.replace('.pdf', '.png')}"
            output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
            cv2.imwrite(output_path, enhanced_image)
            
            # Convert quality_info to JSON-serializable format
            quality_info = convert_to_json_serializable(result['quality_info'])
            
            # Debug: Print image stats before conversion
            print(f"Enhanced image stats - Shape: {enhanced_image.shape}, Mean: {np.mean(enhanced_image):.2f}, Min: {np.min(enhanced_image)}, Max: {np.max(enhanced_image)}")
            
            # Prepare response
            response = {
                'success': True,
                'is_multi_page': False,
                'original_image': image_to_base64(original_image),
                'enhanced_image': image_to_base64(enhanced_image),
                'quality_info': quality_info,
                'method_used': result['method_used'],
                'original_filename': filename,
                'output_filename': output_filename,
                'debug_info': {
                    'enhanced_mean': float(np.mean(enhanced_image)),
                    'enhanced_min': int(np.min(enhanced_image)),
                    'enhanced_max': int(np.max(enhanced_image))
                }
            }
            
            return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    try:
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 404

@app.route('/compare', methods=['POST'])
def compare_images():
    """Compare multiple preprocessing methods"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Load image
        file_ext = filename.rsplit('.', 1)[1].lower()
        
        if file_ext == 'pdf':
            if not pdf_processor.is_pdf_supported():
                return jsonify({'error': 'PDF processing not available'}), 400
            images = pdf_processor.pdf_to_images(filepath)
            original_image = images[0]
        else:
            original_image = cv2.imread(filepath)
        
        # Process with different methods
        methods = ['auto', 'opencv', 'scikit']
        results = {}
        
        for method in methods:
            result = enhancer.process(original_image.copy(), method=method)
            results[method] = {
                'image': image_to_base64(result['enhanced_image']),
                'quality_info': convert_to_json_serializable(result['quality_info'])
            }
        
        response = {
            'success': True,
            'original_image': image_to_base64(original_image),
            'results': results
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/bulk_upload', methods=['POST'])
def bulk_upload():
    """Process multiple documents in bulk"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        
        if len(files) == 0:
            return jsonify({'error': 'No files selected'}), 400
        
        # Get preprocessing method
        method = request.form.get('method', 'auto')
        
        results = []
        errors = []
        
        for idx, file in enumerate(files):
            try:
                if file.filename == '':
                    continue
                
                if not allowed_file(file.filename):
                    errors.append({
                        'filename': file.filename,
                        'error': 'Invalid file type'
                    })
                    continue
                
                # Save uploaded file
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                unique_filename = f"{timestamp}_{idx}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
                file.save(filepath)
                
                # Process based on file type
                file_ext = unique_filename.rsplit('.', 1)[1].lower()
                
                if file_ext == 'pdf':
                    if not pdf_processor.is_pdf_supported():
                        errors.append({
                            'filename': file.filename,
                            'error': 'PDF processing not available'
                        })
                        continue
                    
                    # Process all pages of PDF
                    pdf_dpi = PROCESSING_CONFIG.get('pdf_dpi', 200)
                    images = pdf_processor.pdf_to_images(filepath, dpi=pdf_dpi)
                    if len(images) == 0:
                        errors.append({
                            'filename': file.filename,
                            'error': 'Failed to extract images from PDF'
                        })
                        continue
                    
                    processed_pages = []
                    output_filenames = []
                    
                    for page_idx, original_image in enumerate(images):
                        # Resize large images
                        h, w = original_image.shape[:2]
                        max_dimension = PROCESSING_CONFIG.get('max_image_dimension', 2000)
                        if max(h, w) > max_dimension:
                            scale = max_dimension / max(h, w)
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                            original_image = cv2.resize(original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        
                        # Process image
                        result = enhancer.process(original_image, method=method)
                        enhanced_image = result['enhanced_image']
                        
                        # Validate enhanced image
                        if enhanced_image is None or enhanced_image.size == 0:
                            continue
                        
                        if enhanced_image.dtype != np.uint8:
                            enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
                        
                        if np.all(enhanced_image == 0):
                            continue
                        
                        # Save output
                        base_name = unique_filename.replace('.pdf', '')
                        output_filename = f"enhanced_{base_name}_page{page_idx + 1}.png"
                        output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                        cv2.imwrite(output_path, enhanced_image)
                        
                        processed_pages.append({
                            'page_number': page_idx + 1,
                            'preview_image': image_to_base64(enhanced_image),
                            'output_filename': output_filename,
                            'quality_info': convert_to_json_serializable(result['quality_info'])
                        })
                        output_filenames.append(output_filename)
                    
                    if len(processed_pages) == 0:
                        errors.append({
                            'filename': file.filename,
                            'error': 'Failed to process any pages'
                        })
                        continue
                    
                    results.append({
                        'original_filename': file.filename,
                        'is_multi_page': True,
                        'total_pages': len(processed_pages),
                        'pages': processed_pages,
                        'output_filenames': output_filenames,
                        'method_used': method,
                        'preview_image': processed_pages[0]['preview_image']  # First page for preview
                    })
                else:
                    # Process single image
                    original_image = cv2.imread(filepath)
                    if original_image is None:
                        errors.append({
                            'filename': file.filename,
                            'error': 'Failed to load image'
                        })
                        continue
                    
                    # Resize large images
                    h, w = original_image.shape[:2]
                    max_dimension = PROCESSING_CONFIG.get('max_image_dimension', 2000)
                    if max(h, w) > max_dimension:
                        scale = max_dimension / max(h, w)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        original_image = cv2.resize(original_image, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
                    # Process image
                    result = enhancer.process(original_image, method=method)
                    enhanced_image = result['enhanced_image']
                    
                    # Validate enhanced image
                    if enhanced_image is None or enhanced_image.size == 0:
                        errors.append({
                            'filename': file.filename,
                            'error': 'Processing failed - got empty result'
                        })
                        continue
                    
                    if enhanced_image.dtype != np.uint8:
                        enhanced_image = np.clip(enhanced_image, 0, 255).astype(np.uint8)
                    
                    if np.all(enhanced_image == 0):
                        errors.append({
                            'filename': file.filename,
                            'error': 'Processing resulted in black image'
                        })
                        continue
                    
                    # Save output
                    output_filename = f"enhanced_{unique_filename.replace('.pdf', '.png')}"
                    output_path = os.path.join(app.config['OUTPUT_FOLDER'], output_filename)
                    cv2.imwrite(output_path, enhanced_image)
                    
                    # Convert quality_info to JSON-serializable
                    quality_info = convert_to_json_serializable(result['quality_info'])
                    
                    results.append({
                        'original_filename': file.filename,
                        'is_multi_page': False,
                        'output_filename': output_filename,
                        'quality_info': quality_info,
                        'method_used': result['method_used'],
                        'preview_image': image_to_base64(enhanced_image)
                    })
                
            except Exception as e:
                errors.append({
                    'filename': file.filename,
                    'error': str(e)
                })
                continue
        
        return jsonify({
            'success': True,
            'processed': len(results),
            'errors': len(errors),
            'results': results,
            'error_details': errors
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/bulk_download', methods=['POST'])
def bulk_download():
    """Create a zip file of processed documents"""
    try:
        import zipfile
        import tempfile
        
        data = request.get_json()
        filenames = data.get('filenames', [])
        
        if len(filenames) == 0:
            return jsonify({'error': 'No files specified'}), 400
        
        # Create temporary zip file
        temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
        temp_zip.close()
        
        with zipfile.ZipFile(temp_zip.name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename in filenames:
                filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
                if os.path.exists(filepath):
                    zipf.write(filepath, filename)
        
        return send_file(
            temp_zip.name,
            as_attachment=True,
            download_name='enhanced_documents.zip',
            mimetype='application/zip'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
