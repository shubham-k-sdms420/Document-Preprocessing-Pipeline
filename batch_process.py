#!/usr/bin/env python3
"""
Batch processor for documents
Process multiple documents in a folder
"""

import os
import cv2
import argparse
from pathlib import Path
from document_preprocessor import ImageEnhancer, PDFProcessor
from tqdm import tqdm
import json
from datetime import datetime

def process_batch(input_dir, output_dir, method='auto', save_metrics=True):
    """
    Process all documents in input_dir
    
    Args:
        input_dir: Directory containing input documents
        output_dir: Directory to save processed documents
        method: Processing method ('auto', 'opencv', 'scikit')
        save_metrics: Save quality metrics to JSON file
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize processors
    enhancer = ImageEnhancer()
    pdf_processor = PDFProcessor()
    
    # Supported extensions
    image_extensions = ['.png', '.jpg', '.jpeg']
    pdf_extensions = ['.pdf']
    all_extensions = image_extensions + pdf_extensions
    
    # Find all documents
    documents = []
    for ext in all_extensions:
        documents.extend(input_path.glob(f'*{ext}'))
        documents.extend(input_path.glob(f'*{ext.upper()}'))
    
    if len(documents) == 0:
        print(f"No documents found in {input_dir}")
        return
    
    print(f"Found {len(documents)} documents to process")
    print(f"Using method: {method}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    # Process documents
    metrics_data = []
    success_count = 0
    error_count = 0
    
    for doc_path in tqdm(documents, desc="Processing documents"):
        try:
            # Load document
            if doc_path.suffix.lower() in pdf_extensions:
                if not pdf_processor.is_pdf_supported():
                    print(f"\nPDF processing not available for: {doc_path.name}")
                    error_count += 1
                    continue
                
                # Convert PDF to image
                images = pdf_processor.pdf_to_images(str(doc_path), dpi=300)
                if len(images) == 0:
                    print(f"\nFailed to extract images from: {doc_path.name}")
                    error_count += 1
                    continue
                image = images[0]  # Process first page
                output_filename = f"{doc_path.stem}_page1_enhanced.png"
            else:
                # Load image
                image = cv2.imread(str(doc_path))
                if image is None:
                    print(f"\nFailed to load: {doc_path.name}")
                    error_count += 1
                    continue
                output_filename = f"{doc_path.stem}_enhanced{doc_path.suffix}"
            
            # Process image
            result = enhancer.process(image, method=method)
            
            # Save enhanced image
            output_file = output_path / output_filename
            cv2.imwrite(str(output_file), result['enhanced_image'])
            
            # Store metrics
            if save_metrics:
                quality_info = result['quality_info']
                metrics_data.append({
                    'filename': doc_path.name,
                    'output_filename': output_filename,
                    'quality_class': quality_info['quality_class'],
                    'blur_score': float(quality_info['blur_score']),
                    'contrast': float(quality_info['contrast']),
                    'brightness': float(quality_info['brightness']),
                    'noise_level': float(quality_info['noise_level']),
                    'method_used': result['method_used'],
                    'processed_at': datetime.now().isoformat()
                })
            
            success_count += 1
            
        except Exception as e:
            print(f"\nError processing {doc_path.name}: {str(e)}")
            error_count += 1
            continue
    
    # Save metrics to JSON
    if save_metrics and len(metrics_data) > 0:
        metrics_file = output_path / 'processing_metrics.json'
        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"\nMetrics saved to: {metrics_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("PROCESSING SUMMARY")
    print("=" * 60)
    print(f"Total documents: {len(documents)}")
    print(f"Successfully processed: {success_count}")
    print(f"Errors: {error_count}")
    
    if save_metrics and len(metrics_data) > 0:
        # Quality distribution
        good_count = sum(1 for m in metrics_data if m['quality_class'] == 'GOOD')
        bad_count = sum(1 for m in metrics_data if m['quality_class'] == 'BAD')
        print(f"\nQuality Distribution:")
        print(f"  Good quality: {good_count}")
        print(f"  Bad quality: {bad_count}")
        
        # Average metrics
        avg_blur = sum(m['blur_score'] for m in metrics_data) / len(metrics_data)
        avg_contrast = sum(m['contrast'] for m in metrics_data) / len(metrics_data)
        avg_brightness = sum(m['brightness'] for m in metrics_data) / len(metrics_data)
        
        print(f"\nAverage Metrics:")
        print(f"  Blur score: {avg_blur:.2f}")
        print(f"  Contrast: {avg_contrast:.2f}")
        print(f"  Brightness: {avg_brightness:.2f}")
    
    print("=" * 60)

def main():
    parser = argparse.ArgumentParser(
        description='Batch process documents',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all documents in 'input' folder
  python batch_process.py -i input -o output
  
  # Use specific method
  python batch_process.py -i input -o output -m opencv
  
  # Don't save metrics
  python batch_process.py -i input -o output --no-metrics
        """
    )
    
    parser.add_argument('-i', '--input', required=True, help='Input directory with documents')
    parser.add_argument('-o', '--output', required=True, help='Output directory for enhanced documents')
    parser.add_argument('-m', '--method', default='auto', 
                        choices=['auto', 'opencv', 'scikit'],
                        help='Processing method (default: auto)')
    parser.add_argument('--no-metrics', action='store_true', 
                        help='Don\'t save quality metrics')
    
    args = parser.parse_args()
    
    # Check if input directory exists
    if not os.path.isdir(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        return 1
    
    # Process batch
    process_batch(
        input_dir=args.input,
        output_dir=args.output,
        method=args.method,
        save_metrics=not args.no_metrics
    )
    
    return 0

if __name__ == '__main__':
    exit(main())
