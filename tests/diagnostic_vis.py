"""
Diagnostic script to visualize PaddleOCR detection boxes with current settings.
Useful for debugging why detection is inaccurate (merging lines, noise, etc).
"""

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_pipeline.detector.paddle_detector import PaddleDetector
from ocr_pipeline.config import settings

def main():
    # 1. Setup
    detector = PaddleDetector.from_settings()
    
    sample_dir = Path("data/crawled/image")
    output_dir = Path("tests/output/diagnostic")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get 5 samples
    samples = sorted(list(sample_dir.glob("*.jpg")))[:5]
    if not samples:
        print("No samples found in data/crawled/image")
        return

    print(f"Current Settings: det_db_thresh={settings.det_db_thresh}, det_db_box_thresh={settings.det_db_box_thresh}")
    
    for img_path in samples:
        print(f"Processing {img_path.name}...")
        img = Image.open(img_path).convert("RGB")
        
        # Run detection
        result = detector.detect(img)
        
        # Visualize
        draw = ImageDraw.Draw(img)
        for poly in result.polygons:
            # Convert polygon to list of points for ImageDraw
            points = [(p[0], p[1]) for p in poly]
            draw.polygon(points, outline="red", width=3)
            
        # Save
        out_path = output_dir / f"det_{img_path.name}"
        img.save(out_path)
        print(f"  Saved visualization to {out_path}")

if __name__ == "__main__":
    main()
