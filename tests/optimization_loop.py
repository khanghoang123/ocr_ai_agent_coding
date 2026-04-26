import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_pipeline.detector.paddle_detector import PaddleDetector

def run_optimization_loop(img_path, output_dir):
    # Parameters to sweep (Recall focused)
    # We test combinations of det_db_thresh (T) and det_db_box_thresh (B)
    # with a fixed good unclip_ratio.
    thresh_grid = [0.05, 0.1, 0.15, 0.2]
    box_thresh_grid = [0.1, 0.2, 0.3]
    unclip_grid = [1.6, 2.0]
    
    img_pil_orig = Image.open(img_path).convert("RGB")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    print("-" * 50)
    print(f"{'T (Thresh)':<10} | {'B (BoxTh)':<10} | {'Count':<10} | {'Filename'}")
    print("-" * 50)

    for i, t in enumerate(thresh_grid):
        for j, b in enumerate(box_thresh_grid):
            for k, u in enumerate(unclip_grid):
                # Init detector with sweep params
                detector = PaddleDetector(
                    use_gpu=False,
                    db_thresh=t,
                    db_box_thresh=b,
                    unclip_ratio=u
                )
                
                # Predict
                res = detector.detect(img_pil_orig)
                count = len(res.polygons)
                
                # Visualize
                img_vis = img_pil_orig.copy()
                draw = ImageDraw.Draw(img_vis)
                for poly in res.polygons:
                    points = [(p[0], p[1]) for p in poly]
                    draw.polygon(points, outline="red", width=3)
                
                fname = f"iter_T{t}_B{b}_U{u}.jpg"
                img_vis.save(output_dir / fname)
                
                results.append({"t": t, "b": b, "u": u, "count": count, "fname": fname})
                print(f"{t:<10} | {b:<10} | {u:<10} | {count:<10} | {fname}")

    print("-" * 50)
    print(f"Loop finished. {len(results)} variants saved to {output_dir}")

if __name__ == "__main__":
    img_path = "/home/khang/Projects/OCR_project/ocr_ai_agent_coding/tests/tải xuống.jpg"
    out_dir = Path("tests/output/optimization_loop")
    run_optimization_loop(img_path, out_dir)
