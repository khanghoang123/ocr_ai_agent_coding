import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def run_sweep(img_path, output_dir):
    from paddleocr import PaddleOCR
    
    # Image loading
    img_pil = Image.open(img_path).convert("RGB")
    img_np = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    # Parameters to test
    thresh_list = [0.3, 0.5, 0.6]
    box_thresh_list = [0.3, 0.5, 0.7]
    unclip_ratio_list = [1.5, 1.8, 2.0]
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_log = []
    
    for thresh in thresh_list:
        for b_thresh in box_thresh_list:
            for unclip in unclip_ratio_list:
                print(f"Testing: thresh={thresh}, box_thresh={b_thresh}, unclip={unclip}")
                
                # We need to re-init PaddleOCR or use its internal parameters if possible. 
                # For simplicity in sweep, we re-init (though slow).
                ocr = PaddleOCR(
                    use_det=True, use_rec=True, use_gpu=False,
                    det_db_thresh=thresh,
                    det_db_box_thresh=b_thresh,
                    det_db_unclip_ratio=unclip,
                    show_log=False
                )
                
                res = ocr.ocr(img_np, det=True, rec=True, cls=False)
                
                # Draw
                vis_img = img_pil.copy()
                draw = ImageDraw.Draw(vis_img)
                
                count = 0
                if res is not None and len(res) > 0 and res[0] is not None:
                    for item in res[0]:
                        pts = item[0]
                        points = [(p[0], p[1]) for p in pts]
                        draw.polygon(points, outline="red", width=3)
                        count += 1
                
                fname = f"t{thresh}_b{b_thresh}_u{unclip}.jpg"
                vis_img.save(output_dir / fname)
                results_log.append({"params": (thresh, b_thresh, unclip), "count": count, "file": fname})

    print("\nSweep Complete. Results:")
    for r in results_log:
        print(f"Params {r['params']}: {r['count']} boxes -> {r['file']}")

if __name__ == "__main__":
    test_img = "/home/khang/Projects/OCR_project/ocr_ai_agent_coding/tests/tải xuống.jpg"
    out_dir = Path("tests/output/sweep_results")
    run_sweep(test_img, out_dir)
