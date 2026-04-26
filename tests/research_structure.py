import os
import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

def test_structure(img_path, output_path):
    from paddleocr import PPStructure, draw_structure_result, save_structure_res
    
    # Initialize PPStructure
    # table=False if we only care about layout (titles, paragraphs)
    engine = PPStructure(show_log=False, image_orientation=True, lang='vi', layout=True, table=False)
    
    img = cv2.imread(img_path)
    result = engine(img)
    
    # Analyze results
    print(f"PP-Structure found {len(result)} regions.")
    for i, res in enumerate(result):
        res_type = res['type']
        res_bbox = res['bbox'] # [x1, y1, x2, y2]
        print(f"Region {i}: Type={res_type}, BBox={res_bbox}")
        
        # If it has text content
        if 'res' in res and res['res']:
            print(f"  Contains {len(res['res'])} lines.")
            # Sample first line
            # first_line = res['res'][0]['text']
            # print(f"  Sample Text: {first_line[:50]}...")
    
    # Visualization
    from paddleocr.utils.vis import draw_structure_result
    # result is a list of dicts
    h, w, _ = img.shape
    im_show = draw_structure_result(img, result, font_path=None) # Default font
    im_show = Image.fromarray(im_show)
    im_show.save(output_path)
    print(f"Visualization saved to {output_path}")

if __name__ == "__main__":
    test_img = "/home/khang/Projects/OCR_project/ocr_ai_agent_coding/tests/tải xuống.jpg"
    out_vis = "tests/output/structure_test.jpg"
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    
    try:
        test_structure(test_img, out_vis)
    except Exception as e:
        print(f"PPStructure Test Failed: {e}")
