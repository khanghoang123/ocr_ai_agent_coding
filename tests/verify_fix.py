import sys
from pathlib import Path
from PIL import Image, ImageDraw

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ocr_pipeline.detector.paddle_detector import PaddleDetector
from ocr_pipeline.config import settings

def verify():
    test_img_path = "/home/khang/Projects/OCR_project/ocr_ai_agent_coding/tests/tải xuống.jpg"
    output_path = "tests/output/verification_fixed.jpg"
    Path("tests/output").mkdir(parents=True, exist_ok=True)
    
    print("Verifying with optimized settings:")
    print(f"  thresh={settings.det_db_thresh}")
    print(f"  box_thresh={settings.det_db_box_thresh}")
    print(f"  unclip={settings.det_db_unclip_ratio}")
    
    detector = PaddleDetector.from_settings()
    img = Image.open(test_img_path).convert("RGB")
    
    # Run detection
    result = detector.detect(img)
    
    # Visualize
    draw = ImageDraw.Draw(img)
    for poly in result.polygons:
        points = [(p[0], p[1]) for p in poly]
        draw.polygon(points, outline="green", width=4) # Use green for 'fixed'
        
    img.save(output_path)
    print(f"Done. Detected {len(result.polygons)} lines.")
    print(f"Verification image saved to {output_path}")

if __name__ == "__main__":
    verify()
