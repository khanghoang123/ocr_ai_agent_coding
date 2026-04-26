import cv2
from PIL import Image, ImageDraw
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path.cwd() / "src"))
from ocr_pipeline.detector.paddle_detector import PaddleDetector

def apply_clahe(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_chan = clahe.apply(l_chan)
    enhanced = cv2.merge((l_chan, a_chan, b_chan))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def run_tournament():
    test_dir = Path("tests/test")
    output_base = Path("tests/output/tournament")
    output_base.mkdir(parents=True, exist_ok=True)
    
    # Identify real images (skip identifier files)
    image_files = [f for f in test_dir.glob("*.jpg") if ":Zone.Identifier" not in str(f)]
    
    # Initialize shared detector
    # Using very low thresholds to ensure model 'eyes' are wide open
    detector = PaddleDetector(db_thresh=0.08, db_box_thresh=0.1, unclip_ratio=1.6)
    detector._ensure_loaded()
    
    print(f"Starting Tournament on {len(image_files)} images...")
    
    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        
        h, w = img_bgr.shape[:2]
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        results = {}
        
        # --- Method A: Vanilla ---
        raw_a = detector._run_detection(img_bgr)
        results["Method_A_Vanilla"] = ([b[0] for b in raw_a[0]] if raw_a and raw_a[0] else [], "Baseline")
        
        # --- Method B: CLAHE ---
        img_enhanced = apply_clahe(img_bgr)
        raw_b = detector._run_detection(img_enhanced)
        results["Method_B_CLAHE"] = ([b[0] for b in raw_b[0]] if raw_b and raw_b[0] else [], "Enhanced")
        
        # --- Method C: Patch-wise (Top/Bottom) ---
        top_crop = img_bgr[0:int(h*0.6), :]
        bot_crop = img_bgr[int(h*0.4):, :]
        
        raw_c_top = detector._run_detection(top_crop)
        raw_c_bot = detector._run_detection(bot_crop)
        
        boxes_c = []
        if raw_c_top and raw_c_top[0]:
            boxes_c.extend([b[0] for b in raw_c_top[0]])
        if raw_c_bot and raw_c_bot[0]:
            # Shift
            for b in raw_c_bot[0]:
                shifted = [[p[0], p[1] + int(h*0.4)] for p in b[0]]
                boxes_c.append(shifted)
        results["Method_C_Patchwise"] = (boxes_c, "Split")
        
        # --- Method D: Hybrid (CLAHE + Patch) ---
        img_enh_top = apply_clahe(top_crop)
        img_enh_bot = apply_clahe(bot_crop)
        
        raw_d_top = detector._run_detection(img_enh_top)
        raw_d_bot = detector._run_detection(img_enh_bot)
        
        boxes_d = []
        if raw_d_top and raw_d_top[0]:
            boxes_d.extend([b[0] for b in raw_d_top[0]])
        if raw_d_bot and raw_d_bot[0]:
            for b in raw_d_bot[0]:
                shifted = [[p[0], p[1] + int(h*0.4)] for p in b[0]]
                boxes_d.append(shifted)
        results["Method_D_Hybrid"] = (boxes_d, "Full")
        
        # --- Create Comparison Grid ---
        # Draw on 4 separate copies
        vis_imgs = []
        for name, (boxes, label) in results.items():
            tmp = img_pil.copy()
            draw = ImageDraw.Draw(tmp)
            for pts in boxes:
                draw.polygon([(p[0], p[1]) for p in pts], outline="red", width=3)
            # Add Label
            vis_imgs.append(tmp)
            print(f"  {name}: {len(boxes)} lines")

        # Combine into 2x2
        grid_w = w * 2
        grid_h = h * 2
        grid = Image.new("RGB", (grid_w, grid_h))
        grid.paste(vis_imgs[0], (0, 0))
        grid.paste(vis_imgs[1], (w, 0))
        grid.paste(vis_imgs[2], (0, h))
        grid.paste(vis_imgs[3], (w, h))
        
        grid.save(output_base / f"compare_{img_path.stem}.jpg")

    print(f"Tournament results saved to {output_base}")

if __name__ == "__main__":
    run_tournament()
