"""
Streamlit frontend for Vietnamese Handwritten OCR.

Features:
  - Upload one or multiple files (JPG, PNG, PDF)
  - Fetch available trained models from the backend
  - Run standard OCR or single-file debug OCR
  - Display OCR results with bounding boxes highlighted
  - Visualize raw vs refined detections for debugging
  - Download result as txt, json, or zip (for multiple files)
"""

from __future__ import annotations

import base64
import io
import json
import time

import requests
import streamlit as st
from PIL import Image, ImageDraw

# ── Config ────────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"
SUPPORTED_TYPES = ["jpg", "jpeg", "png", "pdf"]
PAGE_TITLE = "Vietnamese Handwritten OCR"


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .result-text {
        background: #f8fafc;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #e2e8f0;
        font-family: 'Courier New', monospace;
        font-size: 0.9rem;
        white-space: pre-wrap;
        max-height: 400px;
        overflow-y: auto;
    }
</style>
""",
    unsafe_allow_html=True,
)


# ── API helpers ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=30)
def get_api_health():
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        return response.json() if response.ok else None
    except Exception:
        return None


@st.cache_data(ttl=30)
def get_model_catalog():
    try:
        response = requests.get(f"{API_BASE}/models", timeout=5)
        return response.json() if response.ok else None
    except Exception:
        return None


def decode_preview(preview_base64: str) -> bytes:
    return base64.b64decode(preview_base64.encode("ascii"))


def _polygon_to_bbox(polygon: list[list[float]]) -> tuple[float, float, float, float]:
    if not polygon:
        return 0.0, 0.0, 0.0, 0.0
    xs = [pt[0] for pt in polygon]
    ys = [pt[1] for pt in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def _format_bbox(bbox: dict | tuple[float, float, float, float] | None) -> str:
    if bbox is None:
        return "n/a"
    if isinstance(bbox, dict):
        values = (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"])
    else:
        values = bbox
    return ", ".join(f"{value:.1f}" for value in values)


def draw_debug_overlay(image: Image.Image, page_debug: dict) -> Image.Image:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay)

    for polygon in page_debug.get("raw_polygons", []):
        draw.polygon([(pt[0], pt[1]) for pt in polygon], outline="#f59e0b", width=3)

    for bbox in page_debug.get("refined_boxes", []):
        draw.rectangle(
            [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]],
            outline="#2563eb",
            width=2,
        )

    for decision in page_debug.get("decisions", []):
        projection_bbox = decision.get("projection_bbox")
        if projection_bbox:
            draw.rectangle(
                [
                    projection_bbox["x1"],
                    projection_bbox["y1"],
                    projection_bbox["x2"],
                    projection_bbox["y2"],
                ],
                outline="#8b5cf6",
                width=2,
            )
        tight_bbox = decision.get("tight_bbox")
        if tight_bbox:
            draw.rectangle(
                [tight_bbox["x1"], tight_bbox["y1"], tight_bbox["x2"], tight_bbox["y2"]],
                outline="#10b981",
                width=2,
            )
        clamped_bbox = decision.get("clamped_bbox")
        if clamped_bbox:
            draw.rectangle(
                [clamped_bbox["x1"], clamped_bbox["y1"], clamped_bbox["x2"], clamped_bbox["y2"]],
                outline="#ec4899",
                width=2,
            )

    return overlay


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">📝 Vietnamese Handwritten OCR</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-title">Upload handwritten Vietnamese text images or PDFs '
    "to compare trained checkpoints and inspect line-segmentation quality.</div>",
    unsafe_allow_html=True,
)

health = get_api_health()
model_catalog = get_model_catalog()

if health:
    st.success(
        f"✅ API Online — Model: **{health['model_loaded']}** | "
        f"Device: **{health['device']}** | GPU: **{health['gpu_available']}**"
    )
else:
    st.error(
        "❌ Cannot connect to OCR API. "
        f"Make sure it is running at `{API_BASE}`. "
        "Run: `uvicorn app.main:app --host 0.0.0.0 --port 8000`"
    )

if not model_catalog or not model_catalog.get("models"):
    st.error("❌ Cannot load the model catalog from the OCR API.")
    st.stop()

model_options = model_catalog["models"]
model_lookup = {item["key"]: item for item in model_options}
default_model_key = model_catalog.get("default_model", model_options[0]["key"])
default_index = next(
    (idx for idx, item in enumerate(model_options) if item["key"] == default_model_key),
    0,
)

st.divider()


# ── Sidebar settings ──────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.selectbox(
        "Model",
        options=[item["key"] for item in model_options],
        index=default_index,
        format_func=lambda key: model_lookup[key]["display_name"],
        help="Choose one of the trained OCR checkpoints exposed by the backend.",
    )

    export_fmt = st.selectbox(
        "Export Format",
        options=["txt", "json", "pdf"],
        index=0,
        help=(
            "txt: plain text. "
            "json: structured with bboxes and confidence. "
            "pdf: searchable PDF with the original image as background "
            "and per-line text overlaid at the detected coordinates."
        ),
    )

    show_bboxes = st.toggle("Show bounding boxes", value=True)
    show_confidence = st.toggle("Show confidence scores", value=False)
    debug_view = st.toggle(
        "Debug line refinement",
        value=False,
        help="Single-file mode only. Shows raw detections, refined boxes, and final crops.",
    )
    enable_postprocess = st.toggle(
        "Post-process (SymSpell + BARTpho)",
        value=False,
        help="Apply two post-processing methods and display both outputs.",
    )

    st.divider()
    st.markdown("**Model info**")
    selected_model = model_lookup[model_choice]
    st.markdown(f"- {selected_model['display_name']}")
    if selected_model.get("cer") is not None and selected_model.get("wer") is not None:
        st.markdown(
            f"- CER: **{selected_model['cer']:.2f}%** | WER: **{selected_model['wer']:.2f}%**"
        )
    if selected_model.get("exact_match") is not None:
        st.markdown(f"- Exact match: **{selected_model['exact_match']:.2f}%**")
    if selected_model.get("training_iters") is not None:
        st.markdown(f"- Training: **{selected_model['training_iters']:,} iters**")
    st.caption(selected_model.get("description", ""))


# ── File upload ───────────────────────────────────────────────────────────────
uploaded_files = st.file_uploader(
    "📁 Upload files",
    type=SUPPORTED_TYPES,
    accept_multiple_files=True,
    help="Upload one or more JPG, PNG, or PDF files.",
)

if not uploaded_files:
    st.info("👆 Upload one or more handwritten Vietnamese image files to get started.")
    st.stop()

if debug_view and len(uploaded_files) != 1:
    st.warning("Debug line refinement only supports exactly one uploaded file.")
    st.stop()


# ── Preview ───────────────────────────────────────────────────────────────────
st.subheader(f"📎 {len(uploaded_files)} file(s) ready")

preview_cols = st.columns(min(len(uploaded_files), 4))
for idx, file in enumerate(uploaded_files):
    with preview_cols[idx % 4]:
        if file.type and file.type.startswith("image"):
            image = Image.open(io.BytesIO(file.read()))
            file.seek(0)
            st.image(image, caption=file.name, width=320)
        else:
            st.markdown(f"📄 `{file.name}` (PDF)")

st.divider()


# ── Run OCR ───────────────────────────────────────────────────────────────────
run_col, _ = st.columns([1, 4])
with run_col:
    run_ocr = st.button("🚀 Run OCR", type="primary", width="stretch")

if not run_ocr:
    st.stop()

if not health:
    st.error("API is not available. Cannot run OCR.")
    st.stop()


# ── Send to API ───────────────────────────────────────────────────────────────
with st.spinner("Running OCR pipeline..."):
    t0 = time.time()
    files_payload = [
        ("files", (file.name, file.read(), file.type or "application/octet-stream"))
        for file in uploaded_files
    ]
    for file in uploaded_files:
        file.seek(0)

    endpoint = "/ocr/debug" if debug_view else "/ocr"
    payload = {"model": model_choice}
    if not debug_view:
        payload["postprocess"] = str(enable_postprocess).lower()

    try:
        response = requests.post(
            f"{API_BASE}{endpoint}",
            files=files_payload,
            data=payload,
            timeout=120,
        )
        elapsed = time.time() - t0

        if not response.ok:
            st.error(f"API error {response.status_code}: {response.text}")
            st.stop()

        api_data = response.json()
    except requests.exceptions.Timeout:
        st.error("Request timed out. Try with smaller files or fewer pages.")
        st.stop()
    except Exception as exc:
        st.error(f"Connection failed: {exc}")
        st.stop()


# ── Display results ───────────────────────────────────────────────────────────
results = api_data.get("results", [])
debug_results = api_data.get("debug_results", [])

st.success(f"✅ OCR complete — {len(results)} file(s) processed in {elapsed:.1f}s")


def _line_count(page: dict) -> int:
    return len(page.get("lines", []))


total_lines = sum(sum(_line_count(page) for page in result.get("pages", [])) for result in results)
total_pages = sum(result["total_pages"] for result in results)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Files", len(results))
m2.metric("Pages", total_pages)
m3.metric("Lines detected", total_lines)
m4.metric("Processing time", f"{elapsed:.1f}s")

st.divider()

for result in results:
    line_count = sum(_line_count(page) for page in result.get("pages", []))
    with st.expander(
        f"📄 {result['filename']} — {result['total_pages']} page(s), {line_count} lines",
        expanded=True,
    ):
        for page in result["pages"]:
            if result["total_pages"] > 1:
                st.markdown(f"**Page {page['page_number']}**")

            col_text, col_img = st.columns([1, 1])

            with col_text:
                st.markdown("**Extracted text:**")
                page_text = page.get("full_text")
                if page_text is None:
                    page_text = "\n".join(line.get("text", "") for line in page.get("lines", []))
                st.markdown(
                    f'<div class="result-text">{page_text}</div>',
                    unsafe_allow_html=True,
                )

                postprocessed = page.get("postprocessed", {})
                if enable_postprocess and postprocessed:
                    st.markdown("**Post-processed outputs:**")
                    pp_left, pp_right = st.columns(2)
                    with pp_left:
                        st.markdown("**SymSpell**")
                        st.markdown(
                            f'<div class="result-text">{postprocessed.get("symspell", "")}</div>',
                            unsafe_allow_html=True,
                        )
                    with pp_right:
                        st.markdown("**BARTpho**")
                        st.markdown(
                            f'<div class="result-text">{postprocessed.get("bartpho", "")}</div>',
                            unsafe_allow_html=True,
                        )

                if show_confidence and page["lines"]:
                    st.markdown("**Line details:**")
                    for line in page["lines"][:10]:
                        st.markdown(f"`[{line['confidence']:.2f}]` {line['text']}")
                    if len(page["lines"]) > 10:
                        st.caption(f"... and {len(page['lines']) - 10} more lines")

            with col_img:
                matching = next((file for file in uploaded_files if file.name == result["filename"]), None)
                if matching and matching.type and matching.type.startswith("image"):
                    matching.seek(0)
                    image = Image.open(io.BytesIO(matching.read())).convert("RGB")
                    matching.seek(0)

                    if show_bboxes and page["lines"]:
                        draw = ImageDraw.Draw(image)
                        for line in page["lines"]:
                            bbox = line["bbox"]
                            draw.rectangle(
                                [bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]],
                                outline="#ef4444",
                                width=2,
                            )
                    st.image(image, caption=f"Page {page['page_number']}", width=520)

if debug_view and debug_results:
    st.divider()
    st.subheader("🧪 Line Refinement Debug")

    debug_item = debug_results[0]
    source_file = uploaded_files[0]
    source_file.seek(0)
    source_image = Image.open(io.BytesIO(source_file.read())).convert("RGB")
    source_file.seek(0)

    for page_debug in debug_item.get("debug_pages", []):
        st.markdown(f"**Page {page_debug['page_number']}**")
        overlay = draw_debug_overlay(source_image, page_debug)

        col_overlay, col_decisions = st.columns([1.2, 1.0])
        with col_overlay:
            st.image(
                overlay,
                caption="Orange: raw polygons | Blue: final line boxes | Purple: projection | Green: tight | Pink: clamped",
                width=700,
            )
        with col_decisions:
            st.markdown("**Refinement decisions**")
            for decision in page_debug.get("decisions", [])[:12]:
                stage = decision.get("decision_stage", "full_page")
                curve_score = decision.get("curve_score", 0.0)
                fallback_reason = decision.get("fallback_reason")
                note = decision.get("note", "")
                raw_bbox = _polygon_to_bbox(decision.get("raw_polygon", []))
                tight_bbox = decision.get("tight_bbox")
                projection_bbox = decision.get("projection_bbox")
                clamped_bbox = decision.get("clamped_bbox")
                suffix = (
                    f" | stage={stage} | curve={curve_score:.3f}"
                    f" | notebook={decision.get('notebook_mode', False)}"
                )
                if fallback_reason:
                    suffix += f" | fallback={fallback_reason}"
                st.markdown(f"- `#{decision['source_index']}` {decision['action']}: {note}{suffix}")
                st.caption(
                    f"raw_bbox=({_format_bbox(raw_bbox)}) | "
                    f"projection_bbox=({_format_bbox(projection_bbox)}) | "
                    f"tight_bbox=({_format_bbox(tight_bbox)}) | "
                    f"clamped_bbox=({_format_bbox(clamped_bbox)}) | "
                    f"refined_boxes={len(decision.get('refined_boxes', []))}"
                )
                st.caption(
                    f"neighbor={decision.get('neighbor_strategy') or 'n/a'} | "
                    f"projection_threshold={decision.get('projection_threshold', 0.0):.2f} | "
                    f"ruled_line_mode={decision.get('ruled_line_mode', False)}"
                )
                if decision.get("original_text"):
                    st.caption(
                        f"Original score {decision.get('original_score', 0.0):.2f}: "
                        f"{decision['original_text']}"
                    )
                if decision.get("split_texts"):
                    split_summary = " | ".join(
                        f"{text} ({score:.2f})"
                        for text, score in zip(
                            decision.get("split_texts", []),
                            decision.get("split_scores", []),
                        )
                    )
                    st.caption(split_summary)
                preview_cols = st.columns(2)
                with preview_cols[0]:
                    if decision.get("mask_preview_base64"):
                        st.image(
                            decode_preview(decision["mask_preview_base64"]),
                            caption="Mask preview",
                            width="stretch",
                        )
                with preview_cols[1]:
                    if decision.get("rectified_preview_base64"):
                        st.image(
                            decode_preview(decision["rectified_preview_base64"]),
                            caption="Rectified preview",
                            width="stretch",
                        )

        final_crops = page_debug.get("final_crops", [])
        if final_crops:
            st.markdown("**Final crops**")
            crop_cols = st.columns(3)
            for idx, crop in enumerate(final_crops):
                with crop_cols[idx % 3]:
                    st.image(
                        decode_preview(crop["preview_base64"]),
                        caption=f"Line {crop['line_index']}",
                        width="stretch",
                    )
                    st.caption(
                        f"rec={crop.get('recognition_score', 0.0):.2f} | "
                        f"det={crop['detection_confidence']:.2f} | "
                        f"decode={crop.get('decode_mode', 'n/a')}"
                    )
                    st.caption(
                        f"ratio {crop.get('pre_norm_ratio', 0.0):.2f}"
                        f"→{crop.get('post_norm_ratio', 0.0):.2f} | "
                        f"baseline_conf={crop.get('baseline_confidence', 0.0):.2f} | "
                        f"offset={crop.get('baseline_offset', 0.0):.1f}"
                    )
                    if crop.get("normalization_preview_base64"):
                        st.image(
                            decode_preview(crop["normalization_preview_base64"]),
                            caption="Normalized preview",
                            width="stretch",
                        )
                    st.markdown(crop["text"])


# ── Download ──────────────────────────────────────────────────────────────────
st.divider()
st.subheader("⬇️ Download Results")

dl_col1, dl_col2 = st.columns(2)

with dl_col1:
    for file in uploaded_files:
        file.seek(0)
    export_files = [
        ("files", (file.name, file.read(), file.type or "application/octet-stream"))
        for file in uploaded_files
    ]
    try:
        export_response = requests.post(
            f"{API_BASE}/ocr/export",
            files=export_files,
            data={"fmt": export_fmt, "model": model_choice},
            timeout=120,
        )
        if export_response.ok:
            ext = "zip" if len(uploaded_files) > 1 else export_fmt
            dl_filename = (
                "ocr_results.zip"
                if len(uploaded_files) > 1
                else f"ocr_result.{export_fmt}"
            )
            mime_by_ext = {
                "zip": "application/zip",
                "txt": "text/plain",
                "json": "application/json",
                "pdf": "application/pdf",
            }
            st.download_button(
                label=f"⬇️ Download as .{ext}",
                data=export_response.content,
                file_name=dl_filename,
                mime=mime_by_ext.get(ext, "application/octet-stream"),
                width="stretch",
            )
    except Exception as exc:
        st.warning(f"Export failed: {exc}")

with dl_col2:
    st.download_button(
        label="⬇️ Download full API response (.json)",
        data=json.dumps(api_data, ensure_ascii=False, indent=2).encode("utf-8"),
        file_name="ocr_api_response.json",
        mime="application/json",
        width="stretch",
    )
