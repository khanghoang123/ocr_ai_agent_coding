from __future__ import annotations

import ast
import py_compile
from pathlib import Path


FRONTEND_PATH = Path(__file__).resolve().parents[1] / "frontend" / "app.py"


def _constant_assignments(path: Path) -> dict[str, ast.AST]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    values: dict[str, ast.AST] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name):
                values[target.id] = node.value
    return values


def test_frontend_app_compiles():
    py_compile.compile(str(FRONTEND_PATH), doraise=True)


def test_frontend_config_constants_exist():
    values = _constant_assignments(FRONTEND_PATH)
    assert "API_BASE" in values
    assert "SUPPORTED_TYPES" in values
    assert "PAGE_TITLE" in values


def test_frontend_uses_dynamic_model_catalog():
    content = FRONTEND_PATH.read_text(encoding="utf-8")
    assert "/models" in content
    assert "/ocr/debug" in content
    assert "tight_bbox" in content
    assert "curve_score" in content
    assert "mask_preview_base64" in content
    assert "rectified_preview_base64" in content
    assert "use_column_width" not in content
