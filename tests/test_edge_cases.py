"""
Edge-case tests for extraction: garbage files, empty sheets, multi-product
documents, and multi-file payloads with mixed layouts. No API key needed.
"""

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import importlib.util
spec = importlib.util.spec_from_file_location(
    "agent_logic", os.path.join(os.path.dirname(__file__), "..", "main.py")
)
_m = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(_m)
except Exception:
    pass  # wf.init_state / init_ui fail outside the framework runtime

from openpyxl import Workbook


def _xlsx_bytes(rows):
    wb = Workbook()
    ws = wb.active
    for row in rows:
        ws.append(row)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _payload_entry(name, data):
    return {"name": name, "type": "application/vnd.ms-excel", "data": data}


TABULAR = _xlsx_bytes([
    ["product name", "category", "claim", "claim", "claim"],
    ["AcmeGlow LED Desk Lamp", "lighting", "energy-efficient", "high CRI", "long-lasting"],
    ["AcmeBeam Floor Lamp", "lighting", "dimmable", "modern design", "sturdy base"],
])

KEY_VALUE = _xlsx_bytes([
    ["Product Name:", "AcmeTask Clip Lamp"],
    ["Category:", "portable lighting"],
    ["Specifications:", "5W, USB powered, clip mount"],
    ["Target Audience:", "students"],
])

EMPTY = _xlsx_bytes([])
GARBAGE = b"This is not an xlsx file at all, just plain text bytes."


def test_garbage_file():
    try:
        _m.extract_product_data(GARBAGE)
        assert False, "garbage should raise"
    except Exception as e:
        assert not isinstance(e, AssertionError)
    print("PASS garbage single file raises (blueprint shows friendly message)")


def test_empty_sheet():
    try:
        _m.extract_product_data(EMPTY)
        assert False, "empty should raise"
    except ValueError as e:
        assert "no data" in str(e)
    print("PASS empty sheet raises ValueError('...no data.')")


def test_multi_product_single_file():
    data = _m.extract_product_data(TABULAR)
    assert data["layout"] == "tabular"
    assert len(data["products"]) == 2, data
    assert data["products"][0]["product name"] == "AcmeGlow LED Desk Lamp"
    assert data["products"][0]["claim 2"] == "high CRI"  # header dedupe intact
    print("PASS multi-product tabular file -> 2 products, dedupe intact")


def test_multi_file_mixed_layouts():
    payload = [
        _payload_entry("catalog.xlsx", TABULAR),
        _payload_entry("clip_lamp.xlsx", KEY_VALUE),
    ]
    info = _m.extract_products_from_payload(payload)
    assert len(info["products"]) == 3, info
    assert info["files_ok"] == ["catalog.xlsx", "clip_lamp.xlsx"]
    assert info["files_failed"] == []
    labels = [_m.product_label(p, i) for i, p in enumerate(info["products"])]
    assert labels[0] == "AcmeGlow LED Desk Lamp · catalog.xlsx"
    assert labels[2] == "AcmeTask Clip Lamp · clip_lamp.xlsx"
    print(f"PASS 2 files, mixed layouts -> 3 products: {labels}")


def test_multi_file_with_garbage():
    payload = [
        _payload_entry("good.xlsx", KEY_VALUE),
        _payload_entry("broken.pdf", GARBAGE),
        _payload_entry("empty.xlsx", EMPTY),
    ]
    info = _m.extract_products_from_payload(payload)
    assert len(info["products"]) == 1
    assert info["files_ok"] == ["good.xlsx"]
    assert sorted(info["files_failed"]) == ["broken.pdf", "empty.xlsx"]
    print("PASS mixed batch -> good file parsed, 2 bad files skipped and reported")


def test_all_files_bad():
    info = _m.extract_products_from_payload([_payload_entry("junk.txt", GARBAGE)])
    assert info["products"] == []
    assert info["files_failed"] == ["junk.txt"]
    print("PASS all-bad batch -> empty products, failure reported (no crash)")


def test_product_without_name_column():
    anon = _xlsx_bytes([["spec", "value2", "value3"], ["a", "b", "c"]])
    info = _m.extract_products_from_payload([_payload_entry("anon.xlsx", anon)])
    label = _m.product_label(info["products"][0], 0)
    assert label.startswith("Product 1")
    print(f"PASS product with no name column -> fallback label '{label}'")


if __name__ == "__main__":
    test_garbage_file()
    test_empty_sheet()
    test_multi_product_single_file()
    test_multi_file_mixed_layouts()
    test_multi_file_with_garbage()
    test_all_files_bad()
    test_product_without_name_column()
    print("\nALL EDGE CASES PASS")
