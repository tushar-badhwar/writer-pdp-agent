"""
Pipeline test harness.

Offline mode (default): runs the full pipeline with a mock LLM, including a
deliberately over-limit response to exercise the validate -> repair -> truncate
path. No API key needed.

Live mode: set WRITER_API_KEY and run `python tests/test_pipeline.py --live`
to iterate on the actual prompts against palmyra-x5 before pasting them into
Text Generation blocks.
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Import logic functions without triggering wf.init_state at import time
import importlib.util
spec = importlib.util.spec_from_file_location(
    "agent_logic", os.path.join(os.path.dirname(__file__), "..", "main.py")
)
_m = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(_m)
except Exception:
    pass  # wf.init_state may fail outside the framework runtime; functions still bound

SEO_KEYWORDS = (
    "LED Desk Lamp, Energy-efficient Lighting, Adjustable Desk Lamp, "
    "High CRI Lighting, Modern Desk Lamp"
)

SAMPLE_XLSX = os.path.join(
    os.path.dirname(__file__), "..", "sample_data",
    "Product_Information_Document_Sample.xlsx",  # real assignment sample
)


def mock_generate(prompt: str) -> str:
    """Deterministic mock: returns copy with an over-limit title and one
    over-limit bullet so the repair path gets exercised."""
    return json.dumps({
        "title": ("Acme Lumina Pro Modern LED Desk Lamp with Adjustable Arm, 5 Color "
                  "Temperature Modes, Stepless Dimming, USB-C Charging, High CRI Lighting "
                  "and Energy-efficient 12W Design for Home Offices"),  # >150 chars
        "description": ("Work comfortably through long sessions with the Lumina Pro, a modern "
                        "LED desk lamp built for how you actually work. Its fully adjustable arm "
                        "and head direct energy-efficient lighting exactly where you need it, "
                        "while CRI 95+ high CRI lighting keeps colors true for design work and "
                        "reading alike. Five color temperature modes and stepless dimming adapt "
                        "from focused daylight to warm evening light, and the built-in USB-C port "
                        "keeps your devices charged. At just 12W, it delivers the brightness of a "
                        "100W bulb for a fraction of the energy."),
        "bullets": [
            "Fully adjustable desk lamp arm and head position light precisely, reducing glare and eye strain during long work sessions",
            ("High CRI 95+ lighting renders colors accurately for design, photography, and detail work, "
             "with a flicker-free anti-glare diffuser panel that stays comfortable across marathon sessions, "
             "late nights, early mornings and everything in between"),  # >200 chars
            "Energy-efficient 12W LED delivers 100W-equivalent brightness and a 50,000 hour lifespan",
            "Five color temperature modes (2700K-6500K) with stepless dimming and memory function",
            "Built-in USB-C charging port keeps your phone or tablet powered while you work",
        ],
    })


def live_generate(prompt: str) -> str:
    from writerai import Writer
    client = Writer()  # reads WRITER_API_KEY
    resp = client.chat.chat(
        model="palmyra-x5",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content


def run_pipeline(generate_fn, label: str):
    print(f"\n{'='*60}\nPIPELINE RUN: {label}\n{'='*60}")

    with open(SAMPLE_XLSX, "rb") as f:
        data = _m.extract_product_data(f.read())
    product = data["products"][0]
    print(f"[1] Extracted {len(product)} fields (layout={data['layout']})")

    prompt = _m.build_generation_prompt(product, SEO_KEYWORDS)
    raw = generate_fn(prompt)
    copy = _m.parse_copy_output(raw)
    print(f"[2] Generated copy: title={len(copy['title'])} chars, "
          f"{len(copy['bullets'])} bullets")

    attempts = 0
    violations = _m.validate_copy(copy)
    while violations and attempts < _m.MAX_REPAIR_ATTEMPTS:
        attempts += 1
        print(f"[3] Violations found (attempt {attempts}):")
        for v in violations:
            print(f"      - {v[:100]}")
        repair_prompt = _m.build_repair_prompt(copy, violations)
        if generate_fn is mock_generate:
            copy = _m.hard_truncate(copy)  # mock can't actually repair; simulate
        else:
            copy = _m.parse_copy_output(generate_fn(repair_prompt))
        violations = _m.validate_copy(copy)

    if violations:
        print("[4] Repair attempts exhausted -> hard_truncate fallback")
        copy = _m.hard_truncate(copy)
        violations = _m.validate_copy(copy)

    assert not violations, f"Pipeline failed to enforce limits: {violations}"
    print(f"[5] COMPLIANT ✓  title={len(copy['title'])} chars; "
          f"bullets={[len(b) for b in copy['bullets']]}")
    print("\n" + _m.format_copy_markdown(copy))
    return copy


if __name__ == "__main__":
    if "--live" in sys.argv:
        assert os.environ.get("WRITER_API_KEY"), "Set WRITER_API_KEY for live mode"
        run_pipeline(live_generate, "LIVE (palmyra-x5)")
    else:
        run_pipeline(mock_generate, "OFFLINE (mock LLM)")
