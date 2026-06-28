"""Verify all .py files parse and import cleanly."""
import ast, sys, importlib, traceback, os, pathlib

ROOT = pathlib.Path(r"C:\Users\bhrgv\OneDrive\Desktop\DAWN\hacathon\car\C-RET_Engine")
sys.path.insert(0, str(ROOT))

py_files = sorted(ROOT.rglob("*.py"))
print("=== AST parse ===")
parse_fail = 0
for p in py_files:
    try:
        ast.parse(p.read_text(encoding="utf-8"), filename=str(p))
        print("  OK  %s" % p.relative_to(ROOT))
    except SyntaxError as e:
        parse_fail += 1
        print("  ERR %s: %s" % (p.relative_to(ROOT), e))
print("Parse failures: %d" % parse_fail)

print()
print("=== Module import ===")
modules = [
    "node3.normalize",
    "node3.models",
    "node3.generate_training_data",
    "node3.train_autoencoder",
    "node3.train_risk_mlp",
    "node3.export_to_onnx",
    "node3.risk_engine",
]
imp_fail = 0
for m in modules:
    try:
        importlib.import_module(m)
        print("  OK  %s" % m)
    except Exception:
        imp_fail += 1
        print("  ERR %s" % m)
        traceback.print_exc()
print("Import failures: %d" % imp_fail)

print()
print("=== Pipeline entrypoint smoke ===")
try:
    spec = importlib.util.spec_from_file_location(
        "main_inference", str(ROOT / "core" / "main_inference.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("  OK  core.main_inference")
    print("       _extract_top_detections = %s" % mod._extract_top_detections.__name__)
except Exception:
    print("  ERR core.main_inference")
    traceback.print_exc()

print()
print("Summary: parse=%s, imports=%s" % (parse_fail == 0, imp_fail == 0))