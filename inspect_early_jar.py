"""Inspect the publisher-supplied jar without executing the game.

Dependency: jawa==2.2.0. Run from this directory or pass a jar path.
The generated report is static bytecode evidence, not a game runtime test.
"""
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

from jawa.cf import ClassFile

root = Path(__file__).resolve().parent
jar = Path(sys.argv[1]) if len(sys.argv) > 1 else root / "artifacts/rd-132211.mojang.jar"
data = jar.read_bytes()
expected = "393e8d4b4d708587e2accd7c5221db65365e1075"
assert hashlib.sha1(data).hexdigest() == expected, "Unexpected source jar"
report = [f"JAR SHA1 {expected}"]
summary = {"sha1": expected, "bytes": len(data), "random_or_clock_calls": []}
with zipfile.ZipFile(io.BytesIO(data)) as archive:
    for filename in archive.namelist():
        if not filename.endswith(".class"):
            continue
        cf = ClassFile(io.BytesIO(archive.read(filename)))
        if filename.endswith("/Level.class"):
            summary["level_fields"] = [(f.name.value, f.descriptor.value) for f in cf.fields]
            report.append("FIELDS " + repr(summary["level_fields"]))
        for method in cf.methods:
            if method.code is None:
                continue
            selected = filename.endswith("/Level.class") and method.name.value in ("<init>", "load", "save")
            if selected:
                report.append(f"\nMETHOD {method.name.value} {method.descriptor.value}")
            for ins in method.code.disassemble():
                details = ""
                if ins.mnemonic.startswith(("invoke", "getfield", "putfield")) or ins.mnemonic in ("new", "ldc", "ldc_w", "ldc2_w"):
                    details = str(cf.constants[ins.operands[0].value])
                if selected:
                    report.append(str(ins) + (" " + details if details else ""))
                if ins.mnemonic.startswith("invoke") and any(term in details for term in ("random", "Random", "currentTimeMillis", "nanoTime")):
                    entry = {"class": filename, "method": method.name.value, "offset": ins.pos, "reference": details}
                    summary["random_or_clock_calls"].append(entry)
                    report.append("RNG/TIME " + json.dumps(entry))
(root / "artifacts/rd-132211.bytecode-analysis.txt").write_text("\n".join(report) + "\n")
(root / "artifacts/rd-132211.analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({"sha1": expected, "bytes": len(data), "random_or_clock_calls": len(summary["random_or_clock_calls"])}))
