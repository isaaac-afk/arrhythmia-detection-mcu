"""Prove the UART dump reproduces ecg_samples[] bit-for-bit."""
import re, struct, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
log  = (ROOT / "run.log").read_text(errors="replace")
hdr  = (ROOT / "stage13" / "ecg_data.h").read_text()

body   = log.split("=== DUMP BEGIN")[1].split("=== DUMP END")[0]
hexes  = re.findall(r"^([0-9A-F]{16})\s*$", body, re.M)
dumped = np.array([struct.unpack(">d", bytes.fromhex(h))[0] for h in hexes])

txt  = hdr.split("ecg_samples[ECG_N] = {")[1].split("};")[0]
orig = np.array([float(x) for x in txt.replace("\n", "").split(",") if x.strip()])

print(f"dumped over UART : {len(dumped)}")
print(f"in ecg_data.h    : {len(orig)}")
if len(dumped) != len(orig):
    sys.exit("LENGTH MISMATCH -- dropped bytes on the wire")

same = (dumped.view(np.uint64) == orig.view(np.uint64)).all()
if same:
    print("BIT-EXACT ROUND TRIP -- harness verified")
else:
    bad = np.nonzero(dumped.view(np.uint64) != orig.view(np.uint64))[0]
    print(f"MISMATCH at {len(bad)} indices; first at {bad[0]}: "
          f"{dumped[bad[0]]!r} vs {orig[bad[0]]!r}")
