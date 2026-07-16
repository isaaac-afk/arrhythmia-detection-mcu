"""Capture the board's UART output to run.log.
Waits indefinitely for the first line, then stops after 3 s of silence.
"""
import sys, serial

PORT, BAUD, OUT = "COM3", 115200, "run.log"

with serial.Serial(PORT, BAUD, timeout=None) as s, open(OUT, "w") as f:
    print(f"listening on {PORT} @ {BAUD} -- press RESET on the board now")
    print("(waiting for first line; Ctrl-C to abort)")
    line = s.readline().decode(errors="replace")   # blocks until data
    sys.stdout.write(line)
    f.write(line)
    s.timeout = 3                                   # now detect end-of-run
    while True:
        line = s.readline().decode(errors="replace")
        if not line:
            break
        sys.stdout.write(line)
        f.write(line)
print(f"\nwrote {OUT}")
