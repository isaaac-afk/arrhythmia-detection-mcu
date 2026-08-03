"use client";

/* ecg-mcu · real-time arrhythmia monitor -----------------------------------
 * Web Serial dashboard for the STM32 arrhythmia detector.
 *
 * Reads the board's serial stream at 115200 baud and renders a live ECG trace
 * (chart-recorder scroll, right-to-left) plus a heart-rate readout.
 *
 * Wire format (matches firmware app_ecg.c, mode 2 / TIM2-driven replay):
 *   S <int>                         one waveform sample (integer; unit-agnostic)
 *   R-peak @ <idx>   inst BPM <n>    an R-peak with instantaneous BPM
 *   R-peak @ <idx>   (first)         the first R-peak (no interval yet)
 *   # ... / === ... / --- ...        meta / banner lines (ignored)
 *
 * SIGNATURE — live PQRST callout: for the most recently settled beat, we snap
 * P, Q, R, S, T to the real local extrema in the buffered samples around the
 * detected R-peak and label them on the trace. It reads the actual waveform,
 * not fixed offsets — a live version of the textbook ECG-morphology diagram.
 *
 * The trace autoscales Y, so replay (integer microvolts) and live ADC (raw
 * uint16 counts) both render correctly with zero code change.
 *
 * Drop in as app/page.tsx. Run `npm run dev`, open in Chrome or Edge (Web
 * Serial isn't in Firefox/Safari). Only one process can hold the COM port —
 * quit miniterm before connecting.
 * ------------------------------------------------------------------------- */

import { useCallback, useEffect, useRef, useState } from "react";

// ---- stream geometry -------------------------------------------------------
const FS = 360;                 // sample rate (Hz) — matches the firmware
const WINDOW_S = 6;             // seconds of trace on screen
const L = FS * WINDOW_S;        // ring-buffer length in samples
const BAUD = 115200;

// ---- Web Serial types (not in the default TS DOM lib) ----------------------
type SerialPortLike = {
  open: (o: { baudRate: number }) => Promise<void>;
  close: () => Promise<void>;
  readable: ReadableStream<Uint8Array> | null;
};
type SerialLike = { requestPort: () => Promise<SerialPortLike> };
function getSerial(): SerialLike | null {
  if (typeof navigator === "undefined") return null;
  return (navigator as unknown as { serial?: SerialLike }).serial ?? null;
}

type Status = "idle" | "connecting" | "serial" | "sim" | "error";
type Beat = { idx: number };

export default function Page() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  // High-rate data lives in refs so 360 samples/sec never trigger React renders.
  const disp = useRef<Float32Array>(new Float32Array(L));
  const writePos = useRef(0);
  const nSamples = useRef(0);
  const scale = useRef({ min: -1, max: 1 });
  const beats = useRef<Beat[]>([]);
  const lastBeatIdx = useRef<number | null>(null);
  const cssSize = useRef({ w: 0, h: 0 });

  // Low-rate UI state.
  const [status, setStatus] = useState<Status>("idle");
  const [message, setMessage] = useState("Connect the board, or start a simulation.");
  const [bpm, setBpm] = useState<number | null>(null);
  const [peaks, setPeaks] = useState(0);
  const [lastRR, setLastRR] = useState<number | null>(null);
  const [beatKey, setBeatKey] = useState(0);
  const [clock, setClock] = useState(0);
  const [annotate, setAnnotate] = useState(true);
  const [supported, setSupported] = useState(true);

  // Mirror state the render loop needs into refs, so the loop mounts once.
  const statusRef = useRef(status);
  const messageRef = useRef(message);
  const annotateRef = useRef(annotate);
  useEffect(() => void (statusRef.current = status), [status]);
  useEffect(() => void (messageRef.current = message), [message]);
  useEffect(() => void (annotateRef.current = annotate), [annotate]);
  useEffect(() => setSupported(getSerial() !== null), []);

  // ---- data ingest ---------------------------------------------------------
  const pushSample = useCallback((v: number) => {
    disp.current[writePos.current] = v;
    writePos.current = (writePos.current + 1) % L;
    nSamples.current += 1;
  }, []);

  const onBeat = useCallback((idx: number, beatBpm: number | null) => {
    beats.current.push({ idx });
    if (beats.current.length > 16) beats.current.shift();
    setPeaks((p) => p + 1);
    setBeatKey((k) => k + 1);
    if (lastBeatIdx.current !== null && idx > lastBeatIdx.current) {
      setLastRR(Math.round(((idx - lastBeatIdx.current) / FS) * 1000));
    }
    lastBeatIdx.current = idx;
    if (beatBpm !== null) setBpm(beatBpm);
  }, []);

  const parseLine = useCallback(
    (line: string) => {
      const s = line.trim();
      if (!s) return;
      if (s[0] === "S") {
        const m = s.match(/^S\s+(-?\d+)/);
        if (m) pushSample(parseInt(m[1], 10));
        return;
      }
      if (s[0] === "R") {
        let m = s.match(/^R-peak @ (-?\d+)\s+inst BPM (\d+)/);
        if (m) return onBeat(parseInt(m[1], 10), parseInt(m[2], 10));
        m = s.match(/^R-peak @ (-?\d+)\s+\(first\)/);
        if (m) return onBeat(parseInt(m[1], 10), null);
      }
    },
    [pushSample, onBeat]
  );

  const resetStream = useCallback(() => {
    disp.current.fill(0);
    writePos.current = 0;
    nSamples.current = 0;
    beats.current = [];
    lastBeatIdx.current = null;
    scale.current = { min: -1, max: 1 };
    setBpm(null);
    setPeaks(0);
    setLastRR(null);
    setClock(0);
  }, []);

  // ---- Web Serial ----------------------------------------------------------
  const serialAbort = useRef<{ cancel: () => void } | null>(null);
  const portRef = useRef<SerialPortLike | null>(null);
  const simRef = useRef<number | null>(null);

  const stopSim = useCallback(() => {
    if (simRef.current !== null) {
      cancelAnimationFrame(simRef.current);
      simRef.current = null;
    }
  }, []);

  const disconnect = useCallback(async () => {
    stopSim();
    serialAbort.current?.cancel();
    serialAbort.current = null;
    try {
      await portRef.current?.close();
    } catch {
      /* already gone */
    }
    portRef.current = null;
    setStatus("idle");
    setMessage("Disconnected. Connect the board, or start a simulation.");
  }, [stopSim]);

  const connectSerial = useCallback(async () => {
    const serial = getSerial();
    if (!serial) {
      setStatus("error");
      setMessage("This browser can't open serial ports. Open the dashboard in Chrome or Edge.");
      return;
    }
    stopSim();
    setStatus("connecting");
    setMessage("Select the STM32 port in the browser prompt…");
    let port: SerialPortLike;
    try {
      port = await serial.requestPort();
    } catch {
      setStatus("idle");
      setMessage("No port selected. Click Connect and choose the STM32 port.");
      return;
    }
    try {
      await port.open({ baudRate: BAUD });
    } catch (e) {
      setStatus("error");
      const name = e instanceof Error ? e.name : "";
      setMessage(
        name === "InvalidStateError" || name === "NetworkError"
          ? "That port is busy — close miniterm or any other serial tool, then Connect again."
          : "Couldn't open the port. Close other serial tools and retry, or re-seat the USB cable."
      );
      return;
    }

    portRef.current = port;
    resetStream();
    setStatus("serial");
    setMessage("Connected. Waiting for the board — tap RESET (B2) and pick mode 2 if it's idle.");

    const reader = port.readable!.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let alive = true;
    serialAbort.current = {
      cancel: () => {
        alive = false;
        reader.cancel().catch(() => {});
      },
    };

    (async () => {
      try {
        while (alive) {
          const { value, done } = await reader.read();
          if (done) break;
          buf += decoder.decode(value, { stream: true });
          let nl: number;
          while ((nl = buf.indexOf("\n")) >= 0) {
            parseLine(buf.slice(0, nl));
            buf = buf.slice(nl + 1);
          }
          if (buf.length > 4096) buf = buf.slice(-1024);
        }
      } catch {
        if (alive) {
          setStatus("error");
          setMessage("Serial link dropped. Re-seat the cable and Connect again.");
        }
      } finally {
        try {
          reader.releaseLock();
        } catch {
          /* noop */
        }
      }
    })();
  }, [parseLine, resetStream, stopSim]);

  // ---- simulate mode (no hardware) ----------------------------------------
  const startSim = useCallback(() => {
    disconnect().then(() => {
      resetStream();
      setStatus("sim");
      setMessage("Simulating a clean ECG — the same view you'll get from the board.");

      const bump = (x: number, c: number, w: number, a: number) => {
        const z = (x - c) / w;
        return a * Math.exp(-0.5 * z * z);
      };
      // One heartbeat in microvolts, phase p in [0,1). Baseline ≈ -300 µV,
      // R peak ≈ +1150 µV — same ballpark as record 100 so autoscale matches.
      const beatWave = (p: number) => {
        let v = -300;
        v += bump(p, 0.18, 0.028, 130);   // P
        v -= bump(p, 0.40, 0.012, 220);   // Q
        v += bump(p, 0.45, 0.012, 1450);  // R
        v -= bump(p, 0.50, 0.016, 480);   // S
        v += bump(p, 0.68, 0.055, 320);   // T
        return v;
      };

      let phase = 0;
      let rr = FS * (60 / 75);
      let firstDone = false;
      let idx = 0;
      let last = performance.now();
      let acc = 0;

      const tick = (now: number) => {
        const due = ((now - last) / 1000) * FS + acc;
        const n = Math.floor(due);
        acc = due - n;
        last = now;
        for (let k = 0; k < n; k++) {
          const p = phase / rr;
          const noise = (Math.random() - 0.5) * 22;
          pushSample(beatWave(p) + noise);
          const rIndex = Math.round(0.45 * rr);
          if (phase === rIndex) {
            onBeat(idx, firstDone ? Math.round((60 * FS) / rr) : null);
            firstDone = true;
          }
          phase += 1;
          idx += 1;
          if (phase >= rr) {
            phase = 0;
            rr = FS * (60 / (72 + Math.random() * 8));
          }
        }
        simRef.current = requestAnimationFrame(tick);
      };
      simRef.current = requestAnimationFrame(tick);
    });
  }, [disconnect, onBeat, pushSample, resetStream]);

  // ---- elapsed clock (stream time = samples / rate) -----------------------
  useEffect(() => {
    const id = setInterval(() => setClock(Math.floor(nSamples.current / FS)), 250);
    return () => clearInterval(id);
  }, []);

  // ---- render loop ---------------------------------------------------------
  useEffect(() => {
    const canvas = canvasRef.current!;
    const ctx = canvas.getContext("2d")!;
    const reduce =
      typeof matchMedia !== "undefined" &&
      matchMedia("(prefers-reduced-motion: reduce)").matches;

    const ro = new ResizeObserver((entries) => {
      const r = entries[0].contentRect;
      cssSize.current = { w: r.width, h: r.height };
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(r.width * dpr);
      canvas.height = Math.round(r.height * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    });
    ro.observe(canvas.parentElement!);

    // palette (light ECG paper)
    const TRACE = "#D3243A";
    const INK = "#17202A";
    const SLATE = "#5C6B7A";
    const gridMinor = "rgba(211,36,58,0.10)";
    const gridMajor = "rgba(211,36,58,0.22)";

    let raf = 0;
    const draw = () => {
      const { w: W, h: H } = cssSize.current;
      if (W > 0 && H > 0) {
        ctx.clearRect(0, 0, W, H);
        const px = W / (L - 1);

        // --- ECG-paper grid (0.2 s major / 0.04 s minor on X) ---
        const majorX = FS * 0.2 * px;
        const minorX = majorX / 5;
        const rowMinor = H / 25;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let x = 0; x <= W + 0.5; x += minorX) {
          ctx.moveTo(Math.round(x) + 0.5, 0);
          ctx.lineTo(Math.round(x) + 0.5, H);
        }
        for (let y = 0; y <= H + 0.5; y += rowMinor) {
          ctx.moveTo(0, Math.round(y) + 0.5);
          ctx.lineTo(W, Math.round(y) + 0.5);
        }
        ctx.strokeStyle = gridMinor;
        ctx.stroke();
        ctx.beginPath();
        for (let x = 0; x <= W + 0.5; x += majorX) {
          ctx.moveTo(Math.round(x) + 0.5, 0);
          ctx.lineTo(Math.round(x) + 0.5, H);
        }
        for (let y = 0; y <= H + 0.5; y += rowMinor * 5) {
          ctx.moveTo(0, Math.round(y) + 0.5);
          ctx.lineTo(W, Math.round(y) + 0.5);
        }
        ctx.strokeStyle = gridMajor;
        ctx.stroke();

        const filled = Math.min(nSamples.current, L);

        // read a sample `age` steps back from newest (0 = newest)
        const sampleAtAge = (age: number): number | null => {
          if (age < 0 || age >= filled) return null;
          return disp.current[(writePos.current - 1 - age + L * 2) % L];
        };
        const xOfAge = (age: number) => W - age * px;

        if (filled < 3) {
          ctx.fillStyle = SLATE;
          ctx.font = "500 14px 'Roboto', system-ui, sans-serif";
          ctx.textAlign = "center";
          ctx.fillText(
            statusRef.current === "connecting" ? "Waiting for the board…" : messageRef.current,
            W / 2,
            H / 2
          );
          ctx.textAlign = "left";
        } else {
          // --- autoscale toward the visible range (smoothed) ---
          let mn = Infinity,
            mx = -Infinity;
          for (let a = 0; a < filled; a++) {
            const v = sampleAtAge(a)!;
            if (v < mn) mn = v;
            if (v > mx) mx = v;
          }
          if (mx - mn < 1) {
            mx += 1;
            mn -= 1;
          }
          const pad = (mx - mn) * 0.16;
          mn -= pad;
          mx += pad;
          const sc = scale.current;
          const k = reduce ? 1 : 0.12;
          sc.min += (mn - sc.min) * k;
          sc.max += (mx - sc.max) * k;
          const topPad = 14;
          const usableH = H - topPad * 2;
          const yOf = (v: number) =>
            topPad + (1 - (v - sc.min) / (sc.max - sc.min)) * usableH;

          // --- the trace (newest at right, scrolls left) ---
          ctx.strokeStyle = TRACE;
          ctx.lineWidth = 1.9;
          ctx.lineJoin = "round";
          ctx.lineCap = "round";
          ctx.beginPath();
          for (let a = filled - 1; a >= 0; a--) {
            const x = xOfAge(a);
            const y = yOf(sampleAtAge(a)!);
            if (a === filled - 1) ctx.moveTo(x, y);
            else ctx.lineTo(x, y);
          }
          ctx.stroke();

          // --- SIGNATURE: live PQRST callout on the most settled beat ---
          if (annotateRef.current) {
            // pick newest beat whose whole P..T span is on screen
            let chosen: number | null = null;
            for (let i = beats.current.length - 1; i >= 0; i--) {
              const ageR = nSamples.current - beats.current[i].idx;
              if (ageR >= 0.34 * FS && ageR + 0.18 * FS < filled) {
                chosen = ageR;
                break;
              }
            }
            if (chosen !== null) {
              const findExt = (center: number, half: number, wantMax: boolean) => {
                let bestA = center;
                let bestV = wantMax ? -Infinity : Infinity;
                for (let a = center - half; a <= center + half; a++) {
                  const v = sampleAtAge(a);
                  if (v === null) continue;
                  if (wantMax ? v > bestV : v < bestV) {
                    bestV = v;
                    bestA = a;
                  }
                }
                return { a: bestA, v: bestV };
              };
              // snap each fiducial to a real local extremum near textbook offsets
              const R = findExt(chosen, Math.round(0.03 * FS), true);
              const Q = findExt(R.a + Math.round(0.04 * FS), Math.round(0.03 * FS), false);
              const S = findExt(R.a - Math.round(0.04 * FS), Math.round(0.03 * FS), false);
              const P = findExt(R.a + Math.round(0.16 * FS), Math.round(0.05 * FS), true);
              const T = findExt(R.a - Math.round(0.3 * FS), Math.round(0.08 * FS), true);
              const pts = [
                { l: "P", e: P, up: true },
                { l: "Q", e: Q, up: false },
                { l: "R", e: R, up: true },
                { l: "S", e: S, up: false },
                { l: "T", e: T, up: true },
              ];

              // faint band grouping the complex
              const xL = xOfAge(P.a) - 6;
              const xR2 = xOfAge(T.a) + 6;
              ctx.fillStyle = "rgba(19,41,61,0.010)";
              ctx.fillRect(xL, 0, xR2 - xL, H);

              for (const { l, e, up } of pts) {
                const x = xOfAge(e.a);
                const y = yOf(e.v);
                const emph = l === "R";
                ctx.fillStyle = emph ? TRACE : INK;
                ctx.beginPath();
                ctx.arc(x, y, emph ? 3.6 : 2.6, 0, Math.PI * 2);
                ctx.fill();
                ctx.strokeStyle = emph ? "rgba(211,36,58,0.5)" : "rgba(23,32,42,0.28)";
                ctx.lineWidth = 1;
                ctx.beginPath();
                const ly = up ? y - 9 : y + 9;
                ctx.moveTo(x, y);
                ctx.lineTo(x, ly);
                ctx.stroke();
                ctx.fillStyle = emph ? TRACE : INK;
                ctx.font = `700 ${emph ? 13 : 11}px 'Roboto', system-ui, sans-serif`;
                ctx.textAlign = "center";
                ctx.textBaseline = up ? "bottom" : "top";
                ctx.fillText(l, x, up ? ly - 1 : ly + 1);
              }
              ctx.textAlign = "left";
              ctx.textBaseline = "alphabetic";
            }
          }
        }

        // --- time ticks along the bottom (now → −5 s), mode-independent ---
        ctx.fillStyle = SLATE;
        ctx.font = "500 10px 'Roboto', system-ui, sans-serif";
        ctx.textBaseline = "bottom";
        for (let sSec = 0; sSec <= WINDOW_S - 1; sSec++) {
          const x = W - sSec * FS * px;
          ctx.strokeStyle = "rgba(92,107,122,0.3)";
          ctx.lineWidth = 1;
          ctx.beginPath();
          ctx.moveTo(x + 0.5, H - 8);
          ctx.lineTo(x + 0.5, H);
          ctx.stroke();
          ctx.textAlign = sSec === 0 ? "right" : "center";
          ctx.fillText(sSec === 0 ? "now" : `−${sSec}s`, sSec === 0 ? x - 3 : x, H - 9);
        }
        ctx.textAlign = "left";
        ctx.textBaseline = "alphabetic";
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, []);

  useEffect(() => {
    return () => {
      serialAbort.current?.cancel();
      if (simRef.current !== null) cancelAnimationFrame(simRef.current);
      portRef.current?.close().catch(() => {});
    };
  }, []);

  const live = status === "serial" || status === "sim";
  const dot =
    live ? "#2FA36B" : status === "error" ? "#D3243A" : status === "connecting" ? "#C98A12" : "#9AA7B2";
  const statusLabel =
    status === "serial"
      ? "Connected"
      : status === "sim"
      ? "Simulating"
      : status === "connecting"
      ? "Connecting"
      : status === "error"
      ? "Error"
      : "Idle";
  const mm = String(Math.floor(clock / 60)).padStart(2, "0");
  const ss = String(clock % 60).padStart(2, "0");

  return (
    <main className="wrap">
      <div className="device">
        <header className="titlebar">
          <div className="brand">
            <span className="mark">ecg-mcu</span>
            <span className="sub">real-time arrhythmia monitor</span>
          </div>
          <div className="right">
            <span className="tval">{FS} Hz · {BAUD.toLocaleString()} 8N1</span>
            <span className="clock">{mm}:{ss}</span>
            <span className="pill">
              <span className="led" style={{ background: dot }} />
              {statusLabel}
            </span>
          </div>
        </header>

        <div className="charthead">
          <span className="lead">Lead II · single-lead</span>
          <button
            className={`toggle ${annotate ? "on" : ""}`}
            aria-pressed={annotate}
            onClick={() => setAnnotate((a) => !a)}
          >
            <span className="tdot" />
            Annotate P-Q-R-S-T
          </button>
        </div>

        <section className="screen">
          <canvas ref={canvasRef} />
        </section>

        <section className="readout">
          <div className="hr">
            <div className="hrhead">
              <span className="eyebrow">heart rate</span>
              <span key={beatKey} className="heart" style={{ color: live ? TRACE_CSS : "#C9CFD5" }}>
                ♥
              </span>
            </div>
            <div className="bpm">
              <span className="num" style={{ color: bpm !== null ? TRACE_CSS : "#C2C8CE" }}>
                {bpm !== null ? bpm : "––"}
              </span>
              <span className="unit">bpm</span>
            </div>
          </div>
          <div className="metrics">
            <Metric label="beats" value={String(peaks)} />
            <Metric label="last R-R" value={lastRR !== null ? `${lastRR} ms` : "––"} />
            <Metric label="window" value={`${WINDOW_S}.0 s`} />
          </div>
        </section>

        <section className="controls">
          <div className="btns">
            {!live ? (
              <>
                <button
                  className="btn primary"
                  onClick={connectSerial}
                  disabled={status === "connecting"}
                >
                  {status === "connecting" ? "Connecting…" : "Connect board"}
                </button>
                <button className="btn ghost" onClick={startSim}>
                  Simulate
                </button>
              </>
            ) : (
              <button className="btn ghost" onClick={disconnect}>
                {status === "sim" ? "Stop simulation" : "Disconnect"}
              </button>
            )}
          </div>
          <p className="msg">
            {message}
            {!supported && !live ? " · Web Serial needs Chrome or Edge." : ""}
          </p>
        </section>
      </div>

      <style jsx global>{`
        @import url("https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap");

        :root {
          --paper: #f7f1ee;
          --card: #ffffff;
          --ink: #17202a;
          --slate: #5c6b7a;
          --line: #e7e2dd;
          --navy: #13293d;
          --trace: #d3243a;
        }
        * { box-sizing: border-box; }
        html, body { margin: 0; padding: 0; }
        body {
          background: radial-gradient(1200px 600px at 50% -10%, #fdf8f6, var(--paper));
          min-height: 100vh;
          font-family: "Roboto", ui-sans-serif, system-ui, sans-serif;
          color: var(--ink);
          -webkit-font-smoothing: antialiased;
        }
        .wrap {
          max-width: 1120px;
          margin: 0 auto;
          padding: clamp(16px, 3.5vw, 44px);
        }
        .device {
          background: var(--card);
          border: 1px solid var(--line);
          border-radius: 16px;
          overflow: hidden;
          box-shadow: 0 30px 60px -40px rgba(19, 41, 61, 0.5), 0 2px 0 rgba(255, 255, 255, 0.6) inset;
        }

        /* navy instrument title bar */
        .titlebar {
          display: flex;
          align-items: center;
          justify-content: space-between;
          gap: 16px;
          flex-wrap: wrap;
          background: linear-gradient(180deg, #17324a, var(--navy));
          color: #eef4f8;
          padding: 15px clamp(16px, 3vw, 26px);
        }
        .brand { display: flex; align-items: baseline; gap: 12px; }
        .mark {
          font-family: "Roboto", system-ui, sans-serif;
          font-weight: 700;
          font-size: 19px;
          letter-spacing: 0.02em;
        }
        .sub {
          font-size: 11px;
          text-transform: uppercase;
          letter-spacing: 0.22em;
          color: #9db4c6;
        }
        .right { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
        .tval {
          font-family: "Roboto", system-ui, sans-serif;
          font-size: 12px;
          color: #9db4c6;
        }
        .clock {
          font-family: "Roboto", system-ui, sans-serif;
          font-size: 14px;
          color: #d7e5ef;
          font-variant-numeric: tabular-nums;
        }
        .pill {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          font-family: "Roboto", system-ui, sans-serif;
          font-size: 13px;
          padding: 6px 12px;
          border-radius: 999px;
          background: rgba(255, 255, 255, 0.08);
          border: 1px solid rgba(255, 255, 255, 0.14);
        }
        .led { width: 8px; height: 8px; border-radius: 50%; box-shadow: 0 0 7px currentColor; }

        /* chart header strip */
        .charthead {
          display: flex;
          align-items: center;
          justify-content: space-between;
          padding: 12px clamp(16px, 3vw, 26px);
          border-bottom: 1px solid var(--line);
        }
        .lead {
          font-family: "Roboto", system-ui, sans-serif;
          font-size: 12px;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: var(--slate);
        }
        .toggle {
          display: inline-flex;
          align-items: center;
          gap: 8px;
          font-family: "Roboto", sans-serif;
          font-size: 12px;
          font-weight: 500;
          letter-spacing: 0.02em;
          padding: 7px 12px;
          border-radius: 999px;
          border: 1px solid var(--line);
          background: #fff;
          color: var(--slate);
          cursor: pointer;
          transition: all 0.15s ease;
        }
        .toggle .tdot {
          width: 7px; height: 7px; border-radius: 50%;
          background: #c9cfd5; transition: all 0.15s ease;
        }
        .toggle.on { color: var(--navy); border-color: #c3d3df; background: #f2f7fb; }
        .toggle.on .tdot { background: var(--trace); box-shadow: 0 0 6px rgba(211,36,58,0.5); }
        .toggle:focus-visible { outline: 2px solid var(--navy); outline-offset: 2px; }

        /* the screen (ECG paper) */
        .screen {
          position: relative;
          background: #fffdfc;
          height: clamp(280px, 42vh, 420px);
          border-bottom: 1px solid var(--line);
        }
        .screen canvas { display: block; width: 100%; height: 100%; }

        /* readout */
        .readout {
          display: flex;
          align-items: center;
          gap: clamp(20px, 5vw, 64px);
          flex-wrap: wrap;
          padding: 22px clamp(16px, 3vw, 26px);
          border-bottom: 1px solid var(--line);
        }
        .hr { display: flex; flex-direction: column; gap: 2px; }
        .hrhead { display: flex; align-items: center; gap: 9px; }
        .eyebrow {
          font-size: 11px; text-transform: uppercase; letter-spacing: 0.22em; color: var(--slate);
        }
        .heart { font-size: 15px; display: inline-block; animation: beat 0.5s ease-out; }
        @keyframes beat {
          0% { transform: scale(1); }
          22% { transform: scale(1.5); }
          100% { transform: scale(1); }
        }
        .bpm { display: flex; align-items: baseline; gap: 10px; }
        .num {
          font-family: "Roboto", system-ui, sans-serif;
          font-weight: 700;
          font-size: clamp(58px, 10vw, 92px);
          line-height: 0.9;
          font-variant-numeric: tabular-nums;
          letter-spacing: -0.02em;
        }
        .unit { font-family: "Roboto", system-ui, sans-serif; font-size: 15px; color: var(--slate); }
        .metrics { display: flex; gap: clamp(20px, 4vw, 46px); flex-wrap: wrap; }
        .metric .ml {
          font-size: 10px; text-transform: uppercase; letter-spacing: 0.18em; color: var(--slate);
        }
        .metric .mv {
          font-family: "Roboto", system-ui, sans-serif; font-size: 24px; color: var(--ink);
          font-variant-numeric: tabular-nums; margin-top: 2px;
        }

        /* controls */
        .controls {
          display: flex; align-items: center; justify-content: space-between;
          gap: 16px; flex-wrap: wrap;
          padding: 16px clamp(16px, 3vw, 26px);
          background: #fbfaf8;
        }
        .btns { display: flex; gap: 10px; }
        .btn {
          font-family: "Roboto", sans-serif;
          font-size: 14px; font-weight: 500;
          padding: 11px 18px; border-radius: 10px; cursor: pointer;
          border: 1px solid var(--line); background: #fff; color: var(--ink);
          transition: transform 0.08s ease, background 0.15s ease, border-color 0.15s ease;
          white-space: nowrap;
        }
        .btn:active { transform: translateY(1px); }
        .btn:disabled { opacity: 0.55; cursor: default; }
        .btn.primary {
          background: var(--navy); color: #fff; border-color: var(--navy); font-weight: 700;
        }
        .btn.primary:hover:not(:disabled) { background: #1c3c58; }
        .btn.ghost:hover { border-color: #cbd3da; background: #f4f6f8; }
        .btn:focus-visible { outline: 2px solid var(--navy); outline-offset: 2px; }
        .msg {
          margin: 0; font-size: 13px; color: var(--slate);
          font-family: "Roboto", system-ui, sans-serif;
        }

        @media (max-width: 640px) {
          .controls { flex-direction: column; align-items: flex-start; }
        }
        @media (prefers-reduced-motion: reduce) {
          .heart { animation: none; }
        }
      `}</style>
    </main>
  );
}

const TRACE_CSS = "#d3243a";

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric">
      <div className="ml">{label}</div>
      <div className="mv">{value}</div>
    </div>
  );
}
