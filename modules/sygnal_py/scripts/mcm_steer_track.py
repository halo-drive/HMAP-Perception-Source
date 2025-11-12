#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, time, csv, math, argparse, queue, threading, signal, json
from dataclasses import dataclass
from typing import List, Tuple, Optional
import can
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ========================== USER CONFIG (EDIT THESE) ==========================
CONFIG = dict(
    # Map your rig: MCM control+heartbeat on one bus, SAS on the other
    TX_CHANNEL="can2",        # where MCM lives (enable + command + heartbeat)
    ANGLE_CHANNEL="can3",     # where SAS11 is

    # --- MCM Heartbeat (from Heartbeat.dbc you posted) ---
    HEARTBEAT_ID=0x170,       # 368 decimal

    # --- MCM Control-Enable message ---
    # If your platform requires a seed/key handshake, do that outside this test.
    # Here we just send an "interface enable" frame at 10 Hz until MCM reports MCM Control.
    ENABLE_ID=0x180,          # <<< PUT YOUR ACTUAL CONTROL-ENABLE FRAME ID
    ENABLE_PAYLOAD_BYTES=b"\x01\x00\x00\x00\x00\x00\x00\x00",  # <<< EDIT if needed
    ENABLE_RATE_HZ=10.0,      # resend enable at this rate while not in MCM Control
    ENABLE_CONTINUOUS=True,   # keep sending enable even after control is achieved

    # --- MCM Torque Command message ---
    CMD_ID=0x181,             # <<< PUT YOUR ACTUAL TORQUE COMMAND FRAME ID
    # Payload packing: here we encode torque (Nm) as a signed 16-bit at bytes 0..1 with a scale of 10 LSB/Nm.
    TORQUE_SCALE_LSB_PER_NM=10.0,   # e.g., 10 means +100Nm -> raw +1000
    TORQUE_MIN_NM=-6.0,             # clamp
    TORQUE_MAX_NM=+6.0,
    APPEND_COUNTER=True,            # rolling 0..15 at byte 2
    APPEND_ENABLE_BYTE=True,        # 1 at byte 3 to mean "command valid"

    # --- SAS11 decoding (example: angle at 0x130/0x140/… your log shows many IDs) ---
    SAS_ID=0x153,             # <<< PUT YOUR ACTUAL SAS ANGLE FRAME ID
    SAS_START_BIT=0,          # <<< bit offset of the angle signal within the frame
    SAS_LENGTH=16,            # <<< length in bits
    SAS_ENDIAN="motorola",    # "intel" or "motorola"
    SAS_SIGNED=True,
    SAS_SCALE=0.1,            # <<< LSB -> degrees (example)
    SAS_OFFSET=0.0,

    # --- Control strategy ---
    LOOP_HZ=50.0,
    KP=0.07,                  # Nm/deg (proportional gain angle->torque)
    DEADZONE_DEG=1.0,
    SETTLE_TOL_DEG=2.0,
    HOLD_TIME_S=0.50,
    MAX_SETTLE_S=2.5,

    # --- Override handling ---
    OVERRIDE_COOLDOWN_S=1.0,  # stop sending for this time after 253 "Human Override"

    # --- Logging/plotting ---
    OUTDIR="./logs_mcm_track",
)

# ============================== UTIL / IO ====================================

def now_s() -> float:
    return time.monotonic()

def ensure_outdir(p):
    os.makedirs(p, exist_ok=True)

@dataclass
class Target:
    t: float
    angle: float

def load_targets(path: str) -> List[Target]:
    """
    Accepts:
      JSON: list/{"targets": [...] } with fields t (s) & angle_deg
      CSV with headers (any of):
        time:  t | time | time_s | sec | seconds | timestamp | iso_time
        angle: angle_deg | angle | deg | sas_deg | steering_deg | steering_angle_deg
      CSV without headers: first two numeric cols = time(s), angle(deg)

    Special handling:
      • timestamp: epoch seconds (float/int) -> converted to relative t (start at 0.0)
      • iso_time: ISO-8601 string -> converted to relative t (start at 0.0)
    """
    # JSON
    if path.lower().endswith(".json"):
        with open(path, "r") as f:
            obj = json.load(f)
        rows = obj["targets"] if isinstance(obj, dict) and "targets" in obj else obj
        return [Target(float(r["t"]), float(r["angle_deg"])) for r in rows]

    # CSV
    with open(path, newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        has_header = False
        try:
            has_header = csv.Sniffer().has_header(sample)
        except Exception:
            pass

        # Helper to finalize and normalize to [Target]
        def finish_rows(rows, time_kind="t"):
            if not rows:
                raise ValueError("No valid target rows found.")
            # If absolute time, normalize to t=0
            if time_kind in ("timestamp", "iso_time"):
                t0 = rows[0][0]
                norm = [Target(r[0] - t0, r[1]) for r in rows]
            else:
                norm = [Target(r[0], r[1]) for r in rows]
            return norm

        if has_header:
            rd = csv.DictReader(f)
            names = {n.strip().lower(): n for n in (rd.fieldnames or [])}

            time_keys = [k for k in names if k in (
                "t","time","time_s","sec","seconds","timestamp","iso_time"
            )]
            angle_keys = [k for k in names if k in (
                "angle_deg","angle","deg","sas_deg","steering_deg","steering_angle_deg"
            )]
            if not time_keys or not angle_keys:
                raise ValueError(
                    f"Could not find time/angle columns in {rd.fieldnames}. "
                    "Try headers like: t/time/timestamp/iso_time and angle_deg/steering_angle_deg."
                )
            tk = names[time_keys[0]]
            ak = names[angle_keys[0]]

            rows = []
            from datetime import datetime
            for r in rd:
                tv = r.get(tk, "")
                av = r.get(ak, "")
                if tv == "" or av == "":
                    continue

                # Parse time
                time_kind = "t"
                try:
                    if tk.lower() == "iso_time":
                        # 2025-09-02T11:08:05.885399
                        dt = datetime.fromisoformat(str(tv))
                        tval = dt.timestamp()
                        time_kind = "iso_time"
                    elif tk.lower() == "timestamp":
                        tval = float(tv)
                        time_kind = "timestamp"
                    else:
                        tval = float(tv)
                        time_kind = "t"
                    aval = float(av)
                except Exception:
                    continue
                rows.append((tval, aval))

            return finish_rows(rows, time_kind=time_kind)

        # No header: assume first two numeric columns = t(s), angle(deg)
        f.seek(0)
        rd = csv.reader(f)
        rows = []
        for row in rd:
            if len(row) < 2:
                continue
            try:
                rows.append((float(row[0]), float(row[1])))
            except ValueError:
                continue
        return finish_rows(rows, time_kind="t")

def open_bus(channel: str, tag: str) -> can.Bus:
    print(f"[{tag}] Opening {channel} via socketcan …")
    bus = can.ThreadSafeBus(channel=channel, bustype="socketcan")
    print(f"[{tag}] Connected to {channel} (socketcan)")
    return bus

# ============================= HEARTBEAT RX ==================================

class HBWatcher:
    """Listens to 0x170 Heartbeat and exposes SystemState/InterfaceState."""
    def __init__(self, bus: can.Bus, hb_id: int):
        self.bus = bus
        self.hb_id = hb_id
        self._state = 0
        self._last_rx = 0.0
        self._q = queue.Queue()
        self._stop = threading.Event()
        self._listener = can.BufferedReader()
        self._notifier = can.Notifier(bus, [self._listener], timeout=1.0)
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def _parse_state(self, data: bytes) -> int:
        # Heartbeat.dbc: SystemState is at byte 2 (bits 16..23), unsigned
        # We read data[2].
        if len(data) >= 3:
            return data[2]
        return 0

    def _run(self):
        try:
            while not self._stop.is_set():
                msg = self._listener.get_message(timeout=0.5)
                if msg and msg.arbitration_id == self.hb_id:
                    st = self._parse_state(msg.data)
                    self._state = st
                    self._last_rx = now_s()
                    # Pretty print once a second max
                    self._q.put(st)
        except Exception as e:
            print(f"[HB] Warning: heartbeat rx failed ({e}). Continuing without hb monitoring.")

    def stop(self):
        self._stop.set()
        try:
            self._notifier.stop()
        except Exception:
            pass

    def last_state(self) -> int:
        return self._state

# ============================== SAS LISTENER =================================

def _extract_bits(data: bytes, start: int, length: int, endian: str) -> int:
    # crude but simple extractor for 64-bit payloads
    raw = int.from_bytes(data, byteorder="big") if endian == "motorola" else int.from_bytes(data, byteorder="little")
    mask = (1 << length) - 1
    return (raw >> (64 - start - length if endian == "motorola" else start)) & mask

class SASListener:
    def __init__(self, bus: can.Bus, cfg: dict):
        self.bus = bus
        self.cfg = cfg
        self.angle = 0.0
        self._listener = can.BufferedReader()
        self._notifier = can.Notifier(bus, [self._listener], timeout=1.0)
        self._stop = threading.Event()
        self._thr = threading.Thread(target=self._run, daemon=True)
        self._thr.start()

    def _run(self):
        try:
            while not self._stop.is_set():
                msg = self._listener.get_message(timeout=0.5)
                if not msg or msg.arbitration_id != self.cfg["SAS_ID"]:
                    continue
                sb = self.cfg["SAS_START_BIT"]
                ln = self.cfg["SAS_LENGTH"]
                endian = self.cfg["SAS_ENDIAN"]
                raw = _extract_bits(msg.data.ljust(8, b"\x00"), sb, ln, endian)
                if self.cfg["SAS_SIGNED"]:
                    signbit = 1 << (ln - 1)
                    if raw & signbit:
                        raw -= (1 << ln)
                angle = raw * self.cfg["SAS_SCALE"] + self.cfg["SAS_OFFSET"]
                self.angle = angle
        except Exception as e:
            print(f"[SAS] Warning: rx failed ({e}).")

    def stop(self):
        self._stop.set()
        try:
            self._notifier.stop()
        except Exception:
            pass

# ========================== PACKERS (ENABLE/CMD) =============================

def pack_enable(cfg: dict, counter: int) -> can.Message:
    payload = bytearray(8)
    b = cfg["ENABLE_PAYLOAD_BYTES"]
    payload[:min(8, len(b))] = b[:min(8, len(b))]
    # Optional: drop a rolling counter into last byte so you can see it on the wire
    payload[7] = counter & 0xFF
    return can.Message(arbitration_id=cfg["ENABLE_ID"], data=bytes(payload), is_extended_id=False)

def pack_torque_cmd(cfg: dict, torque_nm: float, counter: int) -> can.Message:
    tq = max(cfg["TORQUE_MIN_NM"], min(cfg["TORQUE_MAX_NM"], torque_nm))
    raw = int(round(tq * cfg["TORQUE_SCALE_LSB_PER_NM"]))
    raw = max(-32768, min(32767, raw))
    if raw < 0:
        raw = (1 << 16) + raw
    payload = bytearray(8)
    # torque at bytes [0..1] little-endian (common); adjust if your layout differs
    payload[0] = raw & 0xFF
    payload[1] = (raw >> 8) & 0xFF
    if cfg["APPEND_COUNTER"]:
        payload[2] = counter & 0x0F
    if cfg["APPEND_ENABLE_BYTE"]:
        payload[3] = 0x01
    return can.Message(arbitration_id=cfg["CMD_ID"], data=bytes(payload), is_extended_id=False)

# ============================= FOLLOW TARGETS =================================

def follow_targets(tx_bus: can.Bus, angle_bus: can.Bus, cfg: dict, targets: List[Target]):
    ensure_outdir(cfg["OUTDIR"])
    hb = HBWatcher(tx_bus, cfg["HEARTBEAT_ID"])
    sas = SASListener(angle_bus, cfg)

    log_rows = []
    t0 = now_s()
    loop_dt = 1.0 / cfg["LOOP_HZ"]
    i = 0
    target = targets[0].angle
    next_t = targets[0].t
    state = 0
    last_enable = 0.0
    ctr = 0
    override_until = 0.0

    print(f"[INFO] loaded {len(targets)} target points. First/last t = {targets[0].t:.3f}s / {targets[-1].t:.3f}s")

    try:
        while True:
            t = now_s() - t0
            # advance target based on time
            while i + 1 < len(targets) and targets[i + 1].t <= t:
                i += 1
            target = targets[i].angle
            next_t = targets[i].t

            # heartbeat state
            state = hb.last_state()  # 0 Human Control, 1 MCM Control, 253 Human Override, etc.

            # enable strategy
            if state != 1:
                # if override, pause
                if state == 253:
                    if override_until < now_s():
                        override_until = now_s() + cfg["OVERRIDE_COOLDOWN_S"]
                        print("[CTRL] Human Override detected → pausing commands and re-enabling after cooldown…")
                # periodically send enable
                if cfg["ENABLE_RATE_HZ"] > 0 and (now_s() - last_enable) >= (1.0 / cfg["ENABLE_RATE_HZ"]):
                    msg = pack_enable(cfg, ctr)
                    try:
                        tx_bus.send(msg)
                        print(f"[EN] sent enable  id=0x{msg.arbitration_id:X} data={msg.data.hex(' ')}")
                    except can.CanError as e:
                        print(f"[EN] send error: {e}")
                    last_enable = now_s()

            can_send_ok = (state == 1) and (now_s() >= override_until)

            # compute torque for current target/error
            sas_angle = sas.angle
            err = target - sas_angle
            if abs(err) < cfg["DEADZONE_DEG"]:
                err = 0.0
            torque = cfg["KP"] * err

            if can_send_ok:
                msg = pack_torque_cmd(cfg, torque, ctr)
                try:
                    tx_bus.send(msg)
                    print(f"[TX] t={t:6.3f}s  target={target:7.2f}  sas={sas_angle:7.2f}  err={err:6.2f}  tq={torque:5.2f}  ctr={ctr&0x0F}")
                except can.CanError as e:
                    print(f"[TX] send error: {e}")

            ctr = (ctr + 1) & 0xFF

            # log row
            log_rows.append(dict(
                t=t,
                target_deg=target,
                sas_deg=sas_angle,
                err_deg=(target - sas_angle),
                torque_nm=max(cfg["TORQUE_MIN_NM"], min(cfg["TORQUE_MAX_NM"], torque)),
                system_state=state
            ))

            # finish when we reach the last waypoint (with settle or timeout)
            if i == len(targets) - 1:
                # optional: stop after last timestamp + grace
                if t > (targets[-1].t + 2.0):
                    break

            time.sleep(loop_dt)
    finally:
        hb.stop()
        sas.stop()

    # save log & plot
    ts = time.strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(cfg["OUTDIR"], f"tracklog_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        w.writeheader()
        w.writerows(log_rows)
    print(f"[LOG] saved: {csv_path}")

    png_path = os.path.join(cfg["OUTDIR"], f"trackplot_{ts}.png")
    tA = [r["t"] for r in log_rows]
    tgtA = [r["target_deg"] for r in log_rows]
    sasA = [r["sas_deg"] for r in log_rows]
    stA  = [r["system_state"] for r in log_rows]
    plt.figure(figsize=(10,5))
    plt.title("MCM steer tracking")
    plt.plot(tA, tgtA, label="Target (deg)", marker=".")
    plt.plot(tA, sasA, label="SAS (deg)", alpha=0.8)
    plt.ylabel("Angle (deg)")
    plt.xlabel("Time (s)")
    ax2 = plt.twinx()
    ax2.plot(tA, stA, label="SystemState", linestyle="--", alpha=0.5)
    ax2.set_ylabel("SystemState")
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    plt.legend(lines+lines2, labels+labels2, loc="best")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(png_path, dpi=120)
    print(f"[PLOT] saved: {png_path}")

# =============================== CLI =========================================

def main():
    ap = argparse.ArgumentParser(description="Track MCM steering to recorded targets (enable + torque + SAS verify)")
    ap.add_argument("--targets", required=True, help="CSV or JSON with columns: t, angle_deg")
    args = ap.parse_args()

    targets = load_targets(args.targets)
    if not targets:
        print("No targets found.")
        return

    cfg = CONFIG.copy()
    ensure_outdir(cfg["OUTDIR"])

    tx_bus = open_bus(cfg["TX_CHANNEL"], "TX")
    angle_bus = open_bus(cfg["ANGLE_CHANNEL"], "SAS")

    def sigint(_1,_2):
        print("\n[SYS] CTRL-C, exiting…")
        try: tx_bus.shutdown()
        except: pass
        try: angle_bus.shutdown()
        except: pass
        sys.exit(0)
    signal.signal(signal.SIGINT, sigint)

    follow_targets(tx_bus, angle_bus, cfg, targets)

if __name__ == "__main__":
    main()

