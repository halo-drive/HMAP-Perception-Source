#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steering SAS calibration:
- Sends stepped steering commands to MCM (Control.dbc / Heartbeat.dbc)
- Reads SAS11 (SAS_Angle / SAS_Speed) + optional TCS12 (SA_COUNT) and optional MDPS frames
- Computes mapping: deg per command %, deg per command norm (-1..+1), optional deg per count
- Auto-saves JSON + auto-generates plots

Requirements:
  pip3 install python-can cantools crc8 numpy matplotlib
Hardware:
  MCM on can2, vehicle/SAS on can3 (defaults can be changed below)
"""

import os
import sys
import json
import time
import math
import signal
import asyncio
import crc8
import can
import cantools
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict

# ----------------------------- Configuration ---------------------------------

MCM_CHANNEL = os.environ.get("MCM_CH", "can2")
SAS_CHANNEL = os.environ.get("SAS_CH", "can3")

# Flip this on only if you actually have MDPS frames on one of your buses
USE_MDPS = False
MDPS_ON_MCM_BUS = False  # if False, it will listen on SAS_CHANNEL

# Try to read ESC TCS12 for raw steering sensor counts (nice-to-have)
USE_TCS12_COUNTS = True

DBC_MCM_HEARTBEAT = "./sygnal_dbc/mcm/Heartbeat.dbc"
DBC_MCM_CONTROL   = "./sygnal_dbc/mcm/Control.dbc"
DBC_VEHICLE       = "./sygnal_dbc/vehicle/vehicle.dbc"   # contains SAS11/TCS12/etc if available

# Sweep sequence (in percent)
SWEEP_PERCENT = [
    0, 10, 20, 30, 40, 50,
    -10, -20, -30, -40, -50,
    70, 90, 100,
    -70, -90, -100,
    0
]

WARMUP_SEC = 2.0
SETTLING_SPEED_THRESH_DPS = 8.0     # deg/sec (SAS_Speed uses 4.0 units/bit; see fallback)
SETTLING_HOLD_SEC = 2.0
SETTLING_TIMEOUT_SEC = 15.0
DEADBAND_SCAN_MAX_PCT = 5.0
DEADBAND_SCAN_STEP_PCT = 0.5
DEADBAND_ANGLE_THRESH_DEG = 0.5

SAVE_PLOTS = True

# ----------------------------- Datamodel -------------------------------------

@dataclass
class CalPoint:
    t: float
    cmd_percent: float
    cmd_norm: float
    sas_angle_deg: float
    sas_speed: float
    settled: bool
    # Optional data
    sa_count: Optional[int] = None
    sa_z_count: Optional[int] = None
    mdps_angle_deg: Optional[float] = None
    mdps_torque_nm: Optional[float] = None
    vsm_torque_req_nm: Optional[float] = None
    spas_cmd_deg: Optional[float] = None

# --------------------------- Helper: CRC / timing ----------------------------

def calc_crc8_for_last_byte(data: bytearray) -> int:
    h = crc8.crc8()
    h.update(data[:-1])
    return h.digest()[0]

# ----------------------------- Controller (MCM) -------------------------------

class MCMController:
    def __init__(self, channel: str):
        self.db = cantools.database.Database()
        self.db.add_dbc_file(DBC_MCM_HEARTBEAT)
        self.db.add_dbc_file(DBC_MCM_CONTROL)
        try:
            self.bus = can.Bus(channel=channel, bustype='socketcan', bitrate=500000)
            print(f"[MCM] Connected to {channel}")
        except Exception as e:
            print(f"[MCM] Failed to connect: {e}")
            sys.exit(1)

        self.bus_address = 1
        self.interface_id = 2
        self.control_count = 0
        self.last_norm = 0.0
        self.enabled = False

        # Cache frames
        self.msg_enable = self.db.get_message_by_name('ControlEnable')
        self.msg_cmd    = self.db.get_message_by_name('ControlCommand')
        self.msg_hb     = self.db.get_message_by_name('Heartbeat') if self.db.get_message_by_name('Heartbeat') else None

    async def enable_control(self):
        if self.enabled:
            return
        data = bytearray(self.msg_enable.encode({
            'BusAddress': self.bus_address,
            'InterfaceID': self.interface_id,
            'Enable': 1,
            'CRC': 0
        }))
        data[-1] = calc_crc8_for_last_byte(data)
        self.bus.send(can.Message(arbitration_id=self.msg_enable.frame_id, is_extended_id=False, data=data))
        await asyncio.sleep(0.02)
        self.enabled = True

    async def send_norm_value(self, norm: float):
        """norm in [-1.0, +1.0]"""
        await self.enable_control()
        data = bytearray(self.msg_cmd.encode({
            'BusAddress': self.bus_address,
            'InterfaceID': self.interface_id,
            'Count8': self.control_count,
            'Value': float(max(min(norm, 1.0), -1.0)),  # DBC encodes this to raw
            'CRC': 0
        }))
        data[-1] = calc_crc8_for_last_byte(data)
        self.bus.send(can.Message(arbitration_id=self.msg_cmd.frame_id, is_extended_id=False, data=data))
        self.control_count = (self.control_count + 1) % 256
        self.last_norm = norm

    async def send_percent(self, percent: float, step_size=0.05, step_delay=0.08):
        """percent [-100..100] with smooth stepping (like your controller)."""
        target = max(min(percent, 100.0), -100.0) / 100.0
        current = self.last_norm
        total = abs(target - current)
        nsteps = max(int(total / step_size), 1)
        dv = (target - current) / nsteps
        for _ in range(nsteps):
            current += dv
            current = max(min(current, 1.0), -1.0)
            await self.send_norm_value(current)
            await asyncio.sleep(step_delay)
        if abs(current - target) > 1e-3:
            await self.send_norm_value(target)

# ----------------------------- Sensor Reader ----------------------------------

class SensorReader:
    """Reads SAS11 (and optionally TCS12, MDPS11/12/VSM11/SPAS11) off a given bus."""
    def __init__(self, sas_channel: str, mcm_bus: can.Bus, use_mdps: bool, mdps_on_mcm_bus: bool):
        self.sas_channel = sas_channel
        self.use_mdps = use_mdps
        self.mdps_on_mcm_bus = mdps_on_mcm_bus

        # Load DBC for vehicle bus if available
        self.vehicle_db: Optional[cantools.database.Database] = None
        if os.path.exists(DBC_VEHICLE):
            try:
                self.vehicle_db = cantools.database.load_file(DBC_VEHICLE)
            except Exception as e:
                print(f"[SAS] Failed to load vehicle DBC: {e}")

        # Open SAS bus
        try:
            self.sas_bus = can.Bus(channel=sas_channel, bustype='socketcan', bitrate=500000)
            print(f"[SAS] Connected to {sas_channel}")
        except Exception as e:
            print(f"[SAS] Failed to connect: {e}")
            sys.exit(1)

        # Optionally, also read MDPS on the MCM bus
        self.mcm_bus = mcm_bus

        # Live values
        self.sas_angle = 0.0
        self.sas_speed = 0.0
        self.sas_valid = False

        self.sa_count: Optional[int] = None
        self.sa_z_count: Optional[int] = None

        self.mdps_angle: Optional[float] = None
        self.mdps_torque: Optional[float] = None
        self.vsm_tq_req: Optional[float] = None
        self.spas_cmd: Optional[float] = None

        # IDs (decimal)
        self.ID_SAS11 = 0x2B0  # 688
        self.ID_TCS12 = 0x393  # 915
        self.ID_MDPS11 = 0x381 # 897
        self.ID_MDPS12 = 0x251 # 593
        self.ID_VSM11  = 0x164 # 356
        self.ID_SPAS11 = 0x390 # 912

    def _decode_sas11(self, arb_id: int, data: bytes):
        if arb_id != self.ID_SAS11:
            return
        decoded = None
        if self.vehicle_db:
            try:
                decoded = self.vehicle_db.decode_message(arb_id, data)
                self.sas_angle = float(decoded.get('SAS_Angle', self.sas_angle))
                self.sas_speed = float(decoded.get('SAS_Speed', self.sas_speed))
                self.sas_valid = True
                return
            except Exception:
                pass
        # Fallback manual (from your bit defs):
        # SAS_Angle: 0|16@1- (0.1, 0) little-endian signed
        raw_angle = int.from_bytes(data[0:2], 'little', signed=True)
        self.sas_angle = raw_angle * 0.1
        # SAS_Speed: 16|8@1+ (4.0, 0.0) -> data[2] unsigned * 4.0
        self.sas_speed = data[2] * 4.0
        self.sas_valid = True

    def _decode_tcs12(self, arb_id: int, data: bytes):
        if arb_id != self.ID_TCS12:
            return
        # SA_COUNT: 0|16@1+ (2.0, -32768)
        raw = int.from_bytes(data[0:2], 'little', signed=False)
        self.sa_count = int(raw)
        # SA_Z_COUNT: 16|15@1+ (2.0, -32768)
        raw2 = int.from_bytes(data[2:4], 'little', signed=False)
        self.sa_z_count = int(raw2 & 0x7FFF)  # lower 15 bits

    def _decode_mdps11(self, arb_id: int, data: bytes):
        if arb_id != self.ID_MDPS11:
            return
        if self.vehicle_db:
            try:
                dec = self.vehicle_db.decode_message(arb_id, data)
                self.mdps_angle = float(dec.get('CR_Mdps_StrAng', self.mdps_angle))
                return
            except Exception:
                pass
        # Fallback manual (24|16@1- (0.1, 0)) — bytes 3..4 if you bit-pack; skip manual unless needed.

    def _decode_mdps12(self, arb_id: int, data: bytes):
        if arb_id != self.ID_MDPS12:
            return
        if self.vehicle_db:
            try:
                dec = self.vehicle_db.decode_message(arb_id, data)
                # CR_Mdps_StrTq : 40|12@1+ (0.01, -20.48)
                self.mdps_torque = float(dec.get('CR_Mdps_StrTq', self.mdps_torque))
                return
            except Exception:
                pass

    def _decode_vsm11(self, arb_id: int, data: bytes):
        if arb_id != self.ID_VSM11:
            return
        if self.vehicle_db:
            try:
                dec = self.vehicle_db.decode_message(arb_id, data)
                self.vsm_tq_req = float(dec.get('CR_Esc_StrTqReq', self.vsm_tq_req))
                return
            except Exception:
                pass

    def _decode_spas11(self, arb_id: int, data: bytes):
        if arb_id != self.ID_SPAS11:
            return
        if self.vehicle_db:
            try:
                dec = self.vehicle_db.decode_message(arb_id, data)
                self.spas_cmd = float(dec.get('CR_Spas_StrAngCmd', self.spas_cmd))
                return
            except Exception:
                pass

    async def run(self):
        """Run two tiny pollers: one on SAS bus (always), and optionally on MCM bus if MDPS there."""
        async def _poll(bus: can.Bus, name: str):
            while True:
                try:
                    msg = bus.recv(timeout=0.02)
                    if not msg:
                        await asyncio.sleep(0.001)
                        continue
                    aid = msg.arbitration_id
                    d = msg.data
                    if aid == self.ID_SAS11:
                        self._decode_sas11(aid, d)
                    elif USE_TCS12_COUNTS and aid == self.ID_TCS12:
                        self._decode_tcs12(aid, d)
                    if self.use_mdps:
                        if aid == self.ID_MDPS11: self._decode_mdps11(aid, d)
                        elif aid == self.ID_MDPS12: self._decode_mdps12(aid, d)
                        elif aid == self.ID_VSM11:  self._decode_vsm11(aid, d)
                        elif aid == self.ID_SPAS11: self._decode_spas11(aid, d)
                except can.CanOperationError:
                    pass
                await asyncio.sleep(0.0005)

        tasks = [asyncio.create_task(_poll(self.sas_bus, "SAS"))]
        if self.use_mdps:
            mdps_bus = self.mcm_bus if self.mdps_on_mcm_bus else self.sas_bus
            tasks.append(asyncio.create_task(_poll(mdps_bus, "MDPS")))
        await asyncio.gather(*tasks)

# ----------------------------- Calibration Core -------------------------------

class Calibrator:
    def __init__(self, mcm: MCMController, reader: SensorReader):
        self.mcm = mcm
        self.reader = reader
        self.points: List[CalPoint] = []
        self.zero_angle = 0.0

    async def warmup(self, sec=WARMUP_SEC):
        print("[SYS] Warming up sensors…")
        t0 = time.time()
        while time.time() - t0 < sec or not self.reader.sas_valid:
            await asyncio.sleep(0.01)
        self.zero_angle = self.reader.sas_angle

    async def wait_settle(self, timeout=SETTLING_TIMEOUT_SEC) -> bool:
        start = time.time()
        stable_since = None
        while time.time() - start < timeout:
            spd = abs(self.reader.sas_speed)
            if spd < SETTLING_SPEED_THRESH_DPS:
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= SETTLING_HOLD_SEC:
                    return True
            else:
                stable_since = None
            await asyncio.sleep(0.05)
        return False

    def _rel_angle(self) -> float:
        return self.reader.sas_angle - self.zero_angle

    async def scan_deadband(self) -> float:
        """Symmetric deadband around 0 by angle threshold."""
        print("[CAL] Deadband scan…")
        base = self._rel_angle()
        db_plus = 0.0
        # Positive direction
        pct = 0.0
        while pct <= DEADBAND_SCAN_MAX_PCT:
            await self.mcm.send_percent(pct)
            await asyncio.sleep(0.15)
            if abs(self._rel_angle() - base) >= DEADBAND_ANGLE_THRESH_DEG or abs(self.reader.sas_speed) > SETTLING_SPEED_THRESH_DPS:
                db_plus = pct
                break
            pct += DEADBAND_SCAN_STEP_PCT

        # Negative direction
        base = self._rel_angle()
        db_minus = 0.0
        pct = 0.0
        while pct >= -DEADBAND_SCAN_MAX_PCT:
            await self.mcm.send_percent(pct)
            await asyncio.sleep(0.15)
            if abs(self._rel_angle() - base) >= DEADBAND_ANGLE_THRESH_DEG or abs(self.reader.sas_speed) > SETTLING_SPEED_THRESH_DPS:
                db_minus = -pct
                break
            pct -= DEADBAND_SCAN_STEP_PCT

        deadband = max(db_plus, db_minus)
        # Return to zero
        await self.mcm.send_percent(0.0)
        print(f"[CAL] Deadband: {deadband:.2f}%")
        return deadband

    async def sweep(self, sequence_pct: List[float]):
        print("[CAL] Starting sequence…")
        for i, p in enumerate(sequence_pct, 1):
            tag = f"{p:+.1f}%"
            print(f"[CAL] Step {i}/{len(sequence_pct)}: {tag}")
            await self.mcm.send_percent(p)
            settled = await self.wait_settle()
            pt = CalPoint(
                t=time.time(),
                cmd_percent=p,
                cmd_norm=p/100.0,
                sas_angle_deg=self._rel_angle(),
                sas_speed=self.reader.sas_speed,
                settled=settled,
                sa_count=self.reader.sa_count,
                sa_z_count=self.reader.sa_z_count,
                mdps_angle_deg=self.reader.mdps_angle,
                mdps_torque_nm=self.reader.mdps_torque,
                vsm_torque_req_nm=self.reader.vsm_tq_req,
                spas_cmd_deg=self.reader.spas_cmd
            )
            self.points.append(pt)
            await asyncio.sleep(0.3)
        print("[CAL] Sequence complete.")

    # ---------------------------- Analysis & Plots ----------------------------

    def _fit_xy(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        if len(x) < 2:
            return {"m": float("nan"), "b": float("nan"), "r2": float("nan")}
        coeffs = np.polyfit(x, y, 1)
        m, b = float(coeffs[0]), float(coeffs[1])
        y_pred = m*x + b
        ss_res = float(np.sum((y - y_pred)**2))
        ss_tot = float(np.sum((y - np.mean(y))**2)) if len(y) > 1 else float("nan")
        r2 = 1.0 - ss_res/ss_tot if ss_tot and not math.isnan(ss_tot) else float("nan")
        return {"m": m, "b": b, "r2": r2}

    def analyze(self, deadband_pct: float) -> Dict:
        if not self.points:
            raise RuntimeError("No calibration points recorded.")

        settled = [p for p in self.points if p.settled] or self.points

        x_pct = np.array([p.cmd_percent for p in settled], dtype=float)
        x_norm = np.array([p.cmd_norm for p in settled], dtype=float)
        y_ang = np.array([p.sas_angle_deg for p in settled], dtype=float)

        fit_pct = self._fit_xy(x_pct, y_ang)
        fit_norm = self._fit_xy(x_norm, y_ang)

        # Optional counts fit (deg vs count)
        have_counts = any(p.sa_count is not None for p in settled)
        fit_counts = None
        if have_counts:
            xs = np.array([p.sa_count for p in settled if p.sa_count is not None], dtype=float)
            ys = np.array([p.sas_angle_deg for p in settled if p.sa_count is not None], dtype=float)
            fit = self._fit_xy(xs, ys)
            fit_counts = {"deg_per_count": fit["m"], "offset_deg": fit["b"], "r_squared": fit["r2"]}

        max_angle = float(np.max(np.abs([p.sas_angle_deg for p in self.points])))
        turns = max_angle / 360.0

        summary = {
            "fit_percent_to_angle": {
                "deg_per_percent": fit_pct["m"],
                "offset_deg": fit_pct["b"],
                "r_squared": fit_pct["r2"],
            },
            "fit_norm_to_angle": {
                "deg_per_norm": fit_norm["m"],
                "offset_deg": fit_norm["b"],
                "r_squared": fit_norm["r2"],
            },
            "fit_count_to_angle": fit_counts,
            "max_angle_abs_deg": max_angle,
            "approx_turns": turns,
            "deadband_percent": deadband_pct,
            "num_points": len(self.points),
            "num_settled": len(settled),
            "notes": "Negative slope just indicates opposite sign convention between command and SAS."
        }

        # Console summary
        print("\n=== CALIBRATION SUMMARY ===")
        print(f"deg/percent: {fit_pct['m']:+.3f} (r²={fit_pct['r2']:.3f})")
        print(f"deg/norm:    {fit_norm['m']:+.3f} (r²={fit_norm['r2']:.3f})")
        if fit_counts:
            print(f"deg/count:   {fit_counts['deg_per_count']:+.6f} (r²={fit_counts['r_squared']:.3f})")
        print(f"max angle:   {max_angle:.1f} deg (~{turns:.3f} turns)")
        print(f"deadband:    {deadband_pct:.2f}%")

        return summary

    def save(self, analysis: Dict) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"steering_calibration_{stamp}.json"
        out = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "mcm_channel": MCM_CHANNEL,
                "sas_channel": SAS_CHANNEL,
                "use_mdps": USE_MDPS,
                "mdps_on_mcm_bus": MDPS_ON_MCM_BUS,
                "use_tcs12_counts": USE_TCS12_COUNTS,
                "sweep_percent": SWEEP_PERCENT,
                "settling_speed_thresh_dps": SETTLING_SPEED_THRESH_DPS,
                "settling_hold_sec": SETTLING_HOLD_SEC,
                "settling_timeout_sec": SETTLING_TIMEOUT_SEC,
            },
            "analysis": analysis,
            "raw_points": [asdict(p) for p in self.points]
        }
        with open(fname, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[CAL] Saved: {fname}")
        return fname

    def _plot_fit(self, x, y, m, b, r2, xlabel, title, deadband_pct=None, outname=None):
        plt.figure(figsize=(10,6))
        plt.scatter(x, y, label="points", alpha=0.8)
        xf = np.linspace(float(min(x)), float(max(x)), 200)
        yf = m*xf + b
        plt.plot(xf, yf, label=f"fit: y = {m:.3f}x + {b:.2f} (r²={r2:.3f})")
        if deadband_pct is not None and xlabel.startswith("Command (%)"):
            plt.axvspan(-deadband_pct, deadband_pct, alpha=0.1, label=f"deadband ±{deadband_pct:.2f}%")
        plt.xlabel(xlabel)
        plt.ylabel("SAS angle (deg)")
        plt.title(title)
        plt.grid(True)
        plt.legend()
        if SAVE_PLOTS and outname:
            plt.savefig(outname, dpi=150, bbox_inches="tight")

    def _plot_timeline(self, outname=None):
        t0 = self.points[0].t
        t = [p.t - t0 for p in self.points]
        cmd = [p.cmd_percent for p in self.points]
        ang = [p.sas_angle_deg for p in self.points]

        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(111)
        ax.plot(t, cmd, marker='o', label="Command (%)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Command (%)")
        ax.grid(True)
        ax2 = ax.twinx()
        ax2.plot(t, ang, marker='s', label="Angle (deg)", alpha=0.8)
        ax2.set_ylabel("Angle (deg)")

        # Compose a joint legend
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc="best")

        ax.set_title("Calibration sequence timeline")
        if SAVE_PLOTS and outname:
            plt.savefig(outname, dpi=150, bbox_inches="tight")

    def _plot_counts(self, outname=None):
        pts = [p for p in self.points if p.sa_count is not None]
        if not pts:
            return
        x = np.array([p.sa_count for p in pts], dtype=float)
        y = np.array([p.sas_angle_deg for p in pts], dtype=float)
        fit = self._fit_xy(x, y)
        self._plot_fit(
            x, y, fit["m"], fit["b"], fit["r2"],
            xlabel="SA_COUNT (raw counts)",
            title="SAS angle vs raw steering counts",
            outname=outname
        )

    def plots(self, analysis: Dict, json_fname: str):
        # Angle vs Command (%)
        pts = self.points
        x_pct = np.array([p.cmd_percent for p in pts], dtype=float)
        y_ang = np.array([p.sas_angle_deg for p in pts], dtype=float)
        f = analysis["fit_percent_to_angle"]
        out1 = json_fname.replace(".json", "_angle_vs_percent.png")
        self._plot_fit(
            x_pct, y_ang, f["deg_per_percent"], f["offset_deg"], f["r_squared"],
            xlabel="Command (%)",
            title="SAS angle vs Command (%)",
            deadband_pct=analysis["deadband_percent"],
            outname=out1
        )

        # Angle vs Command (norm)
        x_norm = np.array([p.cmd_norm for p in pts], dtype=float)
        fn = analysis["fit_norm_to_angle"]
        out2 = json_fname.replace(".json", "_angle_vs_norm.png")
        self._plot_fit(
            x_norm, y_ang, fn["deg_per_norm"], fn["offset_deg"], fn["r_squared"],
            xlabel="Command (normalized −1…+1)",
            title="SAS angle vs Command (norm)",
            outname=out2
        )

        # Timeline
        out3 = json_fname.replace(".json", "_timeline.png")
        self._plot_timeline(outname=out3)

        # Counts (if present)
        if analysis.get("fit_count_to_angle"):
            out4 = json_fname.replace(".json", "_angle_vs_counts.png")
            self._plot_counts(outname=out4)

        plt.show()

# --------------------------------- Main --------------------------------------

async def main():
    # Build MCM controller
    mcm = MCMController(MCM_CHANNEL)

    # Sensor reader (SAS + optional MDPS/ESC)
    reader = SensorReader(
        sas_channel=SAS_CHANNEL,
        mcm_bus=mcm.bus,
        use_mdps=USE_MDPS,
        mdps_on_mcm_bus=MDPS_ON_MCM_BUS
    )

    # Launch sensor tasks
    sensor_task = asyncio.create_task(reader.run())

    # Handle Ctrl+C for safe save
    stop_now = {"flag": False}
    def _cleanup(sig=None, frame=None):
        stop_now["flag"] = True
        # We let the normal flow finish; plots & save happen below.

    signal.signal(signal.SIGINT, _cleanup)

    calib = Calibrator(mcm, reader)

    # Warmup and zeroing
    await calib.warmup()

    # Deadband
    deadband = await calib.scan_deadband()

    # Sweep
    await calib.sweep(SWEEP_PERCENT)

    # Analyze + Save + Plot
    analysis = calib.analyze(deadband)
    json_file = calib.save(analysis)
    calib.plots(analysis, json_file)

    # Cancel sensor task
    sensor_task.cancel()
    try:
        await sensor_task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(main())
