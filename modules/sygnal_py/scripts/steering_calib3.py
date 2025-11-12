#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Steering SAS calibration with human-override handling.

- Sends stepped steering position/percent commands to MCM (requires your MCM DBCs).
- Reads SAS11 on the vehicle bus (default can3) and optionally MDPS11/12/VSM11/SPAS11.
- Detects "override spikes" (jerky fast rotations) and "latched" angles afterward.
- Excludes those samples from regression, optionally auto-caps future setpoints.
- Saves JSON and auto-generates plots, marking override samples clearly.

Dependencies:
  pip3 install python-can cantools crc8 numpy matplotlib

Typical run:
  python3 steering_calibmulti.py

Env overrides:
  MCM_CH=can2 SAS_CH=can3 USE_MDPS=0/1 MDPS_ON_MCM_BUS=1/0

Author: you + ChatGPT
"""

import os, sys, time, json, math, signal, asyncio
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict

# Back-end choice to avoid GUI issues if no DISPLAY is present
import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import can, cantools, crc8

# ----------------------------- Config ----------------------------------------

MCM_CHANNEL = os.environ.get("MCM_CH", "can2")
SAS_CHANNEL = os.environ.get("SAS_CH", "can3")

USE_MDPS = os.environ.get("USE_MDPS", "0") == "1"   # default off
MDPS_ON_MCM_BUS = os.environ.get("MDPS_ON_MCM_BUS", "1") == "1"

USE_TCS12_COUNTS = True

DBC_MCM_HEARTBEAT = "./sygnal_dbc/mcm/Heartbeat.dbc"
DBC_MCM_CONTROL   = "./sygnal_dbc/mcm/Control.dbc"
DBC_VEHICLE       = "./sygnal_dbc/vehicle/vehicle.dbc"

# Sweep (percent)
SWEEP_PERCENT = [
    0, 10, 20, 30, 40, 50,
    -10, -20, -30, -40, -50,
    70, 90, 100,
    -70, -90, -100,
    0
]

WARMUP_SEC = 2.0

# Settling
SETTLING_SPEED_THRESH_DPS = 8.0     # considered "still"
SETTLING_HOLD_SEC = 2.0
SETTLING_TIMEOUT_SEC = 15.0

# Deadband scan
DEADBAND_SCAN_MAX_PCT = 6.0
DEADBAND_SCAN_STEP_PCT = 0.5
DEADBAND_ANGLE_THRESH_DEG = 0.5

# Override detection (kinematic-only)
# A rapid spike followed by latched angle while command != 0
OVR_SPIKE_SPEED_PEAK_DPS = 120.0     # peak |ω| to flag a spike
OVR_SPIKE_ANGLE_JUMP_DEG = 40.0      # or |Δθ| within short window
OVR_SPIKE_WINDOW_S       = 0.35
OVR_LATCH_SPEED_DPS      = 6.0       # afterward we're "stopped"
OVR_LATCH_HOLD_S         = 0.8       # must hold low speed this long
OVR_MIN_CMD_ABS_PCT      = 10.0      # only consider if command is non-zero-ish

# After an override trigger, optionally cap future commands to last safe magnitude - margin
AUTO_CAP_AFTER_OVERRIDE = True
OVR_CAP_MARGIN_PCT      = 5.0

SAVE_PLOTS = True

# ----------------------------- Data model ------------------------------------

@dataclass
class CalPoint:
    t: float
    cmd_percent: float
    cmd_norm: float
    sas_angle_deg: float
    sas_speed: float
    settled: bool
    # Extra decoded bits
    sa_count: Optional[int] = None
    sa_z_count: Optional[int] = None
    mdps_angle_deg: Optional[float] = None
    mdps_torque_nm: Optional[float] = None
    vsm_torque_req_nm: Optional[float] = None
    spas_cmd_deg: Optional[float] = None
    # Override annotations
    override_spike: bool = False
    override_latched: bool = False
    note: Optional[str] = None

# --------------------------- Helper: CRC -------------------------------------

def crc8_last_byte(data: bytearray) -> int:
    h = crc8.crc8()
    h.update(data[:-1])
    return h.digest()[0]

# ----------------------------- MCM Controller --------------------------------

class MCMController:
    def __init__(self, channel: str):
        self.db = cantools.database.Database()
        try:
            self.db.add_dbc_file(DBC_MCM_HEARTBEAT)
            self.db.add_dbc_file(DBC_MCM_CONTROL)
        except Exception as e:
            print(f"[MCM] DBC load failed: {e}")
            sys.exit(1)

        try:
            self.bus = can.Bus(channel=channel, bustype="socketcan", bitrate=500000)
            print(f"[MCM] Connected to {channel}")
        except Exception as e:
            print(f"[MCM] Connect failed: {e}")
            sys.exit(1)

        self.msg_enable = self.db.get_message_by_name("ControlEnable")
        self.msg_cmd    = self.db.get_message_by_name("ControlCommand")
        self.msg_hb     = self.db.get_message_by_name("Heartbeat") if self.db.get_message_by_name("Heartbeat") else None

        self.bus_address = 1
        self.interface_id = 2
        self.count8 = 0
        self.last_norm = 0.0
        self.enabled = False

    async def enable(self):
        if self.enabled:
            return
        data = bytearray(self.msg_enable.encode({
            "BusAddress": self.bus_address,
            "InterfaceID": self.interface_id,
            "Enable": 1,
            "CRC": 0
        }))
        data[-1] = crc8_last_byte(data)
        self.bus.send(can.Message(arbitration_id=self.msg_enable.frame_id, data=data, is_extended_id=False))
        await asyncio.sleep(0.02)
        self.enabled = True

    async def send_norm(self, norm: float):
        await self.enable()
        norm = max(min(norm, 1.0), -1.0)
        data = bytearray(self.msg_cmd.encode({
            "BusAddress": self.bus_address,
            "InterfaceID": self.interface_id,
            "Count8": self.count8,
            "Value": float(norm),
            "CRC": 0
        }))
        data[-1] = crc8_last_byte(data)
        self.bus.send(can.Message(arbitration_id=self.msg_cmd.frame_id, data=data, is_extended_id=False))
        self.count8 = (self.count8 + 1) & 0xFF
        self.last_norm = norm

    async def send_percent(self, percent: float, step=0.05, dt=0.08):
        target = max(min(percent, 100.0), -100.0) / 100.0
        cur = self.last_norm
        total = abs(target - cur)
        n = max(int(total / step), 1)
        dv = (target - cur) / n
        for _ in range(n):
            cur = max(min(cur + dv, 1.0), -1.0)
            await self.send_norm(cur)
            await asyncio.sleep(dt)
        if abs(cur - target) > 1e-3:
            await self.send_norm(target)

# ----------------------------- Sensor Reader ---------------------------------

class SensorReader:
    def __init__(self, sas_channel: str, mcm_bus: can.Bus, use_mdps: bool, mdps_on_mcm_bus: bool):
        # DBC
        self.db = None
        if os.path.exists(DBC_VEHICLE):
            try:
                self.db = cantools.database.load_file(DBC_VEHICLE)
            except Exception as e:
                print(f"[SAS] vehicle.dbc load failed: {e}")
        # Buses
        try:
            self.sas_bus = can.Bus(channel=sas_channel, bustype="socketcan", bitrate=500000)
            print(f"[SAS] Connected to {sas_channel}")
        except Exception as e:
            print(f"[SAS] Connect failed: {e}")
            sys.exit(1)
        self.mcm_bus = mcm_bus
        self.use_mdps = use_mdps
        self.mdps_on_mcm_bus = mdps_on_mcm_bus

        # State
        self.sas_angle = 0.0
        self.sas_speed = 0.0
        self.sas_valid = False

        self.sa_count: Optional[int] = None
        self.sa_z_count: Optional[int] = None
        self.mdps_angle: Optional[float] = None
        self.mdps_torque: Optional[float] = None
        self.vsm_tq_req: Optional[float] = None
        self.spas_cmd: Optional[float] = None

        # IDs
        self.ID_SAS11 = 0x2B0
        self.ID_TCS12 = 0x393
        self.ID_MDPS11 = 0x381
        self.ID_MDPS12 = 0x251
        self.ID_VSM11  = 0x164
        self.ID_SPAS11 = 0x390

    # --- decoders ---
    def _dec_sas11(self, data: bytes):
        if self.db:
            try:
                d = self.db.decode_message(self.ID_SAS11, data)
                self.sas_angle = float(d.get("SAS_Angle", self.sas_angle))
                self.sas_speed = float(d.get("SAS_Speed", self.sas_speed))
                self.sas_valid = True
                return
            except Exception:
                pass
        # manual fallback
        raw_angle = int.from_bytes(data[0:2], "little", signed=True)
        self.sas_angle = raw_angle * 0.1
        self.sas_speed = data[2] * 4.0
        self.sas_valid = True

    def _dec_tcs12(self, data: bytes):
        raw = int.from_bytes(data[0:2], "little", signed=False)
        self.sa_count = int(raw)
        raw2 = int.from_bytes(data[2:4], "little", signed=False)
        self.sa_z_count = int(raw2 & 0x7FFF)

    def _dec_mdps11(self, data: bytes):
        if not self.db: return
        try:
            d = self.db.decode_message(self.ID_MDPS11, data)
            self.mdps_angle = float(d.get("CR_Mdps_StrAng", self.mdps_angle))
        except Exception:
            pass

    def _dec_mdps12(self, data: bytes):
        if not self.db: return
        try:
            d = self.db.decode_message(self.ID_MDPS12, data)
            self.mdps_torque = float(d.get("CR_Mdps_StrTq", self.mdps_torque))
        except Exception:
            pass

    def _dec_vsm11(self, data: bytes):
        if not self.db: return
        try:
            d = self.db.decode_message(self.ID_VSM11, data)
            self.vsm_tq_req = float(d.get("CR_Esc_StrTqReq", self.vsm_tq_req))
        except Exception:
            pass

    def _dec_spas11(self, data: bytes):
        if not self.db: return
        try:
            d = self.db.decode_message(self.ID_SPAS11, data)
            self.spas_cmd = float(d.get("CR_Spas_StrAngCmd", self.spas_cmd))
        except Exception:
            pass

    async def run(self):
        async def poll(bus: can.Bus, allow_mdps: bool):
            while True:
                msg = bus.recv(timeout=0.02)
                if msg is None:
                    await asyncio.sleep(0.001)
                    continue
                aid = msg.arbitration_id
                d = msg.data
                if aid == self.ID_SAS11:
                    self._dec_sas11(d)
                elif USE_TCS12_COUNTS and aid == self.ID_TCS12:
                    self._dec_tcs12(d)
                if allow_mdps and self.use_mdps:
                    if aid == self.ID_MDPS11: self._dec_mdps11(d)
                    elif aid == self.ID_MDPS12: self._dec_mdps12(d)
                    elif aid == self.ID_VSM11:  self._dec_vsm11(d)
                    elif aid == self.ID_SPAS11: self._dec_spas11(d)
                await asyncio.sleep(0.0005)

        tasks = [asyncio.create_task(poll(self.sas_bus, allow_mdps=not MDPS_ON_MCM_BUS))]
        if self.use_mdps and MDPS_ON_MCM_BUS:
            tasks.append(asyncio.create_task(poll(self.mcm_bus, allow_mdps=True)))
        await asyncio.gather(*tasks)

# ----------------------------- Calibrator ------------------------------------

class Calibrator:
    def __init__(self, mcm: MCMController, reader: SensorReader):
        self.mcm = mcm
        self.reader = reader
        self.points: List[CalPoint] = []
        self.zero = 0.0
        self.max_safe_abs_pct = 100.0  # auto-capped when override is seen

    # --- utilities ---
    def _rel_angle(self) -> float:
        return self.reader.sas_angle - self.zero

    async def warmup(self):
        print("[SYS] Warming up sensors…")
        t0 = time.time()
        while time.time() - t0 < WARMUP_SEC or not self.reader.sas_valid:
            await asyncio.sleep(0.01)
        self.zero = self.reader.sas_angle

    async def scan_deadband(self) -> float:
        print("[CAL] Deadband scan…")
        base = self._rel_angle()
        plus = 0.0
        pct = 0.0
        # + direction
        while pct <= DEADBAND_SCAN_MAX_PCT:
            await self.mcm.send_percent(pct)
            await asyncio.sleep(0.15)
            if abs(self._rel_angle() - base) >= DEADBAND_ANGLE_THRESH_DEG or abs(self.reader.sas_speed) > SETTLING_SPEED_THRESH_DPS:
                plus = pct
                break
            pct += DEADBAND_SCAN_STEP_PCT
        # - direction
        base = self._rel_angle()
        minus = 0.0
        pct = 0.0
        while pct >= -DEADBAND_SCAN_MAX_PCT:
            await self.mcm.send_percent(pct)
            await asyncio.sleep(0.15)
            if abs(self._rel_angle() - base) >= DEADBAND_ANGLE_THRESH_DEG or abs(self.reader.sas_speed) > SETTLING_SPEED_THRESH_DPS:
                minus = -pct
                break
            pct -= DEADBAND_SCAN_STEP_PCT
        dead = max(plus, minus)
        await self.mcm.send_percent(0.0)
        print(f"[CAL] Deadband: {dead:.2f}%")
        return dead

    # --- override-aware settling ---
    async def wait_until_settled_or_override(self, current_cmd_pct: float):
        """
        Returns: (settled: bool, override_spike: bool, override_latched: bool)
        """
        t_start = time.time()
        # Spike detection window
        spike_window = []
        spike_hit = False
        last_angle = self._rel_angle()
        last_t = time.time()
        latched_since = None

        while time.time() - t_start < SETTLING_TIMEOUT_SEC:
            now = time.time()
            ang = self._rel_angle()
            spd = float(self.reader.sas_speed)

            # windowed speed & angle jump
            spike_window.append((now, ang, spd))
            # prune
            while spike_window and (now - spike_window[0][0]) > OVR_SPIKE_WINDOW_S:
                spike_window.pop(0)

            # compute peak ω and Δθ within window
            if len(spike_window) >= 2:
                w_peak = max(abs(s[2]) for s in spike_window)
                dtheta = abs(spike_window[-1][1] - spike_window[0][1])
                if (w_peak >= OVR_SPIKE_SPEED_PEAK_DPS or dtheta >= OVR_SPIKE_ANGLE_JUMP_DEG) and abs(current_cmd_pct) >= OVR_MIN_CMD_ABS_PCT:
                    spike_hit = True

            # Latch detector: after spike, angle sits still with small speed
            if spike_hit:
                if abs(spd) <= OVR_LATCH_SPEED_DPS:
                    if latched_since is None:
                        latched_since = now
                    elif (now - latched_since) >= OVR_LATCH_HOLD_S:
                        return (False, True, True)   # override spike + latched
                else:
                    latched_since = None

            # Normal settle
            if abs(spd) < SETTLING_SPEED_THRESH_DPS:
                # hold still for SETTLING_HOLD_SEC
                if now - last_t >= SETTLING_HOLD_SEC:
                    return (True, False, False)
            else:
                last_t = now

            await asyncio.sleep(0.02)

        # timeout: treat as not-settled; if spike already happened, mark it
        return (False, spike_hit, False)

    async def sweep(self, sequence_pct: List[float]):
        print("[CAL] Starting sequence…")
        for i, raw_pct in enumerate(sequence_pct, 1):
            # honor auto-cap if enabled
            pct = raw_pct
            if AUTO_CAP_AFTER_OVERRIDE and abs(pct) > self.max_safe_abs_pct:
                pct = math.copysign(self.max_safe_abs_pct, pct)

            print(f"[CAL] Step {i}/{len(sequence_pct)}: {pct:+.1f}%")
            await self.mcm.send_percent(pct)
            settled, ovr_spike, ovr_latch = await self.wait_until_settled_or_override(pct)

            # compute "safe cap" if we hit override at this magnitude
            if ovr_spike or ovr_latch:
                new_cap = max(0.0, abs(pct) - OVR_CAP_MARGIN_PCT)
                if new_cap < self.max_safe_abs_pct:
                    self.max_safe_abs_pct = new_cap

            pt = CalPoint(
                t=time.time(),
                cmd_percent=pct,
                cmd_norm=pct/100.0,
                sas_angle_deg=self._rel_angle(),
                sas_speed=self.reader.sas_speed,
                settled=settled,
                sa_count=self.reader.sa_count,
                sa_z_count=self.reader.sa_z_count,
                mdps_angle_deg=self.reader.mdps_angle,
                mdps_torque_nm=self.reader.mdps_torque,
                vsm_torque_req_nm=self.reader.vsm_tq_req,
                spas_cmd_deg=self.reader.spas_cmd,
                override_spike=ovr_spike,
                override_latched=ovr_latch,
                note=("override spike/latched" if (ovr_spike or ovr_latch) else None)
            )
            self.points.append(pt)

            # If override occurred, retreat to zero and cool down
            if ovr_spike or ovr_latch:
                await self.mcm.send_percent(0.0)
                await asyncio.sleep(1.0)

            await asyncio.sleep(0.3)

        print("[CAL] Sequence complete.")

    # --- analysis & plotting ---
    def _fit_xy(self, x, y) -> Dict[str, float]:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if len(x) < 2:
            return {"m": float("nan"), "b": float("nan"), "r2": float("nan")}
        m, b = np.polyfit(x, y, 1)
        yhat = m*x + b
        ss_res = float(np.sum((y - yhat)**2))
        ss_tot = float(np.sum((y - np.mean(y))**2)) if len(y) > 1 else float("nan")
        r2 = 1.0 - ss_res/ss_tot if ss_tot else float("nan")
        return {"m": float(m), "b": float(b), "r2": float(r2)}

    def analyze(self, deadband_pct: float) -> Dict:
        if not self.points:
            raise RuntimeError("No points collected")

        # Use only valid, settled, non-override points
        valid = [p for p in self.points if p.settled and not p.override_spike and not p.override_latched]
        # Fallback if too strict
        if len(valid) < 3:
            valid = [p for p in self.points if not p.override_spike and not p.override_latched]

        x_pct  = [p.cmd_percent for p in valid]
        x_norm = [p.cmd_norm for p in valid]
        y_ang  = [p.sas_angle_deg for p in valid]

        fit_pct = self._fit_xy(x_pct, y_ang)
        fit_norm = self._fit_xy(x_norm, y_ang)

        # Optional raw counts
        counts_pts = [p for p in valid if p.sa_count is not None]
        fit_counts = None
        if len(counts_pts) >= 3:
            xs = [p.sa_count for p in counts_pts]
            ys = [p.sas_angle_deg for p in counts_pts]
            f = self._fit_xy(xs, ys)
            fit_counts = {"deg_per_count": f["m"], "offset_deg": f["b"], "r_squared": f["r2"]}

        max_ang = float(max(abs(p.sas_angle_deg) for p in self.points))
        turns = max_ang / 360.0

        print("\n=== CALIBRATION SUMMARY ===")
        print(f"deg/percent: {fit_pct['m']:+.3f} (r²={fit_pct['r2']:.3f})")
        print(f"deg/norm:    {fit_norm['m']:+.3f} (r²={fit_norm['r2']:.3f})")
        if fit_counts:
            print(f"deg/count:   {fit_counts['deg_per_count']:+.6f} (r²={fit_counts['r_squared']:.3f})")
        print(f"max angle:   {max_ang:.1f} deg (~{turns:.3f} turns)")
        print(f"deadband:    {deadband_pct:.2f}%")
        bad = [p for p in self.points if p.override_spike or p.override_latched]
        if bad:
            print(f"excluded points (override): {len(bad)}")
        if AUTO_CAP_AFTER_OVERRIDE and self.max_safe_abs_pct < 100.0:
            print(f"auto-capped subsequent setpoints to ±{self.max_safe_abs_pct:.1f}% after override")

        return {
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
            "max_angle_abs_deg": max_ang,
            "approx_turns": turns,
            "deadband_percent": deadband_pct,
            "num_points": len(self.points),
            "num_points_used": len(valid),
            "num_points_excluded_override": len([p for p in self.points if p.override_spike or p.override_latched]),
            "auto_cap_after_override": AUTO_CAP_AFTER_OVERRIDE,
            "cap_limit_abs_percent": self.max_safe_abs_pct
        }

    def save(self, analysis: Dict) -> str:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"steering_calibration_{stamp}.json"
        payload = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "mcm_channel": MCM_CHANNEL,
                "sas_channel": SAS_CHANNEL,
                "use_mdps": USE_MDPS,
                "mdps_on_mcm_bus": MDPS_ON_MCM_BUS,
                "use_tcs12_counts": USE_TCS12_COUNTS,
                "override_thresholds": {
                    "spike_speed_peak_dps": OVR_SPIKE_SPEED_PEAK_DPS,
                    "spike_angle_jump_deg": OVR_SPIKE_ANGLE_JUMP_DEG,
                    "spike_window_s": OVR_SPIKE_WINDOW_S,
                    "latch_speed_dps": OVR_LATCH_SPEED_DPS,
                    "latch_hold_s": OVR_LATCH_HOLD_S
                },
                "auto_cap_after_override": AUTO_CAP_AFTER_OVERRIDE,
                "cap_margin_percent": OVR_CAP_MARGIN_PCT,
                "sweep_percent": SWEEP_PERCENT
            },
            "analysis": analysis,
            "raw_points": [asdict(p) for p in self.points]
        }
        with open(fname, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[CAL] Saved: {fname}")
        return fname

    # ---- plotting ------------------------------------------------------------
    def _scatter_fit(self, x, y, m, b, r2, xlabel, title, deadband=None, outname=None):
        plt.figure(figsize=(10,6))
        plt.scatter(x, y, alpha=0.85, label="points")
        if len(x) >= 2 and not math.isnan(m):
            xs = np.linspace(min(x), max(x), 200)
            ys = m*xs + b
            plt.plot(xs, ys, label=f"fit: y={m:.3f}x+{b:.2f} (r²={r2:.3f})")
        if deadband is not None and "Command (%)" in xlabel:
            plt.axvspan(-deadband, deadband, alpha=0.1, label=f"deadband ±{deadband:.2f}%")
        plt.grid(True); plt.xlabel(xlabel); plt.ylabel("SAS angle (deg)"); plt.title(title)
        plt.legend()
        if SAVE_PLOTS and outname:
            plt.savefig(outname, dpi=150, bbox_inches="tight")

    def _timeline(self, outname=None):
        t0 = self.points[0].t
        t = [p.t - t0 for p in self.points]
        cmd = [p.cmd_percent for p in self.points]
        ang = [p.sas_angle_deg for p in self.points]

        fig = plt.figure(figsize=(12,6))
        ax = fig.add_subplot(111)
        ax.plot(t, cmd, marker='o', label="Command (%)")
        ax.set_xlabel("Time (s)"); ax.set_ylabel("Command (%)"); ax.grid(True)

        ax2 = ax.twinx()
        ax2.plot(t, ang, marker='s', label="Angle (deg)", alpha=0.85)
        ax2.set_ylabel("Angle (deg)")

        # mark override points
        for i, p in enumerate(self.points):
            if p.override_spike or p.override_latched:
                ax2.plot(t[i], ang[i], 'rx', markersize=10)
                ax2.annotate("OVR", (t[i], ang[i]), textcoords="offset points", xytext=(4,4), fontsize=8)

        lines, labels = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + l2, labels + lb2, loc="best")
        ax.set_title("Calibration sequence timeline")
        if SAVE_PLOTS and outname:
            plt.savefig(outname, dpi=150, bbox_inches="tight")

    def _counts_plot(self, outname=None):
        pts = [p for p in self.points if p.sa_count is not None]
        if len(pts) < 2:
            return
        x = [p.sa_count for p in pts]
        y = [p.sas_angle_deg for p in pts]
        f = self._fit_xy(x, y)
        self._scatter_fit(x, y, f["m"], f["b"], f["r2"],
                          xlabel="SA_COUNT (raw)", title="SAS angle vs raw count", outname=outname)

    def plots(self, analysis: Dict, json_name: str):
        # valid (non-override) for the primary scatter fits
        valid = [p for p in self.points if not p.override_spike and not p.override_latched]
        if len(valid) < 2:
            valid = self.points[:]

        x_pct = [p.cmd_percent for p in valid]
        y_ang = [p.sas_angle_deg for p in valid]
        f1 = analysis["fit_percent_to_angle"]

        self._scatter_fit(
            x_pct, y_ang, f1["deg_per_percent"], f1["offset_deg"], f1["r_squared"],
            xlabel="Command (%)",
            title="SAS angle vs Command (%)",
            deadband=analysis["deadband_percent"],
            outname=json_name.replace(".json", "_angle_vs_percent.png")
        )

        x_n = [p.cmd_norm for p in valid]
        f2 = analysis["fit_norm_to_angle"]
        self._scatter_fit(
            x_n, y_ang, f2["deg_per_norm"], f2["offset_deg"], f2["r_squared"],
            xlabel="Command (normalized −1…+1)",
            title="SAS angle vs Command (norm)",
            outname=json_name.replace(".json", "_angle_vs_norm.png")
        )

        self._timeline(outname=json_name.replace(".json", "_timeline.png"))

        if analysis.get("fit_count_to_angle"):
            self._counts_plot(outname=json_name.replace(".json", "_angle_vs_counts.png"))

        plt.show()

# --------------------------------- Main --------------------------------------

async def main():
    mcm = MCMController(MCM_CHANNEL)
    reader = SensorReader(
        sas_channel=SAS_CHANNEL,
        mcm_bus=mcm.bus,
        use_mdps=USE_MDPS,
        mdps_on_mcm_bus=MDPS_ON_MCM_BUS
    )

    sensor_task = asyncio.create_task(reader.run())

    # allow Ctrl+C to stop after save
    def _sigint(sig, frame):
        pass
    signal.signal(signal.SIGINT, _sigint)

    calib = Calibrator(mcm, reader)
    await calib.warmup()
    dead = await calib.scan_deadband()
    await calib.sweep(SWEEP_PERCENT)
    analysis = calib.analyze(dead)
    fname = calib.save(analysis)
    calib.plots(analysis, fname)

    sensor_task.cancel()
    try:
        await sensor_task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(main())
