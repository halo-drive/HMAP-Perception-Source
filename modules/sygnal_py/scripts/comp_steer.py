#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAS-focused calibration with explicit MCM state tracking.

- Monitors MCM Heartbeat (0x368) to gate commands on state transitions:
  Human Control (0) -> MCM Control (1). Detects Human Override (253).
- Re-arms with ControlEnable and waits for heartbeat confirmation.
- Logs each transmit with: percent, physical value, raw integer, and payload bytes.
- Logs vehicle reaction from SAS11 (angle, speed), and optional MDPS/VSM torques.
- Fits integer_cmd->angle and percent->angle, saves JSON + plots, shows graphs.

Dependencies: python-can, cantools, crc8, numpy, matplotlib
"""

import os, sys, time, json, math, signal, asyncio, binascii
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import numpy as np
import can, cantools, crc8

import matplotlib
if not os.environ.get("DISPLAY"):
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------------------- Config --------------------
MCM_CHANNEL = os.environ.get("MCM_CH", "can2")
SAS_CHANNEL = os.environ.get("SAS_CH", "can3")

USE_MDPS = os.environ.get("USE_MDPS", "0") == "1"
MDPS_ON_MCM_BUS = os.environ.get("MDPS_ON_MCM_BUS", "1") == "1"

VERBOSE = os.environ.get("VERBOSE", "0") == "1"

# Command mode: 'torque' maps percent -> raw integer (±TORQUE_MAX_RAW).
#               'norm'   maps percent -> [-1..+1] float (kept for back-compat)
CMD_MODE = os.environ.get("CMD_MODE", "torque").lower()
TORQUE_MAX_RAW = int(os.environ.get("TORQUE_MAX_RAW", "1000"))

# DBC files (adjust paths if needed)
DBC_MCM_HEARTBEAT = "./sygnal_dbc/mcm/Heartbeat.dbc"
DBC_MCM_CONTROL   = "./sygnal_dbc/mcm/Control.dbc"
DBC_VEHICLE       = "./sygnal_dbc/vehicle/vehicle.dbc"

# Which MCM interface we use (for Interface{N}State in the heartbeat)
MCM_INTERFACE_ID = int(os.environ.get("MCM_INTERFACE_ID", "1"))

# Sweep profile
SWEEP_PERCENT = [
    0, 10, 20, 30, 40, 50,
    -10, -20, -30, -40, -50,
    70, 90, 100,
    -70, -90, -100,
    0
]

# Timing & thresholds
WARMUP_SEC = 2.0
SETTLING_SPEED_THRESH_DPS = 8.0
SETTLING_HOLD_SEC = 2.0
SETTLING_TIMEOUT_SEC = 15.0

DEADBAND_SCAN_MAX_PCT = 6.0
DEADBAND_SCAN_STEP_PCT = 0.5
DEADBAND_ANGLE_THRESH_DEG = 0.5

# Override detection via motion + heartbeat
OVR_SPIKE_SPEED_PEAK_DPS = 120.0
OVR_SPIKE_ANGLE_JUMP_DEG = 40.0
OVR_SPIKE_WINDOW_S       = 0.35
OVR_LATCH_SPEED_DPS      = 6.0
OVR_LATCH_HOLD_S         = 0.8
OVR_MIN_CMD_ABS_PCT      = 10.0

REARM_COOLDOWN_S         = 1.0
REARM_PROBE_PCT          = 5.0
REARM_PROBE_TIMEOUT_S    = 2.0
REARM_MAX_ATTEMPTS       = 3

AUTO_CAP_AFTER_OVERRIDE  = True
OVR_CAP_MARGIN_PCT       = 5.0

SAVE_PLOTS = True

# -------------------- Dataclasses --------------------
@dataclass
class CalPoint:
    t: float
    cmd_percent: float
    cmd_phys: float
    cmd_raw_int: Optional[int]
    cmd_bytes_hex: str
    sas_angle_deg: float
    sas_speed: float
    settled: bool
    # optional vehicle-side extra signals
    sa_count: Optional[int] = None
    sa_z_count: Optional[int] = None
    mdps_angle_deg: Optional[float] = None
    mdps_torque_nm: Optional[float] = None
    vsm_torque_req_nm: Optional[float] = None
    spas_cmd_deg: Optional[float] = None
    # state / override info
    mcm_system_state: Optional[int] = None
    mcm_interface_state: Optional[int] = None
    override_spike: bool = False
    override_latched: bool = False
    note: Optional[str] = None

# -------------------- Utilities --------------------
def crc8_last_byte(data: bytearray) -> int:
    h = crc8.crc8()
    h.update(data[:-1])
    return h.digest()[0]

def choice_label(choices: Dict[int, str], value: Optional[int]) -> str:
    if value is None: return "unknown"
    return choices.get(int(value), str(value))

# -------------------- MCM Heartbeat Monitor --------------------
class McmHeartbeat:
    """Decode 0x368 Heartbeat and expose states."""
    def __init__(self, dbc_path: str, interface_id: int):
        self.db = cantools.database.load_file(dbc_path)
        self.msg = self.db.get_message_by_frame_id(0x368)
        self.interface_id = interface_id

        # choices for pretty printing
        self.sys_choices = self.msg.get_signal_by_name("SystemState").choices or {}
        self.if_choices  = self.msg.get_signal_by_name("OverallInterfaceState").choices or {}

        self.system_state: Optional[int] = None
        self.overall_iface: Optional[int] = None
        self.iface_state: Optional[int] = None
        self.count16: Optional[int] = None
        self.last_rx_ts: Optional[float] = None

        name = f"Interface{interface_id}State"
        self._iface_sig = self.msg.get_signal_by_name(name)

    def decode(self, data: bytes):
        d = self.msg.decode(data)
        self.system_state = int(d.get("SystemState", self.system_state if self.system_state is not None else 0))
        self.overall_iface = int(d.get("OverallInterfaceState", self.overall_iface if self.overall_iface is not None else 0))
        if self._iface_sig:
            self.iface_state = int(d.get(self._iface_sig.name, self.iface_state if self.iface_state is not None else 0))
        self.count16 = int(d.get("Count16", self.count16 if self.count16 is not None else 0))
        self.last_rx_ts = time.time()

    @property
    def system_label(self) -> str:
        return choice_label(self.sys_choices, self.system_state)

    @property
    def iface_label(self) -> str:
        # prefer per-interface; fall back to overall
        src = self.if_choices if self._iface_sig else self.if_choices
        val = self.iface_state if self._iface_sig else self.overall_iface
        return choice_label(src, val)

    def is_ready_for_commands(self) -> bool:
        """MCM Control AND InterfaceN == MCM Control."""
        return (self.system_state == 1) and ((self.iface_state or 0) == 1)

    def is_human_override(self) -> bool:
        return self.system_state == 253

# -------------------- Controller (send) --------------------
class MCMController:
    def __init__(self, bus: can.Bus, heartbeat: McmHeartbeat):
        self.bus = bus
        self.heartbeat = heartbeat

        # Load control DBC
        self.db = cantools.database.load_file(DBC_MCM_CONTROL)
        self.msg_enable = self.db.get_message_by_name("ControlEnable")
        self.msg_cmd    = self.db.get_message_by_name("ControlCommand")
        self.msg_resp   = self.db.get_message_by_name("ControlResponse") if self._has_msg("ControlResponse") else None

        # pick a command signal name
        self.cmd_signal_name = None
        for name in ("Value", "Torque", "Cmd", "TorqueCmd", "DesiredTorque"):
            if self._has_sig(self.msg_cmd, name):
                self.cmd_signal_name = name
                break
        if not self.cmd_signal_name:
            raise RuntimeError("No usable command signal found in ControlCommand.")

        self.cmd_sig = self.msg_cmd.get_signal_by_name(self.cmd_signal_name)
        self.count8 = 0
        self.bus_address = 1
        self.interface_id = MCM_INTERFACE_ID
        self.enabled = False
        self.last_phys_value = 0.0

    def _has_msg(self, name: str) -> bool:
        try:
            return self.db.get_message_by_name(name) is not None
        except Exception:
            return False

    @staticmethod
    def _has_sig(msg, name: str) -> bool:
        try:
            return msg.get_signal_by_name(name) is not None
        except Exception:
            return False

    def _phys_to_raw_int(self, phys_value: float) -> Optional[int]:
        """Reverse-apply scaling to estimate on-wire integer for the command signal."""
        try:
            scale = self.cmd_sig.scale if self.cmd_sig.scale is not None else 1.0
            offset = self.cmd_sig.offset if self.cmd_sig.offset is not None else 0.0
            raw = int(round((phys_value - offset) / scale))
            # clamp within bit length
            width = self.cmd_sig.length
            signed = self.cmd_sig.is_signed
            if signed:
                min_raw = -(1 << (width - 1))
                max_raw = (1 << (width - 1)) - 1
            else:
                min_raw = 0
                max_raw = (1 << width) - 1
            return max(min(raw, max_raw), min_raw)
        except Exception:
            return None

    def _encode_command(self, phys_value: float) -> Tuple[bytearray, Optional[int]]:
        payload = {
            "BusAddress": self.bus_address,
            "InterfaceID": self.interface_id,
            "Count8": self.count8,
            self.cmd_signal_name: float(phys_value),
            "CRC": 0
        }
        data = bytearray(self.msg_cmd.encode(payload))
        data[-1] = crc8_last_byte(data)
        raw_int = self._phys_to_raw_int(phys_value)
        return data, raw_int

    async def wait_for_mcm_ready(self, timeout=3.0) -> bool:
        """Wait until heartbeat shows MCM Control for our interface."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.heartbeat.is_ready_for_commands():
                return True
            await asyncio.sleep(0.02)
        return self.heartbeat.is_ready_for_commands()

    async def enable(self) -> bool:
        """Send ControlEnable=1, then wait for MCM Control state."""
        if not self.msg_enable:
            return False
        data = bytearray(self.msg_enable.encode({
            "BusAddress": self.bus_address,
            "InterfaceID": self.interface_id,
            "Enable": 1,
            "CRC": 0
        }))
        data[-1] = crc8_last_byte(data)
        self.bus.send(can.Message(arbitration_id=self.msg_enable.frame_id, data=data, is_extended_id=False))
        if VERBOSE:
            print(f"[TX] ControlEnable -> {self.msg_enable.frame_id:#x} data={binascii.hexlify(data).decode()}")
        self.enabled = True
        ok = await self.wait_for_mcm_ready(timeout=2.0)
        if VERBOSE:
            s = self.heartbeat.system_label
            i = self.heartbeat.iface_label
            print(f"[MCM] enable wait -> sys={s}, iface={i}, ready={ok}")
        return ok

    async def disable(self):
        if not self.msg_enable:
            return
        data = bytearray(self.msg_enable.encode({
            "BusAddress": self.bus_address,
            "InterfaceID": self.interface_id,
            "Enable": 0,
            "CRC": 0
        }))
        data[-1] = crc8_last_byte(data)
        self.bus.send(can.Message(arbitration_id=self.msg_enable.frame_id, data=data, is_extended_id=False))
        if VERBOSE:
            print(f"[TX] ControlDisable -> {self.msg_enable.frame_id:#x} data={binascii.hexlify(data).decode()}")
        self.enabled = False

    async def send_percent(self, percent: float) -> Tuple[float, Optional[int], str]:
        """
        Convert percent to physical value (torque or norm), encode, transmit.
        Returns (phys_value, raw_int, bytes_hex).
        """
        # Map percent to physical
        if CMD_MODE == "torque":
            raw = int(round(max(min(percent, 100.0), -100.0) / 100.0 * TORQUE_MAX_RAW))
            # turn raw back to physical using DBC scale/offset
            scale = self.cmd_sig.scale if self.cmd_sig.scale is not None else 1.0
            offset = self.cmd_sig.offset if self.cmd_sig.offset is not None else 0.0
            phys = raw * scale + offset
        else:
            phys = max(min(percent / 100.0, 1.0), -1.0)
            raw = self._phys_to_raw_int(phys)

        # Ensure ready
        if not self.heartbeat.is_ready_for_commands():
            await self.enable()
        data, raw_int_est = self._encode_command(phys)
        self.bus.send(can.Message(arbitration_id=self.msg_cmd.frame_id, data=data, is_extended_id=False))
        self.count8 = (self.count8 + 1) & 0xFF
        self.last_phys_value = phys

        raw_to_log = raw if raw is not None else raw_int_est
        bytes_hex = binascii.hexlify(data).decode()
        if VERBOSE:
            s = self.heartbeat.system_label
            i = self.heartbeat.iface_label
            print(f"[TX] cmd {percent:+.1f}% | phys={phys:+.3f} | raw_int={raw_to_log} | "
                  f"ID={self.msg_cmd.frame_id:#x} data={bytes_hex} | hb sys={s} if={i}")
        return phys, raw_to_log, bytes_hex

    async def rearm_after_override(self, reader) -> bool:
        if VERBOSE:
            print("[MCM] Re-arming after override: cooldown, enable, probe.")
        for attempt in range(1, REARM_MAX_ATTEMPTS + 1):
            await asyncio.sleep(REARM_COOLDOWN_S)
            ok = await self.enable()
            if not ok:
                if VERBOSE: print(f"[MCM] enable did not confirm MCM Control (attempt {attempt}).")
                continue

            # Small probe
            pre_ang = reader.sas_angle
            await self.send_percent(math.copysign(REARM_PROBE_PCT, 1.0))
            t0 = time.time()
            seen_motion = False
            while time.time() - t0 < REARM_PROBE_TIMEOUT_S:
                if abs(reader.sas_speed) > SETTLING_SPEED_THRESH_DPS:
                    seen_motion = True
                    break
                await asyncio.sleep(0.02)
            await self.send_percent(0.0)
            if seen_motion and abs(reader.sas_angle - pre_ang) > 0.2:
                if VERBOSE: print(f"[MCM] Re-arm confirmed (attempt {attempt}).")
                return True
            if VERBOSE: print(f"[MCM] Probe inconclusive (attempt {attempt}); retrying.")
        print("[MCM] Re-arm failed; staying safe.")
        return False

# -------------------- Sensor Reader (SAS + MDPS + Heartbeat) --------------------
class SensorReader:
    def __init__(self, sas_channel: str, mcm_bus: can.Bus, interface_id: int):
        self.vehicle_db = cantools.database.load_file(DBC_VEHICLE) if Path(DBC_VEHICLE).exists() else None
        self.sas_bus = can.Bus(channel=sas_channel, bustype="socketcan")
        self.mcm_bus = mcm_bus

        # SAS / optional signals
        self.sas_angle = 0.0
        self.sas_speed = 0.0
        self.sas_valid = False
        self.sa_count: Optional[int] = None
        self.sa_z_count: Optional[int] = None
        self.mdps_angle: Optional[float] = None
        self.mdps_torque: Optional[float] = None
        self.vsm_tq_req: Optional[float] = None
        self.spas_cmd: Optional[float] = None

        # Heartbeat monitor
        self.hb = McmHeartbeat(DBC_MCM_HEARTBEAT, interface_id)

        # IDs
        self.ID_SAS11 = 0x2B0
        self.ID_TCS12 = 0x393
        self.ID_MDPS11 = 0x381
        self.ID_MDPS12 = 0x251
        self.ID_VSM11  = 0x164
        self.ID_SPAS11 = 0x390
        self.ID_HB     = 0x368

    def _dec_vehicle(self, aid: int, data: bytes):
        db = self.vehicle_db
        if db:
            try:
                dec = db.decode_message(aid, data)
            except Exception:
                dec = None
        else:
            dec = None

        if aid == self.ID_SAS11:
            if dec:
                self.sas_angle = float(dec.get("SAS_Angle", self.sas_angle))
                self.sas_speed = float(dec.get("SAS_Speed", self.sas_speed))
            else:
                raw_angle = int.from_bytes(data[0:2], "little", signed=True)
                self.sas_angle = raw_angle * 0.1
                self.sas_speed = data[2] * 4.0
            self.sas_valid = True

        elif aid == self.ID_TCS12:
            raw = int.from_bytes(data[0:2], "little", signed=False)
            self.sa_count = int(raw)
            raw2 = int.from_bytes(data[2:4], "little", signed=False)
            self.sa_z_count = int(raw2 & 0x7FFF)

        elif USE_MDPS:
            if aid == self.ID_MDPS11 and dec:
                self.mdps_angle = float(dec.get("CR_Mdps_StrAng", self.mdps_angle))
            elif aid == self.ID_MDPS12 and dec:
                self.mdps_torque = float(dec.get("CR_Mdps_StrTq", self.mdps_torque))
            elif aid == self.ID_VSM11 and dec:
                self.vsm_tq_req = float(dec.get("CR_Esc_StrTqReq", self.vsm_tq_req))
            elif aid == self.ID_SPAS11 and dec:
                self.spas_cmd = float(dec.get("CR_Spas_StrAngCmd", self.spas_cmd))

    async def run(self):
        async def poll_sas():
            while True:
                msg = self.sas_bus.recv(timeout=0.02)
                if msg:
                    self._dec_vehicle(msg.arbitration_id, msg.data)
                await asyncio.sleep(0.001)

        async def poll_mcm():
            while True:
                msg = self.mcm_bus.recv(timeout=0.02)
                if msg:
                    if msg.arbitration_id == self.ID_HB:
                        self.hb.decode(msg.data)
                        if VERBOSE:
                            # Only print on change (Count16 increments)
                            if self.hb.count16 is not None and (self.hb.count16 % 10 == 0):
                                print(f"[HB] sys={self.hb.system_label} if={self.hb.iface_label} cnt={self.hb.count16}")
                    else:
                        # Also decode MDPS/VSM if they ride this bus
                        self._dec_vehicle(msg.arbitration_id, msg.data)
                await asyncio.sleep(0.001)

        await asyncio.gather(asyncio.create_task(poll_sas()),
                             asyncio.create_task(poll_mcm()))

# -------------------- Calibration Core --------------------
class Calibrator:
    def __init__(self, mcm: MCMController, reader: SensorReader):
        self.mcm = mcm
        self.reader = reader
        self.points: List[CalPoint] = []
        self.zero = 0.0
        self.max_safe_abs_pct = 100.0

    def _rel_angle(self) -> float:
        return self.reader.sas_angle - self.zero

    async def warmup(self):
        print("[SYS] Warming up sensors…")
        t0 = time.time()
        while (time.time() - t0 < WARMUP_SEC) or not self.reader.sas_valid:
            await asyncio.sleep(0.02)

        self.zero = self.reader.sas_angle
        # Ensure we are in MCM Control before starting
        if not self.reader.hb.is_ready_for_commands():
            print("[MCM] Initial state:", self.reader.hb.system_label, "/", self.reader.hb.iface_label)
            await self.mcm.enable()

    async def scan_deadband(self) -> float:
        print("[CAL] Deadband scan…")
        base = self._rel_angle()
        # + direction
        plus = 0.0
        pct = 0.0
        while pct <= DEADBAND_SCAN_MAX_PCT:
            await self.mcm.send_percent(pct)
            await asyncio.sleep(0.15)
            moved = (abs(self._rel_angle() - base) >= DEADBAND_ANGLE_THRESH_DEG) or \
                    (abs(self.reader.sas_speed) > SETTLING_SPEED_THRESH_DPS)
            if moved:
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
            moved = (abs(self._rel_angle() - base) >= DEADBAND_ANGLE_THRESH_DEG) or \
                    (abs(self.reader.sas_speed) > SETTLING_SPEED_THRESH_DPS)
            if moved:
                minus = -pct
                break
            pct -= DEADBAND_SCAN_STEP_PCT

        await self.mcm.send_percent(0.0)
        dead = max(plus, minus)
        print(f"[CAL] Deadband: {dead:.2f}%")
        return dead

    async def wait_until_settled_or_override(self, current_cmd_pct: float):
        """Return (settled, ovr_spike, ovr_latch)."""
        t_start = time.time()
        spike_window = []
        spike_hit = False
        latched_since = None
        last_still = time.time()

        while time.time() - t_start < SETTLING_TIMEOUT_SEC:
            now = time.time()
            ang = self._rel_angle()
            spd = float(self.reader.sas_speed)

            # heartbeat says override?
            if self.reader.hb.is_human_override():
                return (False, True, True)

            # spike detection from SAS motion
            spike_window.append((now, ang, spd))
            while spike_window and (now - spike_window[0][0]) > OVR_SPIKE_WINDOW_S:
                spike_window.pop(0)

            if len(spike_window) >= 2:
                w_peak = max(abs(s[2]) for s in spike_window)
                dtheta = abs(spike_window[-1][1] - spike_window[0][1])
                if (w_peak >= OVR_SPIKE_SPEED_PEAK_DPS or dtheta >= OVR_SPIKE_ANGLE_JUMP_DEG) and abs(current_cmd_pct) >= OVR_MIN_CMD_ABS_PCT:
                    spike_hit = True

            if spike_hit:
                if abs(spd) <= OVR_LATCH_SPEED_DPS:
                    if latched_since is None:
                        latched_since = now
                    elif (now - latched_since) >= OVR_LATCH_HOLD_S:
                        return (False, True, True)
                else:
                    latched_since = None

            # settled condition
            if abs(spd) < SETTLING_SPEED_THRESH_DPS:
                if now - last_still >= SETTLING_HOLD_SEC:
                    return (True, False, False)
            else:
                last_still = now

            await asyncio.sleep(0.02)

        return (False, spike_hit, False)

    async def sweep(self, sequence_pct: List[float]) -> bool:
        print("[CAL] Starting sequence…")
        for idx, raw_pct in enumerate(sequence_pct, 1):
            # Magnitude cap after first override to avoid re-triggering
            pct = raw_pct
            if AUTO_CAP_AFTER_OVERRIDE and abs(pct) > self.max_safe_abs_pct:
                pct = math.copysign(self.max_safe_abs_pct, pct)

            print(f"[CAL] Step {idx}/{len(sequence_pct)}: {pct:+.1f}%")

            # Ensure state before sending
            if not self.reader.hb.is_ready_for_commands():
                ok = await self.mcm.enable()
                if not ok:
                    print("[MCM] Not ready (heartbeat). Aborting.")
                    return False

            # Send command and wait
            phys, raw_int, bytes_hex = await self.mcm.send_percent(pct)
            settled, ovr_spike, ovr_latch = await self.wait_until_settled_or_override(pct)

            # Record point
            pt = CalPoint(
                t=time.time(),
                cmd_percent=pct,
                cmd_phys=phys,
                cmd_raw_int=raw_int,
                cmd_bytes_hex=bytes_hex,
                sas_angle_deg=self._rel_angle(),
                sas_speed=self.reader.sas_speed,
                settled=settled,
                sa_count=self.reader.sa_count,
                sa_z_count=self.reader.sa_z_count,
                mdps_angle_deg=self.reader.mdps_angle,
                mdps_torque_nm=self.reader.mdps_torque,
                vsm_torque_req_nm=self.reader.vsm_tq_req,
                spas_cmd_deg=self.reader.spas_cmd,
                mcm_system_state=self.reader.hb.system_state,
                mcm_interface_state=self.reader.hb.iface_state,
                override_spike=ovr_spike,
                override_latched=ovr_latch,
                note=("override; aborted step" if (ovr_spike or ovr_latch) else None)
            )
            self.points.append(pt)

            # Verbose reaction log
            if VERBOSE:
                print(f"[RX] SAS θ={pt.sas_angle_deg:+.2f}° ω={self.reader.sas_speed:+.1f}°/s | "
                      f"HB sys={self.reader.hb.system_label} if={self.reader.hb.iface_label}")

            # If overridden, stop output, re-arm, maybe cap and continue
            if ovr_spike or ovr_latch:
                await self.mcm.send_percent(0.0)
                new_cap = max(0.0, abs(pct) - OVR_CAP_MARGIN_PCT)
                if new_cap < self.max_safe_abs_pct:
                    self.max_safe_abs_pct = new_cap
                ok = await self.mcm.rearm_after_override(self.reader)
                if not ok:
                    return False
            else:
                await asyncio.sleep(0.25)

        print("[CAL] Sequence complete.")
        return True

    # ---------- Analysis & plots ----------
    @staticmethod
    def _fit_xy(x, y) -> Dict[str, float]:
        x = np.asarray(x, float)
        y = np.asarray(y, float)
        if len(x) < 2:
            return {"m": float("nan"), "b": float("nan"), "r2": float("nan")}
        m, b = np.polyfit(x, y, 1)
        yhat = m * x + b
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) if len(y) > 1 else float("nan")
        r2 = 1.0 - ss_res / ss_tot if ss_tot else float("nan")
        return {"m": float(m), "b": float(b), "r2": float(r2)}

    def analyze(self, deadband_pct: float) -> Dict:
        valid = [p for p in self.points if p.settled and not p.override_latched]
        if len(valid) < 3:
            valid = [p for p in self.points if not p.override_latched]

        # percent → angle
        x_pct = [p.cmd_percent for p in valid]
        y_ang = [p.sas_angle_deg for p in valid]
        fit_pct = self._fit_xy(x_pct, y_ang)

        # integer (raw) → angle (only where we have raw ints)
        pts_int = [p for p in valid if p.cmd_raw_int is not None]
        fit_int = None
        if len(pts_int) >= 3:
            xi = [p.cmd_raw_int for p in pts_int]
            yi = [p.sas_angle_deg for p in pts_int]
            fi = self._fit_xy(xi, yi)
            fit_int = {"deg_per_raw": fi["m"], "offset_deg": fi["b"], "r_squared": fi["r2"]}

        max_ang = float(max(abs(p.sas_angle_deg) for p in self.points)) if self.points else 0.0
        turns = max_ang / 360.0

        print("\n=== CALIBRATION SUMMARY ===")
        print(f"deg/percent: {fit_pct['m']:+.3f} (r²={fit_pct['r2']:.3f})")
        if fit_int:
            print(f"deg/raw_int: {fit_int['deg_per_raw']:+.6f} (r²={fit_int['r_squared']:.3f})")
        print(f"max angle:   {max_ang:.1f} deg (~{turns:.3f} turns)")
        print(f"deadband:    {deadband_pct:.2f}%")
        if AUTO_CAP_AFTER_OVERRIDE and self.max_safe_abs_pct < 100.0:
            print(f"auto-cap after override: ±{self.max_safe_abs_pct:.1f}%")
        return {
            "fit_percent_to_angle": {
                "deg_per_percent": fit_pct["m"], "offset_deg": fit_pct["b"], "r_squared": fit_pct["r2"]
            },
            "fit_rawint_to_angle": fit_int,
            "max_angle_abs_deg": max_ang,
            "approx_turns": turns,
            "deadband_percent": deadband_pct,
            "num_points": len(self.points),
            "num_points_used": len(valid)
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
                "cmd_mode": CMD_MODE,
                "torque_max_raw": TORQUE_MAX_RAW,
                "sweep_percent": SWEEP_PERCENT,
                "interface_id": MCM_INTERFACE_ID
            },
            "analysis": analysis,
            "raw_points": [asdict(p) for p in self.points]
        }
        with open(fname, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"[CAL] Saved: {fname}")
        return fname

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
        if not self.points: return
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

        for i, p in enumerate(self.points):
            if p.override_latched:
                ax2.plot(t[i], ang[i], 'rx', markersize=10)
                ax2.annotate("OVR", (t[i], ang[i]), textcoords="offset points", xytext=(4,4), fontsize=8)

        lines, labels = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + l2, labels + lb2, loc="best")
        ax.set_title("Calibration sequence timeline")
        if SAVE_PLOTS and outname:
            plt.savefig(outname, dpi=150, bbox_inches="tight")

    def plots(self, analysis: Dict, json_name: str):
        valid = [p for p in self.points if not p.override_latched]
        if len(valid) < 2:
            valid = self.points[:]

        # Percent -> angle
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

        # Raw integer -> angle
        pts_int = [p for p in valid if p.cmd_raw_int is not None]
        if len(pts_int) >= 2 and analysis.get("fit_rawint_to_angle"):
            xi = [p.cmd_raw_int for p in pts_int]
            yi = [p.sas_angle_deg for p in pts_int]
            fr = analysis["fit_rawint_to_angle"]
            self._scatter_fit(
                xi, yi, fr["deg_per_raw"], fr["offset_deg"], fr["r_squared"],
                xlabel="Command (raw integer)",
                title="SAS angle vs Command (raw int)",
                outname=json_name.replace(".json", "_angle_vs_rawint.png")
            )

        self._timeline(outname=json_name.replace(".json", "_timeline.png"))
        plt.show()

# -------------------- Main --------------------
async def main():
    # Build buses
    try:
        mcm_bus = can.Bus(channel=MCM_CHANNEL, bustype="socketcan")
        print(f"[MCM] Connected to {MCM_CHANNEL}")
    except Exception as e:
        print(f"[MCM] Connect failed: {e}")
        sys.exit(1)

    try:
        reader = SensorReader(SAS_CHANNEL, mcm_bus, MCM_INTERFACE_ID)
        print(f"[SAS] Connected to {SAS_CHANNEL}")
    except Exception as e:
        print(f"[SAS] Connect failed: {e}")
        sys.exit(1)

    # Spawn readers
    sensor_task = asyncio.create_task(reader.run())

    # Controller
    mcm = MCMController(mcm_bus, reader.hb)
    calib = Calibrator(mcm, reader)

    # Warmup + deadband
    await calib.warmup()
    dead = await calib.scan_deadband()

    # Sweep
    ok = await calib.sweep(SWEEP_PERCENT)
    analysis = calib.analyze(dead)
    fname = calib.save(analysis)
    calib.plots(analysis, fname)

    if not ok:
        print("[CAL] Ended early due to override re-arm failure or heartbeat not ready.")

    sensor_task.cancel()
    try:
        await sensor_task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    # gentle SIGINT
    def _sigint(sig, frame):
        print("\n[SYS] SIGINT received; wrapping up…")
    signal.signal(signal.SIGINT, _sigint)
    asyncio.run(main())
