import asyncio
import can
import cantools
import crc8
import json
import math
import signal
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional, List, Dict, Tuple

import numpy as np

# -----------------------------
# Config (adjust names/IDs here)
# -----------------------------
class Config:
    # CAN channels
    MCM_CHANNEL = "can2"
    SAS_CHANNEL = "can3"

    # DBC paths
    DBC_MCM_HEARTBEAT = "./sygnal_dbc/mcm/Heartbeat.dbc"
    DBC_MCM_CONTROL   = "./sygnal_dbc/mcm/Control.dbc"
    DBC_VEHICLE       = "./sygnal_dbc/vehicle/vehicle.dbc"   # contains SAS

    # MCM message names
    MSG_ENABLE_NAME   = "ControlEnable"
    MSG_COMMAND_NAME  = "ControlCommand"
    MSG_RESPONSE_NAME = "ControlCommandResponse"  # optional

    # MCM fields
    FIELD_BUS_ADDR    = "BusAddress"
    FIELD_IFACE_ID    = "InterfaceID"
    FIELD_ENABLE      = "Enable"
    FIELD_COUNT8      = "Count8"
    FIELD_VALUE       = "Value"
    FIELD_CRC         = "CRC"

    # SAS message / signals
    # If you don't have DBC decoding available, the fallback assumes:
    #   ID 0x2B0 (688) with SAS_Angle: little-endian int16 * 0.1 deg,
    #   SAS_Speed: uint8 * 4.0 deg/s
    SAS_MSG_NAME      = "SAS11"            # optional when using DBC
    SAS_FRAME_ID      = 0x2B0             # 688 decimal
    SAS_ANGLE_NAME    = "SAS_Angle"
    SAS_SPEED_NAME    = "SAS_Speed"

    # Addressing
    BUS_ADDRESS       = 1
    IFACE_ID          = 2

    # Timing
    ENABLE_PERIOD_S   = 0.1       # keep-enable loop
    STEP_SLEEP_S      = 0.08      # after each step in a ramp
    RESPONSE_WAIT_S   = 1.0
    SETTLING_HOLD_S   = 2.0       # must remain under speed threshold for this long
    SETTLING_TIMEOUT  = 15.0
    SPEED_THRESH_DEGS = 2.0       # |deg/s| under this = stationary

    # Calibration sequence (percent commands)
    SEQUENCE_PERCENT = [
        0,
        10, 20, 30, 40, 50,
        -10, -20, -30, -40, -50,
        70, 90, 100,
        -70, -90, -100,
        0,
    ]

    # Deadband detection
    DEADBAND_ANGLE_THRESH_DEG = 1.0
    DEADBAND_TEST_POINTS = [-5, -3, -2, -1, 0, 1, 2, 3, 5]

    # Hysteresis analysis: compare +x% vs -x% outcomes for these magnitudes
    HYST_MAG_LIST = [10, 20, 40, 70]

    # Int32 scaling (if you want deg per LSB)
    INT32_FULLSCALE = 2147483647.0


# -----------------------------
# Data model
# -----------------------------
@dataclass
class CalibPoint:
    t_cmd: float
    cmd_percent: float          # input in percent (-100..100)
    cmd_norm: float             # normalized [-1..+1]
    mcm_resp_value: Optional[float]  # normalized from response (if available)
    sas_angle_deg: float        # relative to initial at start of sweep
    sas_speed_deg_s: float
    settled: bool


# -----------------------------
# Controller 
# -----------------------------
class Controller:
    def __init__(self, channel=Config.MCM_CHANNEL):
        self.db = cantools.database.Database()
        self.db.add_dbc_file(Config.DBC_MCM_HEARTBEAT)
        self.db.add_dbc_file(Config.DBC_MCM_CONTROL)

        try:
            self.bus = can.Bus(channel=channel, bustype="socketcan", bitrate=500000)
            print(f"[MCM] Connected to {channel}")
        except Exception as e:
            print(f"Failed to connect MCM bus: {e}")
            sys.exit(1)

        self.control_count = 0
        self.bus_address = Config.BUS_ADDRESS
        self.last_steer_value = 0.0  # normalized [-1..+1]
        self.control_enabled = False

        # Cache frames
        self.msg_enable  = self.db.get_message_by_name(Config.MSG_ENABLE_NAME)
        self.msg_command = self.db.get_message_by_name(Config.MSG_COMMAND_NAME)
        # Response is optional; we’ll guard its presence
        try:
            self.msg_response = self.db.get_message_by_name(Config.MSG_RESPONSE_NAME)
            self.response_id = self.msg_response.frame_id
        except Exception:
            self.msg_response = None
            self.response_id = None

        # Latest response (normalized)
        self.latest_mcm_resp_norm: Optional[float] = None

    def calc_crc8(self, data: bytearray) -> int:
        h = crc8.crc8()
        h.update(data[:-1])
        return h.digest()[0]

    async def enable_control_once(self):
        data = bytearray(self.msg_enable.encode({
            Config.FIELD_BUS_ADDR: self.bus_address,
            Config.FIELD_IFACE_ID: Config.IFACE_ID,
            Config.FIELD_ENABLE:   1,
            Config.FIELD_CRC:      0
        }))
        data[7] = self.calc_crc8(data)
        self.bus.send(can.Message(
            arbitration_id=self.msg_enable.frame_id,
            is_extended_id=False,
            data=data
        ))
        self.control_enabled = True

    async def enable_keepalive(self):
        while True:
            try:
                await self.enable_control_once()
            except Exception as e:
                print(f"[MCM] Enable failed: {e}")
            await asyncio.sleep(Config.ENABLE_PERIOD_S)

    async def send_single_command_norm(self, norm_value: float):
        """norm_value in [-1, +1]"""
        # NOTE: This assumes your DBC’s 'Value' signal is defined to accept the physical
        # normalized float. If your DBC expects raw int32, you’d need to scale here.
        data = bytearray(self.msg_command.encode({
            Config.FIELD_BUS_ADDR: self.bus_address,
            Config.FIELD_IFACE_ID: Config.IFACE_ID,
            Config.FIELD_COUNT8:   self.control_count,
            Config.FIELD_VALUE:    norm_value,
            Config.FIELD_CRC:      0
        }))
        data[7] = self.calc_crc8(data)

        # Make sure control is enabled before each send
        if not self.control_enabled:
            await self.enable_control_once()
            await asyncio.sleep(0.02)

        self.bus.send(can.Message(
            arbitration_id=self.msg_command.frame_id,
            is_extended_id=False,
            data=data
        ))
        self.control_count = (self.control_count + 1) % 256

    async def send_command_percent(self, percent: float, step_size=0.05):
        """
        Percent -> stepped normalized commands using your approach.
        """
        target = max(-100.0, min(100.0, percent)) / 100.0
        current = self.last_steer_value

        total_distance = abs(target - current)
        steps = max(int(total_distance / step_size), 1)
        step = (target - current) / steps

        for _ in range(steps):
            current += step
            current = max(-1.0, min(1.0, current))
            await self.send_single_command_norm(current)
            await asyncio.sleep(Config.STEP_SLEEP_S)

        if abs(current - target) > 1e-3:
            await self.send_single_command_norm(target)

        self.last_steer_value = target
        return target  # normalized value actually commanded

    async def monitor_responses(self):
        """Optional: capture latest MCM response normalized value, if DBC provided."""
        if self.response_id is None:
            # No response message configured
            return
        while True:
            try:
                msg = self.bus.recv(timeout=0.05)
                if msg and msg.arbitration_id == self.response_id:
                    decoded = self.db.decode_message(msg.arbitration_id, msg.data)
                    if (decoded.get(Config.FIELD_BUS_ADDR) == Config.BUS_ADDRESS and
                        decoded.get(Config.FIELD_IFACE_ID) == Config.IFACE_ID):
                        raw = decoded.get(Config.FIELD_VALUE, None)
                        if raw is not None:
                            # Assuming physical already normalized by DBC; otherwise scale here.
                            self.latest_mcm_resp_norm = float(raw)
            except Exception:
                pass
            await asyncio.sleep(0.01)


# -----------------------------
# SAS reader
# -----------------------------
class SASReader:
    def __init__(self, channel=Config.SAS_CHANNEL):
        # Try DBC decode first; fall back to manual decode if needed
        self.db: Optional[cantools.database.Database] = None
        self.sas_frame_id = Config.SAS_FRAME_ID

        try:
            self.db = cantools.database.load_file(Config.DBC_VEHICLE)
        except Exception:
            self.db = None

        if self.db:
            try:
                self.msg = self.db.get_message_by_name(Config.SAS_MSG_NAME)
                self.sas_frame_id = self.msg.frame_id
            except Exception:
                self.msg = None

        try:
            self.bus = can.Bus(channel=channel, bustype="socketcan", bitrate=500000)
            print(f"[SAS] Connected to {channel}")
        except Exception as e:
            print(f"Failed to connect SAS bus: {e}")
            sys.exit(1)

        self.angle_deg = 0.0
        self.speed_deg_s = 0.0

    def _fallback_decode(self, msg: can.Message):
        # SAS_Angle: int16 little-endian * 0.1 deg
        raw_angle = int.from_bytes(msg.data[0:2], "little", signed=True)
        self.angle_deg = raw_angle * 0.1
        # SAS_Speed: uint8 * 4.0 deg/s
        self.speed_deg_s = msg.data[2] * 4.0

    async def run(self):
        while True:
            try:
                msg = self.bus.recv(timeout=0.05)
                if not msg:
                    await asyncio.sleep(0.005)
                    continue

                if msg.arbitration_id != self.sas_frame_id:
                    continue

                if self.db and getattr(self, "msg", None):
                    try:
                        dec = self.db.decode_message(msg.arbitration_id, msg.data)
                        self.angle_deg = float(dec.get(Config.SAS_ANGLE_NAME, 0.0))
                        self.speed_deg_s = float(dec.get(Config.SAS_SPEED_NAME, 0.0))
                    except Exception:
                        self._fallback_decode(msg)
                else:
                    self._fallback_decode(msg)
            except Exception:
                pass
            await asyncio.sleep(0.002)


# -----------------------------
# Calibrator
# -----------------------------
class Calibrator:
    def __init__(self, ctl: Controller, sas: SASReader):
        self.ctl = ctl
        self.sas = sas
        self.points: List[CalibPoint] = []
        self._angle0: Optional[float] = None

    async def _wait_settle(self, timeout=Config.SETTLING_TIMEOUT) -> bool:
        t0 = time.time()
        stable_since: Optional[float] = None

        while time.time() - t0 < timeout:
            if abs(self.sas.speed_deg_s) < Config.SPEED_THRESH_DEGS:
                if stable_since is None:
                    stable_since = time.time()
                elif time.time() - stable_since >= Config.SETTLING_HOLD_S:
                    return True
            else:
                stable_since = None
            await asyncio.sleep(0.05)
        return False

    async def _record_point(self, t_cmd, cmd_pct, cmd_norm, settled) -> None:
        if self._angle0 is None:
            self._angle0 = self.sas.angle_deg

        rel_angle = self.sas.angle_deg - self._angle0
        mcm_resp = self.ctl.latest_mcm_resp_norm
        self.points.append(CalibPoint(
            t_cmd=t_cmd,
            cmd_percent=cmd_pct,
            cmd_norm=cmd_norm,
            mcm_resp_value=mcm_resp,
            sas_angle_deg=rel_angle,
            sas_speed_deg_s=self.sas.speed_deg_s,
            settled=settled
        ))

    async def _send_and_wait(self, cmd_pct: float) -> Tuple[float, float]:
        t_cmd = time.time()
        cmd_norm = await self.ctl.send_command_percent(cmd_pct)
        # brief time to start moving
        await asyncio.sleep(0.2)
        settled = await self._wait_settle()
        await self._record_point(t_cmd, cmd_pct, cmd_norm, settled)
        return cmd_norm, settled

    async def deadband_scan(self) -> Dict[str, float]:
        """
        Find the smallest |percent| that yields |angle| > threshold (after settling).
        """
        hits = []
        for p in Config.DEADBAND_TEST_POINTS:
            await self._send_and_wait(p)
            angle = abs(self.points[-1].sas_angle_deg)
            if angle >= Config.DEADBAND_ANGLE_THRESH_DEG:
                hits.append(abs(p))

        deadband = min(hits) if hits else 0.0
        return {
            "deadband_percent": float(deadband),
            "deadband_norm": float(deadband / 100.0),
        }

    async def run_sequence(self, sequence_percent: List[float]):
        print("[CAL] Starting sequence…")
        # capture initial baseline
        await asyncio.sleep(1.0)
        if self._angle0 is None:
            self._angle0 = self.sas.angle_deg

        for i, pct in enumerate(sequence_percent):
            print(f"[CAL] Step {i+1}/{len(sequence_percent)}: {pct:+.1f}%")
            await self._send_and_wait(pct)
            await asyncio.sleep(0.3)

        print("[CAL] Sequence complete.")

    # -----------------
    # Analysis helpers
    # -----------------
    def _linear_fit(self, x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        coeffs = np.polyfit(x, y, 1)
        m, b = float(coeffs[0]), float(coeffs[1])
        yhat = m * x + b
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2)) if len(y) > 1 else 0.0
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
        return {"slope": m, "offset": b, "r2": r2}

    def analyze(self) -> Dict:
        if len(self.points) < 4:
            return {"error": "Insufficient data"}

        pts = [p for p in self.points if p.settled]
        if len(pts) < 3:
            pts = self.points

        cmd_pct  = np.array([p.cmd_percent for p in pts], dtype=float)
        cmd_norm = np.array([p.cmd_norm for p in pts], dtype=float)
        angle    = np.array([p.sas_angle_deg for p in pts], dtype=float)

        fit_pct  = self._linear_fit(cmd_pct, angle)
        fit_norm = self._linear_fit(cmd_norm, angle)

        # deg per LSB (assume norm -> int32 full scale)
        deg_per_norm = fit_norm["slope"]            # deg per 1.0 normalized
        deg_per_lsb  = deg_per_norm / Config.INT32_FULLSCALE

        # Hysteresis: compare +mag vs -mag (closest available points)
        def mean_angle_for(mag: float, sign: int) -> Optional[float]:
            # pick the closest commanded percent to (sign*mag)
            diffs = np.abs(cmd_pct - (sign * mag))
            if len(diffs) == 0:
                return None
            idx = int(np.argmin(diffs))
            return float(angle[idx])

        hyst = {}
        for mag in Config.HYST_MAG_LIST:
            ap = mean_angle_for(mag, +1)
            an = mean_angle_for(mag, -1)
            if ap is not None and an is not None:
                hyst[str(mag)] = {
                    "pos_angle_deg": ap,
                    "neg_angle_deg": an,
                    "delta_deg": ap - (-an),   # compare sign-corrected magnitudes
                }

        # turns
        max_abs_angle = float(np.max(np.abs(angle))) if len(angle) else 0.0
        max_turns = max_abs_angle / 360.0

        return {
            "fit_percent_to_angle": {
                "deg_per_percent": fit_pct["slope"],
                "offset_deg":      fit_pct["offset"],
                "r_squared":       fit_pct["r2"],
            },
            "fit_norm_to_angle": {
                "deg_per_norm":    fit_norm["slope"],
                "offset_deg":      fit_norm["offset"],
                "r_squared":       fit_norm["r2"],
                "deg_per_int32_lsb": deg_per_lsb,
            },
            "max_abs_angle_deg": max_abs_angle,
            "max_turns": max_turns,
            "points_total": len(self.points),
            "points_settled": len([p for p in self.points if p.settled]),
            "hysteresis_deg": hyst,
        }

    def save_json(self, analysis: Dict, path: Optional[str]=None) -> str:
        if path is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = f"steering_calibration_{ts}.json"

        out = {
            "timestamp": datetime.now().isoformat(),
            "analysis": analysis,
            "raw_points": [asdict(p) for p in self.points],
        }
        with open(path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[CAL] Saved: {path}")
        return path


# -----------------------------
# Main
# -----------------------------
async def main():
    ctl = Controller(channel=Config.MCM_CHANNEL)
    sas = SASReader(channel=Config.SAS_CHANNEL)

    # Tasks
    tasks = [
        asyncio.create_task(ctl.enable_keepalive()),
        asyncio.create_task(sas.run()),
        asyncio.create_task(ctl.monitor_responses()),
    ]

    # Give streams a moment to start
    print("[SYS] Warming up sensors…")
    await asyncio.sleep(2.0)

    calib = Calibrator(ctl, sas)

    # Optional deadband scan near zero
    print("[CAL] Deadband scan…")
    deadband_info = await calib.deadband_scan()
    print(f"[CAL] Deadband: {deadband_info['deadband_percent']:.2f}% "
          f"({deadband_info['deadband_norm']:.4f} norm)")

    # Main sequence
    await calib.run_sequence(Config.SEQUENCE_PERCENT)

    # Analysis
    analysis = calib.analyze()
    analysis["deadband"] = deadband_info
    path = calib.save_json(analysis)

    # Cleanup tasks
    for t in tasks:
        t.cancel()

    print("\n=== CALIBRATION SUMMARY ===")
    if "error" in analysis:
        print(analysis["error"])
        return

    print(f"deg/percent: {analysis['fit_percent_to_angle']['deg_per_percent']:.3f} "
          f"(r²={analysis['fit_percent_to_angle']['r_squared']:.3f})")
    print(f"deg/norm:    {analysis['fit_norm_to_angle']['deg_per_norm']:.3f} "
          f"(r²={analysis['fit_norm_to_angle']['r_squared']:.3f})")
    print(f"deg/LSB:     {analysis['fit_norm_to_angle']['deg_per_int32_lsb']:.6e}")
    print(f"max angle:   {analysis['max_abs_angle_deg']:.1f} deg "
          f"(~{analysis['max_turns']:.3f} turns)")
    print(f"deadband:    {analysis['deadband']['deadband_percent']:.2f}%")

def _graceful(tasks: List[asyncio.Task]):
    for t in tasks:
        t.cancel()
    sys.exit(0)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass

