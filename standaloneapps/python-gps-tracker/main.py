#!/usr/bin/env python3
"""
NovAtel live GPS recorder/follower (print-only guidance).

Deps:
  pip install pyserial novatel_edie

Usage:
  # 1) RECORD a loop/path
  python gps_path_follow.py record --serial /dev/ttyUSB0 --baud 115200 --out loop_ref.jsonl

  # 2) FOLLOW the saved path
  python gps_path_follow.py follow --serial /dev/ttyUSB0 --baud 115200 --ref loop_ref.jsonl
"""

import argparse
import json
import math
import socket
import sys
import time
from datetime import datetime

try:
    import serial  # pyserial
except ImportError:
    serial = None

import novatel_edie as edie

# -----------------------
# Geo helpers (no deps)
# -----------------------
R_EARTH = 6371000.0  # meters

def haversine_m(lat1, lon1, lat2, lon2):
    """Distance in meters."""
    rlat1 = math.radians(lat1); rlat2 = math.radians(lat2)
    dlat = rlat2 - rlat1
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(rlat1)*math.cos(rlat2)*math.sin(dlon/2)**2
    return 2 * R_EARTH * math.asin(math.sqrt(a))

def bearing_deg(lat1, lon1, lat2, lon2):
    """Initial bearing from point1 to point2 in degrees [0..360)."""
    φ1 = math.radians(lat1); φ2 = math.radians(lat2)
    Δλ = math.radians(lon2 - lon1)
    y = math.sin(Δλ) * math.cos(φ2)
    x = math.cos(φ1)*math.cos(φ2) - math.sin(φ1)*math.sin(φ2)*math.cos(Δλ)
    θ = math.degrees(math.atan2(y, x))
    return (θ + 360.0) % 360.0

def wrap180(angle):
    """Wrap angle to [-180, 180)."""
    a = (angle + 180.0) % 360.0 - 180.0
    return a

def cross_track_error_m(lat, lon, lat1, lon1, lat2, lon2):
    """
    Cross-track error from current (lat,lon) to the great-circle path (lat1,lon1)->(lat2,lon2).
    For short segments, this is a good local proxy (sign indicates side).
    """
    d13 = haversine_m(lat1, lon1, lat, lon) / R_EARTH
    θ13 = math.radians(bearing_deg(lat1, lon1, lat, lon))
    θ12 = math.radians(bearing_deg(lat1, lon1, lat2, lon2))
    xte = math.asin(math.sin(d13) * math.sin(θ13 - θ12)) * R_EARTH
    return xte  # meters (left negative, right positive relative to path direction)

# -----------------------
# Stream helpers
# -----------------------
def open_serial(port, baud, timeout=0.1):
    if serial is None:
        raise RuntimeError("pyserial not installed. pip install pyserial")
    return serial.Serial(port=port, baudrate=baud, timeout=timeout)

def open_tcp(host, port, timeout=1.0):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    s.connect((host, port))
    return s

def read_from_stream(stream, chunk=4096, is_serial=True):
    if is_serial:
        return stream.read(chunk)
    else:
        try:
            data = stream.recv(chunk)
        except socket.timeout:
            data = b""
        return data

# -----------------------
# EDIE parser wrapper
# -----------------------
class NovAtelStream:
    def __init__(self):
        self.parser = edie.Parser(encode_format=edie.ENCODE_FORMAT.FLATTENED_BINARY)
        # Filter messages we care about
        self.parser.filter.include_message_name("BESTPOS")   # lat/lon/height
        self.parser.filter.include_message_name("BESTVEL")   # speed/course (if configured)
        self.parser.filter.include_message_name("INSPVAX")   # fused INS/GNSS with heading, speed

        # Snapshot of latest values
        self.last_lat = None
        self.last_lon = None
        self.last_heading = None   # degrees
        self.last_speed = None     # m/s
        self.last_ts = None        # ms or us depending on log (we treat as float seconds downstream)

    def write(self, data: bytes):
        if not data:
            return
        self.parser.write(data)
        for status, meta, msg_data, msg in self.parser:
            if status != edie.STATUS.SUCCESS:
                continue
            name = meta.message_name.decode() if isinstance(meta.message_name, (bytes, bytearray)) else meta.message_name
            # BESTPOS: latitude, longitude (no heading/speed)
            if name == "BESTPOS":
                self.last_lat = getattr(msg.body, "latitude", None)
                self.last_lon = getattr(msg.body, "longitude", None)
                self.last_ts = meta.milliseconds if hasattr(meta, "milliseconds") else None
            # BESTVEL: hor_speed, track_over_ground (course/heading)
            elif name == "BESTVEL":
                # Some databases expose fields as different names. Try common ones:
                sp = getattr(msg.body, "horizontal_speed", None)
                if sp is None:
                    sp = getattr(msg.body, "hor_speed", None)
                crs = getattr(msg.body, "track_over_ground", None)
                if crs is None:
                    crs = getattr(msg.body, "course", None)
                if sp is not None:
                    self.last_speed = float(sp)
                if crs is not None:
                    self.last_heading = float(crs) % 360.0
                self.last_ts = meta.milliseconds if hasattr(meta, "milliseconds") else None
            # INSPVAX: latitude, longitude, heading, speed
            elif name == "INSPVAX":
                self.last_lat = getattr(msg.body, "latitude", self.last_lat)
                self.last_lon = getattr(msg.body, "longitude", self.last_lon)
                hdg = getattr(msg.body, "heading", None)
                if hdg is not None:
                    self.last_heading = float(hdg) % 360.0
                sp = getattr(msg.body, "speed", None)
                if sp is not None:
                    self.last_speed = float(sp)
                self.last_ts = meta.milliseconds if hasattr(meta, "milliseconds") else None

    def current_fix(self):
        """Return (lat, lon, heading_deg, speed_mps, timestamp_s) or None if not ready."""
        if self.last_lat is None or self.last_lon is None:
            return None
        # timestamp can be None; we just stamp wall time if so
        t = time.time() if self.last_ts is None else (self.last_ts / 1000.0 if self.last_ts > 1e6 else float(self.last_ts))
        return (float(self.last_lat), float(self.last_lon),
                float(self.last_heading) if self.last_heading is not None else None,
                float(self.last_speed) if self.last_speed is not None else None,
                t)

# -----------------------
# Record mode
# -----------------------
def record_loop(args):
    # Open transport
    if args.serial:
        stream = open_serial(args.serial, args.baud)
        is_serial = True
        print(f"[record] Reading from serial {args.serial} @ {args.baud}")
    else:
        host, port = args.tcp.split(":")
        stream = open_tcp(host, int(port))
        is_serial = False
        print(f"[record] Reading from TCP {args.tcp}")

    nv = NovAtelStream()
    out_path = args.out
    f = open(out_path, "w", buffering=1)
    print(f"[record] Writing JSONL to {out_path}")
    print("Press Ctrl+C to stop recording.")

    min_move_m = args.min_move   # only log if moved at least this much since last point
    last_logged = None

    try:
        while True:
            data = read_from_stream(stream, is_serial=is_serial)
            nv.write(data)
            fix = nv.current_fix()
            if not fix:
                continue
            lat, lon, heading, speed, ts = fix
            if last_logged:
                d = haversine_m(lat, lon, last_logged["lat"], last_logged["lon"])
                if d < min_move_m:
                    continue
            rec = {
                "timestamp": ts,
                "lat": lat,
                "lon": lon,
                "heading_deg": heading,
                "speed_mps": speed
            }
            f.write(json.dumps(rec) + "\n")
            last_logged = {"lat": lat, "lon": lon}
            print(f"[record] {datetime.fromtimestamp(ts)} lat={lat:.7f} lon={lon:.7f}"
                  + (f" hdg={heading:.1f}° spd={speed:.2f} m/s" if heading is not None and speed is not None else ""))
    except KeyboardInterrupt:
        print("\n[record] Stopped.")
    finally:
        f.close()
        try:
            stream.close()
        except Exception:
            pass

# -----------------------
# Follow mode
# -----------------------
def load_path(jsonl_path):
    path = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                path.append(rec)
            except Exception:
                pass
    return path

def find_closest_index(lat, lon, path):
    best_i = 0
    best_d = float("inf")
    for i, p in enumerate(path):
        d = haversine_m(lat, lon, p["lat"], p["lon"])
        if d < best_d:
            best_d = d; best_i = i
    return best_i, best_d

def follow_loop(args):
    # Load reference path
    ref = load_path(args.ref)
    if len(ref) < 2:
        print(f"[follow] Reference path too short: {args.ref}")
        sys.exit(1)
    print(f"[follow] Loaded {len(ref)} reference points from {args.ref}")

    # Open transport
    if args.serial:
        stream = open_serial(args.serial, args.baud)
        is_serial = True
        print(f"[follow] Reading from serial {args.serial} @ {args.baud}")
    else:
        host, port = args.tcp.split(":")
        stream = open_tcp(host, int(port))
        is_serial = False
        print(f"[follow] Reading from TCP {args.tcp}")

    nv = NovAtelStream()
    print("Press Ctrl+C to stop. Printing guidance...")

    # Tuning knobs
    on_path_threshold_m = args.on_path
    continue_hdg_err_deg = args.hdg_continue
    rejoin_threshold_m = args.rejoin

    try:
        while True:
            data = read_from_stream(stream, is_serial=is_serial)
            nv.write(data)
            fix = nv.current_fix()
            if not fix:
                continue
            lat, lon, heading, speed, ts = fix

            # 1) Find closest reference point and next waypoint index
            idx, d_closest = find_closest_index(lat, lon, ref)
            next_idx = min(idx + 1, len(ref) - 1)

            # 2) Desired bearing along the path
            b_des = bearing_deg(ref[idx]["lat"], ref[idx]["lon"],
                                ref[next_idx]["lat"], ref[next_idx]["lon"])

            # 3) If we have current heading, compute heading error
            if heading is None:
                # Fallback: bearing from current to next ref
                b_cur_to_next = bearing_deg(lat, lon, ref[next_idx]["lat"], ref[next_idx]["lon"])
                hdg_err = wrap180(b_des - b_cur_to_next)
                have_heading = False
            else:
                hdg_err = wrap180(b_des - heading)
                have_heading = True

            # 4) Cross-track error sign to tell left/right when rejoining
            xte = cross_track_error_m(lat, lon,
                                      ref[idx]["lat"], ref[idx]["lon"],
                                      ref[next_idx]["lat"], ref[next_idx]["lon"])

            # 5) Decide command
            cmd = ""
            if abs(xte) > rejoin_threshold_m:
                # We're too far from the line: instruct rejoin direction
                cmd = f"REJOIN PATH ({'left' if xte < 0 else 'right'}) | xtrack={xte:.1f} m"
            else:
                # On/near path: steer by heading error
                if abs(hdg_err) <= continue_hdg_err_deg and d_closest <= on_path_threshold_m:
                    cmd = "CONTINUE"
                elif hdg_err > 0:
                    cmd = f"TURN RIGHT {abs(hdg_err):.0f}°"
                else:
                    cmd = f"TURN LEFT {abs(hdg_err):.0f}°"

            # 6) Print status
            status = (
                f"pos_err={d_closest:.1f} m | xtrack={xte:.1f} m | "
                f"hdg_err={hdg_err:.1f}° | desired_brg={b_des:.1f}° | "
                f"waypoint={idx}/{len(ref)-1}"
            )
            pos_str = f"lat={lat:.7f} lon={lon:.7f}"
            hs = "" if heading is None else f" hdg={heading:.1f}°"
            print(f"[follow] {pos_str}{hs} -> {cmd} || {status}")

            # optional: simple rate capping
            time.sleep(0.02)  # ~50 Hz max printing
    except KeyboardInterrupt:
        print("\n[follow] Stopped.")
    finally:
        try:
            stream.close()
        except Exception:
            pass

# -----------------------
# CLI
# -----------------------
def main():
    p = argparse.ArgumentParser(description="NovAtel path recorder/follower (print-only).")
    sub = p.add_subparsers(dest="mode", required=True)

    pr = sub.add_parser("record", help="Record a reference path (JSONL).")
    pr.add_argument("--serial", help="Serial port path, e.g. /dev/ttyUSB0 or COM3")
    pr.add_argument("--baud", type=int, default=115200, help="Serial baudrate")
    pr.add_argument("--tcp", help="TCP host:port instead of serial, e.g. 192.168.1.10:3001")
    pr.add_argument("--out", required=True, help="Output JSONL file")
    pr.add_argument("--min-move", type=float, default=1.0, help="Min movement (m) between logged points")

    pf = sub.add_parser("follow", help="Follow a saved path (print guidance).")
    pf.add_argument("--serial", help="Serial port path, e.g. /dev/ttyUSB0 or COM3")
    pf.add_argument("--baud", type=int, default=115200, help="Serial baudrate")
    pf.add_argument("--tcp", help="TCP host:port instead of serial, e.g. 192.168.1.10:3001")
    pf.add_argument("--ref", required=True, help="Reference JSONL from record mode")
    pf.add_argument("--on-path", type=float, default=5.0, help="On-path distance threshold (m)")
    pf.add_argument("--rejoin", type=float, default=12.0, help="Rejoin threshold (m) before forcing REJOIN PATH")
    pf.add_argument("--hdg-continue", type=float, default=8.0, help="Heading-error (deg) to still print CONTINUE")

    args = p.parse_args()

    if (args.serial is None) == (args.tcp is None):
        print("ERROR: specify exactly one of --serial or --tcp")
        sys.exit(2)

    if args.mode == "record":
        record_loop(args)
    else:
        follow_loop(args)

if __name__ == "__main__":
    main()