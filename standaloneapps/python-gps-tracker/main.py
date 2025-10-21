#!/usr/bin/env python3
"""
NovAtel live GPS recorder/follower (print-only guidance) - OPTIMIZED VERSION

Features:
  - Automatic loop closure detection
  - Optimized windowed search for large paths
  - Circular path handling with wraparound
  - Smart initial localization

Deps:
  pip install pyserial novatel_edie

Usage:
  # 1) RECORD a loop/path (auto-stops when loop closes)
  python main_optimized.py record --serial /dev/ttyUSB0 --baud 115200 --out loop_ref.jsonl

  # 2) FOLLOW the saved path
  python main_optimized.py follow --serial /dev/ttyUSB0 --baud 115200 --ref loop_ref.jsonl
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
# Loop Closure Detector
# -----------------------
class LoopClosureDetector:
    """Detects when a circular path returns to its starting point."""
    
    def __init__(self, closure_distance_m=10.0, min_points=20):
        """
        Args:
            closure_distance_m: Distance to start point to trigger closure (meters)
            min_points: Minimum number of points before checking for closure
        """
        self.start_lat = None
        self.start_lon = None
        self.closure_distance = closure_distance_m
        self.min_points = min_points
        self.point_count = 0
        self.closure_detected = False
        
    def add_start_point(self, lat, lon):
        """Record the starting position."""
        self.start_lat = lat
        self.start_lon = lon
        self.point_count = 1
        print(f"[loop] Starting position: lat={lat:.7f}, lon={lon:.7f}")
        print(f"[loop] Will auto-close when within {self.closure_distance}m of start (after >{self.min_points} points)")
    
    def check_closure(self, lat, lon):
        """
        Check if current position closes the loop.
        Returns True if loop is closed.
        """
        if self.closure_detected:
            return True
            
        self.point_count += 1
        
        # Don't check until we have minimum points (avoid false trigger at start)
        if self.point_count < self.min_points:
            return False
        
        # Calculate distance to start
        dist_to_start = haversine_m(self.start_lat, self.start_lon, lat, lon)
        
        # Check if we're back at start
        if dist_to_start <= self.closure_distance:
            self.closure_detected = True
            print(f"\n{'='*60}")
            print(f"[loop] ✓ LOOP CLOSED! Distance to start: {dist_to_start:.1f}m")
            print(f"[loop] Total points recorded: {self.point_count}")
            print(f"{'='*60}\n")
            return True
        
        # Periodically show distance to start
        if self.point_count % 50 == 0:
            print(f"[loop] Progress: {self.point_count} points | Distance to start: {dist_to_start:.1f}m")
        
        return False

# -----------------------
# Optimized Path Follower
# -----------------------
class OptimizedPathFollower:
    """Efficient path following with windowed search and circular path handling."""
    
    def __init__(self, path, search_window=100):
        """
        Args:
            path: List of reference points
            search_window: Number of points to search around last position
        """
        self.path = path
        self.path_len = len(path)
        self.search_window = search_window
        self.last_idx = 0
        self.initialized = False
        
        print(f"[follower] Loaded {self.path_len} reference points")
        print(f"[follower] Using windowed search (±{search_window} points)")
        
    def initial_localization(self, lat, lon):
        """
        Perform full search once at startup to find initial position on track.
        This is O(n) but only happens once.
        """
        print(f"[follower] Performing initial localization...")
        best_i = 0
        best_d = float("inf")
        
        for i, p in enumerate(self.path):
            d = haversine_m(lat, lon, p["lat"], p["lon"])
            if d < best_d:
                best_d = d
                best_i = i
        
        self.last_idx = best_i
        self.initialized = True
        print(f"[follower] ✓ Localized at waypoint {best_i}/{self.path_len-1} (distance: {best_d:.1f}m)")
        return best_i, best_d
    
    def find_closest_point(self, lat, lon):
        """
        Find closest point using windowed search.
        For circular paths, handles wraparound at start/end.
        """
        # First time: do full search
        if not self.initialized:
            return self.initial_localization(lat, lon)
        
        # Windowed search around last known position
        best_i = self.last_idx
        best_d = haversine_m(lat, lon, 
                            self.path[best_i]["lat"], 
                            self.path[best_i]["lon"])
        
        # Create search range with wraparound for circular paths
        search_indices = []
        
        for offset in range(-self.search_window, self.search_window + 1):
            idx = (self.last_idx + offset) % self.path_len  # Circular wraparound
            search_indices.append(idx)
        
        # Search only in window
        for i in search_indices:
            d = haversine_m(lat, lon, self.path[i]["lat"], self.path[i]["lon"])
            if d < best_d:
                best_d = d
                best_i = i
        
        self.last_idx = best_i
        return best_i, best_d
    
    def get_next_waypoint(self, current_idx):
        """Get next waypoint with circular wraparound."""
        return (current_idx + 1) % self.path_len

# -----------------------
# Record mode with loop closure
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
    print("Drive in a loop. Recording will auto-stop when loop closes.")
    print("(Or press Ctrl+C to stop manually)\n")

    min_move_m = args.min_move
    loop_detector = LoopClosureDetector(
        closure_distance_m=args.loop_closure,
        min_points=args.min_loop_points
    )
    
    last_logged = None
    first_point = True

    try:
        while True:
            data = read_from_stream(stream, is_serial=is_serial)
            nv.write(data)
            fix = nv.current_fix()
            if not fix:
                continue
            
            lat, lon, heading, speed, ts = fix
            
            # Check if moved enough since last point
            if last_logged:
                d = haversine_m(lat, lon, last_logged["lat"], last_logged["lon"])
                if d < min_move_m:
                    continue
            
            # Record point
            rec = {
                "timestamp": ts,
                "lat": lat,
                "lon": lon,
                "heading_deg": heading,
                "speed_mps": speed
            }
            f.write(json.dumps(rec) + "\n")
            last_logged = {"lat": lat, "lon": lon}
            
            # First point: record as start
            if first_point:
                loop_detector.add_start_point(lat, lon)
                first_point = False
            else:
                # Check for loop closure
                if loop_detector.check_closure(lat, lon):
                    print("[record] Loop complete! Stopping recording.")
                    break
            
            # Print progress
            print(f"[record] {datetime.fromtimestamp(ts)} lat={lat:.7f} lon={lon:.7f}"
                  + (f" hdg={heading:.1f}° spd={speed:.2f} m/s" if heading is not None and speed is not None else ""))
            
    except KeyboardInterrupt:
        print("\n[record] Manually stopped.")
    finally:
        f.close()
        try:
            stream.close()
        except Exception:
            pass
        print(f"[record] Saved to {out_path}")

# -----------------------
# Follow mode with optimized search
# -----------------------
def load_path(jsonl_path):
    """Load path from JSONL file."""
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

def follow_loop(args):
    # Load reference path
    ref = load_path(args.ref)
    if len(ref) < 2:
        print(f"[follow] Reference path too short: {args.ref}")
        sys.exit(1)
    
    # Initialize optimized path follower
    follower = OptimizedPathFollower(ref, search_window=args.search_window)

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
    print("\nWaiting for GPS fix to localize on track...")
    print("Press Ctrl+C to stop.\n")

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

            # 1) Find closest reference point (optimized windowed search)
            idx, d_closest = follower.find_closest_point(lat, lon)
            
            # 2) Get next waypoint (with circular wraparound)
            next_idx = follower.get_next_waypoint(idx)

            # 3) Desired bearing along the path
            b_des = bearing_deg(ref[idx]["lat"], ref[idx]["lon"],
                                ref[next_idx]["lat"], ref[next_idx]["lon"])

            # 4) If we have current heading, compute heading error
            if heading is None:
                # Fallback: bearing from current to next ref
                b_cur_to_next = bearing_deg(lat, lon, ref[next_idx]["lat"], ref[next_idx]["lon"])
                hdg_err = wrap180(b_des - b_cur_to_next)
                have_heading = False
            else:
                hdg_err = wrap180(b_des - heading)
                have_heading = True

            # 5) Cross-track error sign to tell left/right when rejoining
            xte = cross_track_error_m(lat, lon,
                                      ref[idx]["lat"], ref[idx]["lon"],
                                      ref[next_idx]["lat"], ref[next_idx]["lon"])

            # 6) Decide command
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

            # 7) Print status
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
    p = argparse.ArgumentParser(description="NovAtel path recorder/follower with loop closure and optimization.")
    sub = p.add_subparsers(dest="mode", required=True)

    pr = sub.add_parser("record", help="Record a reference path (auto-closes loop).")
    pr.add_argument("--serial", help="Serial port path, e.g. /dev/ttyUSB0 or COM3")
    pr.add_argument("--baud", type=int, default=115200, help="Serial baudrate")
    pr.add_argument("--tcp", help="TCP host:port instead of serial, e.g. 192.168.1.10:3001")
    pr.add_argument("--out", required=True, help="Output JSONL file")
    pr.add_argument("--min-move", type=float, default=1.0, help="Min movement (m) between logged points")
    pr.add_argument("--loop-closure", type=float, default=10.0, help="Distance to start (m) to trigger loop closure")
    pr.add_argument("--min-loop-points", type=int, default=20, help="Minimum points before checking loop closure")

    pf = sub.add_parser("follow", help="Follow a saved path (optimized with windowed search).")
    pf.add_argument("--serial", help="Serial port path, e.g. /dev/ttyUSB0 or COM3")
    pf.add_argument("--baud", type=int, default=115200, help="Serial baudrate")
    pf.add_argument("--tcp", help="TCP host:port instead of serial, e.g. 192.168.1.10:3001")
    pf.add_argument("--ref", required=True, help="Reference JSONL from record mode")
    pf.add_argument("--search-window", type=int, default=100, help="Search window size (points) for optimization")
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