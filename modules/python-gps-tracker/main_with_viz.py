#!/usr/bin/env python3
"""
NovAtel GPS Path Follower with Real-Time Visualization

Features:
  - Automatic loop closure detection
  - Optimized windowed search
  - Real-time OpenCV visualization
  - Recording and following modes with visual feedback

Deps:
  pip install pyserial novatel_edie opencv-python numpy

Usage:
  # RECORD with visualization
  python3 main_with_viz.py record --serial /dev/ttyUSB0 --baud 115200 --out loop.jsonl

  # FOLLOW with visualization
  python3 main_with_viz.py follow --serial /dev/ttyUSB0 --baud 115200 --ref loop.jsonl
"""

import argparse
import json
import math
import socket
import sys
import time
from datetime import datetime

try:
    import serial
except ImportError:
    serial = None

import novatel_edie as edie

# Import visualization module
try:
    from gps_visualizer import GPSVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    print("[WARNING] gps_visualizer.py not found. Running without visualization.")
    print("[WARNING] Place gps_visualizer.py in the same directory as this script.")
    VISUALIZATION_AVAILABLE = False

# -----------------------
# Geo helpers
# -----------------------
R_EARTH = 6371000.0

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
    """Cross-track error from current position to great-circle path."""
    d13 = haversine_m(lat1, lon1, lat, lon) / R_EARTH
    θ13 = math.radians(bearing_deg(lat1, lon1, lat, lon))
    θ12 = math.radians(bearing_deg(lat1, lon1, lat2, lon2))
    xte = math.asin(math.sin(d13) * math.sin(θ13 - θ12)) * R_EARTH
    return xte

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
        # Initialize parser with proper configuration for NovAtel PwrPak7
        # Based on official EDIE examples - need to pass JSON database
        try:
            # Try to initialize with built-in database
            json_db = edie.get_builtin_database()
            self.parser = edie.Parser(json_db)
            print("[LOG] Parser initialized with built-in database")
        except Exception as e:
            # Fallback to basic parser initialization
            self.parser = edie.Parser()
            print(f"[LOG] Parser initialized without database: {e}")
        
        # Configure filter for NovAtel messages (based on examples)
        # Try without filter first to see what messages we get
        # self.parser.filter = edie.Filter()
        # self.parser.filter.add_message_name("BESTPOS")   # Best position solution
        # self.parser.filter.add_message_name("BESTVEL")   # Best velocity solution  
        # self.parser.filter.add_message_name("INSPVAX")   # INS position, velocity, attitude
        # self.parser.filter.add_message_name("INSPVAS")   # INS position, velocity, attitude (short)
        # self.parser.filter.add_message_name("PSRPOS")    # Precise position
        # self.parser.filter.add_message_name("PSRVEL")    # Precise velocity
        
        # Initialize logging
        self.message_count = 0
        self.last_log_time = time.time()

        self.last_lat = None
        self.last_lon = None
        self.last_heading = None
        self.last_speed = None
        self.last_ts = None

    def write(self, data: bytes):
        if not data:
            return
        
        # Log data reception
        current_time = time.time()
        if current_time - self.last_log_time > 5.0:  # Log every 5 seconds
            print(f"[LOG] Received {len(data)} bytes of data")
            # Show first 100 bytes as hex for debugging
            if len(data) > 0:
                hex_data = data[:100].hex()
                print(f"[LOG] Raw data (first 100 bytes): {hex_data}")
                # Also try to decode as ASCII to see if it's text
                try:
                    ascii_data = data[:100].decode('ascii', errors='ignore')
                    print(f"[LOG] ASCII data (first 100 bytes): {ascii_data}")
                except:
                    pass
            self.last_log_time = current_time
        
        self.parser.write(data)
        
        # Parse messages - handle both old and new EDIE API formats
        try:
            # Try the standard parser iteration pattern from examples
            for status, meta_data, message_data, message in self.parser:
                if status != edie.STATUS.SUCCESS:
                    if status != edie.STATUS.INCOMPLETE:  # Don't log incomplete status (normal)
                        print(f"[LOG] Parser status: {status}")
                    continue
                
                self.message_count += 1
                name = meta_data.message_name.decode() if isinstance(meta_data.message_name, (bytes, bytearray)) else meta_data.message_name
                print(f"[LOG] Message #{self.message_count}: {name}")
                
                # Process the message based on its type
                self._process_message(name, message)
                
        except TypeError:
            # Handle case where parser returns message objects directly (newer API)
            for msg in self.parser:
                # Get message type from the message object itself
                msg_type = type(msg).__name__
                self.message_count += 1
                print(f"[LOG] Message #{self.message_count} (direct): {msg_type}")
                
                # Process the message based on its type
                self._process_message(msg_type, msg)
        except Exception as e:
            print(f"[LOG] Parser error: {e}")
    
    def _process_message(self, name, message):
        """Process a parsed message based on its type."""
        if name == "BESTPOS":
            if hasattr(message, 'latitude') and hasattr(message, 'longitude'):
                self.last_lat = getattr(message, "latitude", None)
                self.last_lon = getattr(message, "longitude", None)
                print(f"[LOG] BESTPOS: Lat={self.last_lat}, Lon={self.last_lon}")
            else:
                print(f"[LOG] BESTPOS: No position data available")
        elif name == "INSPVAX":
            if hasattr(message, 'latitude') and hasattr(message, 'longitude'):
                self.last_lat = getattr(message, "latitude", self.last_lat)
                self.last_lon = getattr(message, "longitude", self.last_lon)
                hdg = getattr(message, "azimuth", None)
                if hdg is not None:
                    self.last_heading = float(hdg) % 360.0
                # Calculate speed from velocity components
                north_vel = getattr(message, "north_vel", 0.0)
                east_vel = getattr(message, "east_vel", 0.0)
                if north_vel is not None and east_vel is not None:
                    self.last_speed = (float(north_vel)**2 + float(east_vel)**2)**0.5
                print(f"[LOG] INSPVAX: Lat={self.last_lat}, Lon={self.last_lon}, Heading={self.last_heading}, Speed={self.last_speed}")
                print(f"[LOG] DEBUG: Stored lat={self.last_lat}, lon={self.last_lon} in object")
            else:
                print(f"[LOG] INSPVAX: No position data available")
        elif name == "BESTVEL":
            if hasattr(message, 'horizontal_speed'):
                sp = getattr(message, "horizontal_speed", None)
                if sp is not None:
                    self.last_speed = float(sp)
            if hasattr(message, 'track_over_ground'):
                crs = getattr(message, "track_over_ground", None)
                if crs is not None:
                    self.last_heading = float(crs) % 360.0
            print(f"[LOG] BESTVEL: Speed={self.last_speed}, Heading={self.last_heading}")

    def current_fix(self):
        """Return (lat, lon, heading_deg, speed_mps, timestamp_s) or None."""
        print(f"[LOG] DEBUG: current_fix() called - last_lat={self.last_lat}, last_lon={self.last_lon}")
        if self.last_lat is None or self.last_lon is None:
            print(f"[LOG] DEBUG: current_fix() returning None because lat or lon is None")
            return None
        t = time.time() if self.last_ts is None else (self.last_ts / 1000.0 if self.last_ts > 1e6 else float(self.last_ts))
        result = (float(self.last_lat), float(self.last_lon),
                float(self.last_heading) if self.last_heading is not None else None,
                float(self.last_speed) if self.last_speed is not None else None,
                t)
        print(f"[LOG] DEBUG: current_fix() returning: {result}")
        return result
    
    def get_stats(self):
        """Return parser statistics."""
        return {
            "message_count": self.message_count,
            "last_lat": self.last_lat,
            "last_lon": self.last_lon,
            "last_heading": self.last_heading,
            "last_speed": self.last_speed
        }

# -----------------------
# Loop Closure Detector
# -----------------------
class LoopClosureDetector:
    def __init__(self, closure_distance_m=10.0, min_points=20):
        self.start_lat = None
        self.start_lon = None
        self.closure_distance = closure_distance_m
        self.min_points = min_points
        self.point_count = 0
        self.closure_detected = False
        
    def add_start_point(self, lat, lon):
        self.start_lat = lat
        self.start_lon = lon
        self.point_count = 1
        print(f"[loop] Starting position: lat={lat:.7f}, lon={lon:.7f}")
        print(f"[loop] Will auto-close when within {self.closure_distance}m of start")
    
    def check_closure(self, lat, lon):
        if self.closure_detected:
            return True
            
        self.point_count += 1
        
        if self.point_count < self.min_points:
            return False
        
        dist_to_start = haversine_m(self.start_lat, self.start_lon, lat, lon)
        
        if dist_to_start <= self.closure_distance:
            self.closure_detected = True
            print(f"\n{'='*60}")
            print(f"[loop] ✓ LOOP CLOSED! Distance: {dist_to_start:.1f}m")
            print(f"[loop] Total points: {self.point_count}")
            print(f"{'='*60}\n")
            return True
        
        if self.point_count % 50 == 0:
            print(f"[loop] {self.point_count} points | Dist to start: {dist_to_start:.1f}m")
        
        return False

# -----------------------
# Optimized Path Follower
# -----------------------
class OptimizedPathFollower:
    def __init__(self, path, search_window=100):
        self.path = path
        self.path_len = len(path)
        self.search_window = search_window
        self.last_idx = 0
        self.initialized = False
        
        print(f"[follower] Loaded {self.path_len} reference points")
        print(f"[follower] Using windowed search (±{search_window} points)")
        
    def initial_localization(self, lat, lon):
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
        print(f"[follower] ✓ Localized at waypoint {best_i}/{self.path_len-1} ({best_d:.1f}m)")
        return best_i, best_d
    
    def find_closest_point(self, lat, lon):
        if not self.initialized:
            return self.initial_localization(lat, lon)
        
        best_i = self.last_idx
        best_d = haversine_m(lat, lon, 
                            self.path[best_i]["lat"], 
                            self.path[best_i]["lon"])
        
        search_indices = []
        for offset in range(-self.search_window, self.search_window + 1):
            idx = (self.last_idx + offset) % self.path_len
            search_indices.append(idx)
        
        for i in search_indices:
            d = haversine_m(lat, lon, self.path[i]["lat"], self.path[i]["lon"])
            if d < best_d:
                best_d = d
                best_i = i
        
        self.last_idx = best_i
        return best_i, best_d
    
    def get_next_waypoint(self, current_idx):
        return (current_idx + 1) % self.path_len

# -----------------------
# Record mode with visualization
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

    # Initialize visualization
    viz = None
    if VISUALIZATION_AVAILABLE and not args.no_viz:
        viz = GPSVisualizer(window_name="GPS Path Recorder")
        print("[record] Visualization enabled")
    else:
        print("[record] Running without visualization")

    nv = NovAtelStream()
    out_path = args.out
    f = open(out_path, "w", buffering=1)
    
    # Create log file for debugging
    log_path = out_path.replace('.jsonl', '_debug.log')
    log_file = open(log_path, "w", buffering=1)
    
    print(f"[record] Writing GPS data to {out_path}")
    print(f"[record] Writing debug logs to {log_path}")
    print("Drive in a loop. Will auto-stop when loop closes.")
    print("(Or press Ctrl+C or ESC to stop)\n")
    
    def log_debug(message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_msg = f"[{timestamp}] {message}\n"
        print(log_msg.strip())
        log_file.write(log_msg)
        log_file.flush()

    min_move_m = args.min_move
    loop_detector = LoopClosureDetector(
        closure_distance_m=args.loop_closure,
        min_points=args.min_loop_points
    )
    
    last_logged = None
    first_point = True

    try:
        loop_count = 0
        log_debug("Starting GPS data collection loop")
        
        while True:
            loop_count += 1
            data = read_from_stream(stream, is_serial=is_serial)
            nv.write(data)
            fix = nv.current_fix()
            
            if loop_count % 100 == 0:  # Log every 100 iterations
                stats = nv.get_stats()
                log_debug(f"Loop {loop_count}: Messages received: {stats['message_count']}, "
                         f"Lat: {stats['last_lat']}, Lon: {stats['last_lon']}, "
                         f"Heading: {stats['last_heading']}, Speed: {stats['last_speed']}")
            
            if not fix:
                # Update visualization even without GPS
                if viz:
                    # Check if window is still open
                    if not viz.is_window_open():
                        print("[record] GUI window was closed. Stopping application.")
                        break
                    key = viz.show(mode="record", wait_key=1)
                    if key == 27:  # ESC or window closed
                        break
                continue
            
            lat, lon, heading, speed, ts = fix
            
            # Update visualization with current position
            if viz:
                viz.update_current_position(lat, lon, heading, speed)
            
            # Check if moved enough
            if last_logged:
                d = haversine_m(lat, lon, last_logged["lat"], last_logged["lon"])
                if d < min_move_m:
                    # Still update visualization
                    if viz:
                        key = viz.show(mode="record", wait_key=1)
                        if key == 27:  # ESC
                            break
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
            
            # Add to visualization
            if viz:
                viz.add_recorded_point(lat, lon)
            
            # First point: record as start
            if first_point:
                loop_detector.add_start_point(lat, lon)
                first_point = False
            else:
                # Check for loop closure
                if loop_detector.check_closure(lat, lon):
                    print("[record] Loop complete! Stopping.")
                    break
            
            # Print progress
            print(f"[record] {datetime.fromtimestamp(ts)} lat={lat:.7f} lon={lon:.7f}"
                  + (f" hdg={heading:.1f}° spd={speed:.2f} m/s" if heading and speed else ""))
            
            # Update visualization
            if viz:
                # Check if window is still open
                if not viz.is_window_open():
                    print("[record] GUI window was closed. Stopping application.")
                    break
                key = viz.show(mode="record", wait_key=1)
                if key == 27:  # ESC or window closed
                    break
            
    except KeyboardInterrupt:
        print("\n[record] Manually stopped.")
    except Exception as e:
        print(f"\n[record] Error occurred: {e}")
    finally:
        try:
            f.close()
        except:
            pass
        try:
            log_file.close()
        except:
            pass
        try:
            stream.close()
        except Exception:
            pass
        if viz:
            viz.cleanup()
        
        # Final statistics (only if log_file is still open)
        try:
            final_stats = nv.get_stats()
            log_debug(f"Final stats: Messages received: {final_stats['message_count']}, "
                     f"Final position: Lat={final_stats['last_lat']}, Lon={final_stats['last_lon']}")
        except:
            final_stats = nv.get_stats()
            print(f"[record] Final stats: Messages received: {final_stats['message_count']}, "
                  f"Final position: Lat={final_stats['last_lat']}, Lon={final_stats['last_lon']}")
        
        print(f"[record] Saved GPS data to {out_path}")
        print(f"[record] Saved debug logs to {log_path}")

# -----------------------
# Follow mode with visualization
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

def follow_loop(args):
    # Load reference path
    ref = load_path(args.ref)
    if len(ref) < 2:
        print(f"[follow] Reference path too short: {args.ref}")
        sys.exit(1)
    
    # Initialize path follower
    follower = OptimizedPathFollower(ref, search_window=args.search_window)

    # Initialize visualization
    viz = None
    if VISUALIZATION_AVAILABLE and not args.no_viz:
        viz = GPSVisualizer(window_name="GPS Path Follower")
        viz.set_reference_path(ref)
        print("[follow] Visualization enabled")
    else:
        print("[follow] Running without visualization")

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
    print("\nWaiting for GPS fix...")
    print("Press Ctrl+C or ESC to stop.\n")

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
                # Update visualization even without GPS
                if viz:
                    # Check if window is still open
                    if not viz.is_window_open():
                        print("[follow] GUI window was closed. Stopping application.")
                        break
                    key = viz.show(mode="follow", wait_key=1)
                    if key == 27:  # ESC or window closed
                        break
                continue
            
            lat, lon, heading, speed, ts = fix

            # Find closest reference point
            idx, d_closest = follower.find_closest_point(lat, lon)
            next_idx = follower.get_next_waypoint(idx)

            # Desired bearing
            b_des = bearing_deg(ref[idx]["lat"], ref[idx]["lon"],
                                ref[next_idx]["lat"], ref[next_idx]["lon"])

            # Heading error
            if heading is None:
                b_cur_to_next = bearing_deg(lat, lon, ref[next_idx]["lat"], ref[next_idx]["lon"])
                hdg_err = wrap180(b_des - b_cur_to_next)
            else:
                hdg_err = wrap180(b_des - heading)

            # Cross-track error
            xte = cross_track_error_m(lat, lon,
                                      ref[idx]["lat"], ref[idx]["lon"],
                                      ref[next_idx]["lat"], ref[next_idx]["lon"])

            # Decide command
            if abs(xte) > rejoin_threshold_m:
                cmd = f"REJOIN PATH ({'left' if xte < 0 else 'right'}) | xtrack={xte:.1f} m"
            else:
                if abs(hdg_err) <= continue_hdg_err_deg and d_closest <= on_path_threshold_m:
                    cmd = "CONTINUE"
                elif hdg_err > 0:
                    cmd = f"TURN RIGHT {abs(hdg_err):.0f}°"
                else:
                    cmd = f"TURN LEFT {abs(hdg_err):.0f}°"

            # Print status
            status = (
                f"pos_err={d_closest:.1f} m | xtrack={xte:.1f} m | "
                f"hdg_err={hdg_err:.1f}° | desired_brg={b_des:.1f}° | "
                f"waypoint={idx}/{len(ref)-1}"
            )
            pos_str = f"lat={lat:.7f} lon={lon:.7f}"
            hs = "" if heading is None else f" hdg={heading:.1f}°"
            print(f"[follow] {pos_str}{hs} -> {cmd} || {status}")

            # Update visualization
            if viz:
                # Check if window is still open
                if not viz.is_window_open():
                    print("[follow] GUI window was closed. Stopping application.")
                    break
                viz.update_current_position(lat, lon, heading, speed)
                viz.update_errors(d_closest, xte, hdg_err)
                key = viz.show(mode="follow", wait_key=1)
                if key == 27:  # ESC or window closed
                    break

            time.sleep(0.02)
    except KeyboardInterrupt:
        print("\n[follow] Stopped.")
    except Exception as e:
        print(f"\n[follow] Error occurred: {e}")
    finally:
        try:
            stream.close()
        except Exception:
            pass
        if viz:
            viz.cleanup()

# -----------------------
# CLI
# -----------------------
def main():
    p = argparse.ArgumentParser(description="NovAtel GPS tracker with visualization.")
    sub = p.add_subparsers(dest="mode", required=True)

    pr = sub.add_parser("record", help="Record a path with visualization.")
    pr.add_argument("--serial", help="Serial port")
    pr.add_argument("--baud", type=int, default=115200, help="Baudrate")
    pr.add_argument("--tcp", help="TCP host:port")
    pr.add_argument("--out", required=True, help="Output JSONL file")
    pr.add_argument("--min-move", type=float, default=1.0, help="Min movement (m)")
    pr.add_argument("--loop-closure", type=float, default=10.0, help="Loop closure distance (m)")
    pr.add_argument("--min-loop-points", type=int, default=20, help="Min points before closure check")
    pr.add_argument("--no-viz", action="store_true", help="Disable visualization")

    pf = sub.add_parser("follow", help="Follow a path with visualization.")
    pf.add_argument("--serial", help="Serial port")
    pf.add_argument("--baud", type=int, default=115200, help="Baudrate")
    pf.add_argument("--tcp", help="TCP host:port")
    pf.add_argument("--ref", required=True, help="Reference JSONL")
    pf.add_argument("--search-window", type=int, default=100, help="Search window size")
    pf.add_argument("--on-path", type=float, default=5.0, help="On-path threshold (m)")
    pf.add_argument("--rejoin", type=float, default=12.0, help="Rejoin threshold (m)")
    pf.add_argument("--hdg-continue", type=float, default=8.0, help="Heading error for CONTINUE (deg)")
    pf.add_argument("--no-viz", action="store_true", help="Disable visualization")

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