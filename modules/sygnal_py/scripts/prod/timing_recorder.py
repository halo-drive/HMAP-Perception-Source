#!/usr/bin/env python3
"""
Pure Time-Based Trajectory Recorder
Records trajectory indexed ONLY by elapsed time
Distance is secondary metadata for visualization

Usage:
    python3 time_based_recorder.py --interface can3
"""

import can
import time
import json
import struct
import signal
import sys
import math
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict
from datetime import datetime

@dataclass
class TimeIndexedWaypoint:
    """Waypoint indexed by time"""
    # PRIMARY KEY
    elapsed_time_s: float
    timestamp: float
    
    # Steering command
    steering_angle_deg: float
    
    # Speed profile (for driver to match)
    target_speed_mps: float
    
    # Metadata (for visualization)
    cumulative_distance_m: float
    wheel_speeds_kmh: Dict[str, float]
    yaw_rate_deg_s: Optional[float] = None
    
    # ISO time for debugging
    iso_time: Optional[str] = None

class TimeBasedRecorder:
    """Records trajectory with pure time indexing"""
    
    def __init__(self, interface='can3', time_interval=0.1):
        self.interface = interface
        self.time_interval = time_interval  # Record every 0.1s
        
        # CAN setup
        try:
            self.bus = can.Bus(channel=interface, bustype='socketcan', bitrate=500000)
            print(f"✓ CAN connected: {interface}")
        except Exception as e:
            print(f"✗ CAN failed: {e}")
            sys.exit(1)
        
        # CAN IDs
        self.SAS11_CAN_ID = 0x2B0
        self.WHL_SPD_CAN_ID = 0x386
        self.ESP12_CAN_ID = 0x220
        
        # Scaling
        self.SAS_ANGLE_SCALE = 0.1
        self.WHEEL_SPEED_SCALE = 0.03125
        self.YAW_RATE_SCALE = 0.01
        self.YAW_RATE_OFFSET = -40.95
        
        # State
        self.cumulative_distance_m = 0.0
        self.vehicle_speed_mps = 0.0
        self.wheel_speeds_kmh = {'FL': 0.0, 'FR': 0.0, 'RL': 0.0, 'RR': 0.0}
        self.current_steering_angle = None
        self.yaw_rate_deg_s = None
        self.last_update_time = None
        
        # Temporal state
        self.recording_start_time = None
        self.elapsed_time_s = 0.0
        self.last_recorded_time = -999.0
        
        # Storage
        self.waypoints: List[TimeIndexedWaypoint] = []
        self.recording = False
        
        print(f"✓ Time-based recorder initialized")
        print(f"  Time interval: {time_interval}s")
    
    def parse_sas11_angle(self, data: bytes) -> Optional[float]:
        """Parse SAS11 steering angle"""
        if len(data) != 5:
            return None
        try:
            data_extended = data + b'\x00\x00\x00'
            frame_uint64 = struct.unpack('<Q', data_extended)[0]
            angle_raw = frame_uint64 & 0xFFFF
            if angle_raw & 0x8000:
                angle_raw |= 0xFFFF0000
                angle_raw = struct.unpack('<i', struct.pack('<I', angle_raw & 0xFFFFFFFF))[0]
            steering_angle = angle_raw * self.SAS_ANGLE_SCALE
            if -4000.0 <= steering_angle <= 4000.0:
                return steering_angle
        except:
            pass
        return None
    
    def parse_wheel_speeds(self, data: bytes) -> Optional[Dict[str, float]]:
        """Parse wheel speeds"""
        if len(data) < 8:
            return None
        try:
            frame_uint64 = struct.unpack('<Q', data)[0]
            return {
                'FL': ((frame_uint64 >> 0) & 0x3FFF) * self.WHEEL_SPEED_SCALE,
                'FR': ((frame_uint64 >> 16) & 0x3FFF) * self.WHEEL_SPEED_SCALE,
                'RL': ((frame_uint64 >> 32) & 0x3FFF) * self.WHEEL_SPEED_SCALE,
                'RR': ((frame_uint64 >> 48) & 0x3FFF) * self.WHEEL_SPEED_SCALE,
            }
        except:
            return None
    
    def parse_esp12(self, data: bytes) -> Optional[float]:
        """Parse yaw rate"""
        if len(data) < 8:
            return None
        try:
            frame_uint64 = struct.unpack('<Q', data)[0]
            yaw_rate_raw = (frame_uint64 >> 40) & 0x1FFF
            return yaw_rate_raw * self.YAW_RATE_SCALE + self.YAW_RATE_OFFSET
        except:
            return None
    
    def update_odometry(self, wheel_speeds: Dict[str, float], timestamp: float):
        """Update distance (for visualization only)"""
        self.wheel_speeds_kmh = wheel_speeds.copy()
        rear_avg_kmh = (wheel_speeds['RL'] + wheel_speeds['RR']) / 2.0
        self.vehicle_speed_mps = rear_avg_kmh / 3.6
        
        if self.last_update_time is not None:
            dt = timestamp - self.last_update_time
            if 0.001 <= dt <= 1.0:
                distance_increment = self.vehicle_speed_mps * dt
                self.cumulative_distance_m += distance_increment
                
                if self.recording_start_time:
                    self.elapsed_time_s = timestamp - self.recording_start_time
        
        self.last_update_time = timestamp
    
    def should_record_waypoint(self) -> bool:
        """Record based on time interval"""
        time_delta = self.elapsed_time_s - self.last_recorded_time
        return time_delta >= self.time_interval
    
    def record_waypoint(self, timestamp: float):
        """Record time-indexed waypoint"""
        if self.current_steering_angle is None:
            return
        
        waypoint = TimeIndexedWaypoint(
            elapsed_time_s=self.elapsed_time_s,
            timestamp=timestamp,
            steering_angle_deg=self.current_steering_angle,
            target_speed_mps=self.vehicle_speed_mps,
            cumulative_distance_m=self.cumulative_distance_m,
            wheel_speeds_kmh=self.wheel_speeds_kmh.copy(),
            yaw_rate_deg_s=self.yaw_rate_deg_s,
            iso_time=datetime.fromtimestamp(timestamp).isoformat()
        )
        
        self.waypoints.append(waypoint)
        self.last_recorded_time = self.elapsed_time_s
    
    def start_recording(self):
        """Start recording"""
        self.recording = True
        self.recording_start_time = time.time()
        self.cumulative_distance_m = 0.0
        self.elapsed_time_s = 0.0
        self.waypoints.clear()
        self.last_update_time = None
        
        print(f"\n{'='*80}")
        print(f"TIME-BASED RECORDING STARTED")
        print(f"{'='*80}")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Recording mode: Pure Time (Distance is metadata only)")
        print(f"Press Ctrl+C to stop\n")
    
    def stop_recording(self):
        """Stop recording"""
        self.recording = False
        print(f"\n{'='*80}")
        print(f"RECORDING STOPPED")
        print(f"{'='*80}")
    
    def save_trajectory(self, filename: Optional[str] = None):
        """Save trajectory"""
        if not self.waypoints:
            print("⚠ No waypoints recorded")
            return None
        
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"time_trajectory_{timestamp}.json"
        
        # Calculate speed statistics
        speeds = [wp.target_speed_mps for wp in self.waypoints]
        avg_speed = sum(speeds) / len(speeds)
        max_speed = max(speeds)
        min_speed = min(speeds)
        
        trajectory_data = {
            "format_version": "5.0_pure_time",
            "recording_type": "time_indexed",
            "coordinate_frame": "SAS11_LEFT_POSITIVE",
            
            "metadata": {
                "start_time": self.recording_start_time,
                "end_time": self.waypoints[-1].timestamp,
                "duration_seconds": self.waypoints[-1].elapsed_time_s,
                "total_distance_m": self.waypoints[-1].cumulative_distance_m,
                "number_of_points": len(self.waypoints),
                
                "speed_profile": {
                    "average_speed_mps": avg_speed,
                    "max_speed_mps": max_speed,
                    "min_speed_mps": min_speed,
                    "target_for_driver": "Match speed profile during playback"
                },
                
                "playback_instructions": {
                    "indexing": "Pure time-based (distance ignored)",
                    "driver_task": "Control throttle to match target speed",
                    "steering": "Automatic based on elapsed time",
                    "visualization": "Real-time speed matching display"
                }
            },
            
            "waypoints": [asdict(wp) for wp in self.waypoints]
        }
        
        with open(filename, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"TRAJECTORY SAVED")
        print(f"{'='*80}")
        print(f"File: {filename}")
        print(f"Format: Pure Time-Based (v5.0)")
        print(f"\nStatistics:")
        print(f"  Duration: {self.waypoints[-1].elapsed_time_s:.1f}s")
        print(f"  Waypoints: {len(self.waypoints)}")
        print(f"  Distance (metadata): {self.waypoints[-1].cumulative_distance_m:.2f}m")
        print(f"  Speed: avg={avg_speed:.2f} m/s, range=[{min_speed:.2f}, {max_speed:.2f}]")
        print(f"\n⚠ IMPORTANT: During playback, manually control throttle to match speed!")
        print(f"{'='*80}\n")
        
        return filename
    
    def run(self):
        """Main recording loop"""
        self.start_recording()
        last_status_time = time.time()
        
        try:
            while self.recording:
                message = self.bus.recv(timeout=0.1)
                if message is None:
                    continue
                
                if message.arbitration_id == self.SAS11_CAN_ID:
                    angle = self.parse_sas11_angle(message.data)
                    if angle is not None:
                        self.current_steering_angle = angle
                
                elif message.arbitration_id == self.WHL_SPD_CAN_ID:
                    wheel_speeds = self.parse_wheel_speeds(message.data)
                    if wheel_speeds:
                        self.update_odometry(wheel_speeds, message.timestamp)
                        
                        if self.current_steering_angle is not None and self.should_record_waypoint():
                            self.record_waypoint(message.timestamp)
                
                elif message.arbitration_id == self.ESP12_CAN_ID:
                    yaw_rate = self.parse_esp12(message.data)
                    if yaw_rate is not None:
                        self.yaw_rate_deg_s = yaw_rate
                
                if time.time() - last_status_time >= 2.0:
                    print(f"[Recording] Time: {self.elapsed_time_s:6.1f}s | "
                          f"Speed: {self.vehicle_speed_mps:4.2f}m/s | "
                          f"Angle: {self.current_steering_angle:+6.1f}° | "
                          f"Waypoints: {len(self.waypoints):4d}")
                    last_status_time = time.time()
        
        except KeyboardInterrupt:
            print("\n\n⚠ Recording interrupted")
        
        finally:
            self.stop_recording()
            self.save_trajectory()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Pure Time-Based Trajectory Recorder')
    parser.add_argument('--interface', '-i', default='can3', help='CAN interface')
    parser.add_argument('--interval', '-t', type=float, default=0.1,
                       help='Time interval between waypoints (seconds)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"TIME-BASED RECORDER")
    print(f"{'='*80}")
    print(f"CAN interface: {args.interface}")
    print(f"Time interval: {args.interval}s")
    print(f"{'='*80}\n")
    
    recorder = TimeBasedRecorder(interface=args.interface, time_interval=args.interval)
    
    def signal_handler(sig, frame):
        recorder.recording = False
    
    signal.signal(signal.SIGINT, signal_handler)
    recorder.run()

if __name__ == "__main__":
    main()
