#!/usr/bin/env python3
"""
Hybrid Time-Distance Trajectory Recorder
Records both temporal and spatial indexing for accurate playback

Features:
- Dual indexing: cumulative distance + elapsed time
- Velocity profile capture
- Dynamics data (yaw rate, accelerations)
- Progress metrics computation
- Speed variation analysis

Usage:
    python3 hybrid_recorder.py --interface can3 --distance-threshold 0.5
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
class HybridTrajectoryWaypoint:
    """Waypoint with dual time-distance indexing"""
    # Spatial indexing
    cumulative_distance_m: float
    
    # Temporal indexing  
    elapsed_time_s: float
    timestamp: float
    
    # Steering command
    steering_angle_deg: float
    
    # Velocity context
    vehicle_speed_mps: float
    wheel_speeds_kmh: Dict[str, float]
    
    # Dynamics context
    yaw_rate_deg_s: Optional[float] = None
    lateral_accel_mps2: Optional[float] = None
    longitudinal_accel_mps2: Optional[float] = None
    
    # Progress metrics (computed after recording)
    distance_progress: Optional[float] = None  # 0.0 to 1.0
    time_progress: Optional[float] = None      # 0.0 to 1.0
    
    # Raw data (for debugging)
    raw_sas_data: Optional[str] = None
    iso_time: Optional[str] = None

class HybridTrajectoryRecorder:
    """
    Records trajectory with both time and distance indexing
    Enables velocity-compensated playback
    """
    
    def __init__(self, interface='can3', 
                 distance_threshold=0.5,
                 angle_threshold=2.0):
        
        self.interface = interface
        self.distance_threshold = distance_threshold
        self.angle_threshold = angle_threshold
        
        # CAN setup
        try:
            self.bus = can.Bus(channel=interface, bustype='socketcan', bitrate=500000)
            print(f"✓ CAN interface connected: {interface}")
        except Exception as e:
            print(f"✗ CAN connection failed: {e}")
            sys.exit(1)
        
        # CAN IDs
        self.SAS11_CAN_ID = 0x2B0
        self.WHL_SPD_CAN_ID = 0x386
        self.ESP12_CAN_ID = 0x220
        
        # Scaling factors
        self.SAS_ANGLE_SCALE = 0.1
        self.WHEEL_SPEED_SCALE = 0.03125
        self.YAW_RATE_SCALE = 0.01
        self.YAW_RATE_OFFSET = -40.95
        self.ACCEL_SCALE = 0.01
        self.ACCEL_OFFSET = -10.23
        
        # Odometry state
        self.cumulative_distance_m = 0.0
        self.vehicle_speed_mps = 0.0
        self.wheel_speeds_kmh = {'FL': 0.0, 'FR': 0.0, 'RL': 0.0, 'RR': 0.0}
        self.last_update_time = None
        
        # Temporal state
        self.recording_start_time = None
        self.elapsed_time_s = 0.0
        
        # Current steering angle
        self.current_steering_angle = None
        self.last_sas_timestamp = None
        
        # Vehicle dynamics
        self.yaw_rate_deg_s = None
        self.lateral_accel_mps2 = None
        self.longitudinal_accel_mps2 = None
        
        # Trajectory storage
        self.waypoints: List[HybridTrajectoryWaypoint] = []
        self.last_recorded_distance = -999.0
        self.last_recorded_angle = None
        
        # Recording state
        self.recording = False
        
        # Statistics
        self.total_messages = 0
        self.sas_messages = 0
        self.wheel_messages = 0
        self.esp_messages = 0
        
        print(f"✓ Hybrid recorder initialized")
        print(f"  Distance threshold: {distance_threshold}m")
        print(f"  Angle threshold: {angle_threshold}°")
    
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
            else:
                return None
                
        except Exception as e:
            print(f"SAS11 parsing error: {e}")
            return None
    
    def parse_wheel_speeds(self, data: bytes) -> Optional[Dict[str, float]]:
        """Parse WHL_SPD11 message"""
        if len(data) < 8:
            return None
        
        try:
            frame_uint64 = struct.unpack('<Q', data)[0]
            
            speeds_kmh = {
                'FL': ((frame_uint64 >> 0) & 0x3FFF) * self.WHEEL_SPEED_SCALE,
                'FR': ((frame_uint64 >> 16) & 0x3FFF) * self.WHEEL_SPEED_SCALE,
                'RL': ((frame_uint64 >> 32) & 0x3FFF) * self.WHEEL_SPEED_SCALE,
                'RR': ((frame_uint64 >> 48) & 0x3FFF) * self.WHEEL_SPEED_SCALE,
            }
            
            return speeds_kmh
            
        except Exception as e:
            print(f"Wheel speed parsing error: {e}")
            return None
    
    def parse_esp12(self, data: bytes) -> Optional[Dict[str, float]]:
        """Parse ESP12 for yaw rate and accelerations"""
        if len(data) < 8:
            return None
        
        try:
            frame_uint64 = struct.unpack('<Q', data)[0]
            
            # YAW_RATE: 40|13@1+ (0.01,-40.95)
            yaw_rate_raw = (frame_uint64 >> 40) & 0x1FFF
            yaw_rate = yaw_rate_raw * self.YAW_RATE_SCALE + self.YAW_RATE_OFFSET
            
            # LAT_ACCEL: 23|11@1+ (0.01,-10.23)
            lat_accel_raw = (frame_uint64 >> 23) & 0x7FF
            lat_accel = lat_accel_raw * self.ACCEL_SCALE + self.ACCEL_OFFSET
            
            # LONG_ACCEL: 12|11@1+ (0.01,-10.23)
            long_accel_raw = (frame_uint64 >> 12) & 0x7FF
            long_accel = long_accel_raw * self.ACCEL_SCALE + self.ACCEL_OFFSET
            
            return {
                'yaw_rate_deg_s': yaw_rate,
                'lateral_accel_mps2': lat_accel,
                'longitudinal_accel_mps2': long_accel
            }
            
        except Exception as e:
            print(f"ESP12 parsing error: {e}")
            return None
    
    def update_odometry(self, wheel_speeds: Dict[str, float], timestamp: float):
        """Update distance and time"""
        self.wheel_speeds_kmh = wheel_speeds.copy()
        
        # Rear axle average for vehicle speed
        rear_avg_kmh = (wheel_speeds['RL'] + wheel_speeds['RR']) / 2.0
        self.vehicle_speed_mps = rear_avg_kmh / 3.6
        
        if self.last_update_time is not None:
            dt = timestamp - self.last_update_time
            
            if 0.001 <= dt <= 1.0:
                # Update distance
                distance_increment = self.vehicle_speed_mps * dt
                self.cumulative_distance_m += distance_increment
                
                # Update elapsed time
                if self.recording_start_time:
                    self.elapsed_time_s = timestamp - self.recording_start_time
        
        self.last_update_time = timestamp
    
    def should_record_waypoint(self, current_angle: float) -> bool:
        """Determine if waypoint should be recorded"""
        if not self.waypoints:
            return True  # Always record first point
        
        distance_delta = self.cumulative_distance_m - self.last_recorded_distance
        
        # Record based on distance threshold
        if distance_delta >= self.distance_threshold:
            return True
        
        # Record based on angle threshold
        if self.last_recorded_angle is not None:
            angle_delta = abs(current_angle - self.last_recorded_angle)
            if angle_delta >= self.angle_threshold:
                return True
        
        return False
    
    def record_waypoint(self, timestamp: float):
        """Record waypoint with time and distance"""
        
        if self.current_steering_angle is None:
            return
        
        waypoint = HybridTrajectoryWaypoint(
            cumulative_distance_m=self.cumulative_distance_m,
            elapsed_time_s=self.elapsed_time_s,
            timestamp=timestamp,
            steering_angle_deg=self.current_steering_angle,
            vehicle_speed_mps=self.vehicle_speed_mps,
            wheel_speeds_kmh=self.wheel_speeds_kmh.copy(),
            yaw_rate_deg_s=self.yaw_rate_deg_s,
            lateral_accel_mps2=self.lateral_accel_mps2,
            longitudinal_accel_mps2=self.longitudinal_accel_mps2,
            iso_time=datetime.fromtimestamp(timestamp).isoformat()
        )
        
        self.waypoints.append(waypoint)
        self.last_recorded_distance = self.cumulative_distance_m
        self.last_recorded_angle = self.current_steering_angle
    
    def compute_progress_metrics(self):
        """Compute normalized progress for each waypoint"""
        if len(self.waypoints) < 2:
            return
        
        total_distance = self.waypoints[-1].cumulative_distance_m
        total_time = self.waypoints[-1].elapsed_time_s
        
        for wp in self.waypoints:
            wp.distance_progress = wp.cumulative_distance_m / total_distance if total_distance > 0 else 0.0
            wp.time_progress = wp.elapsed_time_s / total_time if total_time > 0 else 0.0
    
    def compute_speed_variation(self) -> float:
        """Compute coefficient of variation in speed"""
        if len(self.waypoints) < 2:
            return 0.0
        
        speeds = [wp.vehicle_speed_mps for wp in self.waypoints]
        mean_speed = sum(speeds) / len(speeds)
        
        if mean_speed == 0:
            return 0.0
        
        variance = sum((s - mean_speed)**2 for s in speeds) / len(speeds)
        std_dev = math.sqrt(variance)
        
        return std_dev / mean_speed  # Coefficient of variation
    
    def start_recording(self):
        """Start recording with time reference"""
        self.recording = True
        self.recording_start_time = time.time()
        self.cumulative_distance_m = 0.0
        self.elapsed_time_s = 0.0
        self.waypoints.clear()
        self.last_update_time = None
        
        print(f"\n{'='*80}")
        print(f"RECORDING STARTED")
        print(f"{'='*80}")
        print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Recording mode: Hybrid (Time + Distance)")
        print(f"Press Ctrl+C to stop recording\n")
    
    def stop_recording(self):
        """Stop recording"""
        self.recording = False
        print(f"\n{'='*80}")
        print(f"RECORDING STOPPED")
        print(f"{'='*80}")
    
    def save_trajectory(self, filename: Optional[str] = None):
        """Save trajectory with dual indexing"""
        
        if not self.waypoints:
            print("⚠ No waypoints recorded")
            return None
        
        # Compute progress metrics
        self.compute_progress_metrics()
        
        # Generate filename
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"hybrid_trajectory_{timestamp}.json"
        
        # Prepare data structure
        trajectory_data = {
            "format_version": "4.0_hybrid_time_distance",
            "recording_type": "hybrid_indexed",
            "coordinate_frame": "SAS11_LEFT_POSITIVE",
            
            "metadata": {
                "start_time": self.recording_start_time,
                "end_time": self.waypoints[-1].timestamp,
                "duration_seconds": self.waypoints[-1].elapsed_time_s,
                "total_distance_m": self.waypoints[-1].cumulative_distance_m,
                "number_of_points": len(self.waypoints),
                
                # Speed statistics
                "average_speed_mps": self.waypoints[-1].cumulative_distance_m / self.waypoints[-1].elapsed_time_s if self.waypoints[-1].elapsed_time_s > 0 else 0.0,
                "max_speed_mps": max(wp.vehicle_speed_mps for wp in self.waypoints),
                "min_speed_mps": min(wp.vehicle_speed_mps for wp in self.waypoints),
                
                # Angle statistics
                "max_angle_deg": max(wp.steering_angle_deg for wp in self.waypoints),
                "min_angle_deg": min(wp.steering_angle_deg for wp in self.waypoints),
                
                # Indexing strategy recommendation
                "recommended_playback_mode": "time_distance_fusion",
                "speed_variation_coefficient": self.compute_speed_variation(),
                
                # Message statistics
                "total_can_messages": self.total_messages,
                "sas_messages": self.sas_messages,
                "wheel_messages": self.wheel_messages,
                "esp_messages": self.esp_messages
            },
            
            "waypoints": [asdict(wp) for wp in self.waypoints]
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
        
        print(f"\n{'='*80}")
        print(f"TRAJECTORY SAVED")
        print(f"{'='*80}")
        print(f"File: {filename}")
        print(f"Format: Hybrid Time-Distance (v4.0)")
        print(f"\nStatistics:")
        print(f"  Waypoints: {len(self.waypoints)}")
        print(f"  Distance: {self.waypoints[-1].cumulative_distance_m:.2f}m")
        print(f"  Duration: {self.waypoints[-1].elapsed_time_s:.1f}s")
        print(f"  Avg speed: {trajectory_data['metadata']['average_speed_mps']:.2f}m/s")
        print(f"  Speed variation: {trajectory_data['metadata']['speed_variation_coefficient']*100:.1f}%")
        print(f"  Angle range: [{trajectory_data['metadata']['min_angle_deg']:+.1f}°, {trajectory_data['metadata']['max_angle_deg']:+.1f}°]")
        print(f"\nRecommended playback mode: {trajectory_data['metadata']['recommended_playback_mode']}")
        print(f"{'='*80}\n")
        
        return filename
    
    def run(self):
        """Main recording loop"""
        
        self.start_recording()
        
        last_status_time = time.time()
        status_interval = 2.0  # Print status every 2 seconds
        
        try:
            while self.recording:
                # Read CAN message
                message = self.bus.recv(timeout=0.1)
                
                if message is None:
                    continue
                
                self.total_messages += 1
                
                # Process steering angle (SAS11)
                if message.arbitration_id == self.SAS11_CAN_ID:
                    angle = self.parse_sas11_angle(message.data)
                    if angle is not None:
                        self.current_steering_angle = angle
                        self.last_sas_timestamp = message.timestamp
                        self.sas_messages += 1
                
                # Process wheel speeds (WHL_SPD11)
                elif message.arbitration_id == self.WHL_SPD_CAN_ID:
                    wheel_speeds = self.parse_wheel_speeds(message.data)
                    if wheel_speeds:
                        self.update_odometry(wheel_speeds, message.timestamp)
                        self.wheel_messages += 1
                        
                        # Check if should record waypoint
                        if (self.current_steering_angle is not None and 
                            self.should_record_waypoint(self.current_steering_angle)):
                            self.record_waypoint(message.timestamp)
                
                # Process vehicle dynamics (ESP12)
                elif message.arbitration_id == self.ESP12_CAN_ID:
                    dynamics = self.parse_esp12(message.data)
                    if dynamics:
                        self.yaw_rate_deg_s = dynamics['yaw_rate_deg_s']
                        self.lateral_accel_mps2 = dynamics['lateral_accel_mps2']
                        self.longitudinal_accel_mps2 = dynamics['longitudinal_accel_mps2']
                        self.esp_messages += 1
                
                # Print status periodically
                if time.time() - last_status_time >= status_interval:
                    self.print_status()
                    last_status_time = time.time()
        
        except KeyboardInterrupt:
            print("\n\n⚠ Recording interrupted by user")
        
        finally:
            self.stop_recording()
            filename = self.save_trajectory()
            
            if filename:
                print(f"\nTrajectory saved successfully: {filename}")
    
    def print_status(self):
        """Print current recording status"""
        print(f"[Recording] "
              f"Time: {self.elapsed_time_s:6.1f}s | "
              f"Distance: {self.cumulative_distance_m:7.2f}m | "
              f"Speed: {self.vehicle_speed_mps:4.2f}m/s | "
              f"Angle: {self.current_steering_angle:+6.1f}° | "
              f"Waypoints: {len(self.waypoints):4d}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Hybrid Time-Distance Trajectory Recorder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Records trajectory with dual indexing (time + distance) for accurate playback
even with varying throttle/speed.

Example usage:
    # Standard recording
    python3 hybrid_recorder.py --interface can3
    
    # Custom thresholds
    python3 hybrid_recorder.py --interface can3 --distance-threshold 0.3 --angle-threshold 3.0
        '''
    )
    
    parser.add_argument('--interface', '-i', default='can3',
                       help='CAN interface (default: can3)')
    parser.add_argument('--distance-threshold', '-d', type=float, default=0.5,
                       help='Distance threshold for waypoint recording (meters, default: 0.5)')
    parser.add_argument('--angle-threshold', '-a', type=float, default=2.0,
                       help='Angle threshold for waypoint recording (degrees, default: 2.0)')
    parser.add_argument('--output', '-o', default=None,
                       help='Output filename (default: auto-generated)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"HYBRID TRAJECTORY RECORDER")
    print(f"{'='*80}")
    print(f"CAN interface: {args.interface}")
    print(f"Distance threshold: {args.distance_threshold}m")
    print(f"Angle threshold: {args.angle_threshold}°")
    print(f"{'='*80}\n")
    
    # Create recorder
    recorder = HybridTrajectoryRecorder(
        interface=args.interface,
        distance_threshold=args.distance_threshold,
        angle_threshold=args.angle_threshold
    )
    
    # Setup signal handler
    def signal_handler(sig, frame):
        print("\n\n⚠ Stopping recorder...")
        recorder.recording = False
    
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run recorder
    recorder.run()

if __name__ == "__main__":
    main()
