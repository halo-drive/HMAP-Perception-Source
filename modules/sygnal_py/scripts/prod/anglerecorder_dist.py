#!/usr/bin/env python3
"""
Odometry-Enhanced Steering Trajectory Recorder
Distance-based trajectory capture with wheel speed integration

Records steering trajectory indexed by distance traveled rather than time.
Designed for closed-loop path replication with repeatable spatial accuracy.

Key features:
- Distance-based trajectory indexing from wheel speeds
- Yaw rate and acceleration capture for future heading estimation
- Parking lot drift compensation through calibration
- Loop-ready output format with distance normalization
"""

import can
import struct
import time
import argparse
import sys
import csv
import json
import threading
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
from collections import deque
import statistics

@dataclass
class OdometryData:
    """Real-time odometry state"""
    cumulative_distance_m: float
    vehicle_speed_mps: float
    wheel_speeds_kmh: Dict[str, float]
    yaw_rate_deg_s: Optional[float]
    lateral_accel_mps2: Optional[float]
    longitudinal_accel_mps2: Optional[float]

@dataclass
class EnhancedTrajectoryPoint:
    """Distance-indexed trajectory point with full vehicle state"""
    # Core trajectory data
    timestamp: float
    cumulative_distance_m: float
    steering_angle_deg: float
    
    # Odometry data
    vehicle_speed_mps: float
    wheel_speeds_kmh: Dict[str, float]
    
    # Vehicle dynamics
    yaw_rate_deg_s: Optional[float] = None
    lateral_accel_mps2: Optional[float] = None
    longitudinal_accel_mps2: Optional[float] = None
    
    # Delta information
    angle_change_deg: float = 0.0
    distance_increment_m: float = 0.0
    
    # Metadata
    message_count: int = 0
    iso_time: str = ""
    raw_sas_data: str = ""
    
    # Quality metrics
    signal_quality_score: Optional[float] = None
    data_completeness: float = 100.0  # % of expected signals received

@dataclass
class OdometryCalibration:
    """Calibration data for odometry accuracy"""
    wheel_circumference_m: float = 2.0  # Typical for passenger car (205/55R16)
    wheel_slip_factor: float = 1.0      # Correction factor for rough surfaces
    distance_scale_factor: float = 1.0  # Overall calibration from known distance
    yaw_rate_bias_deg_s: float = 0.0    # Sensor bias correction

@dataclass
class TrajectoryMetadata:
    """Complete trajectory session metadata"""
    start_time: float
    end_time: float
    duration_seconds: float
    total_distance_m: float
    average_speed_mps: float
    max_speed_mps: float
    min_speed_mps: float
    number_of_points: int
    coordinate_frame: str = "SAS11_LEFT_POSITIVE"
    loop_closure_error_m: Optional[float] = None  # Distance between start/end
    
class OdometryTrajectoryRecorder:
    def __init__(self, interface='can3', angle_threshold=5.0, distance_threshold=0.5):
        self.interface = interface
        self.angle_threshold = angle_threshold  # degrees
        self.distance_threshold = distance_threshold  # meters
        self.start_time = time.time()
        
        # CAN IDs
        self.SAS11_CAN_ID = 0x2B0   # 688 - Steering angle
        self.WHL_SPD_CAN_ID = 0x386  # 902 - Wheel speeds
        self.ESP12_CAN_ID = 0x220    # 544 - Yaw rate, accelerations
        
        # Scaling factors from DBC
        self.SAS_ANGLE_SCALE = 0.1        # deg per LSB
        self.WHEEL_SPEED_SCALE = 0.03125  # km/h per LSB
        self.YAW_RATE_SCALE = 0.01        # deg/s per LSB
        self.YAW_RATE_OFFSET = -40.95     # deg/s
        self.ACCEL_SCALE = 0.01           # m/s^2 per LSB
        self.ACCEL_OFFSET = -10.23        # m/s^2
        
        # Odometry state
        self.cumulative_distance_m = 0.0
        self.last_distance_update_time = None
        self.vehicle_speed_mps = 0.0
        self.wheel_speeds_kmh = {'FL': 0.0, 'FR': 0.0, 'RL': 0.0, 'RR': 0.0}
        self.yaw_rate_deg_s = None
        self.lateral_accel_mps2 = None
        self.longitudinal_accel_mps2 = None
        
        # Steering state
        self.current_steering_angle = None
        self.last_recorded_angle = None
        self.last_recorded_distance = 0.0
        self.initial_angle = None
        
        # Trajectory storage
        self.trajectory_points: List[EnhancedTrajectoryPoint] = []
        self.message_count = 0
        
        # Message reception tracking
        self.sas_message_count = 0
        self.wheel_speed_message_count = 0
        self.esp_message_count = 0
        self.last_sas_time = None
        self.last_wheel_speed_time = None
        self.last_esp_time = None
        
        # Calibration
        self.calibration = OdometryCalibration()
        
        # Thread synchronization
        self.running = False
        self.data_lock = threading.Lock()
        
        # Logging
        self.setup_logging()
        
        # Connect to CAN
        try:
            self.bus = can.interface.Bus(channel=interface, bustype='socketcan')
            print(f"✓ Odometry recorder connected to {interface}")
            print(f"  Angle threshold: {angle_threshold}°")
            print(f"  Distance threshold: {distance_threshold}m")
        except Exception as e:
            print(f"✗ Failed to connect to {interface}: {e}")
            sys.exit(1)
    
    def setup_logging(self):
        """Initialize logging infrastructure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Main trajectory CSV
        self.trajectory_csv = f"odometry_trajectory_{timestamp}.csv"
        with open(self.trajectory_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'iso_time', 'cumulative_distance_m', 
                'steering_angle_deg', 'angle_change_deg', 'distance_increment_m',
                'vehicle_speed_mps', 'wheel_speed_FL_kmh', 'wheel_speed_FR_kmh',
                'wheel_speed_RL_kmh', 'wheel_speed_RR_kmh',
                'yaw_rate_deg_s', 'lateral_accel_mps2', 'longitudinal_accel_mps2',
                'signal_quality_score', 'data_completeness', 'message_count'
            ])
        
        # Complete trajectory JSON
        self.trajectory_json = f"odometry_trajectory_{timestamp}.json"
        
        # Raw odometry log for debugging
        self.odometry_csv = f"odometry_raw_{timestamp}.csv"
        with open(self.odometry_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'distance_m', 'speed_mps', 
                'wheel_FL', 'wheel_FR', 'wheel_RL', 'wheel_RR',
                'yaw_rate', 'lat_accel', 'long_accel'
            ])
        
        print(f"\nLogging setup:")
        print(f"  Trajectory: {self.trajectory_csv}")
        print(f"  JSON data: {self.trajectory_json}")
        print(f"  Raw odometry: {self.odometry_csv}")
    
    def parse_wheel_speeds(self, data: bytes) -> Optional[Dict[str, float]]:
        """Parse WHL_SPD11 message (ID 0x386)"""
        if len(data) < 8:
            return None
        
        try:
            frame_uint64 = struct.unpack('<Q', data)[0]
            
            # Extract 14-bit wheel speeds
            # WHL_SPD_FL : 0|14@1+
            # WHL_SPD_FR : 16|14@1+
            # WHL_SPD_RL : 32|14@1+
            # WHL_SPD_RR : 48|14@1+
            
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
    
    def parse_esp12(self, data: bytes) -> Optional[Tuple[float, float, float]]:
        """Parse ESP12 message (ID 0x220) for yaw rate and accelerations"""
        if len(data) < 8:
            return None
        
        try:
            frame_uint64 = struct.unpack('<Q', data)[0]
            
            # LAT_ACCEL : 0|11@1+ (0.01,-10.23)
            lat_accel_raw = (frame_uint64 >> 0) & 0x7FF
            lat_accel = lat_accel_raw * self.ACCEL_SCALE + self.ACCEL_OFFSET
            
            # LONG_ACCEL : 13|11@1+ (0.01,-10.23)
            long_accel_raw = (frame_uint64 >> 13) & 0x7FF
            long_accel = long_accel_raw * self.ACCEL_SCALE + self.ACCEL_OFFSET
            
            # YAW_RATE : 40|13@1+ (0.01,-40.95)
            yaw_rate_raw = (frame_uint64 >> 40) & 0x1FFF
            yaw_rate = yaw_rate_raw * self.YAW_RATE_SCALE + self.YAW_RATE_OFFSET
            
            return yaw_rate, lat_accel, long_accel
            
        except Exception as e:
            print(f"ESP12 parsing error: {e}")
            return None
    
    def parse_sas11_angle(self, data: bytes) -> Optional[float]:
        """Extract steering angle from SAS11 (existing logic)"""
        if len(data) != 5:
            return None
        
        try:
            data_extended = data + b'\x00\x00\x00'
            frame_uint64 = struct.unpack('<Q', data_extended)[0]
            
            # SAS_Angle : 0|16@1- (0.1,0.0) - 16-bit signed
            angle_raw = frame_uint64 & 0xFFFF
            if angle_raw & 0x8000:  # Sign extend
                angle_raw |= 0xFFFF0000
                angle_raw = struct.unpack('<i', struct.pack('<I', angle_raw & 0xFFFFFFFF))[0]
            
            steering_angle = angle_raw * self.SAS_ANGLE_SCALE
            
            # Validation
            if -4000.0 <= steering_angle <= 4000.0:
                return steering_angle
            else:
                return None
                
        except Exception as e:
            print(f"SAS11 parsing error: {e}")
            return None
    
    def update_odometry(self, wheel_speeds_kmh: Dict[str, float], timestamp: float):
        """Calculate cumulative distance from wheel speeds"""
        
        with self.data_lock:
            self.wheel_speeds_kmh = wheel_speeds_kmh
            
            # Use rear axle average (more stable during steering)
            rear_avg_kmh = (wheel_speeds_kmh['RL'] + wheel_speeds_kmh['RR']) / 2.0
            
            # Convert to m/s and apply calibration
            vehicle_speed_mps = (rear_avg_kmh / 3.6) * self.calibration.distance_scale_factor
            
            # Integrate distance
            if self.last_distance_update_time is not None:
                dt = timestamp - self.last_distance_update_time
                
                # Sanity check on dt
                if 0.001 <= dt <= 1.0:  # Between 1ms and 1s
                    distance_increment = vehicle_speed_mps * dt
                    self.cumulative_distance_m += distance_increment
                    
                    # Log raw odometry
                    self.log_raw_odometry(timestamp)
            
            self.vehicle_speed_mps = vehicle_speed_mps
            self.last_distance_update_time = timestamp
    
    def update_vehicle_dynamics(self, yaw_rate: float, lat_accel: float, long_accel: float):
        """Update vehicle dynamics state"""
        with self.data_lock:
            self.yaw_rate_deg_s = yaw_rate - self.calibration.yaw_rate_bias_deg_s
            self.lateral_accel_mps2 = lat_accel
            self.longitudinal_accel_mps2 = long_accel
    
    def calculate_data_completeness(self) -> float:
        """Calculate percentage of expected signals received"""
        expected_signals = 3  # SAS, WHL_SPD, ESP12
        received_signals = 0
        
        if self.last_sas_time and (time.time() - self.last_sas_time) < 0.5:
            received_signals += 1
        if self.last_wheel_speed_time and (time.time() - self.last_wheel_speed_time) < 0.5:
            received_signals += 1
        if self.last_esp_time and (time.time() - self.last_esp_time) < 0.5:
            received_signals += 1
        
        return (received_signals / expected_signals) * 100.0
    
    def should_record_point(self, current_angle: float, current_distance: float) -> bool:
        """Determine if trajectory point should be recorded"""
        
        # Always record first point
        if self.last_recorded_angle is None:
            return True
        
        # Check angle change threshold
        angle_change = abs(current_angle - self.last_recorded_angle)
        angle_trigger = angle_change >= self.angle_threshold
        
        # Check distance threshold
        distance_change = current_distance - self.last_recorded_distance
        distance_trigger = distance_change >= self.distance_threshold
        
        # Record if either threshold exceeded
        return angle_trigger or distance_trigger
    
    def record_trajectory_point(self, angle: float, timestamp: float, raw_data: bytes):
        """Record enhanced trajectory point with odometry"""
        
        with self.data_lock:
            # Calculate deltas
            if self.last_recorded_angle is not None:
                angle_change = angle - self.last_recorded_angle
                distance_increment = self.cumulative_distance_m - self.last_recorded_distance
            else:
                angle_change = 0.0
                distance_increment = 0.0
                self.initial_angle = angle
            
            # Create trajectory point
            point = EnhancedTrajectoryPoint(
                timestamp=timestamp,
                cumulative_distance_m=self.cumulative_distance_m,
                steering_angle_deg=angle,
                vehicle_speed_mps=self.vehicle_speed_mps,
                wheel_speeds_kmh=self.wheel_speeds_kmh.copy(),
                yaw_rate_deg_s=self.yaw_rate_deg_s,
                lateral_accel_mps2=self.lateral_accel_mps2,
                longitudinal_accel_mps2=self.longitudinal_accel_mps2,
                angle_change_deg=angle_change,
                distance_increment_m=distance_increment,
                message_count=self.message_count,
                iso_time=datetime.fromtimestamp(timestamp).isoformat(),
                raw_sas_data=raw_data.hex().upper(),
                data_completeness=self.calculate_data_completeness()
            )
            
            self.trajectory_points.append(point)
            self.last_recorded_angle = angle
            self.last_recorded_distance = self.cumulative_distance_m
        
        # Log to CSV
        self.log_trajectory_to_csv(point)
        
        # Display
        print(f"[{len(self.trajectory_points):4d}] "
              f"Dist: {point.cumulative_distance_m:7.2f}m | "
              f"Angle: {angle:+7.1f}° | "
              f"Speed: {point.vehicle_speed_mps:5.2f}m/s | "
              f"Yaw: {point.yaw_rate_deg_s:+6.1f}°/s" if point.yaw_rate_deg_s else "Yaw: N/A")
    
    def log_trajectory_to_csv(self, point: EnhancedTrajectoryPoint):
        """Log trajectory point to CSV"""
        with open(self.trajectory_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                point.timestamp, point.iso_time, point.cumulative_distance_m,
                point.steering_angle_deg, point.angle_change_deg, point.distance_increment_m,
                point.vehicle_speed_mps,
                point.wheel_speeds_kmh['FL'], point.wheel_speeds_kmh['FR'],
                point.wheel_speeds_kmh['RL'], point.wheel_speeds_kmh['RR'],
                point.yaw_rate_deg_s, point.lateral_accel_mps2, point.longitudinal_accel_mps2,
                point.signal_quality_score, point.data_completeness, point.message_count
            ])
    
    def log_raw_odometry(self, timestamp: float):
        """Log raw odometry data for analysis"""
        with open(self.odometry_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, self.cumulative_distance_m, self.vehicle_speed_mps,
                self.wheel_speeds_kmh['FL'], self.wheel_speeds_kmh['FR'],
                self.wheel_speeds_kmh['RL'], self.wheel_speeds_kmh['RR'],
                self.yaw_rate_deg_s, self.lateral_accel_mps2, self.longitudinal_accel_mps2
            ])
    
    def record_trajectory(self):
        """Main recording loop"""
        print(f"\n{'='*80}")
        print(f"ODOMETRY-BASED TRAJECTORY RECORDING")
        print(f"{'='*80}")
        print(f"Interface: {self.interface}")
        print(f"Monitoring:")
        print(f"  - SAS11 (0x{self.SAS11_CAN_ID:03X}): Steering angle")
        print(f"  - WHL_SPD11 (0x{self.WHL_SPD_CAN_ID:03X}): Wheel speeds")
        print(f"  - ESP12 (0x{self.ESP12_CAN_ID:03X}): Yaw rate, accelerations")
        print(f"Recording triggers:")
        print(f"  - Angle change: ≥{self.angle_threshold}°")
        print(f"  - Distance traveled: ≥{self.distance_threshold}m")
        print(f"\nDrive the vehicle at constant 5 mph (~2.2 m/s)")
        print(f"Press Ctrl+C when trajectory complete")
        print(f"{'='*80}\n")
        
        self.running = True
        last_status_time = 0
        
        try:
            while self.running:
                message = self.bus.recv(timeout=0.1)
                
                if message is None:
                    continue
                
                current_time = message.timestamp
                
                # Process wheel speeds (highest priority for odometry)
                if message.arbitration_id == self.WHL_SPD_CAN_ID:
                    self.wheel_speed_message_count += 1
                    self.last_wheel_speed_time = current_time
                    
                    wheel_speeds = self.parse_wheel_speeds(message.data)
                    if wheel_speeds:
                        self.update_odometry(wheel_speeds, current_time)
                
                # Process ESP12 for vehicle dynamics
                elif message.arbitration_id == self.ESP12_CAN_ID:
                    self.esp_message_count += 1
                    self.last_esp_time = current_time
                    
                    esp_data = self.parse_esp12(message.data)
                    if esp_data:
                        yaw_rate, lat_accel, long_accel = esp_data
                        self.update_vehicle_dynamics(yaw_rate, lat_accel, long_accel)
                
                # Process SAS11 for steering angle
                elif message.arbitration_id == self.SAS11_CAN_ID:
                    self.sas_message_count += 1
                    self.last_sas_time = current_time
                    self.message_count += 1
                    
                    angle = self.parse_sas11_angle(message.data)
                    if angle is not None:
                        self.current_steering_angle = angle
                        
                        # Check if we should record this point
                        if self.should_record_point(angle, self.cumulative_distance_m):
                            self.record_trajectory_point(angle, current_time, message.data)
                
                # Periodic status display
                if time.time() - last_status_time >= 2.0:
                    self.display_status()
                    last_status_time = time.time()
                    
        except KeyboardInterrupt:
            print(f"\n\nTrajectory recording stopped by user")
            self.running = False
            self.finalize_recording()
        finally:
            if self.bus:
                self.bus.shutdown()
    
    def display_status(self):
        """Display current recording status"""
        runtime = time.time() - self.start_time
        
        print(f"\n{'─'*80}")
        print(f"Runtime: {runtime:.1f}s | Distance: {self.cumulative_distance_m:.1f}m | "
              f"Speed: {self.vehicle_speed_mps:.2f}m/s | Points: {len(self.trajectory_points)}")
        print(f"Messages: SAS={self.sas_message_count} | Wheels={self.wheel_speed_message_count} | "
              f"ESP={self.esp_message_count}")
        if self.current_steering_angle is not None:
            print(f"Current angle: {self.current_steering_angle:+7.1f}° | "
                  f"Yaw rate: {self.yaw_rate_deg_s:+6.1f}°/s" if self.yaw_rate_deg_s else "Yaw: N/A")
        print(f"{'─'*80}")
    
    def calculate_loop_closure_error(self) -> Optional[float]:
        """Calculate distance between start and end positions (for closed loops)"""
        if len(self.trajectory_points) < 2:
            return None
        
        # Simple metric: difference in cumulative distance vs straight-line displacement
        # For true closed loop, integrate yaw to get heading and calculate XY position
        # This is simplified - just return distance traveled
        return self.trajectory_points[-1].cumulative_distance_m
    
    def finalize_recording(self):
        """Finalize and save complete trajectory data"""
        print(f"\n{'='*80}")
        print(f"FINALIZING TRAJECTORY RECORDING")
        print(f"{'='*80}")
        
        if len(self.trajectory_points) == 0:
            print("No trajectory points recorded!")
            return
        
        # Calculate statistics
        duration = time.time() - self.start_time
        total_distance = self.trajectory_points[-1].cumulative_distance_m
        avg_speed = total_distance / duration if duration > 0 else 0
        
        speeds = [p.vehicle_speed_mps for p in self.trajectory_points]
        max_speed = max(speeds) if speeds else 0
        min_speed = min(speeds) if speeds else 0
        
        # Create metadata
        metadata = TrajectoryMetadata(
            start_time=self.start_time,
            end_time=time.time(),
            duration_seconds=duration,
            total_distance_m=total_distance,
            average_speed_mps=avg_speed,
            max_speed_mps=max_speed,
            min_speed_mps=min_speed,
            number_of_points=len(self.trajectory_points),
            loop_closure_error_m=self.calculate_loop_closure_error()
        )
        
        # Save complete trajectory JSON
        trajectory_data = {
            'format_version': '3.0_odometry',
            'recording_type': 'distance_based',
            'coordinate_frame': 'SAS11_LEFT_POSITIVE',
            'metadata': asdict(metadata),
            'calibration': asdict(self.calibration),
            'trajectory_points': [asdict(p) for p in self.trajectory_points],
            'message_statistics': {
                'sas11_messages': self.sas_message_count,
                'wheel_speed_messages': self.wheel_speed_message_count,
                'esp12_messages': self.esp_message_count,
                'trajectory_points': len(self.trajectory_points)
            }
        }
        
        with open(self.trajectory_json, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
        
        # Print summary
        print(f"\nRecording Summary:")
        print(f"  Duration: {duration:.1f} seconds")
        print(f"  Total distance: {total_distance:.2f} meters")
        print(f"  Average speed: {avg_speed:.2f} m/s ({avg_speed*3.6:.1f} km/h)")
        print(f"  Trajectory points: {len(self.trajectory_points)}")
        print(f"  Initial angle: {self.initial_angle:+.1f}°")
        print(f"  Final angle: {self.trajectory_points[-1].steering_angle_deg:+.1f}°")
        
        print(f"\nMessage Statistics:")
        print(f"  SAS11 (steering): {self.sas_message_count}")
        print(f"  WHL_SPD11 (odometry): {self.wheel_speed_message_count}")
        print(f"  ESP12 (dynamics): {self.esp_message_count}")
        
        print(f"\nOutput Files:")
        print(f"  Trajectory CSV: {self.trajectory_csv}")
        print(f"  Trajectory JSON: {self.trajectory_json}")
        print(f"  Raw odometry: {self.odometry_csv}")
        
        print(f"\nPlayback Command:")
        print(f"  python3 mcm_steer_track2_odometry.py --trajectory {self.trajectory_json}")
        print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(
        description='Odometry-Based Steering Trajectory Recorder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Distance-based trajectory recording for repeatable path replication.

Records steering trajectory indexed by distance traveled from wheel speeds
rather than elapsed time. Designed for closed-loop testing in parking lots
with rough surfaces.

Example usage:
  # Standard recording with 5° angle / 0.5m distance thresholds
  python3 anglerecorder_odometry.py
  
  # High resolution recording
  python3 anglerecorder_odometry.py --angle-threshold 2 --distance-threshold 0.2
  
  # Different CAN interface
  python3 anglerecorder_odometry.py -i can0

Instructions:
  1. Start recorder
  2. Drive vehicle at constant 5 mph (~2.2 m/s)
  3. Complete desired trajectory path
  4. Press Ctrl+C to stop and save
  5. Use generated JSON file for playback
        '''
    )
    
    parser.add_argument('--interface', '-i', default='can3',
                       help='CAN interface (default: can3)')
    parser.add_argument('--angle-threshold', '-a', type=float, default=5.0,
                       help='Angle change threshold in degrees (default: 5.0)')
    parser.add_argument('--distance-threshold', '-d', type=float, default=0.5,
                       help='Distance threshold in meters (default: 0.5)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.angle_threshold <= 0:
        print("Error: Angle threshold must be positive")
        sys.exit(1)
    
    if args.distance_threshold <= 0:
        print("Error: Distance threshold must be positive")
        sys.exit(1)
    
    # Create and run recorder
    recorder = OdometryTrajectoryRecorder(
        interface=args.interface,
        angle_threshold=args.angle_threshold,
        distance_threshold=args.distance_threshold
    )
    
    recorder.record_trajectory()

if __name__ == "__main__":
    main()
