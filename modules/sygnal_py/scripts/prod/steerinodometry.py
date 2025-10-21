#!/usr/bin/env python3
"""
Odometry-Based Trajectory Playback Controller
Distance-indexed steering command execution with progressive brake control

Executes pre-recorded trajectory by monitoring wheel speed odometry and sending
steering + brake commands to replicate recorded path. Designed for repeatable
closed-loop testing with manual lap restart.

Architecture:
  CAN3 (Input):  Wheel speeds (WHL_SPD11), Steering feedback (SAS11)
  CAN2 (Output): Steering commands (IF=2), Brake commands (IF=0)
"""

import can
import cantools
import crc8
import asyncio
import signal
import sys
import json
import struct
import time
import csv
import math
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
import threading
from collections import deque

@dataclass
class TrajectoryTarget:
    """Trajectory waypoint from recorded data"""
    timestamp: float
    cumulative_distance_m: float
    target_angle_deg: float
    vehicle_speed_mps: float

@dataclass
class ValidationPoint:
    """Real-time validation data"""
    timestamp: float
    cumulative_distance_m: float
    target_angle_deg: float
    commanded_steering_pct: float
    commanded_brake_pct: float
    measured_angle_deg: Optional[float]
    measured_speed_mps: Optional[float]
    angle_error_deg: Optional[float]
    distance_error_m: Optional[float]

class OdometryPlaybackController:
    def __init__(self, mcm_channel='can2', sas_channel='can3', 
                 trajectory_file=None, brake_start_distance=8.0):
        # CAN Bus Configuration
        self.mcm_channel = mcm_channel
        self.sas_channel = sas_channel
        
        # MCM Controller Setup (CAN2 output)
        self.mcm_db = cantools.database.Database()
        self.mcm_db.add_dbc_file('./sygnal_dbc/mcm/Heartbeat.dbc')
        self.mcm_db.add_dbc_file('./sygnal_dbc/mcm/Control.dbc')
        
        # CAN Bus Connections
        try:
            self.mcm_bus = can.Bus(channel=mcm_channel, bustype='socketcan', bitrate=500000)
            print(f"✓ MCM controller connected: {mcm_channel}")
        except Exception as e:
            print(f"✗ MCM bus connection failed ({mcm_channel}): {e}")
            sys.exit(1)
            
        try:
            self.sas_bus = can.Bus(channel=sas_channel, bustype='socketcan', bitrate=500000)
            print(f"✓ Sensor monitor connected: {sas_channel}")
        except Exception as e:
            print(f"✗ Sensor bus connection failed ({sas_channel}): {e}")
            sys.exit(1)
        
        # MCM Control State
        self.control_count = 0
        self.bus_address = 1
        self.steering_enabled = False
        self.brake_enabled = False
        
        # CAN IDs for monitoring (CAN3)
        self.SAS11_CAN_ID = 0x2B0
        self.WHL_SPD_CAN_ID = 0x386
        self.ESP12_CAN_ID = 0x220
        
        # Scaling factors
        self.SAS_ANGLE_SCALE = 0.1
        self.WHEEL_SPEED_SCALE = 0.03125
        
        # Odometry State
        self.cumulative_distance_m = 0.0
        self.vehicle_speed_mps = 0.0
        self.wheel_speeds_kmh = {'FL': 0.0, 'FR': 0.0, 'RL': 0.0, 'RR': 0.0}
        self.last_distance_update_time = None
        
        # Steering Feedback
        self.latest_angle_deg = None
        self.latest_angle_timestamp = None
        
        # Trajectory Data
        self.trajectory_targets: List[TrajectoryTarget] = []
        self.trajectory_metadata = {}
        self.total_trajectory_distance = 0.0
        
        # Validation Logging
        self.validation_log: List[ValidationPoint] = []
        
        # Brake Parameters
        self.BRAKE_START_DISTANCE = brake_start_distance  # meters
        self.MAX_BRAKE_INTENSITY = 0.7  # 70% maximum
        
        # State Machine
        self.trajectory_index = 0
        self.braking_active = False
        self.trajectory_complete = False
        
        # Thread Synchronization
        self.running = False
        self.data_lock = threading.Lock()
        
        # Statistics
        self.stats = {
            'commands_sent': 0,
            'brake_commands_sent': 0,
            'distance_errors': [],
            'angle_errors': [],
            'max_distance_error': 0.0,
            'max_angle_error': 0.0
        }
        
        # Load Trajectory
        if trajectory_file:
            self.load_trajectory_data(trajectory_file)
        
        # Setup Logging
        self.setup_validation_logging()
    
    def load_trajectory_data(self, filename: str):
        """Load distance-indexed trajectory from JSON"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Validate format
            if data.get('format_version') != '3.0_odometry':
                print(f"⚠ Warning: Unexpected format version {data.get('format_version')}")
            
            if data.get('recording_type') != 'distance_based':
                print(f"⚠ Warning: Expected distance_based recording type")
            
            # Extract trajectory points
            trajectory_points = data['trajectory_points']
            
            for point in trajectory_points:
                target = TrajectoryTarget(
                    timestamp=point['timestamp'],
                    cumulative_distance_m=point['cumulative_distance_m'],
                    target_angle_deg=point['steering_angle_deg'],
                    vehicle_speed_mps=point['vehicle_speed_mps']
                )
                self.trajectory_targets.append(target)
            
            # Store metadata
            self.trajectory_metadata = data.get('metadata', {})
            self.total_trajectory_distance = self.trajectory_targets[-1].cumulative_distance_m
            
            print(f"\n{'='*80}")
            print(f"TRAJECTORY LOADED")
            print(f"{'='*80}")
            print(f"Trajectory points: {len(self.trajectory_targets)}")
            print(f"Total distance: {self.total_trajectory_distance:.2f}m")
            print(f"Duration (recorded): {self.trajectory_metadata.get('duration_seconds', 0):.1f}s")
            print(f"Average speed: {self.trajectory_metadata.get('average_speed_mps', 0):.2f}m/s")
            
            angle_range = (
                min(t.target_angle_deg for t in self.trajectory_targets),
                max(t.target_angle_deg for t in self.trajectory_targets)
            )
            print(f"Angle range: {angle_range[0]:+.1f}° to {angle_range[1]:+.1f}°")
            print(f"{'='*80}\n")
            
        except FileNotFoundError:
            print(f"✗ Trajectory file not found: {filename}")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Failed to load trajectory: {e}")
            sys.exit(1)
    
    def setup_validation_logging(self):
        """Initialize validation logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.validation_csv = f"playback_validation_{timestamp}.csv"
        
        with open(self.validation_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'cumulative_distance_m', 'target_angle_deg',
                'commanded_steering_pct', 'commanded_brake_pct',
                'measured_angle_deg', 'measured_speed_mps',
                'angle_error_deg', 'distance_error_m'
            ])
        
        print(f"Validation logging: {self.validation_csv}")
    
    def calc_crc8(self, data: bytes) -> int:
        """Calculate CRC8 for MCM messages"""
        hash = crc8.crc8()
        hash.update(data[:-1])
        return hash.digest()[0]
    
    def angle_to_percentage(self, angle_deg: float) -> float:
        """
        Convert steering angle to MCM command percentage
        
        Coordinate frame inversion:
          SAS11: Left=+positive, Right=-negative
          MCM:   Left=-negative, Right=+positive
        
        Scaling assumption: ±400° = ±100%
        """
        # Sign inversion for coordinate frame
        adjusted_angle = -angle_deg
        
        # Scale to percentage
        percentage = adjusted_angle * (100.0 / 400.0)
        
        # Clamp to valid range
        return max(min(percentage, 100.0), -100.0)
    
    def calculate_brake_command(self, remaining_distance: float) -> float:
        """
        Progressive brake intensity calculation
        
        Args:
            remaining_distance: Distance to target stop point (meters)
        
        Returns:
            Brake intensity [0.0 to 0.7] (0% to 70%)
        
        Brake curve (quadratic):
            8m remaining: 0%
            4m remaining: ~18%
            2m remaining: ~35%
            1m remaining: ~50%
            0m remaining: 70%
        """
        if remaining_distance >= self.BRAKE_START_DISTANCE:
            return 0.0
        
        if remaining_distance <= 0.0:
            return self.MAX_BRAKE_INTENSITY
        
        # Normalized progress through brake zone [0.0 to 1.0]
        brake_progress = 1.0 - (remaining_distance / self.BRAKE_START_DISTANCE)
        
        # Quadratic curve: brake = max_intensity × progress^1.5
        brake_intensity = self.MAX_BRAKE_INTENSITY * (brake_progress ** 1.5)
        
        return min(max(brake_intensity, 0.0), self.MAX_BRAKE_INTENSITY)
    
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
    
    def update_odometry(self, wheel_speeds_kmh: Dict[str, float], timestamp: float):
        """Update cumulative distance from wheel speeds"""
        with self.data_lock:
            self.wheel_speeds_kmh = wheel_speeds_kmh
            
            # Rear axle average
            rear_avg_kmh = (wheel_speeds_kmh['RL'] + wheel_speeds_kmh['RR']) / 2.0
            vehicle_speed_mps = rear_avg_kmh / 3.6
            
            # Integrate distance
            if self.last_distance_update_time is not None:
                dt = timestamp - self.last_distance_update_time
                
                if 0.001 <= dt <= 1.0:
                    distance_increment = vehicle_speed_mps * dt
                    self.cumulative_distance_m += distance_increment
            
            self.vehicle_speed_mps = vehicle_speed_mps
            self.last_distance_update_time = timestamp
    
    def get_current_odometry(self) -> Tuple[float, float]:
        """Get current distance and speed (thread-safe)"""
        with self.data_lock:
            return self.cumulative_distance_m, self.vehicle_speed_mps
    
    def get_current_angle(self) -> Tuple[Optional[float], Optional[float]]:
        """Get latest steering angle (thread-safe)"""
        with self.data_lock:
            return self.latest_angle_deg, self.latest_angle_timestamp
    
    async def enable_mcm_interface(self, interface_name: str):
        """
        Enable MCM control interface
        
        Args:
            interface_name: 'steer' (IF=2) or 'brake' (IF=0)
        """
        interface_id = {'brake': 0, 'accel': 1, 'steer': 2}.get(interface_name)
        
        if interface_id is None:
            print(f"✗ Invalid interface: {interface_name}")
            return False
        
        msg = self.mcm_db.get_message_by_name('ControlEnable')
        data = bytearray(msg.encode({
            'BusAddress': self.bus_address,
            'InterfaceID': interface_id,
            'Enable': 1,
            'CRC': 0
        }))
        data[7] = self.calc_crc8(data)
        
        try:
            self.mcm_bus.send(can.Message(
                arbitration_id=msg.frame_id,
                is_extended_id=False,
                data=data
            ))
            
            if interface_name == 'steer':
                self.steering_enabled = True
            elif interface_name == 'brake':
                self.brake_enabled = True
            
            await asyncio.sleep(0.02)
            return True
            
        except Exception as e:
            print(f"✗ Failed to enable {interface_name} interface: {e}")
            return False
    
    async def send_steering_command(self, percentage: float) -> bool:
        """Send steering command to MCM (InterfaceID=2)"""
        percentage = max(min(percentage, 100.0), -100.0)
        
        msg = self.mcm_db.get_message_by_name('ControlCommand')
        data = bytearray(msg.encode({
            'BusAddress': self.bus_address,
            'InterfaceID': 2,  # Steering
            'Count8': self.control_count,
            'Value': percentage / 100.0,  # Normalize to [-1.0, +1.0]
            'CRC': 0
        }))
        data[7] = self.calc_crc8(data)
        
        try:
            if not self.steering_enabled:
                await self.enable_mcm_interface('steer')
            
            self.mcm_bus.send(can.Message(
                arbitration_id=msg.frame_id,
                is_extended_id=False,
                data=data
            ))
            
            self.control_count = (self.control_count + 1) % 256
            self.stats['commands_sent'] += 1
            
            return True
            
        except Exception as e:
            print(f"✗ Steering command failed: {e}")
            return False
    
    async def send_brake_command(self, percentage: float) -> bool:
        """Send brake command to MCM (InterfaceID=0)"""
        percentage = max(min(percentage, 100.0), 0.0)
        
        msg = self.mcm_db.get_message_by_name('ControlCommand')
        data = bytearray(msg.encode({
            'BusAddress': self.bus_address,
            'InterfaceID': 0,  # Brake
            'Count8': self.control_count,
            'Value': percentage / 100.0,  # Normalize to [0.0, 1.0]
            'CRC': 0
        }))
        data[7] = self.calc_crc8(data)
        
        try:
            if not self.brake_enabled:
                await self.enable_mcm_interface('brake')
            
            self.mcm_bus.send(can.Message(
                arbitration_id=msg.frame_id,
                is_extended_id=False,
                data=data
            ))
            
            self.control_count = (self.control_count + 1) % 256
            self.stats['brake_commands_sent'] += 1
            
            return True
            
        except Exception as e:
            print(f"✗ Brake command failed: {e}")
            return False
    
    def monitor_sensors(self):
        """Background thread for sensor monitoring (CAN3)"""
        print("✓ Sensor monitoring thread started")
        
        while self.running:
            try:
                message = self.sas_bus.recv(timeout=0.1)
                
                if message is None:
                    continue
                
                # Process wheel speeds
                if message.arbitration_id == self.WHL_SPD_CAN_ID:
                    wheel_speeds = self.parse_wheel_speeds(message.data)
                    if wheel_speeds:
                        self.update_odometry(wheel_speeds, message.timestamp)
                
                # Process steering feedback
                elif message.arbitration_id == self.SAS11_CAN_ID:
                    angle = self.parse_sas11_angle(message.data)
                    if angle is not None:
                        with self.data_lock:
                            self.latest_angle_deg = angle
                            self.latest_angle_timestamp = message.timestamp
                            
            except Exception as e:
                if self.running:
                    print(f"⚠ Sensor monitoring error: {e}")
                    time.sleep(0.1)
    
    async def execute_trajectory_playback(self):
        """Main trajectory execution state machine"""
        print(f"\n{'='*80}")
        print(f"TRAJECTORY PLAYBACK EXECUTION")
        print(f"{'='*80}")
        print(f"Target distance: {self.total_trajectory_distance:.2f}m")
        print(f"Brake initiation: {self.BRAKE_START_DISTANCE:.1f}m before target")
        print(f"Trajectory points: {len(self.trajectory_targets)}")
        print(f"\nExecuting trajectory...")
        print(f"{'='*80}\n")
        
        playback_start_time = time.time()
        last_status_time = playback_start_time
        
        # Reset state
        self.trajectory_index = 0
        self.braking_active = False
        self.trajectory_complete = False
        
        # Main control loop
        while self.running and not self.trajectory_complete:
            
            # Get current odometry state
            current_distance, current_speed = self.get_current_odometry()
            remaining_distance = self.total_trajectory_distance - current_distance
            
            # === BRAKING LOGIC ===
            brake_command = self.calculate_brake_command(remaining_distance)
            
            if brake_command > 0.0:
                if not self.braking_active:
                    print(f"\n⚠ Braking initiated at {current_distance:.2f}m "
                          f"(remaining: {remaining_distance:.2f}m)")
                    self.braking_active = True
                
                await self.send_brake_command(brake_command * 100.0)
            
            # Check trajectory completion
            if current_distance >= self.total_trajectory_distance:
                print(f"\n✓ Trajectory complete at {current_distance:.2f}m")
                # Hold full brake
                await self.send_brake_command(self.MAX_BRAKE_INTENSITY * 100.0)
                self.trajectory_complete = True
                break
            
            # === STEERING LOGIC ===
            # Advance through trajectory waypoints based on distance
            while (self.trajectory_index < len(self.trajectory_targets) and
                   current_distance >= self.trajectory_targets[self.trajectory_index].cumulative_distance_m):
                
                target = self.trajectory_targets[self.trajectory_index]
                command_percentage = self.angle_to_percentage(target.target_angle_deg)
                
                await self.send_steering_command(command_percentage)
                
                # Validation logging
                measured_angle, _ = self.get_current_angle()
                
                angle_error = None
                if measured_angle is not None:
                    angle_error = measured_angle - target.target_angle_deg
                    self.stats['angle_errors'].append(abs(angle_error))
                    self.stats['max_angle_error'] = max(self.stats['max_angle_error'], abs(angle_error))
                
                distance_error = current_distance - target.cumulative_distance_m
                self.stats['distance_errors'].append(abs(distance_error))
                self.stats['max_distance_error'] = max(self.stats['max_distance_error'], abs(distance_error))
                
                # Log validation point
                validation_point = ValidationPoint(
                    timestamp=time.time(),
                    cumulative_distance_m=current_distance,
                    target_angle_deg=target.target_angle_deg,
                    commanded_steering_pct=command_percentage,
                    commanded_brake_pct=brake_command * 100.0,
                    measured_angle_deg=measured_angle,
                    measured_speed_mps=current_speed,
                    angle_error_deg=angle_error,
                    distance_error_m=distance_error
                )
                self.validation_log.append(validation_point)
                self.log_validation_to_csv(validation_point)
                
                self.trajectory_index += 1
            
            # === STATUS DISPLAY ===
            if time.time() - last_status_time >= 1.0:
                progress_pct = (current_distance / self.total_trajectory_distance) * 100
                measured_angle, _ = self.get_current_angle()
                
                status_line = (f"[{self.trajectory_index:4d}/{len(self.trajectory_targets)}] "
                              f"Dist: {current_distance:6.2f}m ({progress_pct:5.1f}%) | "
                              f"Speed: {current_speed:4.2f}m/s | "
                              f"Remaining: {remaining_distance:5.2f}m")
                
                if self.braking_active:
                    status_line += f" | BRAKE: {brake_command*100:4.1f}%"
                
                if measured_angle is not None:
                    status_line += f" | Angle: {measured_angle:+6.1f}°"
                
                print(status_line)
                last_status_time = time.time()
            
            # Control loop rate: 20Hz
            await asyncio.sleep(0.05)
        
        # === POST-TRAJECTORY ACTIONS ===
        # Hold brake for 2 seconds to ensure full stop
        print(f"\nHolding brake for complete stop...")
        for _ in range(40):  # 2 seconds at 20Hz
            await self.send_brake_command(self.MAX_BRAKE_INTENSITY * 100.0)
            await asyncio.sleep(0.05)
        
        # Release brake
        await self.send_brake_command(0.0)
        
        # Final statistics
        final_distance, final_speed = self.get_current_odometry()
        distance_overshoot = final_distance - self.total_trajectory_distance
        
        print(f"\n{'='*80}")
        print(f"TRAJECTORY EXECUTION COMPLETE")
        print(f"{'='*80}")
        print(f"Target distance: {self.total_trajectory_distance:.2f}m")
        print(f"Actual distance: {final_distance:.2f}m")
        print(f"Overshoot: {distance_overshoot:+.2f}m")
        print(f"Execution time: {time.time() - playback_start_time:.1f}s")
        print(f"{'='*80}\n")
    
    def log_validation_to_csv(self, point: ValidationPoint):
        """Append validation point to CSV"""
        with open(self.validation_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                point.timestamp, point.cumulative_distance_m, point.target_angle_deg,
                point.commanded_steering_pct, point.commanded_brake_pct,
                point.measured_angle_deg, point.measured_speed_mps,
                point.angle_error_deg, point.distance_error_m
            ])
    
    async def maintain_mcm_heartbeat(self):
        """Maintain MCM interface enable signals"""
        while self.running:
            try:
                if self.steering_enabled:
                    await self.enable_mcm_interface('steer')
                if self.brake_enabled:
                    await self.enable_mcm_interface('brake')
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"⚠ Heartbeat maintenance error: {e}")
                await asyncio.sleep(0.5)
    
    async def run_playback_loop(self, num_laps: Optional[int] = None):
        """
        Execute trajectory playback with manual lap restart
        
        Args:
            num_laps: Number of laps to execute (None = infinite until Ctrl+C)
        """
        self.running = True
        
        # Start sensor monitoring thread
        sensor_thread = threading.Thread(target=self.monitor_sensors, daemon=True)
        sensor_thread.start()
        
        # Start MCM heartbeat task
        heartbeat_task = asyncio.create_task(self.maintain_mcm_heartbeat())
        
        # Wait for initial sensor data
        print("Waiting for sensor initialization...")
        timeout_start = time.time()
        while time.time() - timeout_start < 5.0:
            current_distance, current_speed = self.get_current_odometry()
            measured_angle, _ = self.get_current_angle()
            
            if current_speed > 0 or measured_angle is not None:
                print(f"✓ Sensors initialized")
                print(f"  Speed: {current_speed:.2f}m/s")
                print(f"  Angle: {measured_angle:+.1f}°" if measured_angle else "  Angle: waiting...")
                break
            
            await asyncio.sleep(0.1)
        else:
            print("⚠ Sensor timeout - continuing anyway")
        
        # Lap execution loop
        lap_count = 0
        try:
            while True:
                lap_count += 1
                
                if num_laps is not None and lap_count > num_laps:
                    print(f"\n✓ Completed {num_laps} laps")
                    break
                
                print(f"\n{'█'*80}")
                print(f"{'█'*80}")
                print(f"█{'':^78}█")
                print(f"█{f'LAP {lap_count}':^78}█")
                print(f"█{'':^78}█")
                print(f"{'█'*80}")
                print(f"{'█'*80}\n")
                
                # Reset distance counter
                with self.data_lock:
                    self.cumulative_distance_m = 0.0
                    self.last_distance_update_time = None
                
                # Execute trajectory
                await self.execute_trajectory_playback()
                
                # Print lap summary
                self.print_lap_summary(lap_count)
                
                # Manual restart prompt
                if num_laps is None or lap_count < num_laps:
                    print(f"\n{'─'*80}")
                    print(f"Lap {lap_count} complete. Vehicle stopped.")
                    print(f"{'─'*80}")
                    
                    # Wait for user input
                    restart_input = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        input, 
                        "\nPress ENTER to start next lap (or Ctrl+C to exit): "
                    )
                    
                    print(f"\n{'─'*80}")
                    print(f"Restarting lap {lap_count + 1}...")
                    print(f"{'─'*80}\n")
                    
                    # Brief delay before restart
                    await asyncio.sleep(1.0)
        
        except KeyboardInterrupt:
            print(f"\n\n⚠ Playback interrupted by user")
        
        finally:
            self.running = False
            heartbeat_task.cancel()
            
            # Release all controls
            await self.send_brake_command(0.0)
            await self.send_steering_command(0.0)
            
            print(f"\n{'='*80}")
            print(f"PLAYBACK SESSION COMPLETE")
            print(f"{'='*80}")
            print(f"Total laps: {lap_count}")
            print(f"Total commands: {self.stats['commands_sent']}")
            print(f"Brake commands: {self.stats['brake_commands_sent']}")
            print(f"Validation log: {self.validation_csv}")
            print(f"{'='*80}")
    
    def print_lap_summary(self, lap_number: int):
        """Print statistical summary for completed lap"""
        if not self.stats['angle_errors']:
            return
        
        print(f"\n{'─'*80}")
        print(f"LAP {lap_number} SUMMARY")
        print(f"{'─'*80}")
        
        print(f"Distance tracking:")
        if self.stats['distance_errors']:
            mean_dist_err = sum(self.stats['distance_errors']) / len(self.stats['distance_errors'])
            print(f"  Mean error: {mean_dist_err:.3f}m")
            print(f"  Max error: {self.stats['max_distance_error']:.3f}m")
        
        print(f"Angle tracking:")
        if self.stats['angle_errors']:
            mean_angle_err = sum(self.stats['angle_errors']) / len(self.stats['angle_errors'])
            print(f"  Mean error: {mean_angle_err:.2f}°")
            print(f"  Max error: {self.stats['max_angle_error']:.2f}°")
        
        print(f"Commands:")
        print(f"  Steering: {self.stats['commands_sent']}")
        print(f"  Brake: {self.stats['brake_commands_sent']}")
        
        print(f"{'─'*80}")
        
        # Reset stats for next lap
        self.stats['distance_errors'].clear()
        self.stats['angle_errors'].clear()
        self.stats['max_distance_error'] = 0.0
        self.stats['max_angle_error'] = 0.0
        self.stats['commands_sent'] = 0
        self.stats['brake_commands_sent'] = 0

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Odometry-Based Trajectory Playback Controller',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Distance-indexed trajectory playback with progressive brake control.

Architecture:
  CAN2 (MCM): Steering commands (IF=2) + Brake commands (IF=0)
  CAN3 (Sensors): Wheel speeds (odometry) + Steering feedback

Example usage:
  # Single lap execution
  python3 mcm_steer_track2_odometry.py --trajectory recorded_trajectory.json
  
  # Multiple laps (manual restart between laps)
  python3 mcm_steer_track2_odometry.py --trajectory path.json --laps 5
  
  # Custom brake parameters
  python3 mcm_steer_track2_odometry.py --trajectory path.json --brake-distance 10.0

Requirements:
  - Cruise control active at 5 mph (vehicle maintains constant speed)
  - Vehicle positioned at trajectory start location
  - CAN2 and CAN3 interfaces operational
        '''
    )
    
    parser.add_argument('--trajectory', '-t', required=True,
                       help='Trajectory JSON file from odometry recorder')
    parser.add_argument('--mcm', default='can2',
                       help='MCM command CAN interface (default: can2)')
    parser.add_argument('--sas', default='can3',
                       help='Sensor feedback CAN interface (default: can3)')
    parser.add_argument('--brake-distance', type=float, default=8.0,
                       help='Distance before target to initiate braking (default: 8.0m)')
    parser.add_argument('--laps', type=int, default=None,
                       help='Number of laps to execute (default: infinite until Ctrl+C)')
    
    args = parser.parse_args()
    
    # Validate brake distance
    if args.brake_distance <= 0:
        print("✗ Brake distance must be positive")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"ODOMETRY PLAYBACK CONTROLLER")
    print(f"{'='*80}")
    print(f"CAN2 (MCM commands): {args.mcm}")
    print(f"CAN3 (Sensor feedback): {args.sas}")
    print(f"Trajectory file: {args.trajectory}")
    print(f"Brake initiation: {args.brake_distance}m before target")
    if args.laps:
        print(f"Laps configured: {args.laps}")
    else:
        print(f"Laps: Infinite (manual Ctrl+C to stop)")
    print(f"{'='*80}\n")
    
    # Initialize controller
    controller = OdometryPlaybackController(
        mcm_channel=args.mcm,
        sas_channel=args.sas,
        trajectory_file=args.trajectory,
        brake_start_distance=args.brake_distance
    )
    
    # Setup signal handling
    def cleanup(sig=None, frame=None):
        print("\n\n⚠ Stopping playback...")
        controller.running = False
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup)
    
    # Run playback loop
    try:
        asyncio.run(controller.run_playback_loop(num_laps=args.laps))
    except KeyboardInterrupt:
        cleanup()

if __name__ == "__main__":
    main()
