#!/usr/bin/env python3
"""
Steering Command Validation Controller
Sends MCM steering commands on CAN2 and monitors SAS11 feedback on CAN3
Validates command execution against recorded trajectory targets

Architecture:
- CAN2: MCM steering commands (Controller → Steering actuator)  
- CAN3: SAS11 angle feedback (Steering sensor → Controller)
- Real-time validation of command tracking performance
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
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import threading
from collections import deque

@dataclass
class TrajectoryTarget:
    """Target trajectory point from recorded data"""
    timestamp: float
    target_angle_deg: float
    relative_time: float

@dataclass
class ValidationPoint:
    """Real-time validation data point"""
    timestamp: float
    target_angle_deg: float
    commanded_percentage: float
    measured_angle_deg: Optional[float]
    angle_error_deg: Optional[float]
    settling_time_ms: Optional[float]

class SteeringValidator:
    def __init__(self, mcm_channel='can2', sas_channel='can3', trajectory_file=None):
        # CAN Bus Configuration
        self.mcm_channel = mcm_channel
        self.sas_channel = sas_channel
        
        # MCM Controller Setup
        self.mcm_db = cantools.database.Database()
        self.mcm_db.add_dbc_file('./sygnal_dbc/mcm/Heartbeat.dbc')
        self.mcm_db.add_dbc_file('./sygnal_dbc/mcm/Control.dbc')
        
        # CAN Bus Connections
        try:
            self.mcm_bus = can.Bus(channel=mcm_channel, bustype='socketcan', bitrate=500000)
            print(f"MCM Controller connected to {mcm_channel}")
        except Exception as e:
            print(f"Failed to connect MCM bus {mcm_channel}: {e}")
            sys.exit(1)
            
        try:
            self.sas_bus = can.Bus(channel=sas_channel, bustype='socketcan', bitrate=500000)
            print(f"SAS Monitor connected to {sas_channel}")
        except Exception as e:
            print(f"Failed to connect SAS bus {sas_channel}: {e}")
            sys.exit(1)
        
        # MCM Control State
        self.control_count = 0
        self.bus_address = 1
        self.last_steer_percentage = 0.0
        self.control_enabled = False
        
        # SAS11 Monitoring
        self.SAS11_CAN_ID = 0x2B0
        self.SAS_ANGLE_SCALE = 0.1
        self.latest_angle_deg = None
        self.latest_angle_timestamp = None
        
        # Trajectory Data
        self.trajectory_targets: List[TrajectoryTarget] = []
        self.validation_log: List[ValidationPoint] = []
        
        # Validation Parameters
        self.ANGLE_TOLERANCE_DEG = 5.0      # Acceptable position error
        self.SETTLING_TIMEOUT_MS = 2000     # Maximum settling time
        self.MAX_STEERING_DEG = 400.0       # Physical steering limit
        self.COMMAND_UPDATE_RATE = 0.05     # 20Hz command rate
        
        # Conversion Parameters (require calibration)
        self.ANGLE_TO_PERCENTAGE_GAIN = 100.0 / 400.0  # Assume ±400° = ±100%
        self.CENTER_OFFSET_DEG = 0.0        # Steering center offset
        
        # Thread synchronization
        self.running = False
        self.sas_lock = threading.Lock()
        
        # Load trajectory if provided
        if trajectory_file:
            self.load_trajectory_data(trajectory_file)
            
        # Statistics tracking
        self.stats = {
            'commands_sent': 0,
            'successful_tracks': 0,
            'position_errors': [],
            'settling_times': [],
            'max_error': 0.0
        }
        
        # Timing compensation tracking
        self.timing_stats = {
            'command_latencies': [],
            'cumulative_drift_ms': 0.0,
            'max_early_ms': 0.0,
            'max_late_ms': 0.0,
            'timing_corrections': 0,
            'average_execution_delay_ms': 0.0
        }
    
    def load_trajectory_data(self, filename: str):
        """Load recorded trajectory data from JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            trajectory_points = data['trajectory_points']
            start_time = trajectory_points[0]['timestamp']
            
            for point in trajectory_points:
                target = TrajectoryTarget(
                    timestamp=point['timestamp'],
                    target_angle_deg=point['steering_angle_deg'],
                    relative_time=point['timestamp'] - start_time
                )
                self.trajectory_targets.append(target)
            
            print(f"Loaded {len(self.trajectory_targets)} trajectory targets")
            print(f"Trajectory duration: {self.trajectory_targets[-1].relative_time:.1f}s")
            print(f"Angle range: {min(t.target_angle_deg for t in self.trajectory_targets):.1f}° to "
                  f"{max(t.target_angle_deg for t in self.trajectory_targets):.1f}°")
            
        except Exception as e:
            print(f"Failed to load trajectory data: {e}")
            sys.exit(1)
    
    def calc_crc8(self, data: bytes) -> int:
        """Calculate CRC8 for MCM messages"""
        hash = crc8.crc8()
        hash.update(data[:-1])
        return hash.digest()[0]
    
    def angle_to_percentage(self, angle_deg: float) -> float:
        """Convert steering angle to MCM command percentage"""
        # Apply center offset and coordinate frame inversion
        # SAS11: Left=positive, Right=negative
        # MCM:   Left=negative, Right=positive
        adjusted_angle = -(angle_deg - self.CENTER_OFFSET_DEG)
        percentage = adjusted_angle * self.ANGLE_TO_PERCENTAGE_GAIN
        
        # Clamp to valid range
        return max(min(percentage, 100.0), -100.0)
    
    def parse_sas11_angle(self, data: bytes) -> Optional[float]:
        """Extract steering angle from SAS11 CAN message"""
        if len(data) != 5:
            return None
        
        try:
            # Extend to 8 bytes for processing
            data_extended = data + b'\x00\x00\x00'
            frame_uint64 = struct.unpack('<Q', data_extended)[0]
            
            # Extract steering angle (16-bit signed)
            angle_raw = frame_uint64 & 0xFFFF
            if angle_raw & 0x8000:  # Sign extend
                angle_raw |= 0xFFFF0000
                angle_raw = struct.unpack('<i', struct.pack('<I', angle_raw & 0xFFFFFFFF))[0]
            
            steering_angle = angle_raw * self.SAS_ANGLE_SCALE
            
            # Validation range check
            if -self.MAX_STEERING_DEG <= steering_angle <= self.MAX_STEERING_DEG:
                return steering_angle
            else:
                return None
                
        except Exception as e:
            print(f"SAS11 parsing error: {e}")
            return None
    
    async def enable_mcm_control(self):
        """Enable MCM steering control"""
        msg = self.mcm_db.get_message_by_name('ControlEnable')
        data = bytearray(msg.encode({
            'BusAddress': self.bus_address,
            'InterfaceID': 2,
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
            self.control_enabled = True
            await asyncio.sleep(0.02)
        except Exception as e:
            print(f"Failed to enable MCM control: {e}")
    
    async def send_steering_command(self, percentage: float) -> bool:
        """Send single MCM steering command"""
        # Clamp to valid range
        percentage = max(min(percentage, 100.0), -100.0)
        
        msg = self.mcm_db.get_message_by_name('ControlCommand')
        data = bytearray(msg.encode({
            'BusAddress': self.bus_address,
            'InterfaceID': 2,
            'Count8': self.control_count,
            'Value': percentage / 100.0,  # Normalize to -1 to 1
            'CRC': 0
        }))
        data[7] = self.calc_crc8(data)
        
        try:
            await self.enable_mcm_control()
            self.mcm_bus.send(can.Message(
                arbitration_id=msg.frame_id,
                is_extended_id=False,
                data=data
            ))
            
            self.control_count = (self.control_count + 1) % 256
            self.last_steer_percentage = percentage
            self.stats['commands_sent'] += 1
            
            return True
        except Exception as e:
            print(f"Failed to send steering command: {e}")
            return False
    
    def monitor_sas_feedback(self):
        """Background thread for SAS11 angle monitoring"""
        print("Starting SAS11 monitoring thread")
        
        while self.running:
            try:
                message = self.sas_bus.recv(timeout=0.1)
                
                if message and message.arbitration_id == self.SAS11_CAN_ID:
                    angle = self.parse_sas11_angle(message.data)
                    
                    if angle is not None:
                        with self.sas_lock:
                            self.latest_angle_deg = angle
                            self.latest_angle_timestamp = message.timestamp
                            
            except Exception as e:
                if self.running:  # Only log errors if we're supposed to be running
                    print(f"SAS monitoring error: {e}")
                    time.sleep(0.1)
    
    def get_current_angle(self) -> Tuple[Optional[float], Optional[float]]:
        """Get latest measured steering angle with timestamp"""
        with self.sas_lock:
            return self.latest_angle_deg, self.latest_angle_timestamp
    
    async def wait_for_settling(self, target_angle: float, timeout_ms: int = 2000) -> Tuple[bool, float]:
        """Wait for steering to settle at target angle"""
        start_time = time.time() * 1000
        last_stable_time = start_time
        
        while (time.time() * 1000 - start_time) < timeout_ms:
            current_angle, _ = self.get_current_angle()
            
            if current_angle is not None:
                error = abs(current_angle - target_angle)
                
                if error <= self.ANGLE_TOLERANCE_DEG:
                    # Check if we've been stable long enough
                    stable_duration = time.time() * 1000 - last_stable_time
                    if stable_duration >= 100:  # 100ms stability requirement
                        settling_time = time.time() * 1000 - start_time
                        return True, settling_time
                else:
                    last_stable_time = time.time() * 1000
            
            await asyncio.sleep(0.01)  # 10ms check interval
        
        return False, timeout_ms
    
    async def execute_trajectory_validation(self, high_frequency_mode=True):
        """Execute trajectory validation with configurable frequency modes"""
        print("\nStarting trajectory validation sequence...")
        print(f"Total targets: {len(self.trajectory_targets)}")
        print(f"Validation tolerance: ±{self.ANGLE_TOLERANCE_DEG}°")
        print(f"Mode: {'High-frequency' if high_frequency_mode else 'Original timing'}")
        
        validation_start = time.time()
        original_duration = self.trajectory_targets[-1].relative_time
        
        if high_frequency_mode:
            # High-frequency mode: fixed command rate, parallel validation
            await self.execute_high_frequency_validation(validation_start, original_duration)
        else:
            # Original mode: preserve recorded timing with settling waits
            await self.execute_original_timing_validation(validation_start)
        
        print(f"\n✓ Trajectory validation completed in {time.time() - validation_start:.1f}s")
        print(f"  Original trajectory duration: {original_duration:.1f}s")
        self.print_validation_summary()
    
    async def execute_high_frequency_validation(self, start_time: float, target_duration: float):
        """High-frequency validation mode - preserves original trajectory timing"""
        validation_buffer = deque(maxlen=50)  # Rolling validation window
        
        print(f"Preserving original trajectory timing over {target_duration:.1f}s")
        
        last_command_percentage = None
        last_command_time = start_time
        
        for i, target in enumerate(self.trajectory_targets):
            # Use original trajectory timestamp for precise timing
            command_time = start_time + target.relative_time
            current_time = time.time()
            
            # Wait until the exact trajectory time
            if command_time > current_time:
                await asyncio.sleep(command_time - current_time)
            
            # Convert target angle to MCM command
            command_percentage = self.angle_to_percentage(target.target_angle_deg)
            
            # Only send command if it's different from last command (efficiency)
            command_sent = False
            if last_command_percentage is None or abs(command_percentage - last_command_percentage) > 0.5:
                command_sent = await self.send_steering_command(command_percentage)
                last_command_percentage = command_percentage
                last_command_time = time.time()
            else:
                # Same command, just log the validation point without transmission
                command_sent = True
            
            if command_sent:
                # Capture immediate feedback
                measured_angle, measure_timestamp = self.get_current_angle()
                
                # Log validation point
                validation_point = ValidationPoint(
                    timestamp=time.time(),
                    target_angle_deg=target.target_angle_deg,
                    commanded_percentage=command_percentage,
                    measured_angle_deg=measured_angle,
                    angle_error_deg=(measured_angle - target.target_angle_deg) if measured_angle else None,
                    settling_time_ms=None  # No settling measurement in trajectory mode
                )
                self.validation_log.append(validation_point)
                validation_buffer.append(validation_point)
                
                # Update statistics
                if measured_angle is not None:
                    error = abs(measured_angle - target.target_angle_deg)
                    self.stats['position_errors'].append(error)
                    self.stats['max_error'] = max(self.stats['max_error'], error)
                    
                    if error <= self.ANGLE_TOLERANCE_DEG:
                        self.stats['successful_tracks'] += 1
                
    async def execute_high_frequency_validation(self, start_time: float, target_duration: float):
        """High-frequency validation mode - preserves original trajectory timing with compensation"""
        validation_buffer = deque(maxlen=50)  # Rolling validation window
        
        print(f"Preserving original trajectory timing over {target_duration:.1f}s")
        print("Implementing predictive timing compensation...")
        
        last_command_percentage = None
        running_drift_ms = 0.0  # Cumulative timing drift
        average_delay_ms = 5.0  # Initial estimate of system delay
        
        for i, target in enumerate(self.trajectory_targets):
            # Calculate intended execution time from original trajectory
            intended_time = start_time + target.relative_time
            current_time = time.time()
            
            # Apply predictive timing compensation
            # Send command early by the average observed delay plus current drift
            compensation_ms = (average_delay_ms + running_drift_ms) / 1000.0
            compensated_time = intended_time - compensation_ms
            
            # Wait until the compensated command time
            if compensated_time > current_time:
                await asyncio.sleep(compensated_time - current_time)
            
            # Record actual command execution time
            actual_execution_time = time.time()
            
            # Convert target angle to MCM command
            command_percentage = self.angle_to_percentage(target.target_angle_deg)
            
            # Only send command if it's different from last command (efficiency)
            command_sent = False
            if last_command_percentage is None or abs(command_percentage - last_command_percentage) > 0.5:
                # Measure command transmission latency
                cmd_start = time.time()
                command_sent = await self.send_steering_command(command_percentage)
                cmd_end = time.time()
                
                # Track command latency
                cmd_latency_ms = (cmd_end - cmd_start) * 1000
                self.timing_stats['command_latencies'].append(cmd_latency_ms)
                
                last_command_percentage = command_percentage
            else:
                # Same command, just log without transmission
                command_sent = True
            
            if command_sent:
                # Calculate timing error (how early/late we actually executed)
                timing_error_ms = (actual_execution_time - intended_time) * 1000
                
                # Update timing statistics
                if timing_error_ms > 0:
                    self.timing_stats['max_late_ms'] = max(self.timing_stats['max_late_ms'], timing_error_ms)
                else:
                    self.timing_stats['max_early_ms'] = max(self.timing_stats['max_early_ms'], abs(timing_error_ms))
                
                # Update running drift compensation
                if len(self.timing_stats['command_latencies']) >= 3:
                    # Calculate running average of execution delays
                    recent_delays = self.timing_stats['command_latencies'][-10:]  # Last 10 commands
                    average_delay_ms = sum(recent_delays) / len(recent_delays)
                    self.timing_stats['average_execution_delay_ms'] = average_delay_ms
                    
                    # Adjust drift compensation based on timing error
                    if abs(timing_error_ms) > 10:  # Only correct significant errors
                        running_drift_ms += timing_error_ms * 0.3  # Proportional correction
                        self.timing_stats['timing_corrections'] += 1
                
                # Update cumulative drift
                self.timing_stats['cumulative_drift_ms'] = running_drift_ms
                
                # Capture feedback
                measured_angle, measure_timestamp = self.get_current_angle()
                
                # Log validation point with timing data
                validation_point = ValidationPoint(
                    timestamp=actual_execution_time,
                    target_angle_deg=target.target_angle_deg,
                    commanded_percentage=command_percentage,
                    measured_angle_deg=measured_angle,
                    angle_error_deg=(measured_angle - target.target_angle_deg) if measured_angle else None,
                    settling_time_ms=timing_error_ms  # Repurpose for timing error
                )
                self.validation_log.append(validation_point)
                validation_buffer.append(validation_point)
                
                # Update statistics
                if measured_angle is not None:
                    error = abs(measured_angle - target.target_angle_deg)
                    self.stats['position_errors'].append(error)
                    self.stats['max_error'] = max(self.stats['max_error'], error)
                    
                    if error <= self.ANGLE_TOLERANCE_DEG:
                        self.stats['successful_tracks'] += 1
                
                # Real-time progress display with timing info
                if (i + 1) % 10 == 0 or i == len(self.trajectory_targets) - 1:
                    recent_errors = [p.angle_error_deg for p in validation_buffer if p.angle_error_deg is not None]
                    if recent_errors:
                        avg_error = sum(abs(e) for e in recent_errors) / len(recent_errors)
                        success_rate = sum(1 for e in recent_errors if abs(e) <= self.ANGLE_TOLERANCE_DEG) / len(recent_errors) * 100
                        
                        # Calculate interval since last trajectory point  
                        if i > 0:
                            interval = target.relative_time - self.trajectory_targets[i-1].relative_time
                            print(f"[{i+1:3d}/{len(self.trajectory_targets)}] "
                                  f"T+{target.relative_time:.1f}s (Δ{interval:.1f}s) | "
                                  f"Timing: {timing_error_ms:+5.1f}ms | "
                                  f"Target: {target.target_angle_deg:+7.1f}° | "
                                  f"Measured: {measured_angle:+7.1f}° | "
                                  f"Success: {success_rate:.0f}%")
                        else:
                            print(f"[{i+1:3d}/{len(self.trajectory_points)}] "
                                  f"T+{target.relative_time:.1f}s | "
                                  f"Timing: {timing_error_ms:+5.1f}ms | "
                                  f"Target: {target.target_angle_deg:+7.1f}° | "
                                  f"Measured: {measured_angle:+7.1f}° | "
                                  f"Success: {success_rate:.0f}%")
    
    async def execute_original_timing_validation(self, start_time: float):
        """Original timing validation mode - preserves recorded intervals"""
        for i, target in enumerate(self.trajectory_targets):
            # Preserve original trajectory timing
            target_time = start_time + target.relative_time
            current_time = time.time()
            
            if target_time > current_time:
                await asyncio.sleep(target_time - current_time)
            
            # Convert target angle to MCM percentage
            command_percentage = self.angle_to_percentage(target.target_angle_deg)
            
            print(f"\n[{i+1:3d}/{len(self.trajectory_targets)}] "
                  f"Target: {target.target_angle_deg:+7.1f}° → Command: {command_percentage:+6.1f}%")
            
            # Send steering command
            command_sent = await self.send_steering_command(command_percentage)
            
            if not command_sent:
                print("❌ Failed to send command")
                continue
            
            # Wait for system to settle and measure result
            settled, settling_time = await self.wait_for_settling(target.target_angle_deg, timeout_ms=500)
            measured_angle, _ = self.get_current_angle()
            
            # Calculate validation metrics
            angle_error = None
            if measured_angle is not None:
                angle_error = measured_angle - target.target_angle_deg
                self.stats['position_errors'].append(abs(angle_error))
                self.stats['max_error'] = max(self.stats['max_error'], abs(angle_error))
            
            # Log validation point
            validation_point = ValidationPoint(
                timestamp=time.time(),
                target_angle_deg=target.target_angle_deg,
                commanded_percentage=command_percentage,
                measured_angle_deg=measured_angle,
                angle_error_deg=angle_error,
                settling_time_ms=settling_time if settled else None
            )
            self.validation_log.append(validation_point)
            
            # Display results
            if measured_angle is not None and angle_error is not None:
                status = "✓" if abs(angle_error) <= self.ANGLE_TOLERANCE_DEG else "⚠"
                settle_status = "✓" if settled else "⏱"
                
                print(f"    {status} Measured: {measured_angle:+7.1f}° | "
                      f"Error: {angle_error:+6.1f}° | "
                      f"Settle: {settle_status} {settling_time:.0f}ms")
                
                if abs(angle_error) <= self.ANGLE_TOLERANCE_DEG:
                    self.stats['successful_tracks'] += 1
                    self.stats['settling_times'].append(settling_time)
            else:
                print("    ❌ No feedback received")
    
    def print_validation_summary(self):
        """Print comprehensive validation results including timing analysis"""
        print("\n" + "="*80)
        print("STEERING VALIDATION SUMMARY")
        print("="*80)
        
        success_rate = (self.stats['successful_tracks'] / self.stats['commands_sent'] * 100) if self.stats['commands_sent'] > 0 else 0
        
        print(f"Commands Sent: {self.stats['commands_sent']}")
        print(f"Successful Tracking: {self.stats['successful_tracks']} ({success_rate:.1f}%)")
        print(f"Maximum Error: {self.stats['max_error']:.2f}°")
        
        if self.stats['position_errors']:
            mean_error = sum(self.stats['position_errors']) / len(self.stats['position_errors'])
            print(f"Mean Position Error: {mean_error:.2f}°")
            print(f"RMS Position Error: {(sum(e**2 for e in self.stats['position_errors']) / len(self.stats['position_errors']))**0.5:.2f}°")
        
        if self.stats['settling_times']:
            mean_settle = sum(self.stats['settling_times']) / len(self.stats['settling_times'])
            max_settle = max(self.stats['settling_times'])
            print(f"Mean Settling Time: {mean_settle:.0f}ms")
            print(f"Maximum Settling Time: {max_settle:.0f}ms")
        
        # Timing performance analysis
        print("\nTIMING PERFORMANCE:")
        print("-" * 40)
        if self.timing_stats['command_latencies']:
            avg_latency = sum(self.timing_stats['command_latencies']) / len(self.timing_stats['command_latencies'])
            max_latency = max(self.timing_stats['command_latencies'])
            print(f"Average Command Latency: {avg_latency:.1f}ms")
            print(f"Maximum Command Latency: {max_latency:.1f}ms")
        
        print(f"Maximum Early Execution: {self.timing_stats['max_early_ms']:.1f}ms")
        print(f"Maximum Late Execution: {self.timing_stats['max_late_ms']:.1f}ms")
        print(f"Final Cumulative Drift: {self.timing_stats['cumulative_drift_ms']:+.1f}ms")
        print(f"Timing Corrections Applied: {self.timing_stats['timing_corrections']}")
        print(f"Final Avg Execution Delay: {self.timing_stats['average_execution_delay_ms']:.1f}ms")
    
    def save_validation_results(self, filename: str = None):
        """Save validation results to CSV with timing analysis"""
        if filename is None:
            filename = f"steering_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'target_angle_deg', 'commanded_percentage', 
                'measured_angle_deg', 'angle_error_deg', 'timing_error_ms',
                'within_tolerance', 'within_timing_budget'
            ])
            
            for point in self.validation_log:
                within_tolerance = (point.angle_error_deg is not None and 
                                  abs(point.angle_error_deg) <= self.ANGLE_TOLERANCE_DEG)
                within_timing = (point.settling_time_ms is not None and 
                               abs(point.settling_time_ms) <= 50.0)  # 50ms timing tolerance
                
                writer.writerow([
                    point.timestamp, point.target_angle_deg, point.commanded_percentage,
                    point.measured_angle_deg, point.angle_error_deg, point.settling_time_ms,
                    within_tolerance, within_timing
                ])
        
        # Save timing statistics summary
        timing_file = filename.replace('.csv', '_timing_stats.json')
        with open(timing_file, 'w') as f:
            json.dump({
                'timing_performance': self.timing_stats,
                'validation_summary': {
                    'total_commands': self.stats['commands_sent'],
                    'successful_tracks': self.stats['successful_tracks'],
                    'timing_accuracy_rate': sum(1 for p in self.validation_log 
                                               if p.settling_time_ms and abs(p.settling_time_ms) <= 50.0) / len(self.validation_log),
                    'mean_timing_error': sum(abs(p.settling_time_ms) for p in self.validation_log 
                                           if p.settling_time_ms) / len([p for p in self.validation_log if p.settling_time_ms])
                }
            }, f, indent=2)
        
        print(f"Validation results saved to: {filename}")
        print(f"Timing analysis saved to: {timing_file}")
    
    async def maintain_mcm_control(self):
        """Maintain MCM control enable signal"""
        while self.running:
            try:
                await self.enable_mcm_control()
                await asyncio.sleep(0.1)
            except Exception as e:
                print(f"Control maintenance error: {e}")
                await asyncio.sleep(0.5)
    
    async def run_validation(self):
        """Main validation execution"""
        self.running = True
        
        # Start SAS monitoring thread
        sas_thread = threading.Thread(target=self.monitor_sas_feedback, daemon=True)
        sas_thread.start()
        
        # Start MCM control maintenance
        control_task = asyncio.create_task(self.maintain_mcm_control())
        
        # Wait for initial SAS data
        print("Waiting for initial SAS11 data...")
        timeout = 5.0
        start_wait = time.time()
        
        while time.time() - start_wait < timeout:
            if self.latest_angle_deg is not None:
                print(f"✓ SAS11 connected - Initial angle: {self.latest_angle_deg:.1f}°")
                break
            await asyncio.sleep(0.1)
        else:
            print("❌ Timeout waiting for SAS11 data")
            return
        
        try:
            # Execute trajectory validation with configured mode
            high_freq_mode = getattr(self, '_high_frequency_mode', True)
            await self.execute_trajectory_validation(high_frequency_mode=high_freq_mode)
            
        except KeyboardInterrupt:
            print("\n⏹ Validation interrupted by user")
        finally:
            self.running = False
            control_task.cancel()
            self.save_validation_results()
            print("Validation session ended")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Steering Command Validation Controller',
        epilog='''
Validates MCM steering commands against SAS11 feedback using recorded trajectory data.
Provides comprehensive analysis of command tracking performance.

Example:
  python3 steering_validator.py --trajectory trajectory.json --mcm can2 --sas can3
        '''
    )
    parser.add_argument('--trajectory', '-t', required=True,
                       help='Trajectory JSON file from previous recording')
    parser.add_argument('--mcm', default='can2',
                       help='MCM command CAN interface (default: can2)')
    parser.add_argument('--sas', default='can3', 
                       help='SAS feedback CAN interface (default: can3)')
    parser.add_argument('--tolerance', type=float, default=5.0,
                       help='Angle tolerance in degrees (default: 5.0)')
    
    args = parser.parse_args()
    
    print("Steering Command Validation Controller")
    print("MCM Command → SAS11 Feedback Validation")
    print(f"MCM Bus: {args.mcm} | SAS Bus: {args.sas}")
    print(f"Trajectory: {args.trajectory}")
    print()
    
    # Initialize validator
    validator = SteeringValidator(
        mcm_channel=args.mcm,
        sas_channel=args.sas,
        trajectory_file=args.trajectory
    )
    validator.ANGLE_TOLERANCE_DEG = args.tolerance
    
    # Setup signal handling
    def cleanup(sig=None, frame=None):
        print("\n🛑 Stopping validation...")
        validator.running = False
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup)
    
    # Configure execution mode via class attribute
    validator._high_frequency_mode = not getattr(args, 'original_timing', False)
    
    # Run validation
    try:
        asyncio.run(validator.run_validation())
    except KeyboardInterrupt:
        cleanup()

if __name__ == "__main__":
    main()