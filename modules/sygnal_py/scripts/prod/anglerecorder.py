#!/usr/bin/env python3
"""
Enhanced Steering Trajectory Recorder
Advanced SAS11 steering trajectory capture with comprehensive validation support

Captures high-fidelity vehicle trajectory data including:
- Variable-threshold steering angle detection
- Timing characteristics and system latency
- CAN message flow analysis
- Quality metrics and calibration data
- Validation-ready output format

Designed to work seamlessly with the Steering Command Validation Controller
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
class TrajectoryPoint:
    """Enhanced trajectory point with comprehensive metadata"""
    timestamp: float
    steering_angle_deg: float
    angle_change_deg: float
    cumulative_change_deg: float
    message_count: int
    raw_data: str
    iso_time: str
    
    # Enhanced timing data
    inter_message_interval_ms: Optional[float] = None
    can_bus_load_percent: Optional[float] = None
    message_rate_hz: Optional[float] = None
    
    # Quality indicators
    signal_quality_score: Optional[float] = None
    angle_velocity_deg_s: Optional[float] = None
    steering_direction: Optional[str] = None

@dataclass
class SystemCharacteristics:
    """System timing and performance characteristics"""
    average_message_interval_ms: float
    message_rate_statistics: Dict[str, float]
    timing_jitter_ms: float
    max_latency_ms: float
    angle_resolution_deg: float
    coordinate_frame: str  # "SAS11_LEFT_POSITIVE"

@dataclass
class RecordingSession:
    """Complete recording session metadata"""
    start_time: float
    end_time: float
    duration_seconds: float
    interface: str
    angle_threshold: float
    adaptive_threshold_enabled: bool
    initial_angle: float
    final_angle: float
    total_cumulative_change: float
    coordinate_frame_verified: bool

class EnhancedSteeringTrajectoryRecorder:
    def __init__(self, interface='can3', angle_threshold=10.0, adaptive_threshold=True, 
                 quality_monitoring=True):
        self.interface = interface
        self.base_angle_threshold = angle_threshold
        self.adaptive_threshold_enabled = adaptive_threshold
        self.current_threshold = angle_threshold
        self.quality_monitoring = quality_monitoring
        self.start_time = time.time()
        
        # CAN configuration
        self.SAS11_CAN_ID = 0x2B0   # 688 decimal
        self.SAS_ANGLE_SCALE = 0.1  # degrees per LSB
        
        # Enhanced trajectory tracking
        self.last_recorded_angle = None
        self.last_message_time = None
        self.initial_angle = None
        self.cumulative_change = 0.0
        self.trajectory_points: List[TrajectoryPoint] = []
        self.message_count = 0
        
        # Timing analysis
        self.message_intervals = deque(maxlen=1000)  # Rolling window
        self.recent_angles = deque(maxlen=50)        # For velocity calculation
        self.recent_timestamps = deque(maxlen=50)
        self.can_message_buffer = deque(maxlen=100)  # Bus load analysis
        
        # Statistics and quality metrics
        self.total_messages = 0
        self.valid_messages = 0
        self.trajectory_events = 0
        self.quality_scores = []
        self.timing_anomalies = 0
        
        # System characterization
        self.system_characteristics = None
        
        # Thread synchronization
        self.running = False
        self.data_lock = threading.Lock()
        
        # Enhanced logging setup
        self.setup_enhanced_logging()
        
        # Connect to CAN bus
        try:
            self.bus = can.interface.Bus(channel=interface, bustype='socketcan')
            print(f"✓ Enhanced recorder connected to {interface}")
            print(f"Base threshold: {angle_threshold}° | Adaptive: {adaptive_threshold}")
            print(f"Quality monitoring: {quality_monitoring}")
        except Exception as e:
            print(f"✗ Failed to connect to {interface}: {e}")
            sys.exit(1)
    
    def setup_enhanced_logging(self):
        """Initialize comprehensive logging infrastructure"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Primary trajectory data
        self.trajectory_csv = f"enhanced_trajectory_{timestamp}.csv"
        with open(self.trajectory_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'iso_time', 'steering_angle_deg', 
                'angle_change_deg', 'cumulative_change_deg', 
                'message_count', 'raw_data_hex',
                'inter_message_interval_ms', 'can_bus_load_percent',
                'message_rate_hz', 'signal_quality_score',
                'angle_velocity_deg_s', 'steering_direction'
            ])
        
        # Comprehensive session data
        self.trajectory_json = f"enhanced_trajectory_{timestamp}.json"
        
        # Raw timing data for validation calibration
        self.timing_csv = f"timing_analysis_{timestamp}.csv"
        with open(self.timing_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'message_interval_ms', 'angle_deg', 
                'can_id', 'raw_data_hex', 'bus_load_estimate'
            ])
        
        print(f"Enhanced logging:")
        print(f"  Trajectory: {self.trajectory_csv}")
        print(f"  Session data: {self.trajectory_json}")
        print(f"  Timing analysis: {self.timing_csv}")
    
    def parse_sas11_angle(self, data: bytes) -> Optional[float]:
        """Extract steering angle from SAS11 with enhanced validation"""
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
            
            # Enhanced validation with outlier detection
            if -4000.0 <= steering_angle <= 4000.0:
                # Check for unrealistic angle jumps
                if self.recent_angles and len(self.recent_angles) > 0:
                    last_angle = self.recent_angles[-1]
                    angle_jump = abs(steering_angle - last_angle)
                    
                    # Flag suspicious jumps (>50°/message typical at 100Hz = 5000°/s)
                    if angle_jump > 50.0:
                        print(f"⚠ Large angle jump detected: {angle_jump:.1f}° "
                              f"({last_angle:.1f}° → {steering_angle:.1f}°)")
                        return None
                
                return steering_angle
            else:
                return None
            
        except Exception as e:
            print(f"SAS11 parsing error: {e}")
            return None
    
    def calculate_quality_metrics(self, angle: float, timestamp: float) -> Dict[str, Optional[float]]:
        """Calculate comprehensive quality metrics for current measurement"""
        metrics = {
            'signal_quality_score': None,
            'angle_velocity_deg_s': None,
            'steering_direction': None,
            'message_rate_hz': None,
            'inter_message_interval_ms': None
        }
        
        # Calculate inter-message timing
        if self.last_message_time:
            interval_ms = (timestamp - self.last_message_time) * 1000
            metrics['inter_message_interval_ms'] = interval_ms
            self.message_intervals.append(interval_ms)
            
            # Detect timing anomalies
            if len(self.message_intervals) > 10:
                median_interval = statistics.median(self.message_intervals)
                if abs(interval_ms - median_interval) > median_interval * 0.5:
                    self.timing_anomalies += 1
        
        # Calculate angular velocity
        if len(self.recent_angles) >= 2 and len(self.recent_timestamps) >= 2:
            time_span = timestamp - self.recent_timestamps[0]
            if time_span > 0:
                angle_span = angle - self.recent_angles[0]
                metrics['angle_velocity_deg_s'] = angle_span / time_span
        
        # Determine steering direction
        if metrics['angle_velocity_deg_s']:
            if abs(metrics['angle_velocity_deg_s']) > 1.0:  # Threshold for active steering
                metrics['steering_direction'] = "LEFT" if metrics['angle_velocity_deg_s'] > 0 else "RIGHT"
            else:
                metrics['steering_direction'] = "STABLE"
        
        # Calculate message rate
        if len(self.message_intervals) > 5:
            avg_interval_ms = sum(list(self.message_intervals)[-10:]) / min(10, len(self.message_intervals))
            metrics['message_rate_hz'] = 1000.0 / avg_interval_ms if avg_interval_ms > 0 else None
        
        # Calculate signal quality score (0-100)
        quality_factors = []
        
        # Timing consistency (0-40 points)
        if len(self.message_intervals) > 5:
            timing_std = statistics.stdev(list(self.message_intervals)[-20:])
            timing_score = max(0, 40 - (timing_std / 2))  # Penalize jitter
            quality_factors.append(timing_score)
        
        # Angle continuity (0-30 points)
        if len(self.recent_angles) > 3:
            angle_changes = [abs(self.recent_angles[i] - self.recent_angles[i-1]) 
                           for i in range(1, min(5, len(self.recent_angles)))]
            avg_change = sum(angle_changes) / len(angle_changes)
            continuity_score = max(0, 30 - avg_change)  # Penalize erratic changes
            quality_factors.append(continuity_score)
        
        # Message rate stability (0-30 points)
        if metrics['message_rate_hz']:
            target_rate = 100.0  # Expected SAS11 rate
            rate_deviation = abs(metrics['message_rate_hz'] - target_rate)
            rate_score = max(0, 30 - rate_deviation)
            quality_factors.append(rate_score)
        
        if quality_factors:
            metrics['signal_quality_score'] = sum(quality_factors)
            self.quality_scores.append(metrics['signal_quality_score'])
        
        return metrics
    
    def calculate_adaptive_threshold(self) -> float:
        """Dynamically adjust recording threshold based on steering activity"""
        if not self.adaptive_threshold_enabled or len(self.recent_angles) < 10:
            return self.base_angle_threshold
        
        # Analyze recent steering activity
        recent_changes = [abs(self.recent_angles[i] - self.recent_angles[i-1]) 
                         for i in range(1, len(self.recent_angles))]
        
        if not recent_changes:
            return self.base_angle_threshold
        
        avg_change = sum(recent_changes) / len(recent_changes)
        max_change = max(recent_changes)
        
        # Adaptive logic
        if avg_change > 5.0:  # High activity - use smaller threshold for detail
            adaptive_threshold = max(2.0, self.base_angle_threshold * 0.5)
        elif avg_change < 1.0:  # Low activity - use larger threshold to reduce noise
            adaptive_threshold = min(20.0, self.base_angle_threshold * 1.5)
        else:
            adaptive_threshold = self.base_angle_threshold
        
        return adaptive_threshold
    
    def should_record_angle(self, current_angle: float) -> bool:
        """Enhanced angle recording decision with adaptive thresholding"""
        if self.last_recorded_angle is None:
            return True  # Always record first angle
        
        angle_change = abs(current_angle - self.last_recorded_angle)
        self.current_threshold = self.calculate_adaptive_threshold()
        
        return angle_change >= self.current_threshold
    
    def record_trajectory_point(self, angle: float, timestamp: float, raw_data: bytes, 
                               quality_metrics: Dict):
        """Record enhanced trajectory point with comprehensive metadata"""
        # Calculate angle change
        if self.last_recorded_angle is not None:
            angle_change = angle - self.last_recorded_angle
            self.cumulative_change += angle_change
        else:
            angle_change = 0.0
            self.initial_angle = angle
        
        # Create enhanced trajectory point
        point = TrajectoryPoint(
            timestamp=timestamp,
            steering_angle_deg=angle,
            angle_change_deg=angle_change,
            cumulative_change_deg=self.cumulative_change,
            message_count=self.message_count,
            raw_data=raw_data.hex().upper(),
            iso_time=datetime.fromtimestamp(timestamp).isoformat(),
            inter_message_interval_ms=quality_metrics.get('inter_message_interval_ms'),
            can_bus_load_percent=self.estimate_bus_load(),
            message_rate_hz=quality_metrics.get('message_rate_hz'),
            signal_quality_score=quality_metrics.get('signal_quality_score'),
            angle_velocity_deg_s=quality_metrics.get('angle_velocity_deg_s'),
            steering_direction=quality_metrics.get('steering_direction')
        )
        
        with self.data_lock:
            self.trajectory_points.append(point)
            self.trajectory_events += 1
            self.last_recorded_angle = angle
        
        # Log to CSV immediately
        self.log_trajectory_to_csv(point)
        self.log_timing_data(timestamp, quality_metrics.get('inter_message_interval_ms'), 
                           angle, raw_data)
        
        # Real-time display with enhanced info
        velocity = quality_metrics.get('angle_velocity_deg_s') or 0
        quality = quality_metrics.get('signal_quality_score') or 0
        
        print(f"[{self.trajectory_events:3d}] {point.iso_time[-12:]} | "
              f"Angle: {angle:+7.1f}° | "
              f"Change: {angle_change:+6.1f}° | "
              f"Velocity: {velocity:+6.1f}°/s | "
              f"Quality: {quality:3.0f}% | "
              f"Threshold: {self.current_threshold:.1f}°")
    
    def estimate_bus_load(self) -> Optional[float]:
        """Estimate CAN bus load based on message timing"""
        if len(self.can_message_buffer) < 10:
            return None
        
        recent_msgs = list(self.can_message_buffer)[-10:]
        time_span = recent_msgs[-1]['timestamp'] - recent_msgs[0]['timestamp']
        
        if time_span <= 0:
            return None
        
        # Estimate based on message density
        msg_rate = len(recent_msgs) / time_span
        estimated_load = min(100.0, (msg_rate / 1000.0) * 100)  # Assume 1000 msg/s = 100%
        
        return estimated_load
    
    def log_trajectory_to_csv(self, point: TrajectoryPoint):
        """Log enhanced trajectory point to CSV"""
        with open(self.trajectory_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                point.timestamp, point.iso_time, point.steering_angle_deg,
                point.angle_change_deg, point.cumulative_change_deg,
                point.message_count, point.raw_data,
                point.inter_message_interval_ms, point.can_bus_load_percent,
                point.message_rate_hz, point.signal_quality_score,
                point.angle_velocity_deg_s, point.steering_direction
            ])
    
    def log_timing_data(self, timestamp: float, interval_ms: Optional[float], 
                       angle: float, raw_data: bytes):
        """Log raw timing data for validation system calibration"""
        with open(self.timing_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp, interval_ms, angle, self.SAS11_CAN_ID, 
                raw_data.hex().upper(), self.estimate_bus_load()
            ])
    
    def characterize_system(self):
        """Analyze system timing characteristics for validation calibration"""
        if len(self.message_intervals) < 50:
            return None
        
        intervals = list(self.message_intervals)
        
        characteristics = SystemCharacteristics(
            average_message_interval_ms=statistics.mean(intervals),
            message_rate_statistics={
                'mean_hz': 1000.0 / statistics.mean(intervals),
                'median_hz': 1000.0 / statistics.median(intervals),
                'std_dev_ms': statistics.stdev(intervals),
                'min_interval_ms': min(intervals),
                'max_interval_ms': max(intervals)
            },
            timing_jitter_ms=statistics.stdev(intervals),
            max_latency_ms=max(intervals),
            angle_resolution_deg=self.SAS_ANGLE_SCALE,
            coordinate_frame="SAS11_LEFT_POSITIVE"  # Document coordinate frame
        )
        
        self.system_characteristics = characteristics
        return characteristics
    
    def save_comprehensive_trajectory_data(self):
        """Save complete trajectory data with validation-ready format"""
        # Characterize system performance
        system_char = self.characterize_system()
        
        # Prepare comprehensive data structure
        trajectory_data = {
            'format_version': '2.0',
            'coordinate_frame': 'SAS11_LEFT_POSITIVE',  # Critical for validator
            'recording_session': asdict(RecordingSession(
                start_time=self.start_time,
                end_time=time.time(),
                duration_seconds=time.time() - self.start_time,
                interface=self.interface,
                angle_threshold=self.base_angle_threshold,
                adaptive_threshold_enabled=self.adaptive_threshold_enabled,
                initial_angle=self.initial_angle,
                final_angle=self.trajectory_points[-1].steering_angle_deg if self.trajectory_points else 0,
                total_cumulative_change=self.cumulative_change,
                coordinate_frame_verified=True
            )),
            'system_characteristics': asdict(system_char) if system_char else None,
            'quality_statistics': {
                'total_can_messages': self.total_messages,
                'valid_sas11_messages': self.valid_messages,
                'trajectory_events': self.trajectory_events,
                'timing_anomalies': self.timing_anomalies,
                'average_quality_score': statistics.mean(self.quality_scores) if self.quality_scores else None,
                'message_rate_stability': statistics.stdev(self.quality_scores) if len(self.quality_scores) > 1 else None,
                'max_single_change': max([abs(p.angle_change_deg) for p in self.trajectory_points], default=0),
                'total_angle_range': (max([p.steering_angle_deg for p in self.trajectory_points], default=0) - 
                                    min([p.steering_angle_deg for p in self.trajectory_points], default=0)),
                'steering_activity_analysis': self.analyze_steering_patterns()
            },
            'validation_parameters': {
                'recommended_angle_tolerance_deg': self.recommend_angle_tolerance(),
                'recommended_timing_tolerance_ms': self.recommend_timing_tolerance(),
                'suggested_mcm_gain': self.suggest_mcm_conversion_gain(),
                'coordinate_frame_correction': -1.0  # MCM sign inversion factor
            },
            'trajectory_points': [asdict(point) for point in self.trajectory_points]
        }
        
        with open(self.trajectory_json, 'w') as f:
            json.dump(trajectory_data, f, indent=2)
        
        return trajectory_data
    
    def analyze_steering_patterns(self) -> Dict:
        """Analyze steering patterns for validation insights"""
        if not self.trajectory_points:
            return {}
        
        left_turns = sum(1 for p in self.trajectory_points if p.angle_change_deg > 1.0)
        right_turns = sum(1 for p in self.trajectory_points if p.angle_change_deg < -1.0)
        stable_periods = sum(1 for p in self.trajectory_points if abs(p.angle_change_deg) <= 1.0)
        
        angle_velocities = [p.angle_velocity_deg_s for p in self.trajectory_points 
                          if p.angle_velocity_deg_s is not None]
        
        return {
            'left_turn_events': left_turns,
            'right_turn_events': right_turns,
            'stable_periods': stable_periods,
            'max_angular_velocity_deg_s': max(angle_velocities, default=0),
            'avg_angular_velocity_deg_s': statistics.mean(angle_velocities) if angle_velocities else 0,
            'steering_activity_ratio': (left_turns + right_turns) / len(self.trajectory_points),
            'dominant_direction': 'LEFT' if left_turns > right_turns else 'RIGHT' if right_turns > left_turns else 'BALANCED'
        }
    
    def recommend_angle_tolerance(self) -> float:
        """Recommend validation angle tolerance based on recorded data quality"""
        if not self.quality_scores:
            return 5.0  # Default
        
        avg_quality = statistics.mean(self.quality_scores)
        
        if avg_quality > 80:
            return 2.0  # High quality - tight tolerance
        elif avg_quality > 60:
            return 3.0  # Good quality
        else:
            return 5.0  # Lower quality - relaxed tolerance
    
    def recommend_timing_tolerance(self) -> float:
        """Recommend validation timing tolerance based on system characteristics"""
        if not self.message_intervals:
            return 50.0  # Default 50ms
        
        jitter = statistics.stdev(list(self.message_intervals))
        return max(20.0, jitter * 3.0)  # 3-sigma tolerance
    
    def suggest_mcm_conversion_gain(self) -> float:
        """Suggest MCM conversion gain based on recorded angle range"""
        if not self.trajectory_points:
            return 100.0 / 400.0  # Default assumption
        
        max_angle = max([abs(p.steering_angle_deg) for p in self.trajectory_points])
        
        # Assume recorded max angle corresponds to ~80% of system capability
        estimated_max_capability = max_angle / 0.8
        
        return 100.0 / estimated_max_capability
    
    def display_enhanced_status(self):
        """Display comprehensive recording status"""
        print(f"\033[2J\033[H")  # Clear screen
        print("=" * 100)
        print(f"ENHANCED STEERING TRAJECTORY RECORDER - {self.interface}")
        print("=" * 100)
        
        runtime = time.time() - self.start_time
        print(f"Runtime: {runtime:.1f}s | "
              f"Base threshold: {self.base_angle_threshold}° | "
              f"Current: {self.current_threshold:.1f}° | "
              f"Messages: {self.total_messages}")
        print(f"Trajectory Events: {self.trajectory_events} | "
              f"Valid Messages: {self.valid_messages} | "
              f"Timing Anomalies: {self.timing_anomalies}")
        
        # System performance metrics
        if len(self.message_intervals) > 5:
            avg_rate = 1000.0 / statistics.mean(list(self.message_intervals)[-20:])
            print(f"Message Rate: {avg_rate:.1f} Hz | "
                  f"Quality Score: {statistics.mean(self.quality_scores[-10:]) if self.quality_scores else 0:.0f}%")
        
        # Current angle and activity
        if self.trajectory_points:
            latest = self.trajectory_points[-1]
            velocity = latest.angle_velocity_deg_s or 0
            direction = latest.steering_direction or 'UNKNOWN'
            print(f"Current Angle: {latest.steering_angle_deg:+7.1f}° | "
                  f"Velocity: {velocity:+6.1f}°/s | "
                  f"Direction: {direction}")
            print(f"Cumulative Change: {self.cumulative_change:+7.1f}°")
        
        print(f"\nRecent Trajectory Points (Last 5):")
        print("-" * 100)
        for point in self.trajectory_points[-5:]:
            quality = point.signal_quality_score or 0
            velocity = point.angle_velocity_deg_s or 0
            quality_str = f"{quality:.0f}%"
            velocity_str = f"{velocity:+6.1f}"
            print(f"{point.iso_time[-12:]} | {point.steering_angle_deg:+7.1f}° | "
                  f"Δ{point.angle_change_deg:+6.1f}° | "
                  f"V{velocity_str}°/s | "
                  f"Q{quality_str:>4}")
        
        print(f"\nPress Ctrl+C to stop enhanced recording...")
    
    def record_enhanced_trajectory(self):
        """Main enhanced trajectory recording loop"""
        print(f"Starting Enhanced Steering Trajectory Recording")
        print(f"Interface: {self.interface} (SAS11: 0x{self.SAS11_CAN_ID:03X})")
        print(f"Base threshold: {self.base_angle_threshold}° | Adaptive: {self.adaptive_threshold_enabled}")
        print(f"Quality monitoring: {self.quality_monitoring}")
        print("=" * 100)
        
        self.running = True
        last_status_time = 0
        status_interval = 2.0  # Update status every 2 seconds
        
        try:
            while self.running:
                # Receive CAN message
                message = self.bus.recv(timeout=0.1)
                
                if message is None:
                    continue
                
                self.total_messages += 1
                current_time = message.timestamp
                
                # Store message for bus load analysis
                self.can_message_buffer.append({
                    'timestamp': current_time,
                    'arbitration_id': message.arbitration_id,
                    'data_length': len(message.data)
                })
                
                # Process SAS11 messages
                if message.arbitration_id == self.SAS11_CAN_ID:
                    self.message_count += 1
                    
                    # Extract steering angle
                    steering_angle = self.parse_sas11_angle(message.data)
                    
                    if steering_angle is not None:
                        self.valid_messages += 1
                        
                        # Update recent data for analysis
                        with self.data_lock:
                            self.recent_angles.append(steering_angle)
                            self.recent_timestamps.append(current_time)
                        
                        # Calculate quality metrics
                        quality_metrics = self.calculate_quality_metrics(steering_angle, current_time)
                        
                        # Check if we should record this point
                        if self.should_record_angle(steering_angle):
                            self.record_trajectory_point(
                                steering_angle, current_time, message.data, quality_metrics
                            )
                        
                        self.last_message_time = current_time
                
                # Update status display periodically
                if time.time() - last_status_time >= status_interval:
                    self.display_enhanced_status()
                    last_status_time = time.time()
                    
        except KeyboardInterrupt:
            print(f"\nEnhanced trajectory recording stopped")
            self.running = False
            self.print_comprehensive_summary()
            trajectory_data = self.save_comprehensive_trajectory_data()
            self.print_validation_recommendations(trajectory_data)
        finally:
            if self.bus:
                self.bus.shutdown()
    
    def print_comprehensive_summary(self):
        """Print detailed recording session summary"""
        print(f"\nENHANCED TRAJECTORY RECORDING SUMMARY:")
        print("=" * 100)
        
        duration = time.time() - self.start_time
        print(f"Recording Duration: {duration:.1f} seconds")
        print(f"Total CAN Messages: {self.total_messages}")
        print(f"SAS11 Valid Messages: {self.valid_messages} ({self.valid_messages/max(1,self.total_messages)*100:.1f}%)")
        print(f"Trajectory Events Recorded: {self.trajectory_events}")
        print(f"Timing Anomalies Detected: {self.timing_anomalies}")
        
        if self.trajectory_points:
            angles = [p.steering_angle_deg for p in self.trajectory_points]
            velocities = [p.angle_velocity_deg_s for p in self.trajectory_points 
                         if p.angle_velocity_deg_s is not None]
            
            print(f"\nTrajectory Analysis:")
            print(f"  Angle Range: {min(angles):+.1f}° to {max(angles):+.1f}°")
            print(f"  Total Cumulative Change: {self.cumulative_change:+.1f}°")
            print(f"  Initial Angle: {self.initial_angle:+.1f}°")
            print(f"  Final Angle: {self.trajectory_points[-1].steering_angle_deg:+.1f}°")
            
            if velocities:
                print(f"  Max Angular Velocity: {max(velocities):+.1f}°/s")
                print(f"  Avg Angular Velocity: {statistics.mean([abs(v) for v in velocities]):.1f}°/s")
            
            # Steering pattern analysis
            pattern_analysis = self.analyze_steering_patterns()
            print(f"  Left Turn Events: {pattern_analysis.get('left_turn_events', 0)}")
            print(f"  Right Turn Events: {pattern_analysis.get('right_turn_events', 0)}")
            print(f"  Dominant Direction: {pattern_analysis.get('dominant_direction', 'UNKNOWN')}")
        
        # Quality assessment
        if self.quality_scores:
            avg_quality = statistics.mean(self.quality_scores)
            print(f"\nData Quality Assessment:")
            print(f"  Average Quality Score: {avg_quality:.1f}%")
            print(f"  Quality Rating: {'EXCELLENT' if avg_quality > 80 else 'GOOD' if avg_quality > 60 else 'ACCEPTABLE' if avg_quality > 40 else 'POOR'}")
        
        # System characteristics
        if len(self.message_intervals) > 10:
            avg_interval = statistics.mean(list(self.message_intervals))
            jitter = statistics.stdev(list(self.message_intervals))
            print(f"\nSystem Timing Characteristics:")
            print(f"  Average Message Rate: {1000.0/avg_interval:.1f} Hz")
            print(f"  Timing Jitter: {jitter:.2f} ms")
            print(f"  Coordinate Frame: SAS11_LEFT_POSITIVE (Left steering = positive angles)")
        
        print(f"\nOutput Files:")
        print(f"  Enhanced Trajectory: {self.trajectory_csv}")
        print(f"  Comprehensive Data: {self.trajectory_json}")
        print(f"  Timing Analysis: {self.timing_csv}")
    
    def print_validation_recommendations(self, trajectory_data: Dict):
        """Print specific recommendations for validation system"""
        print(f"\nVALIDATION SYSTEM RECOMMENDATIONS:")
        print("=" * 100)
        
        val_params = trajectory_data.get('validation_parameters', {})
        
        print(f"Recommended Settings:")
        print(f"  --tolerance {val_params.get('recommended_angle_tolerance_deg', 5.0):.1f}")
        print(f"  Timing tolerance: {val_params.get('recommended_timing_tolerance_ms', 50.0):.1f} ms")
        print(f"  MCM conversion gain: {val_params.get('suggested_mcm_gain', 0.25):.4f}")
        print(f"  Coordinate frame correction: {val_params.get('coordinate_frame_correction', -1.0):+.1f}")
        
        system_char = trajectory_data.get('system_characteristics')
        if system_char:
            print(f"\nSystem Characteristics for Calibration:")
            print(f"  Message rate: {system_char['message_rate_statistics']['mean_hz']:.1f} Hz")
            print(f"  Timing jitter: {system_char['timing_jitter_ms']:.2f} ms")
            print(f"  Angle resolution: {system_char['angle_resolution_deg']:.1f}°")
        
        print(f"\nCommand Line Usage:")
        print(f"python3 mcm_steer_track2.py --trajectory {self.trajectory_json} \\")
        print(f"    --tolerance {val_params.get('recommended_angle_tolerance_deg', 5.0):.1f} \\")
        print(f"    --mcm can2 --sas can3")

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Steering Trajectory Recorder with Validation Support',
        epilog='''
Advanced trajectory recording with comprehensive analysis for validation system calibration.
Captures high-fidelity steering data with timing characteristics, quality metrics, and
validation-ready output format.

Example usage:
  python3 enhanced_recorder.py --threshold 5 --adaptive    # Adaptive 5° base threshold
  python3 enhanced_recorder.py -i can3 --no-adaptive      # Fixed threshold, can3 interface
  python3 enhanced_recorder.py --threshold 2 --quality     # High detail with quality monitoring
        '''
    )
    parser.add_argument('--interface', '-i', default='can3', 
                       help='CAN interface (default: can3)')
    parser.add_argument('--threshold', '-t', type=float, default=10.0,
                       help='Base angle change threshold in degrees (default: 10.0)')
    parser.add_argument('--adaptive', action='store_true', default=True,
                       help='Enable adaptive threshold adjustment (default: enabled)')
    parser.add_argument('--no-adaptive', action='store_false', dest='adaptive',
                       help='Disable adaptive threshold adjustment')
    parser.add_argument('--quality', action='store_true', default=True,
                       help='Enable quality monitoring (default: enabled)')
    parser.add_argument('--no-quality', action='store_false', dest='quality',
                       help='Disable quality monitoring')
    
    args = parser.parse_args()
    
    if args.threshold <= 0:
        print("Error: Angle threshold must be positive")
        sys.exit(1)
    
    print("Enhanced Steering Trajectory Recorder")
    print("High-Fidelity SAS11 Recording with Validation Support")
    print()
    
    recorder = EnhancedSteeringTrajectoryRecorder(
        interface=args.interface,
        angle_threshold=args.threshold,
        adaptive_threshold=args.adaptive,
        quality_monitoring=args.quality
    )
    recorder.record_enhanced_trajectory()

if __name__ == "__main__":
    main()
