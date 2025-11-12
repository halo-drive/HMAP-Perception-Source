#!/usr/bin/env python3
"""
Hybrid Time-Distance Playback Controller
Three indexing modes: distance-only, velocity-compensated, time-distance fusion

Features:
- Multiple playback strategies
- Real-time OpenCV visualization
- Comprehensive debug metadata
- GPS/IMU integration ready
- Progressive brake control

Usage:
    python3 hybrid_playback.py --trajectory recorded.json --mode time_distance_fusion --alpha 0.6
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
import numpy as np
import cv2
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Tuple
from enum import Enum
import threading
from collections import deque

# ============================================================================
# Enums and Data Structures
# ============================================================================

class PlaybackMode(Enum):
    """Playback indexing strategies"""
    DISTANCE_ONLY = "distance_only"
    VELOCITY_COMPENSATED = "velocity_compensated"
    TIME_DISTANCE_FUSION = "time_distance_fusion"

@dataclass
class HybridTrajectoryWaypoint:
    """Waypoint with dual time-distance indexing"""
    cumulative_distance_m: float
    elapsed_time_s: float
    timestamp: float
    steering_angle_deg: float
    vehicle_speed_mps: float
    wheel_speeds_kmh: Dict[str, float]
    yaw_rate_deg_s: Optional[float] = None
    lateral_accel_mps2: Optional[float] = None
    longitudinal_accel_mps2: Optional[float] = None
    distance_progress: Optional[float] = None
    time_progress: Optional[float] = None
    raw_sas_data: Optional[str] = None
    iso_time: Optional[str] = None

@dataclass
class ValidationPoint:
    """Real-time validation data"""
    timestamp: float
    cumulative_distance_m: float
    elapsed_time_s: float
    target_angle_deg: float
    commanded_steering_pct: float
    commanded_brake_pct: float
    measured_angle_deg: Optional[float]
    measured_speed_mps: Optional[float]
    angle_error_deg: Optional[float]
    distance_error_m: Optional[float]
    time_error_s: Optional[float]
    progress_distance: float
    progress_time: float
    progress_fused: float

@dataclass
class DebugMetadata:
    """Comprehensive debug metadata"""
    session_id: str
    start_timestamp: float
    end_timestamp: Optional[float]
    trajectory_file: str
    playback_mode: str
    fusion_alpha: Optional[float]
    
    target_distance_m: float
    target_duration_s: float
    actual_distance_m: Optional[float]
    actual_duration_s: Optional[float]
    
    lap_number: int
    total_steering_commands: int = 0
    total_brake_commands: int = 0
    
    mean_angle_error_deg: Optional[float] = None
    max_angle_error_deg: Optional[float] = None
    mean_distance_error_m: Optional[float] = None
    max_distance_error_m: Optional[float] = None
    mean_time_error_s: Optional[float] = None
    max_time_error_s: Optional[float] = None
    
    loop_closure_error_m: Optional[float] = None
    final_position_xy: Optional[Tuple[float, float]] = None

# ============================================================================
# Trajectory Visualizer
# ============================================================================

class TrajectoryVisualizer:
    """Real-time OpenCV visualization"""
    
    def __init__(self, waypoints: List[HybridTrajectoryWaypoint], 
                 window_size=(1200, 800), grid_spacing_m=5.0):
        self.waypoints = waypoints
        self.window_size = window_size
        self.grid_spacing_m = grid_spacing_m
        
        # Calculate XY coordinates
        self.trajectory_xy = self.calculate_trajectory_coordinates()
        
        # Bounds
        self.bounds = self.calculate_bounds()
        
        # Current state
        self.current_position = (0.0, 0.0)
        self.current_heading = 0.0
        self.current_distance = 0.0
        self.current_time = 0.0
        self.brake_active = False
        self.trajectory_complete = False
        
        # Telemetry
        self.telemetry = {
            'speed_mps': 0.0,
            'steering_angle_deg': 0.0,
            'brake_pct': 0.0,
            'distance_remaining_m': 0.0,
            'time_remaining_s': 0.0,
            'angle_error_deg': 0.0,
            'waypoint_index': 0,
            'progress_distance': 0.0,
            'progress_time': 0.0,
            'progress_fused': 0.0
        }
        
        # History
        self.position_history = deque(maxlen=500)
        
        # Thread safety
        self.vis_lock = threading.Lock()
        
        # Initialize window
        cv2.namedWindow('Hybrid Trajectory Playback', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Hybrid Trajectory Playback', *window_size)
        
        print(f"✓ Visualization initialized")
    
    def calculate_trajectory_coordinates(self) -> List[Tuple[float, float]]:
        """Calculate XY from waypoints using dead reckoning"""
        trajectory_xy = [(0.0, 0.0)]
        
        x, y = 0.0, 0.0
        heading_deg = 0.0
        
        for i in range(1, len(self.waypoints)):
            prev = self.waypoints[i-1]
            curr = self.waypoints[i]
            
            distance_delta = curr.cumulative_distance_m - prev.cumulative_distance_m
            
            # Update heading from yaw rate
            if curr.yaw_rate_deg_s is not None and prev.yaw_rate_deg_s is not None:
                dt = curr.elapsed_time_s - prev.elapsed_time_s
                avg_yaw_rate = (curr.yaw_rate_deg_s + prev.yaw_rate_deg_s) / 2.0
                heading_deg += avg_yaw_rate * dt
            
            # Update position
            heading_rad = math.radians(heading_deg)
            x += distance_delta * math.sin(heading_rad)
            y += distance_delta * math.cos(heading_rad)
            
            trajectory_xy.append((x, y))
        
        return trajectory_xy
    
    def calculate_bounds(self) -> Dict[str, float]:
        """Calculate visualization bounds"""
        if not self.trajectory_xy:
            return {'x_min': -10, 'x_max': 10, 'y_min': -10, 'y_max': 10}
        
        x_coords = [pos[0] for pos in self.trajectory_xy]
        y_coords = [pos[1] for pos in self.trajectory_xy]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        x_range = max(x_max - x_min, 10.0)
        y_range = max(y_max - y_min, 10.0)
        
        padding_x = x_range * 0.2
        padding_y = y_range * 0.2
        
        return {
            'x_min': x_min - padding_x,
            'x_max': x_max + padding_x,
            'y_min': y_min - padding_y,
            'y_max': y_max + padding_y
        }
    
    def world_to_pixel(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world coordinates to pixel"""
        x_range = self.bounds['x_max'] - self.bounds['x_min']
        y_range = self.bounds['y_max'] - self.bounds['y_min']
        
        scale = min(
            (self.window_size[0] - 100) / x_range,
            (self.window_size[1] - 200) / y_range
        )
        
        center_x = self.window_size[0] // 2
        center_y = self.window_size[1] // 2 + 50
        
        pixel_x = int(center_x + (x - (self.bounds['x_min'] + self.bounds['x_max'])/2) * scale)
        pixel_y = int(center_y - (y - (self.bounds['y_min'] + self.bounds['y_max'])/2) * scale)
        
        return (pixel_x, pixel_y)
    
    def draw_grid(self, frame: np.ndarray):
        """Draw metric grid"""
        # Vertical lines
        x = self.bounds['x_min']
        while x <= self.bounds['x_max']:
            px_start = self.world_to_pixel(x, self.bounds['y_min'])
            px_end = self.world_to_pixel(x, self.bounds['y_max'])
            cv2.line(frame, px_start, px_end, (40, 40, 40), 1, cv2.LINE_AA)
            x += self.grid_spacing_m
        
        # Horizontal lines
        y = self.bounds['y_min']
        while y <= self.bounds['y_max']:
            px_start = self.world_to_pixel(self.bounds['x_min'], y)
            px_end = self.world_to_pixel(self.bounds['x_max'], y)
            cv2.line(frame, px_start, px_end, (40, 40, 40), 1, cv2.LINE_AA)
            y += self.grid_spacing_m
    
    def draw_trajectory(self, frame: np.ndarray):
        """Draw planned trajectory"""
        if len(self.trajectory_xy) < 2:
            return
        
        points = [self.world_to_pixel(x, y) for x, y in self.trajectory_xy]
        
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i+1], (80, 80, 200), 2, cv2.LINE_AA)
    
    def draw_vehicle(self, frame: np.ndarray):
        """Draw vehicle"""
        px = self.world_to_pixel(self.current_position[0], self.current_position[1])
        
        if self.brake_active:
            color = (0, 100, 255)
        elif self.trajectory_complete:
            color = (0, 255, 0)
        else:
            color = (255, 100, 0)
        
        cv2.circle(frame, px, 8, color, -1)
        cv2.circle(frame, px, 10, (255, 255, 255), 2)
        
        # Heading arrow
        heading_rad = math.radians(self.current_heading)
        arrow_length = 25
        arrow_end_x = px[0] + int(arrow_length * math.sin(heading_rad))
        arrow_end_y = px[1] - int(arrow_length * math.cos(heading_rad))
        cv2.arrowedLine(frame, px, (arrow_end_x, arrow_end_y), 
                       (255, 255, 255), 2, tipLength=0.3)
    
    def draw_telemetry(self, frame: np.ndarray):
        """Draw telemetry panel"""
        panel_height = 150
        panel = np.zeros((panel_height, self.window_size[0], 3), dtype=np.uint8)
        
        # Title
        cv2.putText(panel, "HYBRID TRAJECTORY PLAYBACK", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Progress bars
        bar_y = 40
        bar_width = 300
        bar_height = 15
        bar_x = self.window_size[0] - bar_width - 20
        
        # Distance progress
        cv2.putText(panel, "Distance:", (bar_x - 80, bar_y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(panel, (bar_x, bar_y), 
                     (bar_x + int(bar_width * self.telemetry['progress_distance']), bar_y + bar_height),
                     (100, 255, 100), -1)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (150, 150, 150), 1)
        
        # Time progress
        bar_y += 25
        cv2.putText(panel, "Time:", (bar_x - 80, bar_y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(panel, (bar_x, bar_y), 
                     (bar_x + int(bar_width * self.telemetry['progress_time']), bar_y + bar_height),
                     (255, 255, 100), -1)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (150, 150, 150), 1)
        
        # Fused progress
        bar_y += 25
        cv2.putText(panel, "Fused:", (bar_x - 80, bar_y + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        cv2.rectangle(panel, (bar_x, bar_y), 
                     (bar_x + int(bar_width * self.telemetry['progress_fused']), bar_y + bar_height),
                     (100, 200, 255), -1)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (150, 150, 150), 1)
        
        # Metrics
        col1_x = 10
        cv2.putText(panel, f"Distance: {self.current_distance:.1f}m", (col1_x, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, f"Time: {self.current_time:.1f}s", (col1_x, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, f"Speed: {self.telemetry['speed_mps']:.2f}m/s", (col1_x, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        col2_x = 250
        cv2.putText(panel, f"Steering: {self.telemetry['steering_angle_deg']:+.1f}°", (col2_x, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, f"Brake: {self.telemetry['brake_pct']:.1f}%", (col2_x, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, f"Waypoint: {self.telemetry['waypoint_index']}", (col2_x, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Status
        if self.trajectory_complete:
            status_text = "COMPLETE"
            status_color = (0, 255, 0)
        elif self.brake_active:
            status_text = "BRAKING"
            status_color = (0, 150, 255)
        else:
            status_text = "EXECUTING"
            status_color = (255, 200, 0)
        
        cv2.putText(panel, f"Status: {status_text}", (col2_x, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        frame[0:panel_height, :] = panel
    
    def update(self, distance: float, elapsed_time: float, position: Tuple[float, float], 
               heading: float, telemetry: Dict, brake_active: bool, complete: bool):
        """Update visualization state"""
        with self.vis_lock:
            self.current_distance = distance
            self.current_time = elapsed_time
            self.current_position = position
            self.current_heading = heading
            self.telemetry = telemetry.copy()
            self.brake_active = brake_active
            self.trajectory_complete = complete
            self.position_history.append(position)
    
    def render(self) -> np.ndarray:
        """Render current frame"""
        with self.vis_lock:
            frame = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
            frame[:] = (20, 20, 20)
            
            self.draw_grid(frame)
            self.draw_trajectory(frame)
            self.draw_vehicle(frame)
            self.draw_telemetry(frame)
            
            return frame
    
    def show(self):
        """Display frame"""
        frame = self.render()
        cv2.imshow('Hybrid Trajectory Playback', frame)
        cv2.waitKey(1)
    
    def close(self):
        """Clean up"""
        cv2.destroyAllWindows()

# ============================================================================
# Hybrid Playback Controller
# ============================================================================

class HybridPlaybackController:
    """
    Playback controller with multiple indexing strategies
    """
    
    def __init__(self, mcm_channel='can2', sas_channel='can3',
                 trajectory_file=None, 
                 playback_mode: PlaybackMode = PlaybackMode.TIME_DISTANCE_FUSION,
                 fusion_alpha: float = 0.6,
                 brake_start_distance: float = 8.0,
                 enable_visualization: bool = True):
        
        # CAN Configuration
        self.mcm_channel = mcm_channel
        self.sas_channel = sas_channel
        
        # MCM Controller Setup
        self.mcm_db = cantools.database.Database()
        self.mcm_db.add_dbc_file('./sygnal_dbc/mcm/Heartbeat.dbc')
        self.mcm_db.add_dbc_file('./sygnal_dbc/mcm/Control.dbc')
        
        # CAN Bus Connections
        try:
            self.mcm_bus = can.Bus(channel=mcm_channel, bustype='socketcan', bitrate=500000)
            print(f"✓ MCM controller connected: {mcm_channel}")
        except Exception as e:
            print(f"✗ MCM bus connection failed: {e}")
            sys.exit(1)
        
        try:
            self.sas_bus = can.Bus(channel=sas_channel, bustype='socketcan', bitrate=500000)
            print(f"✓ Sensor monitor connected: {sas_channel}")
        except Exception as e:
            print(f"✗ Sensor bus connection failed: {e}")
            sys.exit(1)
        
        # MCM Control State
        self.control_count = 0
        self.bus_address = 1
        self.steering_enabled = False
        self.brake_enabled = False
        
        # CAN IDs
        self.SAS11_CAN_ID = 0x2B0
        self.WHL_SPD_CAN_ID = 0x386
        self.ESP12_CAN_ID = 0x220
        
        # Scaling
        self.SAS_ANGLE_SCALE = 0.1
        self.WHEEL_SPEED_SCALE = 0.03125
        self.YAW_RATE_SCALE = 0.01
        self.YAW_RATE_OFFSET = -40.95
        
        # Odometry State
        self.cumulative_distance_m = 0.0
        self.vehicle_speed_mps = 0.0
        self.last_update_time = None
        
        # Temporal State
        self.playback_start_time = None
        self.elapsed_playback_time_s = 0.0
        
        # Position Estimation
        self.estimated_x_m = 0.0
        self.estimated_y_m = 0.0
        self.estimated_heading_deg = 0.0
        self.yaw_rate_deg_s = None
        
        # Steering Feedback
        self.latest_angle_deg = None
        
        # Playback Mode
        self.playback_mode = playback_mode
        self.fusion_alpha = fusion_alpha
        
        # Trajectory Data
        self.waypoints: List[HybridTrajectoryWaypoint] = []
        self.trajectory_metadata = {}
        self.trajectory_file_path = trajectory_file
        
        # Velocity Tracking
        self.recorded_velocity_profile = []
        self.current_velocity_profile = []
        
        # Brake Parameters
        self.BRAKE_START_DISTANCE = brake_start_distance
        self.MAX_BRAKE_INTENSITY = 0.7
        
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
            'time_errors': [],
            'angle_errors': []
        }
        
        # Debug Metadata
        self.current_lap_metadata: Optional[DebugMetadata] = None
        self.all_laps_metadata: List[DebugMetadata] = []
        
        # Load Trajectory
        if trajectory_file:
            self.load_trajectory(trajectory_file)
        
        # Visualization
        self.enable_visualization = enable_visualization
        self.visualizer: Optional[TrajectoryVisualizer] = None
        
        if self.enable_visualization and len(self.waypoints) > 0:
            try:
                self.visualizer = TrajectoryVisualizer(self.waypoints)
            except Exception as e:
                print(f"⚠ Visualization failed: {e}")
                self.enable_visualization = False
        
        # Setup Logging
        self.setup_logging()
        
        print(f"\n✓ Playback controller initialized")
        print(f"  Mode: {playback_mode.value}")
        if playback_mode == PlaybackMode.TIME_DISTANCE_FUSION:
            print(f"  Fusion α: {fusion_alpha:.2f} (time weight)")
    
    def load_trajectory(self, filename: str):
        """Load hybrid trajectory"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            # Load waypoints
            for wp_data in data['waypoints']:
                wp = HybridTrajectoryWaypoint(**wp_data)
                self.waypoints.append(wp)
            
            self.trajectory_metadata = data['metadata']
            
            # Extract velocity profile
            self.recorded_velocity_profile = [wp.vehicle_speed_mps for wp in self.waypoints]
            
            print(f"\n{'='*80}")
            print(f"TRAJECTORY LOADED")
            print(f"{'='*80}")
            print(f"File: {filename}")
            print(f"Format: {data.get('format_version', 'unknown')}")
            print(f"Waypoints: {len(self.waypoints)}")
            print(f"Distance: {self.trajectory_metadata['total_distance_m']:.2f}m")
            print(f"Duration: {self.trajectory_metadata['duration_seconds']:.1f}s")
            print(f"Avg speed: {self.trajectory_metadata['average_speed_mps']:.2f}m/s")
            print(f"Speed variation: {self.trajectory_metadata.get('speed_variation_coefficient', 0)*100:.1f}%")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"✗ Failed to load trajectory: {e}")
            sys.exit(1)
    
    def setup_logging(self):
        """Initialize logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.validation_csv = f"hybrid_validation_{timestamp}.csv"
        self.debug_metadata_json = f"hybrid_debug_{timestamp}.json"
        
        with open(self.validation_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'cumulative_distance_m', 'elapsed_time_s',
                'target_angle_deg', 'commanded_steering_pct', 'commanded_brake_pct',
                'measured_angle_deg', 'measured_speed_mps',
                'angle_error_deg', 'distance_error_m', 'time_error_s',
                'progress_distance', 'progress_time', 'progress_fused'
            ])
        
        print(f"Validation CSV: {self.validation_csv}")
        print(f"Debug metadata: {self.debug_metadata_json}")
    
    # ... (Include all parsing methods from previous code) ...
    
    def calc_crc8(self, data: bytes) -> int:
        """Calculate CRC8"""
        hash = crc8.crc8()
        hash.update(data[:-1])
        return hash.digest()[0]
    
    def angle_to_percentage(self, angle_deg: float) -> float:
        """Convert steering angle to MCM percentage"""
        adjusted_angle = -angle_deg  # Sign inversion for coordinate frame
        percentage = adjusted_angle * (100.0 / 400.0)
        return max(min(percentage, 100.0), -100.0)
    
    def calculate_brake_command(self, remaining_distance: float) -> float:
        """Progressive brake calculation"""
        if remaining_distance >= self.BRAKE_START_DISTANCE:
            return 0.0
        if remaining_distance <= 0.0:
            return self.MAX_BRAKE_INTENSITY
        
        brake_progress = 1.0 - (remaining_distance / self.BRAKE_START_DISTANCE)
        brake_intensity = self.MAX_BRAKE_INTENSITY * (brake_progress ** 1.5)
        
        return min(max(brake_intensity, 0.0), self.MAX_BRAKE_INTENSITY)
    
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
    
    def parse_sas11_angle(self, data: bytes) -> Optional[float]:
        """Parse steering angle"""
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
        """Update distance"""
        rear_avg_kmh = (wheel_speeds['RL'] + wheel_speeds['RR']) / 2.0
        vehicle_speed_mps = rear_avg_kmh / 3.6
        
        if self.last_update_time is not None:
            dt = timestamp - self.last_update_time
            if 0.001 <= dt <= 1.0:
                distance_increment = vehicle_speed_mps * dt
                with self.data_lock:
                    self.cumulative_distance_m += distance_increment
                    
                    # Update position estimate
                    if self.yaw_rate_deg_s is not None:
                        self.estimated_heading_deg += self.yaw_rate_deg_s * dt
                    heading_rad = math.radians(self.estimated_heading_deg)
                    self.estimated_x_m += distance_increment * math.sin(heading_rad)
                    self.estimated_y_m += distance_increment * math.cos(heading_rad)
        
        with self.data_lock:
            self.vehicle_speed_mps = vehicle_speed_mps
        self.last_update_time = timestamp
    
    def get_current_odometry(self) -> Tuple[float, float]:
        """Get distance and speed"""
        with self.data_lock:
            return self.cumulative_distance_m, self.vehicle_speed_mps
    
    def get_current_position(self) -> Tuple[float, float, float]:
        """Get position estimate"""
        with self.data_lock:
            return self.estimated_x_m, self.estimated_y_m, self.estimated_heading_deg
    
    # ========================================================================
    # Progress Computation Methods
    # ========================================================================
    
    def compute_progress_distance_only(self) -> float:
        """Pure distance-based progress"""
        if self.trajectory_metadata['total_distance_m'] == 0:
            return 0.0
        return self.cumulative_distance_m / self.trajectory_metadata['total_distance_m']
    
    def compute_progress_time_only(self) -> float:
        """Pure time-based progress"""
        if self.trajectory_metadata['duration_seconds'] == 0:
            return 0.0
        return self.elapsed_playback_time_s / self.trajectory_metadata['duration_seconds']
    
    def compute_progress_velocity_compensated(self) -> float:
        """Velocity-compensated distance progress"""
        recorded_avg_speed = self.trajectory_metadata['average_speed_mps']
        if recorded_avg_speed == 0:
            return self.compute_progress_distance_only()
        
        if len(self.current_velocity_profile) == 0:
            current_avg_speed = self.vehicle_speed_mps
        else:
            current_avg_speed = sum(self.current_velocity_profile) / len(self.current_velocity_profile)
        
        velocity_ratio = current_avg_speed / recorded_avg_speed
        effective_distance = self.cumulative_distance_m * velocity_ratio
        
        return effective_distance / self.trajectory_metadata['total_distance_m']
    
    def compute_progress_fusion(self) -> float:
        """Hybrid time-distance fusion"""
        time_progress = self.compute_progress_time_only()
        distance_progress = self.compute_progress_distance_only()
        
        fused_progress = (self.fusion_alpha * time_progress + 
                         (1 - self.fusion_alpha) * distance_progress)
        
        return fused_progress
    
    def get_current_progress(self) -> float:
        """Get progress according to mode"""
        if self.playback_mode == PlaybackMode.DISTANCE_ONLY:
            return self.compute_progress_distance_only()
        elif self.playback_mode == PlaybackMode.VELOCITY_COMPENSATED:
            return self.compute_progress_velocity_compensated()
        elif self.playback_mode == PlaybackMode.TIME_DISTANCE_FUSION:
            return self.compute_progress_fusion()
        return self.compute_progress_distance_only()
    
    def find_next_waypoint(self) -> Tuple[Optional[HybridTrajectoryWaypoint], Optional[int]]:
        """Find next waypoint based on progress"""
        current_progress = self.get_current_progress()
        
        for i in range(self.trajectory_index, len(self.waypoints)):
            wp = self.waypoints[i]
            
            if self.playback_mode == PlaybackMode.TIME_DISTANCE_FUSION:
                waypoint_progress = (self.fusion_alpha * wp.time_progress + 
                                    (1 - self.fusion_alpha) * wp.distance_progress)
            else:
                waypoint_progress = wp.distance_progress
            
            if current_progress >= waypoint_progress:
                return wp, i
        
        return None, None
    
    # ========================================================================
    # MCM Control Methods
    # ========================================================================
    
    async def enable_mcm_interface(self, interface_name: str):
        """Enable MCM interface"""
        interface_id = {'brake': 0, 'accel': 1, 'steer': 2}.get(interface_name)
        if interface_id is None:
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
            self.mcm_bus.send(can.Message(arbitration_id=msg.frame_id, is_extended_id=False, data=data))
            if interface_name == 'steer':
                self.steering_enabled = True
            elif interface_name == 'brake':
                self.brake_enabled = True
            await asyncio.sleep(0.02)
            return True
        except:
            return False
    
    async def send_steering_command(self, percentage: float) -> bool:
        """Send steering command"""
        percentage = max(min(percentage, 100.0), -100.0)
        
        msg = self.mcm_db.get_message_by_name('ControlCommand')
        data = bytearray(msg.encode({
            'BusAddress': self.bus_address,
            'InterfaceID': 2,
            'Count8': self.control_count,
            'Value': percentage / 100.0,
            'CRC': 0
        }))
        data[7] = self.calc_crc8(data)
        
        try:
            if not self.steering_enabled:
                await self.enable_mcm_interface('steer')
            self.mcm_bus.send(can.Message(arbitration_id=msg.frame_id, is_extended_id=False, data=data))
            self.control_count = (self.control_count + 1) % 256
            self.stats['commands_sent'] += 1
            return True
        except:
            return False
    
    async def send_brake_command(self, percentage: float) -> bool:
        """Send brake command"""
        percentage = max(min(percentage, 100.0), 0.0)
        
        msg = self.mcm_db.get_message_by_name('ControlCommand')
        data = bytearray(msg.encode({
            'BusAddress': self.bus_address,
            'InterfaceID': 0,
            'Count8': self.control_count,
            'Value': percentage / 100.0,
            'CRC': 0
        }))
        data[7] = self.calc_crc8(data)
        
        try:
            if not self.brake_enabled:
                await self.enable_mcm_interface('brake')
            self.mcm_bus.send(can.Message(arbitration_id=msg.frame_id, is_extended_id=False, data=data))
            self.control_count = (self.control_count + 1) % 256
            self.stats['brake_commands_sent'] += 1
            return True
        except:
            return False
    
    # ========================================================================
    # Sensor Monitoring
    # ========================================================================
    
    def monitor_sensors(self):
        """Background sensor monitoring"""
        print("✓ Sensor monitoring thread started")
        
        while self.running:
            try:
                message = self.sas_bus.recv(timeout=0.1)
                if message is None:
                    continue
                
                if message.arbitration_id == self.WHL_SPD_CAN_ID:
                    wheel_speeds = self.parse_wheel_speeds(message.data)
                    if wheel_speeds:
                        self.update_odometry(wheel_speeds, message.timestamp)
                
                elif message.arbitration_id == self.SAS11_CAN_ID:
                    angle = self.parse_sas11_angle(message.data)
                    if angle is not None:
                        with self.data_lock:
                            self.latest_angle_deg = angle
                
                elif message.arbitration_id == self.ESP12_CAN_ID:
                    yaw_rate = self.parse_esp12(message.data)
                    if yaw_rate is not None:
                        with self.data_lock:
                            self.yaw_rate_deg_s = yaw_rate
            except:
                if self.running:
                    time.sleep(0.1)
    
    # ========================================================================
    # Main Playback Execution
    # ========================================================================
    
    async def execute_trajectory_playback(self):
        """Main playback execution"""
        print(f"\n{'='*80}")
        print(f"HYBRID PLAYBACK EXECUTION")
        print(f"{'='*80}")
        print(f"Mode: {self.playback_mode.value}")
        print(f"Distance: {self.trajectory_metadata['total_distance_m']:.2f}m")
        print(f"Duration: {self.trajectory_metadata['duration_seconds']:.1f}s")
        print(f"{'='*80}\n")
        
        self.playback_start_time = time.time()
        self.trajectory_index = 0
        self.cumulative_distance_m = 0.0
        self.elapsed_playback_time_s = 0.0
        self.current_velocity_profile.clear()
        
        last_status_time = time.time()
        
        while self.running and not self.trajectory_complete:
            # Update time
            self.elapsed_playback_time_s = time.time() - self.playback_start_time
            
            # Get odometry
            current_distance, current_speed = self.get_current_odometry()
            self.current_velocity_profile.append(current_speed)
            
            # Compute progress
            progress_dist = self.compute_progress_distance_only()
            progress_time = self.compute_progress_time_only()
            progress_fused = self.compute_progress_fusion()
            current_progress = self.get_current_progress()
            
            # Find next waypoint
            waypoint, new_index = self.find_next_waypoint()
            
            if waypoint:
                # Send steering
                command_pct = self.angle_to_percentage(waypoint.steering_angle_deg)
                await self.send_steering_command(command_pct)
                
                # Validation
                measured_angle = self.latest_angle_deg
                angle_error = (measured_angle - waypoint.steering_angle_deg) if measured_angle else None
                distance_error = current_distance - waypoint.cumulative_distance_m
                time_error = self.elapsed_playback_time_s - waypoint.elapsed_time_s
                
                if angle_error:
                    self.stats['angle_errors'].append(abs(angle_error))
                self.stats['distance_errors'].append(abs(distance_error))
                self.stats['time_errors'].append(abs(time_error))
                
                # Log
                validation = ValidationPoint(
                    timestamp=time.time(),
                    cumulative_distance_m=current_distance,
                    elapsed_time_s=self.elapsed_playback_time_s,
                    target_angle_deg=waypoint.steering_angle_deg,
                    commanded_steering_pct=command_pct,
                    commanded_brake_pct=0.0,
                    measured_angle_deg=measured_angle,
                    measured_speed_mps=current_speed,
                    angle_error_deg=angle_error,
                    distance_error_m=distance_error,
                    time_error_s=time_error,
                    progress_distance=progress_dist,
                    progress_time=progress_time,
                    progress_fused=progress_fused
                )
                self.log_validation(validation)
                
                self.trajectory_index = new_index + 1
            
            # Braking
            remaining_distance = self.trajectory_metadata['total_distance_m'] - current_distance
            brake_command = self.calculate_brake_command(remaining_distance)
            
            if brake_command > 0.0:
                if not self.braking_active:
                    print(f"\n⚠ Braking initiated")
                    self.braking_active = True
                await self.send_brake_command(brake_command * 100.0)
            
            # Check completion
            if current_progress >= 0.99:
                print(f"\n✓ Trajectory complete")
                await self.send_brake_command(self.MAX_BRAKE_INTENSITY * 100.0)
                self.trajectory_complete = True
                break
            
            # Visualization
            if self.visualizer and time.time() - last_status_time >= 0.05:
                x, y, heading = self.get_current_position()
                telemetry = {
                    'speed_mps': current_speed,
                    'steering_angle_deg': self.latest_angle_deg or 0.0,
                    'brake_pct': brake_command * 100.0,
                    'distance_remaining_m': remaining_distance,
                    'time_remaining_s': self.trajectory_metadata['duration_seconds'] - self.elapsed_playback_time_s,
                    'angle_error_deg': angle_error or 0.0,
                    'waypoint_index': self.trajectory_index,
                    'progress_distance': progress_dist,
                    'progress_time': progress_time,
                    'progress_fused': progress_fused
                }
                self.visualizer.update(current_distance, self.elapsed_playback_time_s,
                                     (x, y), heading, telemetry, 
                                     self.braking_active, self.trajectory_complete)
                self.visualizer.show()
            
            # Status
            if time.time() - last_status_time >= 1.0:
                print(f"[WP {self.trajectory_index:4d}] "
                      f"Dist: {current_distance:6.2f}m ({progress_dist*100:5.1f}%) | "
                      f"Time: {self.elapsed_playback_time_s:5.1f}s ({progress_time*100:5.1f}%) | "
                      f"Fused: {progress_fused*100:5.1f}%")
                last_status_time = time.time()
            
            await asyncio.sleep(0.05)
        
        # Hold brake
        for _ in range(40):
            await self.send_brake_command(self.MAX_BRAKE_INTENSITY * 100.0)
            await asyncio.sleep(0.05)
        
        await self.send_brake_command(0.0)
        
        # Final stats
        self.print_final_statistics()
    
    def log_validation(self, point: ValidationPoint):
        """Log validation point"""
        with open(self.validation_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                point.timestamp, point.cumulative_distance_m, point.elapsed_time_s,
                point.target_angle_deg, point.commanded_steering_pct, point.commanded_brake_pct,
                point.measured_angle_deg, point.measured_speed_mps,
                point.angle_error_deg, point.distance_error_m, point.time_error_s,
                point.progress_distance, point.progress_time, point.progress_fused
            ])
    
    def print_final_statistics(self):
        """Print final statistics"""
        time_error = self.elapsed_playback_time_s - self.trajectory_metadata['duration_seconds']
        distance_error = self.cumulative_distance_m - self.trajectory_metadata['total_distance_m']
        
        print(f"\n{'='*80}")
        print(f"PLAYBACK COMPLETE")
        print(f"{'='*80}")
        print(f"Time: {self.elapsed_playback_time_s:.1f}s (target: {self.trajectory_metadata['duration_seconds']:.1f}s, error: {time_error:+.1f}s)")
        print(f"Distance: {self.cumulative_distance_m:.1f}m (target: {self.trajectory_metadata['total_distance_m']:.1f}m, error: {distance_error:+.1f}m)")
        
        if self.stats['angle_errors']:
            print(f"Angle error: mean={sum(self.stats['angle_errors'])/len(self.stats['angle_errors']):.2f}°, max={max(self.stats['angle_errors']):.2f}°")
        if self.stats['time_errors']:
            print(f"Time error: mean={sum(self.stats['time_errors'])/len(self.stats['time_errors']):.2f}s, max={max(self.stats['time_errors']):.2f}s")
        
        print(f"Commands: {self.stats['commands_sent']} steering, {self.stats['brake_commands_sent']} brake")
        print(f"{'='*80}\n")
    
    async def maintain_mcm_heartbeat(self):
        """Maintain heartbeat"""
        while self.running:
            try:
                if self.steering_enabled:
                    await self.enable_mcm_interface('steer')
                if self.brake_enabled:
                    await self.enable_mcm_interface('brake')
                await asyncio.sleep(0.1)
            except:
                await asyncio.sleep(0.5)
    
    async def run_playback_loop(self, num_laps: Optional[int] = None):
        """Run playback loop"""
        self.running = True
        
        # Start sensor thread
        sensor_thread = threading.Thread(target=self.monitor_sensors, daemon=True)
        sensor_thread.start()
        
        # Start heartbeat
        heartbeat_task = asyncio.create_task(self.maintain_mcm_heartbeat())
        
        # Wait for sensors
        print("Waiting for sensor initialization...")
        await asyncio.sleep(2.0)
        
        lap_count = 0
        try:
            while True:
                lap_count += 1
                
                if num_laps and lap_count > num_laps:
                    break
                
                print(f"\n{'█'*80}")
                print(f"█{f'LAP {lap_count}':^78}█")
                print(f"{'█'*80}\n")
                
                # Reset
                with self.data_lock:
                    self.cumulative_distance_m = 0.0
                    self.estimated_x_m = 0.0
                    self.estimated_y_m = 0.0
                    self.estimated_heading_deg = 0.0
                    self.last_update_time = None
                
                self.trajectory_index = 0
                self.braking_active = False
                self.trajectory_complete = False
                self.current_velocity_profile.clear()
                
                # Execute
                await self.execute_trajectory_playback()
                
                # Manual restart
                if num_laps is None or lap_count < num_laps:
                    restart = await asyncio.get_event_loop().run_in_executor(
                        None, input, "\nPress ENTER for next lap (Ctrl+C to exit): "
                    )
                    await asyncio.sleep(1.0)
        
        except KeyboardInterrupt:
            print(f"\n\n⚠ Interrupted")
        
        finally:
            self.running = False
            heartbeat_task.cancel()
            await self.send_brake_command(0.0)
            await self.send_steering_command(0.0)
            if self.visualizer:
                self.visualizer.close()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Hybrid Time-Distance Playback Controller',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--trajectory', '-t', required=True,
                       help='Hybrid trajectory JSON file')
    parser.add_argument('--mode', '-m', type=str, 
                       choices=['distance_only', 'velocity_compensated', 'time_distance_fusion'],
                       default='time_distance_fusion',
                       help='Playback indexing mode')
    parser.add_argument('--alpha', '-a', type=float, default=0.6,
                       help='Fusion alpha (0=distance, 1=time, default=0.6)')
    parser.add_argument('--mcm', default='can2', help='MCM CAN interface')
    parser.add_argument('--sas', default='can3', help='Sensor CAN interface')
    parser.add_argument('--brake-distance', type=float, default=8.0,
                       help='Brake initiation distance (m)')
    parser.add_argument('--laps', type=int, default=None,
                       help='Number of laps')
    parser.add_argument('--no-vis', action='store_true',
                       help='Disable visualization')
    
    args = parser.parse_args()
    
    mode_map = {
        'distance_only': PlaybackMode.DISTANCE_ONLY,
        'velocity_compensated': PlaybackMode.VELOCITY_COMPENSATED,
        'time_distance_fusion': PlaybackMode.TIME_DISTANCE_FUSION
    }
    
    print(f"\n{'='*80}")
    print(f"HYBRID PLAYBACK CONTROLLER")
    print(f"{'='*80}")
    print(f"Trajectory: {args.trajectory}")
    print(f"Mode: {args.mode}")
    if args.mode == 'time_distance_fusion':
        print(f"Fusion α: {args.alpha} (time weight)")
    print(f"{'='*80}\n")
    
    controller = HybridPlaybackController(
        mcm_channel=args.mcm,
        sas_channel=args.sas,
        trajectory_file=args.trajectory,
        playback_mode=mode_map[args.mode],
        fusion_alpha=args.alpha,
        brake_start_distance=args.brake_distance,
        enable_visualization=not args.no_vis
    )
    
    def cleanup(sig=None, frame=None):
        controller.running = False
        if controller.visualizer:
            controller.visualizer.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup)
    
    try:
        asyncio.run(controller.run_playback_loop(num_laps=args.laps))
    except KeyboardInterrupt:
        cleanup()

if __name__ == "__main__":
    main()
