#!/usr/bin/env python3
"""
Odometry-Based Trajectory Playback Controller with Real-Time Visualization
Distance-indexed steering command execution with OpenCV trajectory rendering

Features:
  - Real-time OpenCV trajectory visualization
  - XY position estimation from yaw rate integration
  - Comprehensive debug metadata logging
  - Live telemetry overlay
  - Path tracking analysis
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
import threading
from collections import deque

# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class TrajectoryTarget:
    """Trajectory waypoint from recorded data"""
    timestamp: float
    cumulative_distance_m: float
    target_angle_deg: float
    vehicle_speed_mps: float
    yaw_rate_deg_s: Optional[float] = None

@dataclass
class ValidationPoint:
    """Real-time validation data with position estimate"""
    timestamp: float
    cumulative_distance_m: float
    target_angle_deg: float
    commanded_steering_pct: float
    commanded_brake_pct: float
    measured_angle_deg: Optional[float]
    measured_speed_mps: Optional[float]
    angle_error_deg: Optional[float]
    distance_error_m: Optional[float]
    # Position estimation
    estimated_x_m: Optional[float] = None
    estimated_y_m: Optional[float] = None
    estimated_heading_deg: Optional[float] = None

@dataclass
class DebugMetadata:
    """Comprehensive debug metadata for single playback execution"""
    session_id: str
    start_timestamp: float
    end_timestamp: Optional[float]
    trajectory_file: str
    target_distance_m: float
    actual_distance_m: Optional[float]
    execution_time_s: Optional[float]
    lap_number: int
    
    # Control statistics
    total_steering_commands: int = 0
    total_brake_commands: int = 0
    brake_initiated_at_m: Optional[float] = None
    brake_duration_s: Optional[float] = None
    
    # Tracking performance
    mean_angle_error_deg: Optional[float] = None
    max_angle_error_deg: Optional[float] = None
    mean_distance_error_m: Optional[float] = None
    max_distance_error_m: Optional[float] = None
    rms_position_error_m: Optional[float] = None
    
    # Position tracking
    final_position_xy: Optional[Tuple[float, float]] = None
    loop_closure_error_m: Optional[float] = None
    
    # System health
    sensor_dropout_events: int = 0
    control_failure_events: int = 0
    timing_violations: int = 0

# ============================================================================
# Trajectory Visualizer
# ============================================================================

class TrajectoryVisualizer:
    """Real-time OpenCV visualization of trajectory execution"""
    
    def __init__(self, trajectory_targets: List[TrajectoryTarget], 
                 window_size=(1200, 800), grid_spacing_m=5.0):
        self.trajectory_targets = trajectory_targets
        self.window_size = window_size
        self.grid_spacing_m = grid_spacing_m
        
        # Calculate XY coordinates from trajectory
        self.trajectory_xy = self.calculate_trajectory_coordinates()
        
        # Determine visualization bounds
        self.bounds = self.calculate_bounds()
        
        # Current state
        self.current_position = (0.0, 0.0)
        self.current_heading = 0.0
        self.current_distance = 0.0
        self.brake_active = False
        self.trajectory_complete = False
        
        # Telemetry
        self.telemetry = {
            'speed_mps': 0.0,
            'steering_angle_deg': 0.0,
            'brake_pct': 0.0,
            'distance_remaining_m': 0.0,
            'angle_error_deg': 0.0,
            'trajectory_index': 0
        }
        
        # History trail
        self.position_history = deque(maxlen=500)
        
        # Thread safety
        self.vis_lock = threading.Lock()
        
        # Initialize window
        cv2.namedWindow('Trajectory Visualization', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Trajectory Visualization', *window_size)
        
        print(f"✓ Visualization initialized")
        print(f"  Bounds: X[{self.bounds['x_min']:.1f}, {self.bounds['x_max']:.1f}] "
              f"Y[{self.bounds['y_min']:.1f}, {self.bounds['y_max']:.1f}]")
    
    def calculate_trajectory_coordinates(self) -> List[Tuple[float, float]]:
        """
        Calculate XY coordinates from distance-indexed trajectory using yaw rate integration
        
        Dead reckoning algorithm:
          x[i+1] = x[i] + Δd × sin(θ[i])
          y[i+1] = y[i] + Δd × cos(θ[i])
          θ[i+1] = θ[i] + yaw_rate × Δt
        """
        trajectory_xy = [(0.0, 0.0)]  # Start at origin
        
        x, y = 0.0, 0.0
        heading_deg = 0.0  # Initial heading (vehicle pointing "north" = +Y)
        
        for i in range(1, len(self.trajectory_targets)):
            prev = self.trajectory_targets[i-1]
            curr = self.trajectory_targets[i]
            
            # Distance increment
            distance_delta = curr.cumulative_distance_m - prev.cumulative_distance_m
            
            # Update heading from yaw rate integration
            if curr.yaw_rate_deg_s is not None and prev.yaw_rate_deg_s is not None:
                dt = curr.timestamp - prev.timestamp
                avg_yaw_rate = (curr.yaw_rate_deg_s + prev.yaw_rate_deg_s) / 2.0
                heading_deg += avg_yaw_rate * dt
            
            # Calculate position increment
            heading_rad = math.radians(heading_deg)
            x += distance_delta * math.sin(heading_rad)
            y += distance_delta * math.cos(heading_rad)
            
            trajectory_xy.append((x, y))
        
        return trajectory_xy
    
    def calculate_bounds(self) -> Dict[str, float]:
        """Calculate visualization bounds with padding"""
        if not self.trajectory_xy:
            return {'x_min': -10, 'x_max': 10, 'y_min': -10, 'y_max': 10}
        
        x_coords = [pos[0] for pos in self.trajectory_xy]
        y_coords = [pos[1] for pos in self.trajectory_xy]
        
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        
        # Add 20% padding
        x_range = max(x_max - x_min, 10.0)  # Minimum 10m range
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
        """Convert world coordinates (meters) to pixel coordinates"""
        # Map world bounds to pixel space
        x_range = self.bounds['x_max'] - self.bounds['x_min']
        y_range = self.bounds['y_max'] - self.bounds['y_min']
        
        # Maintain aspect ratio
        scale = min(
            (self.window_size[0] - 100) / x_range,
            (self.window_size[1] - 150) / y_range
        )
        
        # Center offset
        center_x = self.window_size[0] // 2
        center_y = self.window_size[1] // 2 + 50
        
        # Transform (Y-axis flipped for screen coordinates)
        pixel_x = int(center_x + (x - (self.bounds['x_min'] + self.bounds['x_max'])/2) * scale)
        pixel_y = int(center_y - (y - (self.bounds['y_min'] + self.bounds['y_max'])/2) * scale)
        
        return (pixel_x, pixel_y)
    
    def draw_grid(self, frame: np.ndarray):
        """Draw metric grid overlay"""
        x_range = self.bounds['x_max'] - self.bounds['x_min']
        y_range = self.bounds['y_max'] - self.bounds['y_min']
        
        # Vertical grid lines
        x = self.bounds['x_min']
        while x <= self.bounds['x_max']:
            px_start = self.world_to_pixel(x, self.bounds['y_min'])
            px_end = self.world_to_pixel(x, self.bounds['y_max'])
            cv2.line(frame, px_start, px_end, (40, 40, 40), 1, cv2.LINE_AA)
            
            # Label
            label_pos = self.world_to_pixel(x, self.bounds['y_min'])
            cv2.putText(frame, f"{x:.0f}m", (label_pos[0]-15, self.window_size[1]-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
            
            x += self.grid_spacing_m
        
        # Horizontal grid lines
        y = self.bounds['y_min']
        while y <= self.bounds['y_max']:
            px_start = self.world_to_pixel(self.bounds['x_min'], y)
            px_end = self.world_to_pixel(self.bounds['x_max'], y)
            cv2.line(frame, px_start, px_end, (40, 40, 40), 1, cv2.LINE_AA)
            
            # Label
            label_pos = self.world_to_pixel(self.bounds['x_min'], y)
            cv2.putText(frame, f"{y:.0f}m", (10, label_pos[1]+5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (80, 80, 80), 1)
            
            y += self.grid_spacing_m
        
        # Origin marker
        origin_px = self.world_to_pixel(0, 0)
        cv2.drawMarker(frame, origin_px, (100, 100, 100), cv2.MARKER_CROSS, 20, 2)
        cv2.putText(frame, "START", (origin_px[0]+10, origin_px[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
    
    def draw_trajectory_path(self, frame: np.ndarray):
        """Draw planned trajectory path"""
        if len(self.trajectory_xy) < 2:
            return
        
        # Draw trajectory line
        points = [self.world_to_pixel(x, y) for x, y in self.trajectory_xy]
        
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i+1], (80, 80, 200), 2, cv2.LINE_AA)
        
        # Draw waypoint markers every 5m
        for i, target in enumerate(self.trajectory_targets):
            if i % 10 == 0:  # Every ~5m (assuming 0.5m spacing)
                x, y = self.trajectory_xy[i]
                px = self.world_to_pixel(x, y)
                cv2.circle(frame, px, 3, (100, 100, 255), -1)
        
        # End point marker
        end_x, end_y = self.trajectory_xy[-1]
        end_px = self.world_to_pixel(end_x, end_y)
        cv2.drawMarker(frame, end_px, (0, 255, 0), cv2.MARKER_STAR, 20, 2)
        cv2.putText(frame, "END", (end_px[0]+10, end_px[1]-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    def draw_position_history(self, frame: np.ndarray):
        """Draw vehicle position trail"""
        if len(self.position_history) < 2:
            return
        
        points = [self.world_to_pixel(x, y) for x, y in self.position_history]
        
        # Fade trail from old (dark) to new (bright)
        for i in range(len(points) - 1):
            alpha = i / len(points)
            color = (0, int(150 + 105*alpha), int(150 + 105*alpha))
            cv2.line(frame, points[i], points[i+1], color, 2, cv2.LINE_AA)
    
    def draw_vehicle(self, frame: np.ndarray):
        """Draw current vehicle position and orientation"""
        px = self.world_to_pixel(self.current_position[0], self.current_position[1])
        
        # Vehicle circle
        if self.brake_active:
            color = (0, 100, 255)  # Orange when braking
        elif self.trajectory_complete:
            color = (0, 255, 0)    # Green when complete
        else:
            color = (255, 100, 0)  # Blue during normal operation
        
        cv2.circle(frame, px, 8, color, -1)
        cv2.circle(frame, px, 10, (255, 255, 255), 2)
        
        # Heading indicator (arrow)
        heading_rad = math.radians(self.current_heading)
        arrow_length = 25
        arrow_end_x = px[0] + int(arrow_length * math.sin(heading_rad))
        arrow_end_y = px[1] - int(arrow_length * math.cos(heading_rad))
        cv2.arrowedLine(frame, px, (arrow_end_x, arrow_end_y), 
                       (255, 255, 255), 2, tipLength=0.3)
    
    def draw_telemetry(self, frame: np.ndarray):
        """Draw telemetry overlay panel"""
        panel_height = 120
        panel = np.zeros((panel_height, self.window_size[0], 3), dtype=np.uint8)
        
        # Title
        cv2.putText(panel, "TRAJECTORY PLAYBACK - LIVE TELEMETRY", (10, 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Column 1: Position/Distance
        col1_x = 10
        cv2.putText(panel, f"Distance: {self.current_distance:.2f}m / {self.trajectory_targets[-1].cumulative_distance_m:.2f}m",
                   (col1_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, f"Remaining: {self.telemetry['distance_remaining_m']:.2f}m",
                   (col1_x, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, f"Position: X={self.current_position[0]:.2f}m, Y={self.current_position[1]:.2f}m",
                   (col1_x, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, f"Heading: {self.current_heading:.1f}°",
                   (col1_x, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Column 2: Control
        col2_x = 350
        cv2.putText(panel, f"Speed: {self.telemetry['speed_mps']:.2f}m/s ({self.telemetry['speed_mps']*3.6:.1f}km/h)",
                   (col2_x, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(panel, f"Steering: {self.telemetry['steering_angle_deg']:+.1f}°",
                   (col2_x, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Brake indicator with color
        brake_color = (0, 100, 255) if self.brake_active else (100, 100, 100)
        cv2.putText(panel, f"Brake: {self.telemetry['brake_pct']:.1f}%",
                   (col2_x, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, brake_color, 1)
        
        cv2.putText(panel, f"Waypoint: {self.telemetry['trajectory_index']}/{len(self.trajectory_targets)}",
                   (col2_x, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Column 3: Errors
        col3_x = 700
        cv2.putText(panel, "TRACKING ACCURACY", (col3_x, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 100), 1)
        
        angle_err_color = (0, 255, 0) if abs(self.telemetry['angle_error_deg']) < 5.0 else (0, 200, 255)
        cv2.putText(panel, f"Angle Error: {self.telemetry['angle_error_deg']:+.2f}°",
                   (col3_x, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, angle_err_color, 1)
        
        # Status indicator
        if self.trajectory_complete:
            status_text = "COMPLETE"
            status_color = (0, 255, 0)
        elif self.brake_active:
            status_text = "BRAKING"
            status_color = (0, 150, 255)
        else:
            status_text = "EXECUTING"
            status_color = (255, 200, 0)
        
        cv2.putText(panel, f"Status: {status_text}", (col3_x, 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        # Progress bar
        progress = min(self.current_distance / self.trajectory_targets[-1].cumulative_distance_m, 1.0)
        bar_width = 200
        bar_x = self.window_size[0] - bar_width - 20
        bar_y = 90
        
        # Background
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), (50, 50, 50), -1)
        # Foreground
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + 15), 
                     (100, 255, 100), -1)
        # Border
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + 15), (150, 150, 150), 1)
        
        cv2.putText(panel, f"{progress*100:.1f}%", (bar_x + bar_width + 10, bar_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        # Composite panel onto frame
        frame[0:panel_height, :] = panel
    
    def update(self, distance: float, position: Tuple[float, float], heading: float,
               telemetry: Dict, brake_active: bool, trajectory_complete: bool):
        """Update visualization state (thread-safe)"""
        with self.vis_lock:
            self.current_distance = distance
            self.current_position = position
            self.current_heading = heading
            self.telemetry = telemetry.copy()
            self.brake_active = brake_active
            self.trajectory_complete = trajectory_complete
            
            # Update position history
            self.position_history.append(position)
    
    def render(self) -> np.ndarray:
        """Render current frame (thread-safe)"""
        with self.vis_lock:
            # Create black canvas
            frame = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
            frame[:] = (20, 20, 20)  # Dark gray background
            
            # Draw layers (bottom to top)
            self.draw_grid(frame)
            self.draw_trajectory_path(frame)
            self.draw_position_history(frame)
            self.draw_vehicle(frame)
            self.draw_telemetry(frame)
            
            return frame
    
    def show(self):
        """Display current frame"""
        frame = self.render()
        cv2.imshow('Trajectory Visualization', frame)
        cv2.waitKey(1)
    
    def close(self):
        """Clean up visualization window"""
        cv2.destroyAllWindows()

# ============================================================================
# Enhanced Playback Controller
# ============================================================================

class OdometryPlaybackController:
    """Enhanced playback controller with visualization and debug metadata"""
    
    def __init__(self, mcm_channel='can2', sas_channel='can3', 
                 trajectory_file=None, brake_start_distance=8.0,
                 enable_visualization=True):
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
        
        # CAN IDs
        self.SAS11_CAN_ID = 0x2B0
        self.WHL_SPD_CAN_ID = 0x386
        self.ESP12_CAN_ID = 0x220
        
        # Scaling factors
        self.SAS_ANGLE_SCALE = 0.1
        self.WHEEL_SPEED_SCALE = 0.03125
        self.YAW_RATE_SCALE = 0.01
        self.YAW_RATE_OFFSET = -40.95
        
        # Odometry State
        self.cumulative_distance_m = 0.0
        self.vehicle_speed_mps = 0.0
        self.wheel_speeds_kmh = {'FL': 0.0, 'FR': 0.0, 'RL': 0.0, 'RR': 0.0}
        self.last_distance_update_time = None
        
        # Vehicle Dynamics
        self.yaw_rate_deg_s = None
        
        # Position Estimation (dead reckoning)
        self.estimated_x_m = 0.0
        self.estimated_y_m = 0.0
        self.estimated_heading_deg = 0.0
        self.last_position_update_time = None
        
        # Steering Feedback
        self.latest_angle_deg = None
        self.latest_angle_timestamp = None
        
        # Trajectory Data
        self.trajectory_targets: List[TrajectoryTarget] = []
        self.trajectory_metadata = {}
        self.total_trajectory_distance = 0.0
        self.trajectory_file_path = trajectory_file
        
        # Validation Logging
        self.validation_log: List[ValidationPoint] = []
        
        # Brake Parameters
        self.BRAKE_START_DISTANCE = brake_start_distance
        self.MAX_BRAKE_INTENSITY = 0.7
        
        # State Machine
        self.trajectory_index = 0
        self.braking_active = False
        self.brake_initiated_at_m = None
        self.brake_start_time = None
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
            'position_errors': [],
            'max_distance_error': 0.0,
            'max_angle_error': 0.0
        }
        
        # Debug Metadata
        self.current_lap_metadata: Optional[DebugMetadata] = None
        self.all_laps_metadata: List[DebugMetadata] = []
        self.sensor_dropout_count = 0
        self.control_failure_count = 0
        
        # Load Trajectory
        if trajectory_file:
            self.load_trajectory_data(trajectory_file)
        
        # Visualization
        self.enable_visualization = enable_visualization
        self.visualizer: Optional[TrajectoryVisualizer] = None
        
        if self.enable_visualization and len(self.trajectory_targets) > 0:
            try:
                self.visualizer = TrajectoryVisualizer(self.trajectory_targets)
            except Exception as e:
                print(f"⚠ Visualization initialization failed: {e}")
                self.enable_visualization = False
        
        # Setup Logging
        self.setup_validation_logging()
    
    def load_trajectory_data(self, filename: str):
        """Load distance-indexed trajectory from JSON"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            if data.get('format_version') != '3.0_odometry':
                print(f"⚠ Warning: Unexpected format version {data.get('format_version')}")
            
            trajectory_points = data['trajectory_points']
            
            for point in trajectory_points:
                target = TrajectoryTarget(
                    timestamp=point['timestamp'],
                    cumulative_distance_m=point['cumulative_distance_m'],
                    target_angle_deg=point['steering_angle_deg'],
                    vehicle_speed_mps=point['vehicle_speed_mps'],
                    yaw_rate_deg_s=point.get('yaw_rate_deg_s')
                )
                self.trajectory_targets.append(target)
            
            self.trajectory_metadata = data.get('metadata', {})
            self.total_trajectory_distance = self.trajectory_targets[-1].cumulative_distance_m
            
            print(f"\n{'='*80}")
            print(f"TRAJECTORY LOADED")
            print(f"{'='*80}")
            print(f"File: {filename}")
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
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def setup_validation_logging(self):
        """Initialize validation logging"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.validation_csv = f"playback_validation_{timestamp}.csv"
        self.debug_metadata_json = f"playback_debug_{timestamp}.json"
        
        with open(self.validation_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'cumulative_distance_m', 'target_angle_deg',
                'commanded_steering_pct', 'commanded_brake_pct',
                'measured_angle_deg', 'measured_speed_mps',
                'angle_error_deg', 'distance_error_m',
                'estimated_x_m', 'estimated_y_m', 'estimated_heading_deg'
            ])
        
        print(f"Validation logging: {self.validation_csv}")
        print(f"Debug metadata: {self.debug_metadata_json}")
    
    def calc_crc8(self, data: bytes) -> int:
        """Calculate CRC8 for MCM messages"""
        hash = crc8.crc8()
        hash.update(data[:-1])
        return hash.digest()[0]
    
    def angle_to_percentage(self, angle_deg: float) -> float:
        """Convert steering angle to MCM command percentage"""
        adjusted_angle = -angle_deg
        percentage = adjusted_angle * (100.0 / 400.0)
        return max(min(percentage, 100.0), -100.0)
    
    def calculate_brake_command(self, remaining_distance: float) -> float:
        """Progressive brake intensity calculation"""
        if remaining_distance >= self.BRAKE_START_DISTANCE:
            return 0.0
        
        if remaining_distance <= 0.0:
            return self.MAX_BRAKE_INTENSITY
        
        brake_progress = 1.0 - (remaining_distance / self.BRAKE_START_DISTANCE)
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
    
    def parse_esp12(self, data: bytes) -> Optional[float]:
        """Parse ESP12 for yaw rate"""
        if len(data) < 8:
            return None
        
        try:
            frame_uint64 = struct.unpack('<Q', data)[0]
            
            # YAW_RATE : 40|13@1+ (0.01,-40.95)
            yaw_rate_raw = (frame_uint64 >> 40) & 0x1FFF
            yaw_rate = yaw_rate_raw * self.YAW_RATE_SCALE + self.YAW_RATE_OFFSET
            
            return yaw_rate
            
        except Exception as e:
            print(f"ESP12 parsing error: {e}")
            return None
    
    def update_odometry(self, wheel_speeds_kmh: Dict[str, float], timestamp: float):
        """Update cumulative distance from wheel speeds"""
        with self.data_lock:
            self.wheel_speeds_kmh = wheel_speeds_kmh
            
            rear_avg_kmh = (wheel_speeds_kmh['RL'] + wheel_speeds_kmh['RR']) / 2.0
            vehicle_speed_mps = rear_avg_kmh / 3.6
            
            if self.last_distance_update_time is not None:
                dt = timestamp - self.last_distance_update_time
                
                if 0.001 <= dt <= 1.0:
                    distance_increment = vehicle_speed_mps * dt
                    self.cumulative_distance_m += distance_increment
                    
                    # Update position estimate
                    self.update_position_estimate(distance_increment, dt)
            
            self.vehicle_speed_mps = vehicle_speed_mps
            self.last_distance_update_time = timestamp
    
    def update_position_estimate(self, distance_delta: float, dt: float):
        """Update XY position estimate using dead reckoning"""
        # Update heading from yaw rate
        if self.yaw_rate_deg_s is not None:
            self.estimated_heading_deg += self.yaw_rate_deg_s * dt
        
        # Update position
        heading_rad = math.radians(self.estimated_heading_deg)
        self.estimated_x_m += distance_delta * math.sin(heading_rad)
        self.estimated_y_m += distance_delta * math.cos(heading_rad)
    
    def get_current_odometry(self) -> Tuple[float, float]:
        """Get current distance and speed (thread-safe)"""
        with self.data_lock:
            return self.cumulative_distance_m, self.vehicle_speed_mps
    
    def get_current_position(self) -> Tuple[float, float, float]:
        """Get current position estimate (thread-safe)"""
        with self.data_lock:
            return self.estimated_x_m, self.estimated_y_m, self.estimated_heading_deg
    
    def get_current_angle(self) -> Tuple[Optional[float], Optional[float]]:
        """Get latest steering angle (thread-safe)"""
        with self.data_lock:
            return self.latest_angle_deg, self.latest_angle_timestamp
    
    async def enable_mcm_interface(self, interface_name: str):
        """Enable MCM control interface"""
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
            self.control_failure_count += 1
            return False
    
    async def send_steering_command(self, percentage: float) -> bool:
        """Send steering command to MCM"""
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
            self.control_failure_count += 1
            return False
    
    async def send_brake_command(self, percentage: float) -> bool:
        """Send brake command to MCM"""
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
            self.control_failure_count += 1
            return False
    
    def monitor_sensors(self):
        """Background thread for sensor monitoring (CAN3)"""
        print("✓ Sensor monitoring thread started")
        
        last_wheel_speed_time = time.time()
        last_sas_time = time.time()
        
        while self.running:
            try:
                message = self.sas_bus.recv(timeout=0.1)
                
                if message is None:
                    # Check for sensor dropouts
                    current_time = time.time()
                    if current_time - last_wheel_speed_time > 1.0:
                        self.sensor_dropout_count += 1
                        last_wheel_speed_time = current_time
                    continue
                
                # Process wheel speeds
                if message.arbitration_id == self.WHL_SPD_CAN_ID:
                    last_wheel_speed_time = time.time()
                    wheel_speeds = self.parse_wheel_speeds(message.data)
                    if wheel_speeds:
                        self.update_odometry(wheel_speeds, message.timestamp)
                
                # Process steering feedback
                elif message.arbitration_id == self.SAS11_CAN_ID:
                    last_sas_time = time.time()
                    angle = self.parse_sas11_angle(message.data)
                    if angle is not None:
                        with self.data_lock:
                            self.latest_angle_deg = angle
                            self.latest_angle_timestamp = message.timestamp
                
                # Process yaw rate
                elif message.arbitration_id == self.ESP12_CAN_ID:
                    yaw_rate = self.parse_esp12(message.data)
                    if yaw_rate is not None:
                        with self.data_lock:
                            self.yaw_rate_deg_s = yaw_rate
                            
            except Exception as e:
                if self.running:
                    print(f"⚠ Sensor monitoring error: {e}")
                    time.sleep(0.1)
    
    def init_lap_metadata(self, lap_number: int):
        """Initialize metadata for new lap"""
        self.current_lap_metadata = DebugMetadata(
            session_id=f"lap_{lap_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            start_timestamp=time.time(),
            end_timestamp=None,
            trajectory_file=self.trajectory_file_path or "unknown",
            target_distance_m=self.total_trajectory_distance,
            actual_distance_m=None,
            execution_time_s=None,
            lap_number=lap_number
        )
    
    def finalize_lap_metadata(self):
        """Finalize and store lap metadata"""
        if self.current_lap_metadata is None:
            return
        
        # Calculate final metrics
        self.current_lap_metadata.end_timestamp = time.time()
        self.current_lap_metadata.execution_time_s = (
            self.current_lap_metadata.end_timestamp - self.current_lap_metadata.start_timestamp
        )
        
        # Control statistics
        self.current_lap_metadata.total_steering_commands = self.stats['commands_sent']
        self.current_lap_metadata.total_brake_commands = self.stats['brake_commands_sent']
        self.current_lap_metadata.brake_initiated_at_m = self.brake_initiated_at_m
        if self.brake_start_time:
            self.current_lap_metadata.brake_duration_s = time.time() - self.brake_start_time
        
        # Tracking performance
        if self.stats['angle_errors']:
            self.current_lap_metadata.mean_angle_error_deg = (
                sum(self.stats['angle_errors']) / len(self.stats['angle_errors'])
            )
            self.current_lap_metadata.max_angle_error_deg = self.stats['max_angle_error']
        
        if self.stats['distance_errors']:
            self.current_lap_metadata.mean_distance_error_m = (
                sum(self.stats['distance_errors']) / len(self.stats['distance_errors'])
            )
            self.current_lap_metadata.max_distance_error_m = self.stats['max_distance_error']
        
        if self.stats['position_errors']:
            rms = math.sqrt(sum(e**2 for e in self.stats['position_errors']) / len(self.stats['position_errors']))
            self.current_lap_metadata.rms_position_error_m = rms
        
        # Position tracking
        current_distance, _ = self.get_current_odometry()
        self.current_lap_metadata.actual_distance_m = current_distance
        
        x, y, heading = self.get_current_position()
        self.current_lap_metadata.final_position_xy = (x, y)
        self.current_lap_metadata.loop_closure_error_m = math.sqrt(x**2 + y**2)
        
        # System health
        self.current_lap_metadata.sensor_dropout_events = self.sensor_dropout_count
        self.current_lap_metadata.control_failure_events = self.control_failure_count
        
        # Store
        self.all_laps_metadata.append(self.current_lap_metadata)
        
        # Save to file
        self.save_debug_metadata()
    
    def save_debug_metadata(self):
        """Save all laps metadata to JSON"""
        metadata_dict = {
            'session_summary': {
                'total_laps': len(self.all_laps_metadata),
                'trajectory_file': self.trajectory_file_path,
                'total_distance_target_m': self.total_trajectory_distance,
                'brake_start_distance_m': self.BRAKE_START_DISTANCE
            },
            'laps': [asdict(lap) for lap in self.all_laps_metadata]
        }
        
        with open(self.debug_metadata_json, 'w') as f:
            json.dump(metadata_dict, f, indent=2)
    
    async def execute_trajectory_playback(self):
        """Main trajectory execution state machine with visualization"""
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
        last_vis_update_time = playback_start_time
        
        # Reset state
        self.trajectory_index = 0
        self.braking_active = False
        self.brake_initiated_at_m = None
        self.brake_start_time = None
        self.trajectory_complete = False
        
        # Main control loop
        while self.running and not self.trajectory_complete:
            
            # Get current state
            current_distance, current_speed = self.get_current_odometry()
            current_x, current_y, current_heading = self.get_current_position()
            remaining_distance = self.total_trajectory_distance - current_distance
            
            # === BRAKING LOGIC ===
            brake_command = self.calculate_brake_command(remaining_distance)
            
            if brake_command > 0.0:
                if not self.braking_active:
                    print(f"\n⚠ Braking initiated at {current_distance:.2f}m "
                          f"(remaining: {remaining_distance:.2f}m)")
                    self.braking_active = True
                    self.brake_initiated_at_m = current_distance
                    self.brake_start_time = time.time()
                
                await self.send_brake_command(brake_command * 100.0)
            
            # Check trajectory completion
            if current_distance >= self.total_trajectory_distance:
                print(f"\n✓ Trajectory complete at {current_distance:.2f}m")
                await self.send_brake_command(self.MAX_BRAKE_INTENSITY * 100.0)
                self.trajectory_complete = True
                break
            
            # === STEERING LOGIC ===
            while (self.trajectory_index < len(self.trajectory_targets) and
                   current_distance >= self.trajectory_targets[self.trajectory_index].cumulative_distance_m):
                
                target = self.trajectory_targets[self.trajectory_index]
                command_percentage = self.angle_to_percentage(target.target_angle_deg)
                
                await self.send_steering_command(command_percentage)
                
                # Validation
                measured_angle, _ = self.get_current_angle()
                
                angle_error = None
                if measured_angle is not None:
                    angle_error = measured_angle - target.target_angle_deg
                    self.stats['angle_errors'].append(abs(angle_error))
                    self.stats['max_angle_error'] = max(self.stats['max_angle_error'], abs(angle_error))
                
                distance_error = current_distance - target.cumulative_distance_m
                self.stats['distance_errors'].append(abs(distance_error))
                self.stats['max_distance_error'] = max(self.stats['max_distance_error'], abs(distance_error))
                
                # Position error (from planned trajectory)
                if self.visualizer:
                    target_xy = self.visualizer.trajectory_xy[min(self.trajectory_index, len(self.visualizer.trajectory_xy)-1)]
                    position_error = math.sqrt((current_x - target_xy[0])**2 + (current_y - target_xy[1])**2)
                    self.stats['position_errors'].append(position_error)
                
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
                    distance_error_m=distance_error,
                    estimated_x_m=current_x,
                    estimated_y_m=current_y,
                    estimated_heading_deg=current_heading
                )
                self.validation_log.append(validation_point)
                self.log_validation_to_csv(validation_point)
                
                self.trajectory_index += 1
            
            # === VISUALIZATION UPDATE ===
            if self.visualizer and (time.time() - last_vis_update_time) >= 0.05:  # 20Hz
                telemetry = {
                    'speed_mps': current_speed,
                    'steering_angle_deg': self.latest_angle_deg or 0.0,
                    'brake_pct': brake_command * 100.0,
                    'distance_remaining_m': remaining_distance,
                    'angle_error_deg': angle_error or 0.0,
                    'trajectory_index': self.trajectory_index
                }
                
                self.visualizer.update(
                    distance=current_distance,
                    position=(current_x, current_y),
                    heading=current_heading,
                    telemetry=telemetry,
                    brake_active=self.braking_active,
                    trajectory_complete=self.trajectory_complete
                )
                self.visualizer.show()
                last_vis_update_time = time.time()
            
            # === STATUS DISPLAY ===
            if time.time() - last_status_time >= 1.0:
                progress_pct = (current_distance / self.total_trajectory_distance) * 100
                measured_angle, _ = self.get_current_angle()
                
                status_line = (f"[{self.trajectory_index:4d}/{len(self.trajectory_targets)}] "
                              f"Dist: {current_distance:6.2f}m ({progress_pct:5.1f}%) | "
                              f"Speed: {current_speed:4.2f}m/s | "
                              f"Pos: ({current_x:5.2f}, {current_y:5.2f})m")
                
                if self.braking_active:
                    status_line += f" | BRAKE: {brake_command*100:4.1f}%"
                
                print(status_line)
                last_status_time = time.time()
            
            # Control loop rate: 20Hz
            await asyncio.sleep(0.05)
        
        # === POST-TRAJECTORY ACTIONS ===
        print(f"\nHolding brake for complete stop...")
        for _ in range(40):
            await self.send_brake_command(self.MAX_BRAKE_INTENSITY * 100.0)
            if self.visualizer:
                self.visualizer.show()
            await asyncio.sleep(0.05)
        
        await self.send_brake_command(0.0)
        
        # Final statistics
        final_distance, final_speed = self.get_current_odometry()
        final_x, final_y, final_heading = self.get_current_position()
        distance_overshoot = final_distance - self.total_trajectory_distance
        loop_closure = math.sqrt(final_x**2 + final_y**2)
        
        print(f"\n{'='*80}")
        print(f"TRAJECTORY EXECUTION COMPLETE")
        print(f"{'='*80}")
        print(f"Target distance: {self.total_trajectory_distance:.2f}m")
        print(f"Actual distance: {final_distance:.2f}m")
        print(f"Overshoot: {distance_overshoot:+.2f}m")
        print(f"Final position: ({final_x:.2f}, {final_y:.2f})m")
        print(f"Loop closure error: {loop_closure:.2f}m")
        print(f"Final heading: {final_heading:.1f}°")
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
                point.angle_error_deg, point.distance_error_m,
                point.estimated_x_m, point.estimated_y_m, point.estimated_heading_deg
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
        """Execute trajectory playback with manual lap restart"""
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
                
                # Initialize lap metadata
                self.init_lap_metadata(lap_count)
                
                # Reset state
                with self.data_lock:
                    self.cumulative_distance_m = 0.0
                    self.estimated_x_m = 0.0
                    self.estimated_y_m = 0.0
                    self.estimated_heading_deg = 0.0
                    self.last_distance_update_time = None
                
                # Reset lap statistics
                self.stats['commands_sent'] = 0
                self.stats['brake_commands_sent'] = 0
                self.stats['distance_errors'].clear()
                self.stats['angle_errors'].clear()
                self.stats['position_errors'].clear()
                self.stats['max_distance_error'] = 0.0
                self.stats['max_angle_error'] = 0.0
                self.sensor_dropout_count = 0
                self.control_failure_count = 0
                
                # Execute trajectory
                await self.execute_trajectory_playback()
                
                # Finalize lap metadata
                self.finalize_lap_metadata()
                
                # Print lap summary
                self.print_lap_summary(lap_count)
                
                # Manual restart prompt
                if num_laps is None or lap_count < num_laps:
                    print(f"\n{'─'*80}")
                    print(f"Lap {lap_count} complete. Vehicle stopped.")
                    print(f"{'─'*80}")
                    
                    restart_input = await asyncio.get_event_loop().run_in_executor(
                        None, 
                        input, 
                        "\nPress ENTER to start next lap (or Ctrl+C to exit): "
                    )
                    
                    print(f"\n{'─'*80}")
                    print(f"Restarting lap {lap_count + 1}...")
                    print(f"{'─'*80}\n")
                    
                    await asyncio.sleep(1.0)
        
        except KeyboardInterrupt:
            print(f"\n\n⚠ Playback interrupted by user")
        
        finally:
            self.running = False
            heartbeat_task.cancel()
            
            # Release controls
            await self.send_brake_command(0.0)
            await self.send_steering_command(0.0)
            
            # Close visualization
            if self.visualizer:
                self.visualizer.close()
            
            print(f"\n{'='*80}")
            print(f"PLAYBACK SESSION COMPLETE")
            print(f"{'='*80}")
            print(f"Total laps: {lap_count}")
            print(f"Validation log: {self.validation_csv}")
            print(f"Debug metadata: {self.debug_metadata_json}")
            print(f"{'='*80}")
    
    def print_lap_summary(self, lap_number: int):
        """Print statistical summary for completed lap"""
        if not self.current_lap_metadata:
            return
        
        meta = self.current_lap_metadata
        
        print(f"\n{'─'*80}")
        print(f"LAP {lap_number} SUMMARY")
        print(f"{'─'*80}")
        
        print(f"Distance tracking:")
        print(f"  Target: {meta.target_distance_m:.2f}m")
        print(f"  Actual: {meta.actual_distance_m:.2f}m" if meta.actual_distance_m else "  Actual: N/A")
        print(f"  Mean error: {meta.mean_distance_error_m:.3f}m" if meta.mean_distance_error_m else "  Mean error: N/A")
        print(f"  Max error: {meta.max_distance_error_m:.3f}m" if meta.max_distance_error_m else "  Max error: N/A")
        
        print(f"Angle tracking:")
        print(f"  Mean error: {meta.mean_angle_error_deg:.2f}°" if meta.mean_angle_error_deg else "  Mean error: N/A")
        print(f"  Max error: {meta.max_angle_error_deg:.2f}°" if meta.max_angle_error_deg else "  Max error: N/A")
        
        print(f"Position tracking:")
        print(f"  Final XY: ({meta.final_position_xy[0]:.2f}, {meta.final_position_xy[1]:.2f})m" if meta.final_position_xy else "  Final XY: N/A")
        print(f"  Loop closure: {meta.loop_closure_error_m:.2f}m" if meta.loop_closure_error_m else "  Loop closure: N/A")
        print(f"  RMS position error: {meta.rms_position_error_m:.3f}m" if meta.rms_position_error_m else "  RMS position error: N/A")
        
        print(f"Control commands:")
        print(f"  Steering: {meta.total_steering_commands}")
        print(f"  Brake: {meta.total_brake_commands}")
        
        print(f"Braking:")
        print(f"  Initiated at: {meta.brake_initiated_at_m:.2f}m" if meta.brake_initiated_at_m else "  Initiated at: N/A")
        print(f"  Duration: {meta.brake_duration_s:.2f}s" if meta.brake_duration_s else "  Duration: N/A")
        
        print(f"System health:")
        print(f"  Sensor dropouts: {meta.sensor_dropout_events}")
        print(f"  Control failures: {meta.control_failure_events}")
        
        print(f"Execution time: {meta.execution_time_s:.1f}s" if meta.execution_time_s else "Execution time: N/A")
        print(f"{'─'*80}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Odometry-Based Trajectory Playback Controller with Visualization',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Distance-indexed trajectory playback with real-time OpenCV visualization
and comprehensive debug metadata logging.

Features:
  - Real-time trajectory visualization with grid overlay
  - XY position estimation via dead reckoning
  - Comprehensive debug metadata per lap
  - Live telemetry panel
  - Path tracking analysis

Example usage:
  # Single lap with visualization
  python3 mcm_steer_track2_odometry.py --trajectory recorded.json
  
  # Multiple laps, no visualization
  python3 mcm_steer_track2_odometry.py --trajectory path.json --laps 5 --no-visualization
  
  # Custom brake distance
  python3 mcm_steer_track2_odometry.py --trajectory path.json --brake-distance 10.0
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
                       help='Number of laps to execute (default: infinite)')
    parser.add_argument('--no-visualization', action='store_true',
                       help='Disable OpenCV visualization')
    
    args = parser.parse_args()
    
    if args.brake_distance <= 0:
        print("✗ Brake distance must be positive")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"ODOMETRY PLAYBACK CONTROLLER WITH VISUALIZATION")
    print(f"{'='*80}")
    print(f"CAN2 (MCM commands): {args.mcm}")
    print(f"CAN3 (Sensor feedback): {args.sas}")
    print(f"Trajectory file: {args.trajectory}")
    print(f"Brake initiation: {args.brake_distance}m before target")
    print(f"Visualization: {'Disabled' if args.no_visualization else 'Enabled'}")
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
        brake_start_distance=args.brake_distance,
        enable_visualization=not args.no_visualization
    )
    
    # Setup signal handling
    def cleanup(sig=None, frame=None):
        print("\n\n⚠ Stopping playback...")
        controller.running = False
        if controller.visualizer:
            controller.visualizer.close()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup)
    
    # Run playback loop
    try:
        asyncio.run(controller.run_playback_loop(num_laps=args.laps))
    except KeyboardInterrupt:
        cleanup()

if __name__ == "__main__":
    main()
