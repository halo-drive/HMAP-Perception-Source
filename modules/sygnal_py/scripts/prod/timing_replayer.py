#!/usr/bin/env python3
"""
Pure Time-Based Playback Controller with Speed Matching Visualization
Commands are sent based on ELAPSED TIME only (odometry ignored for indexing)

Features:
- Pure time-based steering command scheduling
- Real-time speed matching display (like rhythm game)
- Visual feedback: TOO FAST / TOO SLOW / PERFECT
- Throttle control guidance for driver
- Speed deviation alerts

Usage:
    python3 time_based_playback.py --trajectory time_trajectory.json
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
import math
import numpy as np
import cv2
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
import threading
from collections import deque

@dataclass
class TimeIndexedWaypoint:
    elapsed_time_s: float
    timestamp: float
    steering_angle_deg: float
    target_speed_mps: float
    cumulative_distance_m: float
    wheel_speeds_kmh: Dict[str, float]
    yaw_rate_deg_s: Optional[float] = None
    iso_time: Optional[str] = None

class SpeedMatchingVisualizer:
    """
    Real-time speed matching visualization
    Shows driver whether they need to speed up or slow down
    """
    
    def __init__(self, waypoints: List[TimeIndexedWaypoint], window_size=(1400, 900)):
        self.waypoints = waypoints
        self.window_size = window_size
        
        # Extract speed profile
        self.time_profile = [wp.elapsed_time_s for wp in waypoints]
        self.speed_profile = [wp.target_speed_mps for wp in waypoints]
        self.distance_profile = [wp.cumulative_distance_m for wp in waypoints]
        
        # Current state
        self.current_time = 0.0
        self.current_speed = 0.0
        self.target_speed = 0.0
        self.speed_error = 0.0
        self.current_distance = 0.0
        self.current_angle = 0.0
        self.waypoint_index = 0
        
        # Speed history
        self.speed_history = deque(maxlen=200)  # 10 seconds at 20Hz
        self.target_history = deque(maxlen=200)
        self.time_history = deque(maxlen=200)
        
        # Thread safety
        self.vis_lock = threading.Lock()
        
        # Initialize window
        cv2.namedWindow('Speed Matching - Time-Based Playback', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Speed Matching - Time-Based Playback', *window_size)
        
        print(f"✓ Speed matching visualization initialized")
    
    def get_target_speed_at_time(self, elapsed_time: float) -> float:
        """Get target speed at given time"""
        if elapsed_time <= 0:
            return self.speed_profile[0]
        if elapsed_time >= self.time_profile[-1]:
            return self.speed_profile[-1]
        
        # Linear interpolation
        for i in range(len(self.time_profile) - 1):
            if self.time_profile[i] <= elapsed_time <= self.time_profile[i+1]:
                t0, t1 = self.time_profile[i], self.time_profile[i+1]
                v0, v1 = self.speed_profile[i], self.speed_profile[i+1]
                alpha = (elapsed_time - t0) / (t1 - t0)
                return v0 + alpha * (v1 - v0)
        
        return self.speed_profile[-1]
    
    def draw_speed_gauge(self, frame: np.ndarray):
        """
        Draw large speed gauge showing current vs target
        Like a tachometer/speedometer
        """
        gauge_center_x = 200
        gauge_center_y = 400
        gauge_radius = 150
        
        # Background circle
        cv2.circle(frame, (gauge_center_x, gauge_center_y), gauge_radius, (40, 40, 40), -1)
        cv2.circle(frame, (gauge_center_x, gauge_center_y), gauge_radius, (100, 100, 100), 3)
        
        # Speed range (0 to 4 m/s typical)
        max_speed = 4.0
        
        # Target speed arc (green)
        target_angle = -180 + (self.target_speed / max_speed) * 180
        cv2.ellipse(frame, (gauge_center_x, gauge_center_y), 
                   (gauge_radius-10, gauge_radius-10), 
                   0, -180, target_angle, (0, 255, 0), 15)
        
        # Current speed arc (blue or red depending on error)
        current_angle = -180 + (self.current_speed / max_speed) * 180
        
        if abs(self.speed_error) < 0.2:  # Within 0.2 m/s
            color = (0, 255, 0)  # Green - PERFECT
        elif abs(self.speed_error) < 0.5:
            color = (0, 255, 255)  # Yellow - OK
        else:
            color = (0, 0, 255)  # Red - BAD
        
        cv2.ellipse(frame, (gauge_center_x, gauge_center_y),
                   (gauge_radius-30, gauge_radius-30),
                   0, -180, current_angle, color, 10)
        
        # Center text
        cv2.putText(frame, f"{self.current_speed:.2f}", 
                   (gauge_center_x-60, gauge_center_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        cv2.putText(frame, "m/s", 
                   (gauge_center_x-30, gauge_center_y+40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Target label
        cv2.putText(frame, f"Target: {self.target_speed:.2f} m/s",
                   (gauge_center_x-80, gauge_center_y+100),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    def draw_speed_deviation_bar(self, frame: np.ndarray):
        """
        Large horizontal bar showing speed error
        Like a rhythm game hit indicator
        """
        bar_x = 450
        bar_y = 400
        bar_width = 800
        bar_height = 80
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height),
                     (40, 40, 40), -1)
        
        # Perfect zone (center)
        perfect_zone_width = 100
        perfect_x = bar_x + bar_width//2 - perfect_zone_width//2
        cv2.rectangle(frame, (perfect_x, bar_y),
                     (perfect_x + perfect_zone_width, bar_y + bar_height),
                     (0, 100, 0), -1)
        
        # Target line (always at center)
        center_x = bar_x + bar_width//2
        cv2.line(frame, (center_x, bar_y), (center_x, bar_y + bar_height),
                (0, 255, 0), 3)
        
        # Current speed indicator
        # Map speed error to bar position
        max_error = 1.0  # m/s
        error_normalized = max(min(self.speed_error / max_error, 1.0), -1.0)
        indicator_x = center_x + int(error_normalized * (bar_width // 2))
        
        # Indicator color based on magnitude
        if abs(self.speed_error) < 0.2:
            indicator_color = (0, 255, 0)  # Green - PERFECT
            status_text = "PERFECT"
        elif abs(self.speed_error) < 0.5:
            indicator_color = (0, 255, 255)  # Yellow - OK
            status_text = "OK"
        else:
            indicator_color = (0, 0, 255)  # Red - ADJUST
            status_text = "ADJUST"
        
        # Draw indicator triangle
        triangle_top = bar_y - 20
        triangle_bottom = bar_y
        triangle_pts = np.array([
            [indicator_x, triangle_bottom],
            [indicator_x - 15, triangle_top],
            [indicator_x + 15, triangle_top]
        ], np.int32)
        cv2.fillPoly(frame, [triangle_pts], indicator_color)
        
        # Draw thick vertical line through bar
        cv2.line(frame, (indicator_x, bar_y), (indicator_x, bar_y + bar_height),
                indicator_color, 5)
        
        # Status text
        cv2.putText(frame, status_text,
                   (bar_x + bar_width + 30, bar_y + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, indicator_color, 3)
        
        # Instructions
        if self.speed_error > 0.2:
            instruction = "SLOW DOWN"
            instruction_color = (0, 100, 255)
        elif self.speed_error < -0.2:
            instruction = "SPEED UP"
            instruction_color = (0, 100, 255)
        else:
            instruction = "MAINTAIN"
            instruction_color = (0, 255, 0)
        
        cv2.putText(frame, instruction,
                   (bar_x + bar_width//2 - 100, bar_y + bar_height + 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, instruction_color, 3)
    
    def draw_speed_graph(self, frame: np.ndarray):
        """
        Line graph showing target vs actual speed over time
        """
        graph_x = 50
        graph_y = 550
        graph_width = 1300
        graph_height = 300
        
        # Background
        cv2.rectangle(frame, (graph_x, graph_y),
                     (graph_x + graph_width, graph_y + graph_height),
                     (30, 30, 30), -1)
        
        # Grid lines
        for i in range(5):
            y = graph_y + (graph_height // 4) * i
            cv2.line(frame, (graph_x, y), (graph_x + graph_width, y),
                    (50, 50, 50), 1)
        
        # Speed axis labels
        max_speed = 4.0
        for i in range(5):
            speed = max_speed * (1 - i/4)
            y = graph_y + (graph_height // 4) * i
            cv2.putText(frame, f"{speed:.1f}",
                       (graph_x - 40, y + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
        
        # Plot target speed (green line)
        if len(self.target_history) > 1:
            points = []
            for i, target in enumerate(self.target_history):
                x = graph_x + int((i / len(self.target_history)) * graph_width)
                y = graph_y + graph_height - int((target / max_speed) * graph_height)
                points.append((x, y))
            
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i+1], (0, 255, 0), 2)
        
        # Plot actual speed (colored line)
        if len(self.speed_history) > 1:
            for i in range(len(self.speed_history) - 1):
                actual = self.speed_history[i]
                target = self.target_history[i] if i < len(self.target_history) else 0
                
                x1 = graph_x + int((i / len(self.speed_history)) * graph_width)
                x2 = graph_x + int(((i+1) / len(self.speed_history)) * graph_width)
                y1 = graph_y + graph_height - int((actual / max_speed) * graph_height)
                y2 = graph_y + graph_height - int((self.speed_history[i+1] / max_speed) * graph_height)
                
                # Color based on error
                error = actual - target
                if abs(error) < 0.2:
                    color = (0, 255, 0)
                elif abs(error) < 0.5:
                    color = (0, 255, 255)
                else:
                    color = (0, 100, 255)
                
                cv2.line(frame, (x1, y1), (x2, y2), color, 3)
        
        # Current position indicator (vertical line)
        cv2.line(frame, (graph_x + graph_width - 50, graph_y),
                (graph_x + graph_width - 50, graph_y + graph_height),
                (255, 255, 255), 2)
        
        # Labels
        cv2.putText(frame, "Speed History (10s window)",
                   (graph_x + 10, graph_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, "Green = Target | Color = Actual",
                   (graph_x + 10, graph_y + graph_height + 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    def draw_status_panel(self, frame: np.ndarray):
        """Draw status information panel"""
        panel_height = 120
        panel = np.zeros((panel_height, self.window_size[0], 3), dtype=np.uint8)
        
        # Title
        cv2.putText(panel, "TIME-BASED TRAJECTORY PLAYBACK", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Time info
        total_time = self.time_profile[-1] if self.time_profile else 0
        cv2.putText(panel, f"Time: {self.current_time:.1f}s / {total_time:.1f}s",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Progress bar
        progress = self.current_time / total_time if total_time > 0 else 0
        bar_x, bar_y = 10, 75
        bar_width, bar_height = 400, 25
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (50, 50, 50), -1)
        cv2.rectangle(panel, (bar_x, bar_y), 
                     (bar_x + int(bar_width * progress), bar_y + bar_height),
                     (100, 255, 100), -1)
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                     (150, 150, 150), 2)
        
        # Waypoint info
        cv2.putText(panel, f"Waypoint: {self.waypoint_index} / {len(self.waypoints)}",
                   (450, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Distance info (metadata)
        total_distance = self.distance_profile[-1] if self.distance_profile else 0
        cv2.putText(panel, f"Distance: {self.current_distance:.1f}m / {total_distance:.1f}m (metadata)",
                   (450, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Steering info
        cv2.putText(panel, f"Steering: {self.current_angle:+.1f}°",
                   (800, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Speed error magnitude
        cv2.putText(panel, f"Speed Error: {self.speed_error:+.2f} m/s",
                   (800, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Large warning if error too high
        if abs(self.speed_error) > 0.8:
            cv2.putText(panel, "WARNING: SPEED MISMATCH",
                       (1000, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
        
        frame[0:panel_height, :] = panel
    
    def update(self, current_time: float, current_speed: float, target_speed: float,
               current_distance: float, current_angle: float, waypoint_index: int):
        """Update visualization state"""
        with self.vis_lock:
            self.current_time = current_time
            self.current_speed = current_speed
            self.target_speed = target_speed
            self.speed_error = current_speed - target_speed
            self.current_distance = current_distance
            self.current_angle = current_angle
            self.waypoint_index = waypoint_index
            
            # Update history
            self.speed_history.append(current_speed)
            self.target_history.append(target_speed)
            self.time_history.append(current_time)
    
    def render(self) -> np.ndarray:
        """Render frame"""
        with self.vis_lock:
            frame = np.zeros((self.window_size[1], self.window_size[0], 3), dtype=np.uint8)
            frame[:] = (20, 20, 20)
            
            self.draw_status_panel(frame)
            self.draw_speed_gauge(frame)
            self.draw_speed_deviation_bar(frame)
            self.draw_speed_graph(frame)
            
            return frame
    
    def show(self):
        """Display frame"""
        frame = self.render()
        cv2.imshow('Speed Matching - Time-Based Playback', frame)
        cv2.waitKey(1)
    
    def close(self):
        """Clean up"""
        cv2.destroyAllWindows()


class TimeBasedPlaybackController:
    """
    Pure time-based playback controller
    Steering commands triggered by ELAPSED TIME only
    """
    
    def __init__(self, mcm_channel='can2', sas_channel='can3',
                 trajectory_file=None, brake_start_time=5.0,
                 enable_visualization=True):
        
        # CAN Configuration
        self.mcm_channel = mcm_channel
        self.sas_channel = sas_channel
        
        # MCM Setup
        self.mcm_db = cantools.database.Database()
        self.mcm_db.add_dbc_file('./sygnal_dbc/mcm/Heartbeat.dbc')
        self.mcm_db.add_dbc_file('./sygnal_dbc/mcm/Control.dbc')
        
        # CAN Connections
        try:
            self.mcm_bus = can.Bus(channel=mcm_channel, bustype='socketcan', bitrate=500000)
            print(f"✓ MCM connected: {mcm_channel}")
        except Exception as e:
            print(f"✗ MCM failed: {e}")
            sys.exit(1)
        
        try:
            self.sas_bus = can.Bus(channel=sas_channel, bustype='socketcan', bitrate=500000)
            print(f"✓ Sensor connected: {sas_channel}")
        except Exception as e:
            print(f"✗ Sensor failed: {e}")
            sys.exit(1)
        
        # MCM State
        self.control_count = 0
        self.bus_address = 1
        self.steering_enabled = False
        self.brake_enabled = False
        
        # CAN IDs
        self.SAS11_CAN_ID = 0x2B0
        self.WHL_SPD_CAN_ID = 0x386
        self.WHEEL_SPEED_SCALE = 0.03125
        self.SAS_ANGLE_SCALE = 0.1
        
        # Odometry State (for visualization only)
        self.cumulative_distance_m = 0.0
        self.vehicle_speed_mps = 0.0
        self.last_update_time = None
        self.latest_angle_deg = None
        
        # Temporal State (PRIMARY)
        self.playback_start_time = None
        self.elapsed_playback_time_s = 0.0
        
        # Brake Parameters
        self.BRAKE_START_TIME = brake_start_time  # seconds before end
        self.MAX_BRAKE_INTENSITY = 0.7
        
        # Trajectory Data
        self.waypoints: List[TimeIndexedWaypoint] = []
        self.trajectory_metadata = {}
        self.trajectory_file_path = trajectory_file
        
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
            'time_errors': [],
            'speed_errors': []
        }
        
        # Load Trajectory
        if trajectory_file:
            self.load_trajectory(trajectory_file)
        
        # Visualization
        self.enable_visualization = enable_visualization
        self.visualizer: Optional[SpeedMatchingVisualizer] = None
        
        if self.enable_visualization and len(self.waypoints) > 0:
            try:
                self.visualizer = SpeedMatchingVisualizer(self.waypoints)
            except Exception as e:
                print(f"⚠ Visualization failed: {e}")
                self.enable_visualization = False
        
        print(f"\n✓ Time-based playback controller initialized")
    
    def load_trajectory(self, filename: str):
        """Load time-indexed trajectory"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            
            for wp_data in data['waypoints']:
                wp = TimeIndexedWaypoint(**wp_data)
                self.waypoints.append(wp)
            
            self.trajectory_metadata = data['metadata']
            
            print(f"\n{'='*80}")
            print(f"TIME-BASED TRAJECTORY LOADED")
            print(f"{'='*80}")
            print(f"File: {filename}")
            print(f"Format: {data.get('format_version', 'unknown')}")
            print(f"Waypoints: {len(self.waypoints)}")
            print(f"Duration: {self.trajectory_metadata['duration_seconds']:.1f}s")
            print(f"Distance (metadata): {self.trajectory_metadata['total_distance_m']:.1f}m")
            print(f"Speed profile: {self.trajectory_metadata['speed_profile']['average_speed_mps']:.2f} m/s avg")
            print(f"\n⚠ IMPORTANT: Control throttle manually to match target speed!")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"✗ Failed to load trajectory: {e}")
            sys.exit(1)
    
    def calc_crc8(self, data: bytes) -> int:
        hash = crc8.crc8()
        hash.update(data[:-1])
        return hash.digest()[0]
    
    def angle_to_percentage(self, angle_deg: float) -> float:
        adjusted_angle = -angle_deg
        percentage = adjusted_angle * (100.0 / 400.0)
        return max(min(percentage, 100.0), -100.0)
    
    def calculate_brake_command(self, remaining_time: float) -> float:
        """Progressive brake based on remaining TIME"""
        if remaining_time >= self.BRAKE_START_TIME:
            return 0.0
        if remaining_time <= 0.0:
            return self.MAX_BRAKE_INTENSITY
        
        brake_progress = 1.0 - (remaining_time / self.BRAKE_START_TIME)
        brake_intensity = self.MAX_BRAKE_INTENSITY * (brake_progress ** 1.5)
        
        return min(max(brake_intensity, 0.0), self.MAX_BRAKE_INTENSITY)
    
    def parse_wheel_speeds(self, data: bytes) -> Optional[Dict[str, float]]:
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
    
    def update_odometry(self, wheel_speeds: Dict[str, float], timestamp: float):
        """Update odometry (for visualization only)"""
        rear_avg_kmh = (wheel_speeds['RL'] + wheel_speeds['RR']) / 2.0
        vehicle_speed_mps = rear_avg_kmh / 3.6
        
        if self.last_update_time is not None:
            dt = timestamp - self.last_update_time
            if 0.001 <= dt <= 1.0:
                distance_increment = vehicle_speed_mps * dt
                with self.data_lock:
                    self.cumulative_distance_m += distance_increment
        
        with self.data_lock:
            self.vehicle_speed_mps = vehicle_speed_mps
        self.last_update_time = timestamp
    
    def get_current_state(self) -> Tuple[float, float, float]:
        """Get elapsed time, distance, speed"""
        with self.data_lock:
            return (self.elapsed_playback_time_s, 
                   self.cumulative_distance_m, 
                   self.vehicle_speed_mps)
    
    async def enable_mcm_interface(self, interface_name: str):
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
            except:
                if self.running:
                    time.sleep(0.1)
    
    async def execute_trajectory_playback(self):
        """Main playback execution - PURE TIME BASED"""
        print(f"\n{'='*80}")
        print(f"TIME-BASED PLAYBACK EXECUTION")
        print(f"{'='*80}")
        print(f"Duration: {self.trajectory_metadata['duration_seconds']:.1f}s")
        print(f"Steering: Automatic (time-triggered)")
        print(f"Throttle: MANUAL (match speed display)")
        print(f"{'='*80}\n")
        
        self.playback_start_time = time.time()
        self.trajectory_index = 0
        
        # Reset odometry (visualization only)
        with self.data_lock:
            self.cumulative_distance_m = 0.0
            self.last_update_time = None
        
        last_status_time = time.time()
        
        while self.running and not self.trajectory_complete:
            
            # Update elapsed time (PRIMARY INDEX)
            self.elapsed_playback_time_s = time.time() - self.playback_start_time
            
            # Get current state
            elapsed_time, current_distance, current_speed = self.get_current_state()
            
            # Get target speed at current time
            target_speed = self.visualizer.get_target_speed_at_time(elapsed_time) if self.visualizer else 0.0
            speed_error = current_speed - target_speed
            
            # STEERING LOGIC: Pure time-based indexing
            while (self.trajectory_index < len(self.waypoints) and
                   elapsed_time >= self.waypoints[self.trajectory_index].elapsed_time_s):
                
                waypoint = self.waypoints[self.trajectory_index]
                
                # Send steering command
                command_pct = self.angle_to_percentage(waypoint.steering_angle_deg)
                await self.send_steering_command(command_pct)
                
                # Log time error
                time_error = elapsed_time - waypoint.elapsed_time_s
                self.stats['time_errors'].append(abs(time_error))
                self.stats['speed_errors'].append(abs(speed_error))
                
                self.trajectory_index += 1
            
            # BRAKING LOGIC: Time-based
            total_time = self.trajectory_metadata['duration_seconds']
            remaining_time = total_time - elapsed_time
            brake_command = self.calculate_brake_command(remaining_time)
            
            if brake_command > 0.0:
                if not self.braking_active:
                    print(f"\n⚠ Braking initiated (t={elapsed_time:.1f}s)")
                    self.braking_active = True
                await self.send_brake_command(brake_command * 100.0)
            
            # Check completion
            if elapsed_time >= total_time:
                print(f"\n✓ Trajectory complete")
                await self.send_brake_command(self.MAX_BRAKE_INTENSITY * 100.0)
                self.trajectory_complete = True
                break
            
            # VISUALIZATION UPDATE
            if self.visualizer and time.time() - last_status_time >= 0.05:
                self.visualizer.update(
                    current_time=elapsed_time,
                    current_speed=current_speed,
                    target_speed=target_speed,
                    current_distance=current_distance,
                    current_angle=self.latest_angle_deg or 0.0,
                    waypoint_index=self.trajectory_index
                )
                self.visualizer.show()
                last_status_time = time.time()
            
            # Control loop
            await asyncio.sleep(0.05)
        
        # Hold brake
        for _ in range(40):
            await self.send_brake_command(self.MAX_BRAKE_INTENSITY * 100.0)
            if self.visualizer:
                self.visualizer.show()
            await asyncio.sleep(0.05)
        
        await self.send_brake_command(0.0)
        
        # Final stats
        print(f"\n{'='*80}")
        print(f"PLAYBACK COMPLETE")
        print(f"{'='*80}")
        print(f"Target time: {self.trajectory_metadata['duration_seconds']:.1f}s")
        print(f"Actual time: {elapsed_time:.1f}s")
        print(f"Commands sent: {self.stats['commands_sent']}")
        
        if self.stats['time_errors']:
            print(f"Mean time error: {sum(self.stats['time_errors'])/len(self.stats['time_errors']):.3f}s")
        if self.stats['speed_errors']:
            print(f"Mean speed error: {sum(self.stats['speed_errors'])/len(self.stats['speed_errors']):.3f}m/s")
        
        print(f"{'='*80}\n")
    
    async def maintain_mcm_heartbeat(self):
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
        self.running = True
        
        # Start sensor thread
        sensor_thread = threading.Thread(target=self.monitor_sensors, daemon=True)
        sensor_thread.start()
        
        # Start heartbeat
        heartbeat_task = asyncio.create_task(self.maintain_mcm_heartbeat())
        
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
                    self.last_update_time = None
                
                self.trajectory_index = 0
                self.braking_active = False
                self.trajectory_complete = False
                
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
    
    parser = argparse.ArgumentParser(description='Pure Time-Based Playback Controller')
    parser.add_argument('--trajectory', '-t', required=True, help='Time trajectory JSON')
    parser.add_argument('--mcm', default='can2', help='MCM CAN interface')
    parser.add_argument('--sas', default='can3', help='Sensor CAN interface')
    parser.add_argument('--brake-time', type=float, default=5.0,
                       help='Brake start time before end (seconds)')
    parser.add_argument('--laps', type=int, default=None, help='Number of laps')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualization')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f"TIME-BASED PLAYBACK CONTROLLER")
    print(f"{'='*80}")
    print(f"Trajectory: {args.trajectory}")
    print(f"Indexing: Pure TIME (distance ignored)")
    print(f"Driver task: Control throttle to match speed display")
    print(f"{'='*80}\n")
    
    controller = TimeBasedPlaybackController(
        mcm_channel=args.mcm,
        sas_channel=args.sas,
        trajectory_file=args.trajectory,
        brake_start_time=args.brake_time,
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
