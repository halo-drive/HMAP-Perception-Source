#!/usr/bin/env python3
"""
GPS Path Visualization using OpenCV

Provides real-time visualization for:
- Recording mode: Shows waypoints being added and trajectory
- Follow mode: Shows reference path vs actual path comparison
"""

import cv2
import numpy as np
import math
from collections import deque


class GPSVisualizer:
    """Real-time GPS path visualization using OpenCV."""
    
    def __init__(self, window_name="GPS Path Tracker", width=1200, height=800):
        """
        Initialize the visualizer.
        
        Args:
            window_name: Name of the OpenCV window
            width: Window width in pixels
            height: Window height in pixels
        """
        self.window_name = window_name
        self.width = width
        self.height = height
        
        # Map bounds (will be auto-adjusted)
        self.min_lat = None
        self.max_lat = None
        self.min_lon = None
        self.max_lon = None
        
        # Margin around the map (pixels)
        self.margin = 50
        
        # Drawing area (excluding margin)
        self.draw_width = width - 2 * self.margin
        self.draw_height = height - 2 * self.margin
        
        # Colors (BGR format for OpenCV)
        self.COLOR_BACKGROUND = (30, 30, 30)       # Dark gray
        self.COLOR_GRID = (60, 60, 60)             # Lighter gray
        self.COLOR_REFERENCE = (0, 255, 255)       # Yellow - reference path
        self.COLOR_CURRENT = (0, 255, 0)           # Green - current path
        self.COLOR_WAYPOINT = (255, 255, 255)      # White - waypoints
        self.COLOR_POSITION = (0, 0, 255)          # Red - current position
        self.COLOR_START = (255, 0, 255)           # Magenta - start point
        self.COLOR_TEXT = (255, 255, 255)          # White - text
        self.COLOR_ERROR_HIGH = (0, 0, 255)        # Red - high error
        self.COLOR_ERROR_MED = (0, 165, 255)       # Orange - medium error
        self.COLOR_ERROR_LOW = (0, 255, 0)         # Green - low error
        
        # Path storage
        self.reference_points = []  # (lat, lon) tuples for reference
        self.current_points = deque(maxlen=500)  # Recent actual positions (for follow mode)
        self.recorded_points = []  # For recording mode
        
        # Current state
        self.current_lat = None
        self.current_lon = None
        self.current_heading = None
        self.current_speed = None
        
        # Error metrics (for follow mode)
        self.position_error = None
        self.cross_track_error = None
        self.heading_error = None
        
        # Statistics
        self.total_points = 0
        self.distance_traveled = 0.0
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, width, height)
        
        print(f"[viz] OpenCV visualization initialized ({width}x{height})")
    
    def set_reference_path(self, path_points):
        """
        Load reference path from recorded data.
        
        Args:
            path_points: List of dicts with 'lat' and 'lon' keys
        """
        self.reference_points = [(p['lat'], p['lon']) for p in path_points]
        self._update_bounds(self.reference_points)
        print(f"[viz] Loaded {len(self.reference_points)} reference points")
    
    def add_recorded_point(self, lat, lon):
        """
        Add a point during recording mode.
        
        Args:
            lat: Latitude
            lon: Longitude
        """
        self.recorded_points.append((lat, lon))
        self._update_bounds([(lat, lon)])
        self.total_points += 1
    
    def update_current_position(self, lat, lon, heading=None, speed=None):
        """
        Update current vehicle position.
        
        Args:
            lat: Current latitude
            lon: Current longitude
            heading: Current heading in degrees (optional)
            speed: Current speed in m/s (optional)
        """
        self.current_lat = lat
        self.current_lon = lon
        self.current_heading = heading
        self.current_speed = speed
        
        # Add to current path trail (for follow mode)
        if lat and lon:
            self.current_points.append((lat, lon))
            self._update_bounds([(lat, lon)])
    
    def update_errors(self, pos_err, xte, hdg_err):
        """
        Update error metrics for follow mode.
        
        Args:
            pos_err: Position error in meters
            xte: Cross-track error in meters
            hdg_err: Heading error in degrees
        """
        self.position_error = pos_err
        self.cross_track_error = xte
        self.heading_error = hdg_err
    
    def _update_bounds(self, points):
        """Update map bounds to include new points."""
        if not points:
            return
        
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        
        print(f"[VIZ] Updating bounds with points: {points}")
        
        if self.min_lat is None:
            self.min_lat = min(lats)
            self.max_lat = max(lats)
            self.min_lon = min(lons)
            self.max_lon = max(lons)
        else:
            self.min_lat = min(self.min_lat, min(lats))
            self.max_lat = max(self.max_lat, max(lats))
            self.min_lon = min(self.min_lon, min(lons))
            self.max_lon = max(self.max_lon, max(lons))
        
        # Add 10% padding
        lat_range = self.max_lat - self.min_lat
        lon_range = self.max_lon - self.min_lon
        
        if lat_range > 0:
            self.min_lat -= lat_range * 0.1
            self.max_lat += lat_range * 0.1
        else:
            self.min_lat -= 0.0001
            self.max_lat += 0.0001
            
        if lon_range > 0:
            self.min_lon -= lon_range * 0.1
            self.max_lon += lon_range * 0.1
        else:
            self.min_lon -= 0.0001
            self.max_lon += 0.0001
        
        print(f"[VIZ] Updated bounds: lat=[{self.min_lat}, {self.max_lat}], lon=[{self.min_lon}, {self.max_lon}]")
    
    def _gps_to_pixel(self, lat, lon):
        """
        Convert GPS coordinates to pixel coordinates.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            (x, y) pixel coordinates
        """
        if self.min_lat is None or self.max_lat is None:
            return (self.width // 2, self.height // 2)
        
        # Normalize to 0-1 range
        if self.max_lat - self.min_lat > 0:
            norm_lat = (lat - self.min_lat) / (self.max_lat - self.min_lat)
        else:
            norm_lat = 0.5
            
        if self.max_lon - self.min_lon > 0:
            norm_lon = (lon - self.min_lon) / (self.max_lon - self.min_lon)
        else:
            norm_lon = 0.5
        
        # Convert to pixel coordinates (flip Y axis - higher lat = lower Y)
        x = int(self.margin + norm_lon * self.draw_width)
        y = int(self.margin + (1 - norm_lat) * self.draw_height)
        
        return (x, y)
    
    def _draw_grid(self, img):
        """Draw coordinate grid on the map."""
        # Draw grid lines
        grid_spacing = 100  # pixels
        
        # Vertical lines
        for x in range(self.margin, self.width - self.margin, grid_spacing):
            cv2.line(img, (x, self.margin), (x, self.height - self.margin), 
                    self.COLOR_GRID, 1)
        
        # Horizontal lines
        for y in range(self.margin, self.height - self.margin, grid_spacing):
            cv2.line(img, (self.margin, y), (self.width - self.margin, y), 
                    self.COLOR_GRID, 1)
        
        # Draw border
        cv2.rectangle(img, (self.margin, self.margin), 
                     (self.width - self.margin, self.height - self.margin),
                     self.COLOR_GRID, 2)
    
    def _draw_path(self, img, points, color, thickness=2, draw_points=False):
        """
        Draw a path on the map.
        
        Args:
            img: OpenCV image
            points: List of (lat, lon) tuples
            color: BGR color tuple
            thickness: Line thickness
            draw_points: Whether to draw individual points
        """
        if len(points) < 2:
            return
        
        # Convert all points to pixels
        pixel_points = [self._gps_to_pixel(lat, lon) for lat, lon in points]
        
        # Draw lines connecting points
        for i in range(len(pixel_points) - 1):
            cv2.line(img, pixel_points[i], pixel_points[i + 1], color, thickness)
        
        # Optionally draw individual waypoints
        if draw_points:
            for px, py in pixel_points:
                cv2.circle(img, (px, py), 3, color, -1)
    
    def _draw_vehicle(self, img, lat, lon, heading=None, color=None):
        """
        Draw vehicle position with optional heading indicator.
        
        Args:
            img: OpenCV image
            lat: Latitude
            lon: Longitude
            heading: Heading in degrees (optional)
            color: Color override (optional)
        """
        if lat is None or lon is None:
            return
        
        px, py = self._gps_to_pixel(lat, lon)
        
        if color is None:
            color = self.COLOR_POSITION
        
        # Draw position circle
        cv2.circle(img, (px, py), 8, color, -1)
        cv2.circle(img, (px, py), 10, (255, 255, 255), 2)
        
        # Draw heading arrow if available
        if heading is not None:
            arrow_length = 30
            angle_rad = math.radians(90 - heading)  # Convert to math coords
            end_x = int(px + arrow_length * math.cos(angle_rad))
            end_y = int(py - arrow_length * math.sin(angle_rad))
            
            cv2.arrowedLine(img, (px, py), (end_x, end_y), color, 3, tipLength=0.3)
    
    def _draw_stats_panel(self, img, mode="record"):
        """
        Draw statistics and information panel.
        
        Args:
            img: OpenCV image
            mode: "record" or "follow"
        """
        panel_height = 180
        panel_color = (40, 40, 40)
        
        # Draw semi-transparent panel
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (400, panel_height), panel_color, -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        
        # Title
        title = "GPS PATH RECORDER" if mode == "record" else "GPS PATH FOLLOWER"
        cv2.putText(img, title, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   self.COLOR_TEXT, 2)
        
        y_offset = 60
        line_height = 25
        
        if mode == "record":
            # Recording mode stats
            stats = [
                f"Points: {len(self.recorded_points)}",
                f"Position: {self.current_lat:.6f}, {self.current_lon:.6f}" if self.current_lat else "Position: Waiting for GPS...",
                f"Heading: {self.current_heading:.1f}" if self.current_heading else "Heading: N/A",
                f"Speed: {self.current_speed:.1f} m/s" if self.current_speed else "Speed: N/A",
            ]
        else:
            # Follow mode stats
            # Determine error color
            if self.position_error is not None:
                if self.position_error < 3:
                    err_color = self.COLOR_ERROR_LOW
                elif self.position_error < 8:
                    err_color = self.COLOR_ERROR_MED
                else:
                    err_color = self.COLOR_ERROR_HIGH
            else:
                err_color = self.COLOR_TEXT
            
            stats = [
                f"Ref Points: {len(self.reference_points)}",
                f"Position: {self.current_lat:.6f}, {self.current_lon:.6f}" if self.current_lat else "Position: Waiting...",
                f"Pos Error: {self.position_error:.1f} m" if self.position_error is not None else "Pos Error: N/A",
                f"XTE: {self.cross_track_error:.1f} m" if self.cross_track_error is not None else "XTE: N/A",
                f"Hdg Error: {self.heading_error:.1f}" if self.heading_error is not None else "Hdg Error: N/A",
            ]
        
        # Draw stats
        for i, stat in enumerate(stats):
            y = y_offset + i * line_height
            # Use error color for position error line in follow mode
            if mode == "follow" and i == 2 and self.position_error is not None:
                cv2.putText(img, stat, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           err_color, 1)
            else:
                cv2.putText(img, stat, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           self.COLOR_TEXT, 1)
    
    def _draw_legend(self, img, mode="record"):
        """Draw color legend."""
        legend_x = self.width - 250
        legend_y = 20
        
        if mode == "record":
            items = [
                ("Recording Path", self.COLOR_CURRENT),
                ("Current Position", self.COLOR_POSITION),
                ("Start Point", self.COLOR_START),
            ]
        else:
            items = [
                ("Reference Path", self.COLOR_REFERENCE),
                ("Actual Path", self.COLOR_CURRENT),
                ("Current Position", self.COLOR_POSITION),
            ]
        
        for i, (label, color) in enumerate(items):
            y = legend_y + i * 30
            
            # Draw color box
            cv2.rectangle(img, (legend_x, y), (legend_x + 20, y + 15), color, -1)
            cv2.rectangle(img, (legend_x, y), (legend_x + 20, y + 15), 
                         self.COLOR_TEXT, 1)
            
            # Draw label
            cv2.putText(img, label, (legend_x + 30, y + 12), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_TEXT, 1)
    
    def draw_recording_mode(self):
        """
        Draw visualization for recording mode.
        
        Returns:
            OpenCV image
        """
        # Create blank image
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = self.COLOR_BACKGROUND
        
        # Draw grid
        self._draw_grid(img)
        
        # Draw recorded path
        if self.recorded_points:
            self._draw_path(img, self.recorded_points, self.COLOR_CURRENT, 
                          thickness=3, draw_points=True)
            
            # Highlight start point
            if len(self.recorded_points) > 0:
                start_lat, start_lon = self.recorded_points[0]
                px, py = self._gps_to_pixel(start_lat, start_lon)
                cv2.circle(img, (px, py), 12, self.COLOR_START, 3)
                cv2.putText(img, "START", (px - 25, py - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_START, 2)
        
        # Draw current position
        if self.current_lat and self.current_lon:
            self._draw_vehicle(img, self.current_lat, self.current_lon, 
                             self.current_heading)
        
        # Draw stats panel
        self._draw_stats_panel(img, mode="record")
        
        # Draw legend
        self._draw_legend(img, mode="record")
        
        return img
    
    def draw_follow_mode(self):
        """
        Draw visualization for follow mode.
        
        Returns:
            OpenCV image
        """
        # Create blank image
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        img[:] = self.COLOR_BACKGROUND
        
        # Draw grid
        self._draw_grid(img)
        
        # Draw reference path (thinner, semi-transparent)
        if self.reference_points:
            # Draw reference path
            self._draw_path(img, self.reference_points, self.COLOR_REFERENCE, 
                          thickness=2, draw_points=False)
            
            # Mark waypoints along reference path (every 10th point)
            for i in range(0, len(self.reference_points), 10):
                px, py = self._gps_to_pixel(self.reference_points[i][0], 
                                           self.reference_points[i][1])
                cv2.circle(img, (px, py), 2, self.COLOR_REFERENCE, -1)
            
            # Highlight start point
            start_lat, start_lon = self.reference_points[0]
            px, py = self._gps_to_pixel(start_lat, start_lon)
            cv2.circle(img, (px, py), 10, self.COLOR_START, 2)
        
        # Draw actual driven path (thicker, bright)
        if len(self.current_points) > 1:
            self._draw_path(img, list(self.current_points), self.COLOR_CURRENT, 
                          thickness=3, draw_points=False)
        
        # Draw current position
        if self.current_lat and self.current_lon:
            self._draw_vehicle(img, self.current_lat, self.current_lon, 
                             self.current_heading)
        
        # Draw stats panel
        self._draw_stats_panel(img, mode="follow")
        
        # Draw legend
        self._draw_legend(img, mode="follow")
        
        return img
    
    def show(self, mode="record", wait_key=1):
        """
        Display the visualization.
        
        Args:
            mode: "record" or "follow"
            wait_key: Wait time in ms for cv2.waitKey
            
        Returns:
            Key pressed (or -1 if timeout), or 27 if window was closed
        """
        if mode == "record":
            img = self.draw_recording_mode()
        else:
            img = self.draw_follow_mode()
        
        cv2.imshow(self.window_name, img)
        
        # Check if window is still open
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            return 27  # ESC key to signal window was closed
        
        key = cv2.waitKey(wait_key) & 0xFF
        
        # Double check if window was closed during waitKey
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            return 27  # ESC key to signal window was closed
            
        return key
    
    def is_window_open(self):
        """Check if the visualization window is still open."""
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) > 0
        except:
            return False
    
    def cleanup(self):
        """Close all OpenCV windows."""
        cv2.destroyAllWindows()
        print("[viz] Visualization closed")


# Example usage and testing
if __name__ == "__main__":
    import time
    import random
    
    print("Testing GPS Visualizer...")
    
    # Test recording mode
    viz = GPSVisualizer()
    
    # Simulate recording a circular path
    center_lat = 40.7128
    center_lon = -74.0060
    radius = 0.001  # ~100m
    
    print("\nSimulating RECORDING mode (circular path)...")
    for i in range(100):
        angle = (i / 100.0) * 2 * math.pi
        lat = center_lat + radius * math.cos(angle)
        lon = center_lon + radius * math.sin(angle)
        heading = (angle * 180 / math.pi + 90) % 360
        
        viz.add_recorded_point(lat, lon)
        viz.update_current_position(lat, lon, heading, 10.0)
        
        key = viz.show(mode="record", wait_key=50)
        if key == 27:  # ESC
            break
    
    print("Recording simulation complete. Press any key to continue...")
    cv2.waitKey(0)
    
    # Test follow mode
    print("\nSimulating FOLLOW mode...")
    
    # Set reference path from recorded points
    ref_path = [{'lat': p[0], 'lon': p[1]} for p in viz.recorded_points]
    viz.set_reference_path(ref_path)
    
    # Reset current position tracking
    viz.current_points.clear()
    
    # Simulate following with some error
    for i in range(200):
        angle = (i / 100.0) * 2 * math.pi
        # Add some random deviation
        error = random.gauss(0, 0.0001)
        lat = center_lat + radius * math.cos(angle) + error
        lon = center_lon + radius * math.sin(angle) + error
        heading = (angle * 180 / math.pi + 90) % 360
        
        # Simulate errors
        pos_err = random.uniform(0, 5)
        xte = random.gauss(0, 2)
        hdg_err = random.gauss(0, 10)
        
        viz.update_current_position(lat, lon, heading, 10.0)
        viz.update_errors(pos_err, xte, hdg_err)
        
        key = viz.show(mode="follow", wait_key=50)
        if key == 27:  # ESC
            break
    
    print("Follow simulation complete. Press any key to exit...")
    cv2.waitKey(0)
    
    viz.cleanup()