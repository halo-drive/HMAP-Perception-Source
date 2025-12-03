#!/usr/bin/env python3
"""
ESP12 Vehicle Motion Sensor Monitor
CAN ID: 0x220 (544 decimal)
Electronic Stability Program Motion Sensors

Decodes critical vehicle dynamics parameters:
- Lateral Acceleration (±10.24 m/s²)
- Longitudinal Acceleration (±10.24 m/s²) 
- Yaw Rate (±40.96 deg/s)
- Brake Cylinder Pressure (0-409.5 Bar)

DBC Specification Compliance with comprehensive validation
"""

import can
import struct
import time
import argparse
import sys
import json
import csv
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

@dataclass
class ESP12_Signals:
    """ESP12 signal data structure with validation flags"""
    # Motion sensors
    lateral_accel_ms2: float              # m/s² (-10.23 to 10.24)
    lateral_accel_valid: bool             # Status flag
    lateral_accel_diag: bool              # Diagnostic flag
    
    longitudinal_accel_ms2: float         # m/s² (-10.23 to 10.24)
    longitudinal_accel_valid: bool        # Status flag  
    longitudinal_accel_diag: bool         # Diagnostic flag
    
    yaw_rate_degs: float                  # deg/s (-40.95 to 40.96)
    yaw_rate_valid: bool                  # Status flag
    yaw_rate_diag: bool                   # Diagnostic flag
    
    # Brake system
    brake_cylinder_pressure_bar: float    # Bar (0.0 to 409.5)
    brake_pressure_valid: bool            # Status flag
    brake_pressure_diag: bool             # Diagnostic flag
    
    # Message integrity
    checksum: int                         # 4-bit checksum
    alive_counter: int                    # 4-bit rolling counter
    
    # Metadata
    timestamp: float
    can_id: int
    raw_data: bytes

class ESP12_MotionMonitor:
    def __init__(self, interface='can1', log_to_file=True, csv_logging=True):
        self.interface = interface
        self.message_count = 0
        self.start_time = time.time()
        self.log_to_file = log_to_file
        self.csv_logging = csv_logging
        
        # ESP12 specific constants
        self.ESP12_CAN_ID = 0x220
        self.ACCEL_SCALE = 0.01          # m/s² per LSB
        self.ACCEL_OFFSET = -10.23       # m/s² offset
        self.YAW_SCALE = 0.01            # deg/s per LSB  
        self.YAW_OFFSET = -40.95         # deg/s offset
        self.PRESSURE_SCALE = 0.1        # Bar per LSB
        
        # Validation ranges (engineering limits)
        self.ACCEL_MIN = -12.0           # m/s² (beyond DBC for validation)
        self.ACCEL_MAX = 12.0            # m/s²
        self.YAW_MIN = -50.0             # deg/s
        self.YAW_MAX = 50.0              # deg/s
        self.PRESSURE_MAX = 450.0        # Bar
        
        # Statistics tracking
        self.stats = {
            'total_messages': 0,
            'valid_messages': 0,
            'invalid_messages': 0,
            'checksum_errors': 0,
            'out_of_range_errors': 0,
            'max_lateral_accel': 0.0,
            'max_longitudinal_accel': 0.0,
            'max_yaw_rate': 0.0,
            'max_brake_pressure': 0.0
        }
        
        # Initialize logging
        self.setup_logging()
        
        # Connect to CAN bus
        try:
            self.bus = can.interface.Bus(channel=interface, bustype='socketcan')
            print(f"✓ Connected to {interface}")
            self.log_message(f"ESP12 Monitor started on {interface}")
        except Exception as e:
            print(f"✗ Failed to connect to {interface}: {e}")
            sys.exit(1)
    
    def setup_logging(self):
        """Initialize file and CSV logging"""
        if self.log_to_file or self.csv_logging:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        if self.log_to_file:
            self.log_filename = f"esp12_motion_log_{timestamp}.txt"
            with open(self.log_filename, 'w') as f:
                f.write(f"ESP12 Motion Sensor Monitor Log\n")
                f.write(f"Started: {datetime.now().isoformat()}\n")
                f.write(f"Interface: {self.interface}\n")
                f.write("=" * 80 + "\n\n")
        
        if self.csv_logging:
            self.csv_filename = f"esp12_motion_data_{timestamp}.csv"
            with open(self.csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'can_id', 'message_count',
                    'lateral_accel_ms2', 'lateral_accel_valid', 'lateral_accel_diag',
                    'longitudinal_accel_ms2', 'longitudinal_accel_valid', 'longitudinal_accel_diag', 
                    'yaw_rate_degs', 'yaw_rate_valid', 'yaw_rate_diag',
                    'brake_pressure_bar', 'brake_pressure_valid', 'brake_pressure_diag',
                    'checksum', 'alive_counter', 'raw_data_hex'
                ])
    
    def log_message(self, message: str):
        """Log message to file with timestamp"""
        if self.log_to_file:
            with open(self.log_filename, 'a') as f:
                f.write(f"[{datetime.now().isoformat()}] {message}\n")
    
    def parse_esp12_signals(self, data: bytes, timestamp: float) -> Optional[ESP12_Signals]:
        """
        Parse ESP12 CAN message according to DBC specification
        
        ESP12 Signal Layout (64-bit little-endian):
        - LAT_ACCEL: bits [10:0] - 11-bit signed, scale=0.01, offset=-10.23
        - LAT_ACCEL_STAT: bit [11] - status flag
        - LAT_ACCEL_DIAG: bit [12] - diagnostic flag
        - LONG_ACCEL: bits [23:13] - 11-bit signed, scale=0.01, offset=-10.23
        - LONG_ACCEL_STAT: bit [24] - status flag
        - LONG_ACCEL_DIAG: bit [25] - diagnostic flag
        - CYL_PRES: bits [37:26] - 12-bit unsigned, scale=0.1, offset=0.0
        - CYL_PRES_STAT: bit [38] - status flag
        - CYL_PRESS_DIAG: bit [39] - diagnostic flag
        - YAW_RATE: bits [52:40] - 13-bit signed, scale=0.01, offset=-40.95
        - YAW_RATE_STAT: bit [53] - status flag
        - YAW_RATE_DIAG: bit [54] - diagnostic flag
        - ESP12_Checksum: bits [59:56] - 4-bit checksum
        - ESP12_AliveCounter: bits [63:60] - 4-bit counter
        """
        
        if len(data) != 8:
            self.log_message(f"Invalid message length: {len(data)} bytes")
            return None
        
        try:
            # Unpack as little-endian 64-bit unsigned integer
            frame_uint64 = struct.unpack('<Q', data)[0]
            
            # Extract lateral acceleration (11-bit signed)
            lat_accel_raw = (frame_uint64 >> 0) & 0x7FF
            if lat_accel_raw & 0x400:  # Sign extend
                lat_accel_raw |= 0xFFFFF800
                lat_accel_raw = struct.unpack('<i', struct.pack('<I', lat_accel_raw & 0xFFFFFFFF))[0]
            lateral_accel = lat_accel_raw * self.ACCEL_SCALE + self.ACCEL_OFFSET
            lateral_accel_stat = bool((frame_uint64 >> 11) & 0x1)
            lateral_accel_diag = bool((frame_uint64 >> 12) & 0x1)
            
            # Extract longitudinal acceleration (11-bit signed)
            long_accel_raw = (frame_uint64 >> 13) & 0x7FF
            if long_accel_raw & 0x400:  # Sign extend
                long_accel_raw |= 0xFFFFF800
                long_accel_raw = struct.unpack('<i', struct.pack('<I', long_accel_raw & 0xFFFFFFFF))[0]
            longitudinal_accel = long_accel_raw * self.ACCEL_SCALE + self.ACCEL_OFFSET
            longitudinal_accel_stat = bool((frame_uint64 >> 24) & 0x1)
            longitudinal_accel_diag = bool((frame_uint64 >> 25) & 0x1)
            
            # Extract brake cylinder pressure (12-bit unsigned)
            pressure_raw = (frame_uint64 >> 26) & 0xFFF
            brake_pressure = pressure_raw * self.PRESSURE_SCALE
            brake_pressure_stat = bool((frame_uint64 >> 38) & 0x1)
            brake_pressure_diag = bool((frame_uint64 >> 39) & 0x1)
            
            # Extract yaw rate (13-bit signed)
            yaw_raw = (frame_uint64 >> 40) & 0x1FFF
            if yaw_raw & 0x1000:  # Sign extend
                yaw_raw |= 0xFFFFE000
                yaw_raw = struct.unpack('<i', struct.pack('<I', yaw_raw & 0xFFFFFFFF))[0]
            yaw_rate = yaw_raw * self.YAW_SCALE + self.YAW_OFFSET
            yaw_rate_stat = bool((frame_uint64 >> 53) & 0x1)
            yaw_rate_diag = bool((frame_uint64 >> 54) & 0x1)
            
            # Extract message integrity fields
            checksum = (frame_uint64 >> 56) & 0xF
            alive_counter = (frame_uint64 >> 60) & 0xF
            
            # Create signal structure
            signals = ESP12_Signals(
                lateral_accel_ms2=lateral_accel,
                lateral_accel_valid=lateral_accel_stat,
                lateral_accel_diag=lateral_accel_diag,
                longitudinal_accel_ms2=longitudinal_accel,
                longitudinal_accel_valid=longitudinal_accel_stat,
                longitudinal_accel_diag=longitudinal_accel_diag,
                yaw_rate_degs=yaw_rate,
                yaw_rate_valid=yaw_rate_stat,
                yaw_rate_diag=yaw_rate_diag,
                brake_cylinder_pressure_bar=brake_pressure,
                brake_pressure_valid=brake_pressure_stat,
                brake_pressure_diag=brake_pressure_diag,
                checksum=checksum,
                alive_counter=alive_counter,
                timestamp=timestamp,
                can_id=self.ESP12_CAN_ID,
                raw_data=data
            )
            
            # Validate signal ranges
            validation_ok = self.validate_signals(signals)
            
            if validation_ok:
                self.stats['valid_messages'] += 1
                self.update_statistics(signals)
            else:
                self.stats['invalid_messages'] += 1
            
            return signals
            
        except Exception as e:
            self.log_message(f"ESP12 parsing error: {e}")
            self.stats['invalid_messages'] += 1
            return None
    
    def validate_signals(self, signals: ESP12_Signals) -> bool:
        """Validate signal values against engineering limits"""
        validation_errors = []
        
        # Check acceleration ranges
        if not (self.ACCEL_MIN <= signals.lateral_accel_ms2 <= self.ACCEL_MAX):
            validation_errors.append(f"Lateral accel out of range: {signals.lateral_accel_ms2:.3f}")
            self.stats['out_of_range_errors'] += 1
            
        if not (self.ACCEL_MIN <= signals.longitudinal_accel_ms2 <= self.ACCEL_MAX):
            validation_errors.append(f"Longitudinal accel out of range: {signals.longitudinal_accel_ms2:.3f}")
            self.stats['out_of_range_errors'] += 1
            
        # Check yaw rate range
        if not (self.YAW_MIN <= signals.yaw_rate_degs <= self.YAW_MAX):
            validation_errors.append(f"Yaw rate out of range: {signals.yaw_rate_degs:.3f}")
            self.stats['out_of_range_errors'] += 1
            
        # Check brake pressure range
        if not (0 <= signals.brake_cylinder_pressure_bar <= self.PRESSURE_MAX):
            validation_errors.append(f"Brake pressure out of range: {signals.brake_cylinder_pressure_bar:.3f}")
            self.stats['out_of_range_errors'] += 1
        
        if validation_errors:
            self.log_message(f"Validation errors: {', '.join(validation_errors)}")
            return False
            
        return True
    
    def update_statistics(self, signals: ESP12_Signals):
        """Update running statistics"""
        self.stats['max_lateral_accel'] = max(self.stats['max_lateral_accel'], 
                                             abs(signals.lateral_accel_ms2))
        self.stats['max_longitudinal_accel'] = max(self.stats['max_longitudinal_accel'], 
                                                  abs(signals.longitudinal_accel_ms2))
        self.stats['max_yaw_rate'] = max(self.stats['max_yaw_rate'], 
                                        abs(signals.yaw_rate_degs))
        self.stats['max_brake_pressure'] = max(self.stats['max_brake_pressure'], 
                                              signals.brake_cylinder_pressure_bar)
    
    def log_to_csv(self, signals: ESP12_Signals):
        """Log signal data to CSV file"""
        if self.csv_logging:
            with open(self.csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    signals.timestamp, signals.can_id, self.message_count,
                    signals.lateral_accel_ms2, signals.lateral_accel_valid, signals.lateral_accel_diag,
                    signals.longitudinal_accel_ms2, signals.longitudinal_accel_valid, signals.longitudinal_accel_diag,
                    signals.yaw_rate_degs, signals.yaw_rate_valid, signals.yaw_rate_diag,
                    signals.brake_cylinder_pressure_bar, signals.brake_pressure_valid, signals.brake_pressure_diag,
                    signals.checksum, signals.alive_counter, signals.raw_data.hex().upper()
                ])
    
    def display_motion_data(self, signals: ESP12_Signals):
        """Display comprehensive motion sensor data"""
        
        print(f"\033[2J\033[H")  # Clear screen
        print("=" * 90)
        print(f"ESP12 VEHICLE MOTION SENSOR MONITOR - {self.interface}")
        print("=" * 90)
        
        print(f"CAN ID: 0x{signals.can_id:03X} | Raw Data: {signals.raw_data.hex().upper()}")
        print(f"Messages: {self.message_count} | Runtime: {time.time() - self.start_time:.1f}s")
        print(f"Checksum: {signals.checksum} | Alive Counter: {signals.alive_counter}")
        print()
        
        # Motion sensor data
        print("MOTION SENSORS:")
        print("-" * 90)
        print(f"{'Parameter':<25} {'Value':<15} {'Unit':<10} {'Valid':<8} {'Diag':<8} {'Status'}")
        print("-" * 90)
        
        # Lateral acceleration
        lat_status = "✓ GOOD" if signals.lateral_accel_valid and not signals.lateral_accel_diag else "⚠ CHECK"
        print(f"{'Lateral Acceleration':<25} {signals.lateral_accel_ms2:<15.3f} {'m/s²':<10} "
              f"{signals.lateral_accel_valid!s:<8} {signals.lateral_accel_diag!s:<8} {lat_status}")
        
        # Longitudinal acceleration  
        long_status = "✓ GOOD" if signals.longitudinal_accel_valid and not signals.longitudinal_accel_diag else "⚠ CHECK"
        print(f"{'Longitudinal Acceleration':<25} {signals.longitudinal_accel_ms2:<15.3f} {'m/s²':<10} "
              f"{signals.longitudinal_accel_valid!s:<8} {signals.longitudinal_accel_diag!s:<8} {long_status}")
        
        # Yaw rate
        yaw_status = "✓ GOOD" if signals.yaw_rate_valid and not signals.yaw_rate_diag else "⚠ CHECK"  
        print(f"{'Yaw Rate':<25} {signals.yaw_rate_degs:<15.3f} {'deg/s':<10} "
              f"{signals.yaw_rate_valid!s:<8} {signals.yaw_rate_diag!s:<8} {yaw_status}")
        
        # Brake pressure
        brake_status = "✓ GOOD" if signals.brake_pressure_valid and not signals.brake_pressure_diag else "⚠ CHECK"
        print(f"{'Brake Cylinder Pressure':<25} {signals.brake_cylinder_pressure_bar:<15.3f} {'Bar':<10} "
              f"{signals.brake_pressure_valid!s:<8} {signals.brake_pressure_diag!s:<8} {brake_status}")
        
        print()
        
        # Additional derived information
        print("DERIVED INFORMATION:")
        print("-" * 90)
        
        # G-force conversions
        lat_g = signals.lateral_accel_ms2 / 9.81
        long_g = signals.longitudinal_accel_ms2 / 9.81
        total_g = (lat_g**2 + long_g**2)**0.5
        
        print(f"Lateral G-Force: {lat_g:+.3f}g | Longitudinal G-Force: {long_g:+.3f}g | Total: {total_g:.3f}g")
        
        # Motion state assessment
        motion_state = "STATIONARY"
        if abs(signals.lateral_accel_ms2) > 0.5 or abs(signals.longitudinal_accel_ms2) > 0.5:
            motion_state = "ACCELERATING"
        if abs(signals.yaw_rate_degs) > 2.0:
            motion_state = "TURNING"
        if signals.brake_cylinder_pressure_bar > 5.0:
            motion_state = "BRAKING"
            
        print(f"Motion State: {motion_state}")
        print()
        
        # Statistics summary
        print("SESSION STATISTICS:")
        print("-" * 90)
        print(f"Total Messages: {self.stats['total_messages']} | "
              f"Valid: {self.stats['valid_messages']} | "
              f"Invalid: {self.stats['invalid_messages']}")
        print(f"Max Lateral Accel: {self.stats['max_lateral_accel']:.3f} m/s² | "
              f"Max Longitudinal Accel: {self.stats['max_longitudinal_accel']:.3f} m/s²")
        print(f"Max Yaw Rate: {self.stats['max_yaw_rate']:.3f} deg/s | "
              f"Max Brake Pressure: {self.stats['max_brake_pressure']:.3f} Bar")
        
        if self.csv_logging:
            print(f"Data Logging: {self.csv_filename}")
        
        print("\nPress Ctrl+C to stop...")
    
    def monitor(self):
        """Main monitoring loop"""
        
        print(f"Starting ESP12 Vehicle Motion Sensor Monitor")
        print(f"Interface: {self.interface}")
        print(f"Target: CAN ID 0x{self.ESP12_CAN_ID:03X} (ESP12)")
        print(f"Logging: CSV={self.csv_logging}, File={self.log_to_file}")
        print("=" * 90)
        
        last_display_time = 0
        display_interval = 0.2  # Update display every 200ms
        
        try:
            while True:
                # Receive CAN message
                message = self.bus.recv(timeout=0.1)
                
                if message is None:
                    continue
                
                # Check for ESP12 message
                if message.arbitration_id == self.ESP12_CAN_ID:
                    self.message_count += 1
                    self.stats['total_messages'] += 1
                    
                    # Parse ESP12 signals
                    signals = self.parse_esp12_signals(message.data, message.timestamp)
                    
                    if signals:
                        # Log to CSV if enabled
                        self.log_to_csv(signals)
                        
                        # Update display periodically
                        current_time = time.time()
                        if current_time - last_display_time >= display_interval:
                            self.display_motion_data(signals)
                            last_display_time = current_time
                    
        except KeyboardInterrupt:
            print(f"\nESP12 monitoring stopped")
            self.log_message(f"Monitoring stopped - Processed {self.message_count} messages")
            self.print_final_statistics()
        finally:
            if self.bus:
                self.bus.shutdown()
    
    def print_final_statistics(self):
        """Print final session statistics"""
        print(f"\nFINAL SESSION STATISTICS:")
        print("=" * 60)
        print(f"Runtime: {time.time() - self.start_time:.1f} seconds")
        print(f"Total Messages: {self.stats['total_messages']}")
        print(f"Valid Messages: {self.stats['valid_messages']}")
        print(f"Invalid Messages: {self.stats['invalid_messages']}")
        print(f"Checksum Errors: {self.stats['checksum_errors']}")
        print(f"Out of Range Errors: {self.stats['out_of_range_errors']}")
        print(f"Success Rate: {(self.stats['valid_messages']/max(1,self.stats['total_messages']))*100:.1f}%")
        print()
        print(f"Peak Lateral Acceleration: {self.stats['max_lateral_accel']:.3f} m/s² ({self.stats['max_lateral_accel']/9.81:.3f}g)")
        print(f"Peak Longitudinal Acceleration: {self.stats['max_longitudinal_accel']:.3f} m/s² ({self.stats['max_longitudinal_accel']/9.81:.3f}g)")
        print(f"Peak Yaw Rate: {self.stats['max_yaw_rate']:.3f} deg/s")
        print(f"Peak Brake Pressure: {self.stats['max_brake_pressure']:.3f} Bar")
        
        if self.csv_logging:
            print(f"\nData saved to: {self.csv_filename}")
        if self.log_to_file:
            print(f"Log saved to: {self.log_filename}")

def main():
    parser = argparse.ArgumentParser(
        description='ESP12 Vehicle Motion Sensor Monitor - Comprehensive Vehicle Dynamics',
        epilog='''
ESP12 provides critical vehicle motion data:
- Lateral/Longitudinal Acceleration (±10.24 m/s²)
- Yaw Rate (±40.96 deg/s)  
- Brake Cylinder Pressure (0-409.5 Bar)

DBC-compliant parsing with full validation and logging.
        '''
    )
    parser.add_argument('--interface', '-i', default='can1', 
                       help='CAN interface (default: can1)')
    parser.add_argument('--no-csv', action='store_true',
                       help='Disable CSV data logging')
    parser.add_argument('--no-log', action='store_true', 
                       help='Disable text file logging')
    
    args = parser.parse_args()
    
    print("ESP12 Vehicle Motion Sensor Monitor")
    print("Comprehensive Vehicle Dynamics Monitoring")
    print("DBC-Compliant with Validation & Logging")
    print()
    
    monitor = ESP12_MotionMonitor(
        interface=args.interface,
        log_to_file=not args.no_log,
        csv_logging=not args.no_csv
    )
    monitor.monitor()

if __name__ == "__main__":
    main()
