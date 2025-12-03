#!/usr/bin/env python3
"""
Steering System Monitor
CAN IDs: 0x2B0 (SAS11), 0x381 (MDPS11)
Steering Angle Sensor & Motor-Driven Power Steering

Decodes steering system parameters:
- Steering Wheel Angle (±3276.7 degrees)
- Steering Angular Velocity (0-1016 units)
- Driver Applied Torque (±2046 Nm)
- Motor Assistance Torque (±204.7 Nm)

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
class SAS11_Signals:
    """SAS11 Steering Angle Sensor signals"""
    steering_angle_deg: float             # deg (-3276.8 to 3276.7)
    steering_speed: float                 # angular velocity (0-1016)
    steering_status: int                  # sensor status (0-255)
    message_count: int                    # 4-bit counter
    checksum: int                         # 4-bit checksum
    timestamp: float
    can_id: int
    raw_data: bytes

@dataclass  
class MDPS11_Signals:
    """MDPS11 Motor-Driven Power Steering signals"""
    warning_lamp: int                     # 2-bit warning lamp status
    flex_steering: int                    # 3-bit flex steering mode
    flex_display: bool                    # flex steering display
    mdps_status: int                      # 4-bit MDPS status
    driver_torque_nm: float               # Nm (-2048 to 2046)
    alt_request: bool                     # alternator request
    motor_steering_angle_deg: float       # deg (-3276.8 to 3276.7)
    alive_counter: int                    # 8-bit alive counter
    checksum: int                         # 8-bit checksum
    spas_function: bool                   # SPAS function active
    lkas_function: bool                   # LKAS function active
    current_mode: int                     # 2-bit current mode
    mdps_type: int                        # 2-bit MDPS type
    vsm_function: bool                    # VSM function active
    timestamp: float
    can_id: int
    raw_data: bytes

class SteeringSystemMonitor:
    def __init__(self, interface='can1', log_to_file=True, csv_logging=True):
        self.interface = interface
        self.sas_message_count = 0
        self.mdps_message_count = 0
        self.start_time = time.time()
        self.log_to_file = log_to_file
        self.csv_logging = csv_logging
        
        # CAN IDs
        self.SAS11_CAN_ID = 0x2B0   # 688 decimal
        self.MDPS11_CAN_ID = 0x381  # 897 decimal
        
        # SAS11 constants
        self.SAS_ANGLE_SCALE = 0.1          # deg per LSB
        self.SAS_SPEED_SCALE = 4.0          # units per LSB
        
        # MDPS11 constants  
        self.MDPS_ANGLE_SCALE = 0.1         # deg per LSB
        self.MDPS_TORQUE_OFFSET = -2048.0   # Nm offset
        
        # Validation ranges (engineering limits)
        self.ANGLE_MIN = -4000.0            # degrees
        self.ANGLE_MAX = 4000.0             # degrees  
        self.TORQUE_MIN = -100.0            # Nm (typical driver input)
        self.TORQUE_MAX = 100.0             # Nm
        self.SPEED_MAX = 2000.0             # reasonable steering speed limit
        
        # Latest signal data
        self.latest_sas = None
        self.latest_mdps = None
        
        # Statistics tracking
        self.stats = {
            'sas_total_messages': 0,
            'sas_valid_messages': 0,
            'mdps_total_messages': 0,
            'mdps_valid_messages': 0,
            'angle_correlation_errors': 0,
            'max_steering_angle': 0.0,
            'max_driver_torque': 0.0,
            'max_steering_speed': 0.0,
            'checksum_errors': 0
        }
        
        # Initialize logging
        self.setup_logging()
        
        # Connect to CAN bus
        try:
            self.bus = can.interface.Bus(channel=interface, bustype='socketcan')
            print(f"✓ Connected to {interface}")
            self.log_message(f"Steering System Monitor started on {interface}")
        except Exception as e:
            print(f"✗ Failed to connect to {interface}: {e}")
            sys.exit(1)
    
    def setup_logging(self):
        """Initialize file and CSV logging"""
        if self.log_to_file or self.csv_logging:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        if self.log_to_file:
            self.log_filename = f"steering_system_log_{timestamp}.txt"
            with open(self.log_filename, 'w') as f:
                f.write(f"Steering System Monitor Log\n")
                f.write(f"Started: {datetime.now().isoformat()}\n")
                f.write(f"Interface: {self.interface}\n")
                f.write("=" * 80 + "\n\n")
        
        if self.csv_logging:
            self.sas_csv_filename = f"sas11_data_{timestamp}.csv"
            self.mdps_csv_filename = f"mdps11_data_{timestamp}.csv"
            
            # SAS11 CSV header
            with open(self.sas_csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'can_id', 'message_count',
                    'steering_angle_deg', 'steering_speed', 'steering_status',
                    'msg_count', 'checksum', 'raw_data_hex'
                ])
            
            # MDPS11 CSV header  
            with open(self.mdps_csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'can_id', 'message_count',
                    'warning_lamp', 'flex_steering', 'flex_display', 'mdps_status',
                    'driver_torque_nm', 'alt_request', 'motor_steering_angle_deg',
                    'alive_counter', 'checksum', 'spas_function', 'lkas_function',
                    'current_mode', 'mdps_type', 'vsm_function', 'raw_data_hex'
                ])
    
    def log_message(self, message: str):
        """Log message to file with timestamp"""
        if self.log_to_file:
            with open(self.log_filename, 'a') as f:
                f.write(f"[{datetime.now().isoformat()}] {message}\n")
    
    def parse_sas11_signals(self, data: bytes, timestamp: float) -> Optional[SAS11_Signals]:
        """
        Parse SAS11 CAN message according to DBC specification
        
        SAS11 Signal Layout (40-bit little-endian):
        - SAS_Angle: bits [15:0] - 16-bit signed, scale=0.1, offset=0.0
        - SAS_Speed: bits [23:16] - 8-bit unsigned, scale=4.0, offset=0.0  
        - SAS_Stat: bits [31:24] - 8-bit unsigned, scale=1.0, offset=0.0
        - MsgCount: bits [35:32] - 4-bit unsigned
        - CheckSum: bits [39:36] - 4-bit unsigned
        """
        
        if len(data) != 5:
            self.log_message(f"SAS11 Invalid message length: {len(data)} bytes")
            return None
        
        try:
            # Extend to 8 bytes for easier processing
            data_extended = data + b'\x00\x00\x00'
            frame_uint64 = struct.unpack('<Q', data_extended)[0]
            
            # Extract steering angle (16-bit signed)
            angle_raw = frame_uint64 & 0xFFFF
            if angle_raw & 0x8000:  # Sign extend
                angle_raw |= 0xFFFF0000
                angle_raw = struct.unpack('<i', struct.pack('<I', angle_raw & 0xFFFFFFFF))[0]
            steering_angle = angle_raw * self.SAS_ANGLE_SCALE
            
            # Extract other signals
            steering_speed = ((frame_uint64 >> 16) & 0xFF) * self.SAS_SPEED_SCALE
            steering_status = (frame_uint64 >> 24) & 0xFF
            message_count = (frame_uint64 >> 32) & 0xF
            checksum = (frame_uint64 >> 36) & 0xF
            
            # Create signal structure
            signals = SAS11_Signals(
                steering_angle_deg=steering_angle,
                steering_speed=steering_speed,
                steering_status=steering_status,
                message_count=message_count,
                checksum=checksum,
                timestamp=timestamp,
                can_id=self.SAS11_CAN_ID,
                raw_data=data
            )
            
            # Validate signal ranges
            if self.validate_sas_signals(signals):
                self.stats['sas_valid_messages'] += 1
                self.update_sas_statistics(signals)
                return signals
            else:
                return None
            
        except Exception as e:
            self.log_message(f"SAS11 parsing error: {e}")
            return None
    
    def parse_mdps11_signals(self, data: bytes, timestamp: float) -> Optional[MDPS11_Signals]:
        """
        Parse MDPS11 CAN message according to DBC specification
        
        MDPS11 Signal Layout (64-bit little-endian):
        - CF_Mdps_WLmp: bits [1:0] - 2-bit warning lamp
        - CF_Mdps_Flex: bits [4:2] - 3-bit flex steering
        - CF_Mdps_FlexDisp: bit [5] - flex display
        - CF_Mdps_Stat: bits [10:7] - 4-bit MDPS status  
        - CR_Mdps_DrvTq: bits [22:11] - 12-bit signed driver torque
        - CF_Mdps_ALTRequest: bit [23] - alternator request
        - CR_Mdps_StrAng: bits [39:24] - 16-bit signed steering angle
        - CF_Mdps_AliveCnt: bits [47:40] - 8-bit alive counter
        - CF_Mdps_Chksum: bits [55:48] - 8-bit checksum
        - CF_MDPS_VSM_FUNC: bit [56] - VSM function
        - CF_Mdps_SPAS_FUNC: bit [57] - SPAS function
        - CF_Mdps_LKAS_FUNC: bit [58] - LKAS function
        - CF_Mdps_CurrMode: bits [60:59] - 2-bit current mode
        - CF_Mdps_Type: bits [62:61] - 2-bit MDPS type
        """
        
        if len(data) != 8:
            self.log_message(f"MDPS11 Invalid message length: {len(data)} bytes")
            return None
        
        try:
            # Unpack as little-endian 64-bit unsigned integer
            frame_uint64 = struct.unpack('<Q', data)[0]
            
            # Extract signals
            warning_lamp = frame_uint64 & 0x3
            flex_steering = (frame_uint64 >> 2) & 0x7
            flex_display = bool((frame_uint64 >> 5) & 0x1)
            mdps_status = (frame_uint64 >> 7) & 0xF
            
            # Extract driver torque (12-bit signed)
            torque_raw = (frame_uint64 >> 11) & 0xFFF
            if torque_raw & 0x800:  # Sign extend
                torque_raw |= 0xFFFFF000
                torque_raw = struct.unpack('<i', struct.pack('<I', torque_raw & 0xFFFFFFFF))[0]
            driver_torque = torque_raw + self.MDPS_TORQUE_OFFSET
            
            alt_request = bool((frame_uint64 >> 23) & 0x1)
            
            # Extract steering angle (16-bit signed)
            angle_raw = (frame_uint64 >> 24) & 0xFFFF
            if angle_raw & 0x8000:  # Sign extend
                angle_raw |= 0xFFFF0000
                angle_raw = struct.unpack('<i', struct.pack('<I', angle_raw & 0xFFFFFFFF))[0]
            motor_steering_angle = angle_raw * self.MDPS_ANGLE_SCALE
            
            alive_counter = (frame_uint64 >> 40) & 0xFF
            checksum = (frame_uint64 >> 48) & 0xFF
            vsm_function = bool((frame_uint64 >> 56) & 0x1)
            spas_function = bool((frame_uint64 >> 57) & 0x1)
            lkas_function = bool((frame_uint64 >> 58) & 0x1)
            current_mode = (frame_uint64 >> 59) & 0x3
            mdps_type = (frame_uint64 >> 61) & 0x3
            
            # Create signal structure
            signals = MDPS11_Signals(
                warning_lamp=warning_lamp,
                flex_steering=flex_steering,
                flex_display=flex_display,
                mdps_status=mdps_status,
                driver_torque_nm=driver_torque,
                alt_request=alt_request,
                motor_steering_angle_deg=motor_steering_angle,
                alive_counter=alive_counter,
                checksum=checksum,
                spas_function=spas_function,
                lkas_function=lkas_function,
                current_mode=current_mode,
                mdps_type=mdps_type,
                vsm_function=vsm_function,
                timestamp=timestamp,
                can_id=self.MDPS11_CAN_ID,
                raw_data=data
            )
            
            # Validate signal ranges
            if self.validate_mdps_signals(signals):
                self.stats['mdps_valid_messages'] += 1
                self.update_mdps_statistics(signals)
                return signals
            else:
                return None
            
        except Exception as e:
            self.log_message(f"MDPS11 parsing error: {e}")
            return None
    
    def validate_sas_signals(self, signals: SAS11_Signals) -> bool:
        """Validate SAS11 signal values"""
        if not (self.ANGLE_MIN <= signals.steering_angle_deg <= self.ANGLE_MAX):
            self.log_message(f"SAS angle out of range: {signals.steering_angle_deg:.1f}")
            return False
        if signals.steering_speed > self.SPEED_MAX:
            self.log_message(f"SAS speed out of range: {signals.steering_speed:.1f}")
            return False
        return True
    
    def validate_mdps_signals(self, signals: MDPS11_Signals) -> bool:
        """Validate MDPS11 signal values"""
        if not (self.ANGLE_MIN <= signals.motor_steering_angle_deg <= self.ANGLE_MAX):
            self.log_message(f"MDPS angle out of range: {signals.motor_steering_angle_deg:.1f}")
            return False
        if not (self.TORQUE_MIN <= signals.driver_torque_nm <= self.TORQUE_MAX):
            self.log_message(f"MDPS torque out of range: {signals.driver_torque_nm:.1f}")
            return False
        return True
    
    def update_sas_statistics(self, signals: SAS11_Signals):
        """Update SAS11 statistics"""
        self.stats['max_steering_angle'] = max(self.stats['max_steering_angle'], 
                                              abs(signals.steering_angle_deg))
        self.stats['max_steering_speed'] = max(self.stats['max_steering_speed'], 
                                              signals.steering_speed)
    
    def update_mdps_statistics(self, signals: MDPS11_Signals):
        """Update MDPS11 statistics"""
        self.stats['max_driver_torque'] = max(self.stats['max_driver_torque'], 
                                             abs(signals.driver_torque_nm))
    
    def check_angle_correlation(self):
        """Check correlation between SAS and MDPS steering angles"""
        if self.latest_sas and self.latest_mdps:
            angle_diff = abs(self.latest_sas.steering_angle_deg - self.latest_mdps.motor_steering_angle_deg)
            if angle_diff > 10.0:  # More than 10 degrees difference
                self.stats['angle_correlation_errors'] += 1
                self.log_message(f"Angle correlation error: SAS={self.latest_sas.steering_angle_deg:.1f}, MDPS={self.latest_mdps.motor_steering_angle_deg:.1f}")
    
    def log_sas_to_csv(self, signals: SAS11_Signals):
        """Log SAS11 data to CSV"""
        if self.csv_logging:
            with open(self.sas_csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    signals.timestamp, signals.can_id, self.sas_message_count,
                    signals.steering_angle_deg, signals.steering_speed, signals.steering_status,
                    signals.message_count, signals.checksum, signals.raw_data.hex().upper()
                ])
    
    def log_mdps_to_csv(self, signals: MDPS11_Signals):
        """Log MDPS11 data to CSV"""
        if self.csv_logging:
            with open(self.mdps_csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    signals.timestamp, signals.can_id, self.mdps_message_count,
                    signals.warning_lamp, signals.flex_steering, signals.flex_display, signals.mdps_status,
                    signals.driver_torque_nm, signals.alt_request, signals.motor_steering_angle_deg,
                    signals.alive_counter, signals.checksum, signals.spas_function, signals.lkas_function,
                    signals.current_mode, signals.mdps_type, signals.vsm_function, signals.raw_data.hex().upper()
                ])
    
    def display_steering_data(self):
        """Display comprehensive steering system data"""
        
        print(f"\033[2J\033[H")  # Clear screen
        print("=" * 100)
        print(f"STEERING SYSTEM MONITOR - {self.interface}")
        print("=" * 100)
        
        print(f"SAS Messages: {self.sas_message_count} | MDPS Messages: {self.mdps_message_count}")
        print(f"Runtime: {time.time() - self.start_time:.1f}s")
        print()
        
        # SAS11 Data
        print("SAS11 - STEERING ANGLE SENSOR:")
        print("-" * 100)
        if self.latest_sas:
            print(f"CAN ID: 0x{self.latest_sas.can_id:03X} | Raw: {self.latest_sas.raw_data.hex().upper()}")
            print(f"Steering Angle: {self.latest_sas.steering_angle_deg:+7.1f}° | Speed: {self.latest_sas.steering_speed:6.1f}")
            print(f"Status: {self.latest_sas.steering_status} | Count: {self.latest_sas.message_count} | Checksum: {self.latest_sas.checksum}")
        else:
            print("No SAS11 data received")
        print()
        
        # MDPS11 Data
        print("MDPS11 - MOTOR-DRIVEN POWER STEERING:")
        print("-" * 100)
        if self.latest_mdps:
            print(f"CAN ID: 0x{self.latest_mdps.can_id:03X} | Raw: {self.latest_mdps.raw_data.hex().upper()}")
            print(f"Motor Angle: {self.latest_mdps.motor_steering_angle_deg:+7.1f}° | Driver Torque: {self.latest_mdps.driver_torque_nm:+6.1f} Nm")
            print(f"Warning Lamp: {self.latest_mdps.warning_lamp} | MDPS Status: {self.latest_mdps.mdps_status}")
            print(f"Flex Steering: {self.latest_mdps.flex_steering} | Current Mode: {self.latest_mdps.current_mode}")
            
            # Function status
            functions = []
            if self.latest_mdps.spas_function: functions.append("SPAS")
            if self.latest_mdps.lkas_function: functions.append("LKAS") 
            if self.latest_mdps.vsm_function: functions.append("VSM")
            print(f"Active Functions: {', '.join(functions) if functions else 'None'}")
        else:
            print("No MDPS11 data received")
        print()
        
        # Correlation Analysis
        print("CORRELATION ANALYSIS:")
        print("-" * 100)
        if self.latest_sas and self.latest_mdps:
            angle_diff = self.latest_sas.steering_angle_deg - self.latest_mdps.motor_steering_angle_deg
            correlation_status = "✓ GOOD" if abs(angle_diff) < 5.0 else "⚠ CHECK"
            print(f"Angle Difference: {angle_diff:+6.1f}° | Status: {correlation_status}")
            
            # Steering direction analysis
            if abs(self.latest_sas.steering_angle_deg) > 10:
                direction = "LEFT" if self.latest_sas.steering_angle_deg > 0 else "RIGHT"
                print(f"Steering Direction: {direction} | Magnitude: {abs(self.latest_sas.steering_angle_deg):.1f}°")
            else:
                print("Steering: STRAIGHT/CENTER")
                
            # Driver input analysis
            if abs(self.latest_mdps.driver_torque_nm) > 1.0:
                input_direction = "CCW" if self.latest_mdps.driver_torque_nm > 0 else "CW"
                print(f"Driver Input: {input_direction} {abs(self.latest_mdps.driver_torque_nm):.1f} Nm")
            else:
                print("Driver Input: MINIMAL")
        else:
            print("Insufficient data for correlation analysis")
        print()
        
        # Session Statistics
        print("SESSION STATISTICS:")
        print("-" * 100)
        print(f"SAS11 - Total: {self.stats['sas_total_messages']} | Valid: {self.stats['sas_valid_messages']}")
        print(f"MDPS11 - Total: {self.stats['mdps_total_messages']} | Valid: {self.stats['mdps_valid_messages']}")
        print(f"Max Steering Angle: {self.stats['max_steering_angle']:.1f}° | Max Driver Torque: {self.stats['max_driver_torque']:.1f} Nm")
        print(f"Angle Correlation Errors: {self.stats['angle_correlation_errors']}")
        
        if self.csv_logging:
            print(f"Data Logging: {self.sas_csv_filename}, {self.mdps_csv_filename}")
        
        print("\nPress Ctrl+C to stop...")
    
    def monitor(self):
        """Main monitoring loop"""
        
        print(f"Starting Steering System Monitor")
        print(f"Interface: {self.interface}")
        print(f"Targets: SAS11 (0x{self.SAS11_CAN_ID:03X}), MDPS11 (0x{self.MDPS11_CAN_ID:03X})")
        print(f"Logging: CSV={self.csv_logging}, File={self.log_to_file}")
        print("=" * 100)
        
        last_display_time = 0
        display_interval = 0.3  # Update display every 300ms
        
        try:
            while True:
                # Receive CAN message
                message = self.bus.recv(timeout=0.1)
                
                if message is None:
                    continue
                
                # Check for SAS11 message
                if message.arbitration_id == self.SAS11_CAN_ID:
                    self.sas_message_count += 1
                    self.stats['sas_total_messages'] += 1
                    
                    signals = self.parse_sas11_signals(message.data, message.timestamp)
                    if signals:
                        self.latest_sas = signals
                        self.log_sas_to_csv(signals)
                        self.check_angle_correlation()
                
                # Check for MDPS11 message  
                elif message.arbitration_id == self.MDPS11_CAN_ID:
                    self.mdps_message_count += 1
                    self.stats['mdps_total_messages'] += 1
                    
                    signals = self.parse_mdps11_signals(message.data, message.timestamp)
                    if signals:
                        self.latest_mdps = signals
                        self.log_mdps_to_csv(signals)
                        self.check_angle_correlation()
                
                # Update display periodically
                current_time = time.time()
                if current_time - last_display_time >= display_interval:
                    self.display_steering_data()
                    last_display_time = current_time
                    
        except KeyboardInterrupt:
            print(f"\nSteering System monitoring stopped")
            self.log_message(f"Monitoring stopped - SAS: {self.sas_message_count}, MDPS: {self.mdps_message_count}")
            self.print_final_statistics()
        finally:
            if self.bus:
                self.bus.shutdown()
    
    def print_final_statistics(self):
        """Print final session statistics"""
        print(f"\nFINAL STEERING SYSTEM STATISTICS:")
        print("=" * 80)
        print(f"Runtime: {time.time() - self.start_time:.1f} seconds")
        print(f"SAS11 Messages: {self.stats['sas_total_messages']} (Valid: {self.stats['sas_valid_messages']})")
        print(f"MDPS11 Messages: {self.stats['mdps_total_messages']} (Valid: {self.stats['mdps_valid_messages']})")
        print(f"Peak Steering Angle: {self.stats['max_steering_angle']:.1f}°")
        print(f"Peak Driver Torque: {self.stats['max_driver_torque']:.1f} Nm")
        print(f"Angle Correlation Errors: {self.stats['angle_correlation_errors']}")
        
        if self.csv_logging:
            print(f"\nSAS Data: {self.sas_csv_filename}")
            print(f"MDPS Data: {self.mdps_csv_filename}")
        if self.log_to_file:
            print(f"Log: {self.log_filename}")

def main():
    parser = argparse.ArgumentParser(
        description='Steering System Monitor - SAS11 & MDPS11',
        epilog='''
Monitors steering system with dual sensor correlation:
- SAS11: Steering angle sensor data
- MDPS11: Motor-driven power steering data
- Cross-validation and correlation analysis

DBC-compliant parsing with comprehensive validation.
        '''
    )
    parser.add_argument('--interface', '-i', default='can1', 
                       help='CAN interface (default: can1)')
    parser.add_argument('--no-csv', action='store_true',
                       help='Disable CSV data logging')
    parser.add_argument('--no-log', action='store_true', 
                       help='Disable text file logging')
    
    args = parser.parse_args()
    
    print("Steering System Monitor")
    print("SAS11 & MDPS11 Comprehensive Analysis")
    print("DBC-Compliant with Correlation Validation")
    print()
    
    monitor = SteeringSystemMonitor(
        interface=args.interface,
        log_to_file=not args.no_log,
        csv_logging=not args.no_csv
    )
    monitor.monitor()

if __name__ == "__main__":
    main()
