#!/usr/bin/env python3
"""
YRS Inertial Measurement Unit Monitor
CAN IDs: 0x130 (YRS11), 0x140 (YRS12), 0x495 (YRS13)
High-Precision Vehicle Motion Sensors

Decodes ultra-precise IMU parameters:
- Yaw Rate: ±163.84 deg/s (0.005 deg/s resolution)
- Lateral Acceleration: ±4.177g (0.000127465g resolution) 
- Longitudinal Acceleration: ±4.177g (0.000127465g resolution)
- Temperature: -68°C to +187°C (1°C resolution)
- Sensor calibration and diagnostic status

DBC Specification Compliance with advanced sensor fusion validation
"""

import can
import struct
import time
import argparse
import sys
import json
import csv
import math
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

@dataclass
class YRS11_Signals:
    """YRS11 Yaw Rate and Lateral Acceleration signals"""
    yaw_rate_degs: float                  # deg/s (-163.84 to 163.83)
    yaw_rate_status: int                  # 4-bit sensor status
    lateral_accel_g: float                # g (-4.177 to 4.177)
    lateral_accel_status: int             # 4-bit sensor status
    mcu_status: int                       # 4-bit MCU status
    message_count: int                    # 4-bit rolling counter
    crc: int                              # 8-bit CRC
    timestamp: float
    can_id: int
    raw_data: bytes

@dataclass
class YRS12_Signals:
    """YRS12 Longitudinal Acceleration and Temperature signals"""
    longitudinal_accel_g: float           # g (-4.177 to 4.177)
    longitudinal_accel_status: int        # 4-bit sensor status
    imu_reset_status: int                 # 4-bit reset status
    temperature_c: float                  # °C (-68 to 187)
    temperature_status: int               # 4-bit temperature status
    yrs_type: int                         # 4-bit sensor type
    message_count: int                    # 4-bit rolling counter
    crc: int                              # 8-bit CRC
    timestamp: float
    can_id: int
    raw_data: bytes

@dataclass
class YRS13_Signals:
    """YRS13 Serial Number and Identification"""
    serial_number: int                    # 48-bit serial number
    timestamp: float
    can_id: int
    raw_data: bytes

class YRS_IMU_Monitor:
    def __init__(self, interface='can1', log_to_file=True, csv_logging=True):
        self.interface = interface
        self.yrs11_message_count = 0
        self.yrs12_message_count = 0
        self.yrs13_message_count = 0
        self.start_time = time.time()
        self.log_to_file = log_to_file
        self.csv_logging = csv_logging
        
        # CAN IDs
        self.YRS11_CAN_ID = 0x130   # 304 decimal
        self.YRS12_CAN_ID = 0x140   # 320 decimal
        self.YRS13_CAN_ID = 0x495   # 1173 decimal
        
        # YRS11 constants (ultra-high precision)
        self.YAW_SCALE = 0.005              # deg/s per LSB
        self.YAW_OFFSET = -163.84           # deg/s offset
        self.LAT_ACCEL_SCALE = 0.000127465  # g per LSB
        self.LAT_ACCEL_OFFSET = -4.17677312 # g offset
        
        # YRS12 constants
        self.LONG_ACCEL_SCALE = 0.000127465 # g per LSB
        self.LONG_ACCEL_OFFSET = -4.17677312 # g offset
        self.TEMP_OFFSET = -68              # °C offset
        
        # Validation ranges (engineering limits)
        self.YAW_MIN = -200.0               # deg/s
        self.YAW_MAX = 200.0                # deg/s
        self.ACCEL_MIN = -5.0               # g
        self.ACCEL_MAX = 5.0                # g
        self.TEMP_MIN = -80.0               # °C
        self.TEMP_MAX = 200.0               # °C
        
        # Latest signal data
        self.latest_yrs11 = None
        self.latest_yrs12 = None
        self.latest_yrs13 = None
        
        # Statistics tracking
        self.stats = {
            'yrs11_total_messages': 0,
            'yrs11_valid_messages': 0,
            'yrs12_total_messages': 0,
            'yrs12_valid_messages': 0,
            'yrs13_total_messages': 0,
            'crc_errors': 0,
            'sensor_faults': 0,
            'temperature_alerts': 0,
            'max_yaw_rate': 0.0,
            'max_lateral_accel': 0.0,
            'max_longitudinal_accel': 0.0,
            'min_temperature': 200.0,
            'max_temperature': -100.0
        }
        
        # Initialize logging
        self.setup_logging()
        
        # Connect to CAN bus
        try:
            self.bus = can.interface.Bus(channel=interface, bustype='socketcan')
            print(f"✓ Connected to {interface}")
            self.log_message(f"YRS IMU Monitor started on {interface}")
        except Exception as e:
            print(f"✗ Failed to connect to {interface}: {e}")
            sys.exit(1)
    
    def setup_logging(self):
        """Initialize comprehensive logging system"""
        if self.log_to_file or self.csv_logging:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        if self.log_to_file:
            self.log_filename = f"yrs_imu_log_{timestamp}.txt"
            with open(self.log_filename, 'w') as f:
                f.write(f"YRS IMU Monitor Log\n")
                f.write(f"Started: {datetime.now().isoformat()}\n")
                f.write(f"Interface: {self.interface}\n")
                f.write("High-Precision Inertial Measurement Unit Analysis\n")
                f.write("=" * 80 + "\n\n")
        
        if self.csv_logging:
            # YRS11 CSV
            self.yrs11_csv_filename = f"yrs11_data_{timestamp}.csv"
            with open(self.yrs11_csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'can_id', 'message_count',
                    'yaw_rate_degs', 'yaw_rate_status', 
                    'lateral_accel_g', 'lateral_accel_status',
                    'mcu_status', 'msg_count', 'crc', 'raw_data_hex'
                ])
            
            # YRS12 CSV
            self.yrs12_csv_filename = f"yrs12_data_{timestamp}.csv"
            with open(self.yrs12_csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'can_id', 'message_count',
                    'longitudinal_accel_g', 'longitudinal_accel_status',
                    'imu_reset_status', 'temperature_c', 'temperature_status',
                    'yrs_type', 'msg_count', 'crc', 'raw_data_hex'
                ])
            
            # YRS13 CSV
            self.yrs13_csv_filename = f"yrs13_data_{timestamp}.csv"
            with open(self.yrs13_csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'can_id', 'message_count',
                    'serial_number', 'raw_data_hex'
                ])
    
    def log_message(self, message: str):
        """Log message to file with timestamp"""
        if self.log_to_file:
            with open(self.log_filename, 'a') as f:
                f.write(f"[{datetime.now().isoformat()}] {message}\n")
    
    def parse_yrs11_signals(self, data: bytes, timestamp: float) -> Optional[YRS11_Signals]:
        """
        Parse YRS11 CAN message according to DBC specification
        
        YRS11 Signal Layout (64-bit little-endian):
        - CR_Yrs_Yr: bits [15:0] - 16-bit signed, scale=0.005, offset=-163.84
        - CR_Yrs_LatAc: bits [31:16] - 16-bit signed, scale=0.000127465, offset=-4.17677312
        - CF_Yrs_YrStat: bits [35:32] - 4-bit yaw rate status
        - CF_Yrs_LatAcStat: bits [39:36] - 4-bit lateral accel status
        - CF_Yrs_MCUStat: bits [43:40] - 4-bit MCU status
        - CR_Yrs_MsgCnt1: bits [51:48] - 4-bit message counter
        - CR_Yrs_Crc1: bits [63:56] - 8-bit CRC
        """
        
        if len(data) != 8:
            self.log_message(f"YRS11 Invalid message length: {len(data)} bytes")
            return None
        
        try:
            # Unpack as little-endian 64-bit unsigned integer
            frame_uint64 = struct.unpack('<Q', data)[0]
            
            # Extract yaw rate (16-bit signed)
            yaw_raw = frame_uint64 & 0xFFFF
            if yaw_raw & 0x8000:  # Sign extend
                yaw_raw |= 0xFFFF0000
                yaw_raw = struct.unpack('<i', struct.pack('<I', yaw_raw & 0xFFFFFFFF))[0]
            yaw_rate = yaw_raw * self.YAW_SCALE + self.YAW_OFFSET
            
            # Extract lateral acceleration (16-bit signed)
            lat_accel_raw = (frame_uint64 >> 16) & 0xFFFF
            if lat_accel_raw & 0x8000:  # Sign extend
                lat_accel_raw |= 0xFFFF0000
                lat_accel_raw = struct.unpack('<i', struct.pack('<I', lat_accel_raw & 0xFFFFFFFF))[0]
            lateral_accel = lat_accel_raw * self.LAT_ACCEL_SCALE + self.LAT_ACCEL_OFFSET
            
            # Extract status and control fields
            yaw_rate_status = (frame_uint64 >> 32) & 0xF
            lateral_accel_status = (frame_uint64 >> 36) & 0xF
            mcu_status = (frame_uint64 >> 40) & 0xF
            message_count = (frame_uint64 >> 48) & 0xF
            crc = (frame_uint64 >> 56) & 0xFF
            
            # Create signal structure
            signals = YRS11_Signals(
                yaw_rate_degs=yaw_rate,
                yaw_rate_status=yaw_rate_status,
                lateral_accel_g=lateral_accel,
                lateral_accel_status=lateral_accel_status,
                mcu_status=mcu_status,
                message_count=message_count,
                crc=crc,
                timestamp=timestamp,
                can_id=self.YRS11_CAN_ID,
                raw_data=data
            )
            
            # Validate signal ranges and sensor status
            if self.validate_yrs11_signals(signals):
                self.stats['yrs11_valid_messages'] += 1
                self.update_yrs11_statistics(signals)
                return signals
            else:
                return None
            
        except Exception as e:
            self.log_message(f"YRS11 parsing error: {e}")
            return None
    
    def parse_yrs12_signals(self, data: bytes, timestamp: float) -> Optional[YRS12_Signals]:
        """
        Parse YRS12 CAN message according to DBC specification
        
        YRS12 Signal Layout (64-bit little-endian):
        - CR_Yrs_LongAc: bits [15:0] - 16-bit signed, scale=0.000127465, offset=-4.17677312
        - CF_Yrs_LongAcStat: bits [19:16] - 4-bit longitudinal accel status
        - CF_IMU_ResetStat: bits [23:20] - 4-bit IMU reset status
        - YRS_Temp: bits [31:24] - 8-bit signed, scale=1, offset=-68
        - YRS_TempStat: bits [35:32] - 4-bit temperature status
        - CF_Yrs_Type: bits [39:36] - 4-bit YRS type
        - CR_Yrs_MsgCnt2: bits [51:48] - 4-bit message counter
        - CR_Yrs_Crc2: bits [63:56] - 8-bit CRC
        """
        
        if len(data) != 8:
            self.log_message(f"YRS12 Invalid message length: {len(data)} bytes")
            return None
        
        try:
            # Unpack as little-endian 64-bit unsigned integer
            frame_uint64 = struct.unpack('<Q', data)[0]
            
            # Extract longitudinal acceleration (16-bit signed)
            long_accel_raw = frame_uint64 & 0xFFFF
            if long_accel_raw & 0x8000:  # Sign extend
                long_accel_raw |= 0xFFFF0000
                long_accel_raw = struct.unpack('<i', struct.pack('<I', long_accel_raw & 0xFFFFFFFF))[0]
            longitudinal_accel = long_accel_raw * self.LONG_ACCEL_SCALE + self.LONG_ACCEL_OFFSET
            
            # Extract status fields
            longitudinal_accel_status = (frame_uint64 >> 16) & 0xF
            imu_reset_status = (frame_uint64 >> 20) & 0xF
            
            # Extract temperature (8-bit signed)
            temp_raw = (frame_uint64 >> 24) & 0xFF
            if temp_raw & 0x80:  # Sign extend
                temp_raw |= 0xFFFFFF00
                temp_raw = struct.unpack('<i', struct.pack('<I', temp_raw & 0xFFFFFFFF))[0]
            temperature = temp_raw + self.TEMP_OFFSET
            
            # Extract remaining fields
            temperature_status = (frame_uint64 >> 32) & 0xF
            yrs_type = (frame_uint64 >> 36) & 0xF
            message_count = (frame_uint64 >> 48) & 0xF
            crc = (frame_uint64 >> 56) & 0xFF
            
            # Create signal structure
            signals = YRS12_Signals(
                longitudinal_accel_g=longitudinal_accel,
                longitudinal_accel_status=longitudinal_accel_status,
                imu_reset_status=imu_reset_status,
                temperature_c=temperature,
                temperature_status=temperature_status,
                yrs_type=yrs_type,
                message_count=message_count,
                crc=crc,
                timestamp=timestamp,
                can_id=self.YRS12_CAN_ID,
                raw_data=data
            )
            
            # Validate signal ranges and sensor status
            if self.validate_yrs12_signals(signals):
                self.stats['yrs12_valid_messages'] += 1
                self.update_yrs12_statistics(signals)
                return signals
            else:
                return None
            
        except Exception as e:
            self.log_message(f"YRS12 parsing error: {e}")
            return None
    
    def parse_yrs13_signals(self, data: bytes, timestamp: float) -> Optional[YRS13_Signals]:
        """Parse YRS13 Serial Number message"""
        
        if len(data) != 8:
            return None
        
        try:
            # Extract 48-bit serial number starting at bit 16
            frame_uint64 = struct.unpack('<Q', data)[0]
            serial_number = (frame_uint64 >> 16) & 0xFFFFFFFFFFFF
            
            return YRS13_Signals(
                serial_number=serial_number,
                timestamp=timestamp,
                can_id=self.YRS13_CAN_ID,
                raw_data=data
            )
            
        except Exception as e:
            self.log_message(f"YRS13 parsing error: {e}")
            return None
    
    def validate_yrs11_signals(self, signals: YRS11_Signals) -> bool:
        """Validate YRS11 signal values and sensor status"""
        validation_errors = []
        
        # Range validation
        if not (self.YAW_MIN <= signals.yaw_rate_degs <= self.YAW_MAX):
            validation_errors.append(f"Yaw rate out of range: {signals.yaw_rate_degs:.3f}")
        
        if not (self.ACCEL_MIN <= signals.lateral_accel_g <= self.ACCEL_MAX):
            validation_errors.append(f"Lateral accel out of range: {signals.lateral_accel_g:.4f}")
        
        # Sensor status validation
        if signals.yaw_rate_status != 0:  # 0 = OK
            validation_errors.append(f"Yaw rate sensor fault: {signals.yaw_rate_status}")
            self.stats['sensor_faults'] += 1
        
        if signals.lateral_accel_status != 0:  # 0 = OK
            validation_errors.append(f"Lateral accel sensor fault: {signals.lateral_accel_status}")
            self.stats['sensor_faults'] += 1
        
        if signals.mcu_status != 0:  # 0 = OK
            validation_errors.append(f"MCU fault: {signals.mcu_status}")
            self.stats['sensor_faults'] += 1
        
        if validation_errors:
            self.log_message(f"YRS11 validation errors: {', '.join(validation_errors)}")
            return False
            
        return True
    
    def validate_yrs12_signals(self, signals: YRS12_Signals) -> bool:
        """Validate YRS12 signal values and sensor status"""
        validation_errors = []
        
        # Range validation
        if not (self.ACCEL_MIN <= signals.longitudinal_accel_g <= self.ACCEL_MAX):
            validation_errors.append(f"Longitudinal accel out of range: {signals.longitudinal_accel_g:.4f}")
        
        if not (self.TEMP_MIN <= signals.temperature_c <= self.TEMP_MAX):
            validation_errors.append(f"Temperature out of range: {signals.temperature_c:.1f}")
        
        # Status validation
        if signals.longitudinal_accel_status != 0:  # 0 = OK
            validation_errors.append(f"Longitudinal accel sensor fault: {signals.longitudinal_accel_status}")
            self.stats['sensor_faults'] += 1
        
        if signals.imu_reset_status != 0:  # 0 = Normal
            validation_errors.append(f"IMU reset detected: {signals.imu_reset_status}")
            self.stats['sensor_faults'] += 1
        
        if signals.temperature_status != 0:  # 0 = Normal
            validation_errors.append(f"Temperature sensor fault: {signals.temperature_status}")
            self.stats['temperature_alerts'] += 1
        
        # Temperature alert thresholds
        if signals.temperature_c > 85.0:  # High temperature warning
            self.stats['temperature_alerts'] += 1
            self.log_message(f"High IMU temperature: {signals.temperature_c:.1f}°C")
        
        if validation_errors:
            self.log_message(f"YRS12 validation errors: {', '.join(validation_errors)}")
            return False
            
        return True
    
    def update_yrs11_statistics(self, signals: YRS11_Signals):
        """Update YRS11 statistics"""
        self.stats['max_yaw_rate'] = max(self.stats['max_yaw_rate'], 
                                        abs(signals.yaw_rate_degs))
        self.stats['max_lateral_accel'] = max(self.stats['max_lateral_accel'], 
                                             abs(signals.lateral_accel_g))
    
    def update_yrs12_statistics(self, signals: YRS12_Signals):
        """Update YRS12 statistics"""
        self.stats['max_longitudinal_accel'] = max(self.stats['max_longitudinal_accel'], 
                                                  abs(signals.longitudinal_accel_g))
        self.stats['min_temperature'] = min(self.stats['min_temperature'], 
                                           signals.temperature_c)
        self.stats['max_temperature'] = max(self.stats['max_temperature'], 
                                           signals.temperature_c)
    
    def log_to_csv(self, signals, message_type: str):
        """Log signal data to appropriate CSV file"""
        if not self.csv_logging:
            return
            
        if message_type == 'YRS11' and isinstance(signals, YRS11_Signals):
            with open(self.yrs11_csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    signals.timestamp, signals.can_id, self.yrs11_message_count,
                    signals.yaw_rate_degs, signals.yaw_rate_status,
                    signals.lateral_accel_g, signals.lateral_accel_status,
                    signals.mcu_status, signals.message_count, signals.crc,
                    signals.raw_data.hex().upper()
                ])
        
        elif message_type == 'YRS12' and isinstance(signals, YRS12_Signals):
            with open(self.yrs12_csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    signals.timestamp, signals.can_id, self.yrs12_message_count,
                    signals.longitudinal_accel_g, signals.longitudinal_accel_status,
                    signals.imu_reset_status, signals.temperature_c, signals.temperature_status,
                    signals.yrs_type, signals.message_count, signals.crc,
                    signals.raw_data.hex().upper()
                ])
        
        elif message_type == 'YRS13' and isinstance(signals, YRS13_Signals):
            with open(self.yrs13_csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    signals.timestamp, signals.can_id, self.yrs13_message_count,
                    signals.serial_number, signals.raw_data.hex().upper()
                ])
    
    def calculate_total_acceleration(self) -> Tuple[float, float]:
        """Calculate total acceleration magnitude and direction"""
        if not (self.latest_yrs11 and self.latest_yrs12):
            return 0.0, 0.0
        
        lat_g = self.latest_yrs11.lateral_accel_g
        long_g = self.latest_yrs12.longitudinal_accel_g
        
        total_g = math.sqrt(lat_g**2 + long_g**2)
        angle_rad = math.atan2(lat_g, long_g)
        angle_deg = math.degrees(angle_rad)
        
        return total_g, angle_deg
    
    def display_imu_data(self):
        """Display comprehensive IMU sensor data"""
        
        print(f"\033[2J\033[H")  # Clear screen
        print("=" * 110)
        print(f"YRS INERTIAL MEASUREMENT UNIT MONITOR - {self.interface}")
        print("=" * 110)
        
        print(f"YRS11: {self.yrs11_message_count} | YRS12: {self.yrs12_message_count} | YRS13: {self.yrs13_message_count}")
        print(f"Runtime: {time.time() - self.start_time:.1f}s")
        print()
        
        # YRS11 Data (Yaw Rate + Lateral Acceleration)
        print("YRS11 - YAW RATE & LATERAL ACCELERATION:")
        print("-" * 110)
        if self.latest_yrs11:
            print(f"CAN ID: 0x{self.latest_yrs11.can_id:03X} | Raw: {self.latest_yrs11.raw_data.hex().upper()}")
            
            # Format with ultra-high precision
            yaw_status = "✓ OK" if self.latest_yrs11.yaw_rate_status == 0 else f"⚠ FAULT({self.latest_yrs11.yaw_rate_status})"
            lat_status = "✓ OK" if self.latest_yrs11.lateral_accel_status == 0 else f"⚠ FAULT({self.latest_yrs11.lateral_accel_status})"
            mcu_status = "✓ OK" if self.latest_yrs11.mcu_status == 0 else f"⚠ FAULT({self.latest_yrs11.mcu_status})"
            
            print(f"Yaw Rate:         {self.latest_yrs11.yaw_rate_degs:+8.3f} deg/s  | Status: {yaw_status}")
            print(f"Lateral Accel:    {self.latest_yrs11.lateral_accel_g:+8.4f} g      | Status: {lat_status}")
            print(f"MCU Status: {mcu_status} | Count: {self.latest_yrs11.message_count} | CRC: 0x{self.latest_yrs11.crc:02X}")
        else:
            print("No YRS11 data received")
        print()
        
        # YRS12 Data (Longitudinal Acceleration + Temperature)
        print("YRS12 - LONGITUDINAL ACCELERATION & TEMPERATURE:")
        print("-" * 110)
        if self.latest_yrs12:
            print(f"CAN ID: 0x{self.latest_yrs12.can_id:03X} | Raw: {self.latest_yrs12.raw_data.hex().upper()}")
            
            long_status = "✓ OK" if self.latest_yrs12.longitudinal_accel_status == 0 else f"⚠ FAULT({self.latest_yrs12.longitudinal_accel_status})"
            reset_status = "✓ NORMAL" if self.latest_yrs12.imu_reset_status == 0 else f"⚠ RESET({self.latest_yrs12.imu_reset_status})"
            temp_status = "✓ OK" if self.latest_yrs12.temperature_status == 0 else f"⚠ FAULT({self.latest_yrs12.temperature_status})"
            
            # Temperature warning levels
            temp_indicator = "🟢"
            if self.latest_yrs12.temperature_c > 85.0:
                temp_indicator = "🔴"
            elif self.latest_yrs12.temperature_c > 70.0:
                temp_indicator = "🟡"
            
            print(f"Longitudinal Accel: {self.latest_yrs12.longitudinal_accel_g:+8.4f} g      | Status: {long_status}")
            print(f"Temperature:        {self.latest_yrs12.temperature_c:+6.1f} °C    {temp_indicator} | Status: {temp_status}")
            print(f"IMU Reset: {reset_status} | Type: {self.latest_yrs12.yrs_type} | Count: {self.latest_yrs12.message_count}")
        else:
            print("No YRS12 data received")
        print()
        
        # YRS13 Data (Serial Number)
        print("YRS13 - SENSOR IDENTIFICATION:")
        print("-" * 110)
        if self.latest_yrs13:
            print(f"CAN ID: 0x{self.latest_yrs13.can_id:03X} | Serial Number: 0x{self.latest_yrs13.serial_number:012X}")
        else:
            print("No YRS13 data received")
        print()
        
        # Sensor Fusion Analysis
        print("SENSOR FUSION ANALYSIS:")
        print("-" * 110)
        if self.latest_yrs11 and self.latest_yrs12:
            total_g, angle_deg = self.calculate_total_acceleration()
            
            # Convert to m/s²
            lat_ms2 = self.latest_yrs11.lateral_accel_g * 9.81
            long_ms2 = self.latest_yrs12.longitudinal_accel_g * 9.81
            total_ms2 = total_g * 9.81
            
            print(f"Total Acceleration: {total_g:.4f}g ({total_ms2:.2f} m/s²) | Angle: {angle_deg:+6.1f}°")
            print(f"Lateral:  {lat_ms2:+6.2f} m/s² | Longitudinal: {long_ms2:+6.2f} m/s²")
            print(f"Yaw Rate: {self.latest_yrs11.yaw_rate_degs:+6.3f} deg/s")
            
            # Motion state analysis
            motion_state = "STATIONARY"
            if total_g > 0.05:  # >0.05g threshold
                motion_state = "ACCELERATING"
            if abs(self.latest_yrs11.yaw_rate_degs) > 1.0:  # >1 deg/s threshold
                motion_state = "TURNING"
            if total_g > 0.2:  # >0.2g threshold
                motion_state = "DYNAMIC MOTION"
                
            print(f"Motion State: {motion_state}")
            
            # Precision comparison (YRS vs typical automotive sensors)
            print(f"Precision: YRS {self.YAW_SCALE:.3f} deg/s vs Standard 0.1 deg/s (20x better)")
            print(f"           YRS {self.LAT_ACCEL_SCALE:.6f}g vs Standard 0.01g (78x better)")
        else:
            print("Insufficient data for sensor fusion analysis")
        print()
        
        # Session Statistics
        print("SESSION STATISTICS:")
        print("-" * 110)
        print(f"YRS11 - Total: {self.stats['yrs11_total_messages']} | Valid: {self.stats['yrs11_valid_messages']}")
        print(f"YRS12 - Total: {self.stats['yrs12_total_messages']} | Valid: {self.stats['yrs12_valid_messages']}")
        print(f"Sensor Faults: {self.stats['sensor_faults']} | Temperature Alerts: {self.stats['temperature_alerts']}")
        print(f"Peak Yaw Rate: {self.stats['max_yaw_rate']:.3f} deg/s | Peak Lateral Accel: {self.stats['max_lateral_accel']:.4f}g")
        print(f"Peak Longitudinal Accel: {self.stats['max_longitudinal_accel']:.4f}g")
        if self.stats['max_temperature'] > -100:
            print(f"Temperature Range: {self.stats['min_temperature']:.1f}°C to {self.stats['max_temperature']:.1f}°C")
        
        if self.csv_logging:
            print(f"Data Logging: {self.yrs11_csv_filename}, {self.yrs12_csv_filename}, {self.yrs13_csv_filename}")
        
        print("\nPress Ctrl+C to stop...")
    
    def monitor(self):
        """Main monitoring loop"""
        
        print(f"Starting YRS Inertial Measurement Unit Monitor")
        print(f"Interface: {self.interface}")
        print(f"Targets: YRS11 (0x{self.YRS11_CAN_ID:03X}), YRS12 (0x{self.YRS12_CAN_ID:03X}), YRS13 (0x{self.YRS13_CAN_ID:03X})")
        print(f"Ultra-High Precision IMU Analysis")
        print("=" * 110)
        
        last_display_time = 0
        display_interval = 0.2  # Update display every 200ms
        
        try:
            while True:
                # Receive CAN message
                message = self.bus.recv(timeout=0.1)
                
                if message is None:
                    continue
                
                # Check for YRS11 message
                if message.arbitration_id == self.YRS11_CAN_ID:
                    self.yrs11_message_count += 1
                    self.stats['yrs11_total_messages'] += 1
                    
                    signals = self.parse_yrs11_signals(message.data, message.timestamp)
                    if signals:
                        self.latest_yrs11 = signals
                        self.log_to_csv(signals, 'YRS11')
                
                # Check for YRS12 message
                elif message.arbitration_id == self.YRS12_CAN_ID:
                    self.yrs12_message_count += 1
                    self.stats['yrs12_total_messages'] += 1
                    
                    signals = self.parse_yrs12_signals(message.data, message.timestamp)
                    if signals:
                        self.latest_yrs12 = signals
                        self.log_to_csv(signals, 'YRS12')
                
                # Check for YRS13 message
                elif message.arbitration_id == self.YRS13_CAN_ID:
                    self.yrs13_message_count += 1
                    self.stats['yrs13_total_messages'] += 1
                    
                    signals = self.parse_yrs13_signals(message.data, message.timestamp)
                    if signals:
                        self.latest_yrs13 = signals
                        self.log_to_csv(signals, 'YRS13')
                
                # Update display periodically
                current_time = time.time()
                if current_time - last_display_time >= display_interval:
                    self.display_imu_data()
                    last_display_time = current_time
                    
        except KeyboardInterrupt:
            print(f"\nYRS IMU monitoring stopped")
            self.log_message(f"Monitoring stopped - YRS11: {self.yrs11_message_count}, YRS12: {self.yrs12_message_count}")
            self.print_final_statistics()
        finally:
            if self.bus:
                self.bus.shutdown()
    
    def print_final_statistics(self):
        """Print final session statistics"""
        print(f"\nFINAL YRS IMU STATISTICS:")
        print("=" * 80)
        print(f"Runtime: {time.time() - self.start_time:.1f} seconds")
        print(f"YRS11 Messages: {self.stats['yrs11_total_messages']} (Valid: {self.stats['yrs11_valid_messages']})")
        print(f"YRS12 Messages: {self.stats['yrs12_total_messages']} (Valid: {self.stats['yrs12_valid_messages']})")
        print(f"YRS13 Messages: {self.stats['yrs13_total_messages']}")
        print(f"Sensor Faults: {self.stats['sensor_faults']} | Temperature Alerts: {self.stats['temperature_alerts']}")
        print()
        print(f"Peak Yaw Rate: {self.stats['max_yaw_rate']:.3f} deg/s")
        print(f"Peak Lateral Acceleration: {self.stats['max_lateral_accel']:.4f}g ({self.stats['max_lateral_accel']*9.81:.2f} m/s²)")
        print(f"Peak Longitudinal Acceleration: {self.stats['max_longitudinal_accel']:.4f}g ({self.stats['max_longitudinal_accel']*9.81:.2f} m/s²)")
        if self.stats['max_temperature'] > -100:
            print(f"Temperature Range: {self.stats['min_temperature']:.1f}°C to {self.stats['max_temperature']:.1f}°C")
        
        if self.csv_logging:
            print(f"\nYRS11 Data: {self.yrs11_csv_filename}")
            print(f"YRS12 Data: {self.yrs12_csv_filename}")
            print(f"YRS13 Data: {self.yrs13_csv_filename}")
        if self.log_to_file:
            print(f"Log: {self.log_filename}")

def main():
    parser = argparse.ArgumentParser(
        description='YRS Inertial Measurement Unit Monitor - Ultra-High Precision',
        epilog='''
YRS provides ultra-high precision vehicle motion data:
- Yaw Rate: ±163.84 deg/s (0.005 deg/s resolution - 20x better than standard)
- Acceleration: ±4.177g (0.000127465g resolution - 78x better than standard)
- Temperature monitoring with fault detection
- Comprehensive sensor fusion analysis

DBC-compliant parsing with advanced validation.
        '''
    )
    parser.add_argument('--interface', '-i', default='can1', 
                       help='CAN interface (default: can1)')
    parser.add_argument('--no-csv', action='store_true',
                       help='Disable CSV data logging')
    parser.add_argument('--no-log', action='store_true', 
                       help='Disable text file logging')
    
    args = parser.parse_args()
    
    print("YRS Inertial Measurement Unit Monitor")
    print("Ultra-High Precision Vehicle Motion Analysis")
    print("DBC-Compliant with Advanced Sensor Fusion")
    print()
    
    monitor = YRS_IMU_Monitor(
        interface=args.interface,
        log_to_file=not args.no_log,
        csv_logging=not args.no_csv
    )
    monitor.monitor()

if __name__ == "__main__":
    main()
