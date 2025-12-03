#!/usr/bin/env python3
"""
Vehicle Speed Resources Monitor
CAN IDs: 0x316 (EMS11), 0x4F1 (CLU11)
Multi-Source Vehicle Speed Validation

Decodes and cross-validates multiple speed sources:
- EMS11: Engine Management Speed (0-254 km/h)
- CLU11: Cluster Speed with decimal precision (0-255.5 km/h)
- Unit detection (km/h vs MPH)
- Speed correlation analysis and deviation detection

DBC Specification Compliance with multi-source validation
"""

import can
import struct
import time
import argparse
import sys
import json
import csv
import statistics
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
from collections import deque

@dataclass
class EMS11_Speed:
    """EMS11 Engine Management Speed signals"""
    vehicle_speed_kmh: float              # km/h (0-254)
    engine_running: bool                  # F_N_ENG flag
    ignition_switch: bool                 # SWI_IGK flag
    power_unit_control: bool              # PUC_STAT flag
    ac_relay: bool                        # RLY_AC flag
    engine_rpm: float                     # N (0-16383.75 rpm)
    torque_indicated: float               # TQI (0-99.6094 %)
    torque_friction: float                # TQFR (0-99.6094 %)
    throttle_position: float              # TPS (-15.02 to 104.69 %)
    accelerator_pedal: float              # PV_AV_CAN (0-99.6 %)
    timestamp: float
    can_id: int
    raw_data: bytes

@dataclass
class CLU11_Speed:
    """CLU11 Cluster Speed signals"""
    vehicle_speed_kmh: float              # CF_Clu_Vanz (0-255.5 km/h)
    speed_decimal: float                  # CF_Clu_VanzDecimal (0-0.375)
    speed_unit: bool                      # CF_Clu_SPEED_UNIT (0=km/h, 1=MPH)
    cruise_switch_state: int              # CF_Clu_CruiseSwState (0-7)
    cruise_main_switch: bool              # CF_Clu_CruiseSwMain
    detent_output: bool                   # CF_Clu_DetentOut
    rheostat_level: int                   # CF_Clu_RheostatLevel (0-31)
    alive_counter: int                    # CF_Clu_AliveCnt1 (0-15)
    timestamp: float
    can_id: int
    raw_data: bytes

@dataclass
class SpeedCorrelation:
    """Speed correlation analysis data"""
    ems_speed: float
    clu_speed: float
    deviation_kmh: float
    deviation_percent: float
    timestamp: float
    correlation_quality: str

class SpeedResourcesMonitor:
    def __init__(self, interface='can1', log_to_file=True, csv_logging=True):
        self.interface = interface
        self.ems_message_count = 0
        self.clu_message_count = 0
        self.start_time = time.time()
        self.log_to_file = log_to_file
        self.csv_logging = csv_logging
        
        # CAN IDs
        self.EMS11_CAN_ID = 0x316   # 790 decimal
        self.CLU11_CAN_ID = 0x4F1   # 1265 decimal
        
        # EMS11 constants
        self.EMS_SPEED_SCALE = 1.0          # km/h per LSB
        self.EMS_RPM_SCALE = 0.25           # rpm per LSB
        self.EMS_TPS_SCALE = 0.4694836      # % per LSB
        self.EMS_TPS_OFFSET = -15.0234742   # % offset
        self.EMS_PEDAL_SCALE = 0.390625     # % per LSB
        self.EMS_TORQUE_SCALE = 0.390625    # % per LSB
        
        # CLU11 constants
        self.CLU_SPEED_SCALE = 0.5          # km/h per LSB
        self.CLU_DECIMAL_SCALE = 0.125      # decimal fraction per LSB
        
        # Unit conversion
        self.KMH_TO_MPH = 1.0 / 1.609344
        self.MPH_TO_KMH = 1.609344
        
        # Validation ranges
        self.SPEED_MIN = 0.0                # km/h
        self.SPEED_MAX = 300.0              # km/h (reasonable limit)
        self.CORRELATION_THRESHOLD = 5.0    # km/h deviation threshold
        self.RPM_MAX = 8000.0               # rpm
        
        # Latest signal data
        self.latest_ems = None
        self.latest_clu = None
        
        # Speed history for correlation analysis
        self.speed_history = deque(maxlen=100)  # Last 100 correlations
        
        # Statistics tracking
        self.stats = {
            'ems_total_messages': 0,
            'ems_valid_messages': 0,
            'clu_total_messages': 0,
            'clu_valid_messages': 0,
            'correlations_analyzed': 0,
            'good_correlations': 0,
            'poor_correlations': 0,
            'max_ems_speed': 0.0,
            'max_clu_speed': 0.0,
            'max_deviation': 0.0,
            'max_engine_rpm': 0.0,
            'mph_mode_detected': 0,
            'kmh_mode_detected': 0
        }
        
        # Initialize logging
        self.setup_logging()
        
        # Connect to CAN bus
        try:
            self.bus = can.interface.Bus(channel=interface, bustype='socketcan')
            print(f"✓ Connected to {interface}")
            self.log_message(f"Speed Resources Monitor started on {interface}")
        except Exception as e:
            print(f"✗ Failed to connect to {interface}: {e}")
            sys.exit(1)
    
    def setup_logging(self):
        """Initialize comprehensive logging system"""
        if self.log_to_file or self.csv_logging:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        if self.log_to_file:
            self.log_filename = f"speed_resources_log_{timestamp}.txt"
            with open(self.log_filename, 'w') as f:
                f.write(f"Speed Resources Monitor Log\n")
                f.write(f"Started: {datetime.now().isoformat()}\n")
                f.write(f"Interface: {self.interface}\n")
                f.write("Multi-Source Vehicle Speed Validation\n")
                f.write("=" * 80 + "\n\n")
        
        if self.csv_logging:
            # EMS11 CSV
            self.ems_csv_filename = f"ems11_speed_data_{timestamp}.csv"
            with open(self.ems_csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'can_id', 'message_count',
                    'vehicle_speed_kmh', 'engine_running', 'ignition_switch',
                    'puc_status', 'ac_relay', 'engine_rpm', 'torque_indicated',
                    'torque_friction', 'throttle_position', 'accelerator_pedal',
                    'raw_data_hex'
                ])
            
            # CLU11 CSV
            self.clu_csv_filename = f"clu11_speed_data_{timestamp}.csv"
            with open(self.clu_csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'can_id', 'message_count',
                    'vehicle_speed_kmh', 'speed_decimal', 'speed_unit',
                    'cruise_switch_state', 'cruise_main_switch', 'detent_output',
                    'rheostat_level', 'alive_counter', 'raw_data_hex'
                ])
            
            # Correlation CSV
            self.correlation_csv_filename = f"speed_correlation_{timestamp}.csv"
            with open(self.correlation_csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'ems_speed_kmh', 'clu_speed_kmh',
                    'deviation_kmh', 'deviation_percent', 'correlation_quality'
                ])
    
    def log_message(self, message: str):
        """Log message to file with timestamp"""
        if self.log_to_file:
            with open(self.log_filename, 'a') as f:
                f.write(f"[{datetime.now().isoformat()}] {message}\n")
    
    def parse_ems11_signals(self, data: bytes, timestamp: float) -> Optional[EMS11_Speed]:
        """
        Parse EMS11 CAN message according to DBC specification
        
        EMS11 Signal Layout (64-bit little-endian):
        - SWI_IGK: bit [0] - ignition switch
        - F_N_ENG: bit [1] - engine running flag
        - PUC_STAT: bit [3] - power unit control status
        - RLY_AC: bit [6] - AC relay status
        - TQI_ACOR: bits [15:8] - 8-bit torque indicated
        - N: bits [31:16] - 16-bit engine RPM
        - TQI: bits [39:32] - 8-bit torque indicated
        - TQFR: bits [47:40] - 8-bit torque friction
        - VS: bits [55:48] - 8-bit vehicle speed
        - TPS: bits [47:40] - 8-bit throttle position (overlapped, need verification)
        - PV_AV_CAN: bits [55:48] - 8-bit accelerator pedal (overlapped, need verification)
        """
        
        if len(data) != 8:
            self.log_message(f"EMS11 Invalid message length: {len(data)} bytes")
            return None
        
        try:
            # Unpack as little-endian 64-bit unsigned integer
            frame_uint64 = struct.unpack('<Q', data)[0]
            
            # Extract control flags
            ignition_switch = bool(frame_uint64 & 0x1)
            engine_running = bool((frame_uint64 >> 1) & 0x1)
            puc_status = bool((frame_uint64 >> 3) & 0x1)
            ac_relay = bool((frame_uint64 >> 6) & 0x1)
            
            # Extract performance data
            tqi_acor = ((frame_uint64 >> 8) & 0xFF) * self.EMS_TORQUE_SCALE
            engine_rpm = ((frame_uint64 >> 16) & 0xFFFF) * self.EMS_RPM_SCALE
            tqi = ((frame_uint64 >> 32) & 0xFF) * self.EMS_TORQUE_SCALE
            tqfr = ((frame_uint64 >> 40) & 0xFF) * self.EMS_TORQUE_SCALE
            
            # Extract vehicle speed
            vehicle_speed = ((frame_uint64 >> 48) & 0xFF) * self.EMS_SPEED_SCALE
            
            # Note: TPS and PV_AV_CAN locations need DBC verification
            # Using approximations for now
            throttle_position = ((frame_uint64 >> 40) & 0xFF) * self.EMS_TPS_SCALE + self.EMS_TPS_OFFSET
            accelerator_pedal = ((frame_uint64 >> 48) & 0xFF) * self.EMS_PEDAL_SCALE
            
            # Create signal structure
            signals = EMS11_Speed(
                vehicle_speed_kmh=vehicle_speed,
                engine_running=engine_running,
                ignition_switch=ignition_switch,
                power_unit_control=puc_status,
                ac_relay=ac_relay,
                engine_rpm=engine_rpm,
                torque_indicated=tqi,
                torque_friction=tqfr,
                throttle_position=throttle_position,
                accelerator_pedal=accelerator_pedal,
                timestamp=timestamp,
                can_id=self.EMS11_CAN_ID,
                raw_data=data
            )
            
            # Validate signal ranges
            if self.validate_ems11_signals(signals):
                self.stats['ems_valid_messages'] += 1
                self.update_ems_statistics(signals)
                return signals
            else:
                return None
            
        except Exception as e:
            self.log_message(f"EMS11 parsing error: {e}")
            return None
    
    def parse_clu11_signals(self, data: bytes, timestamp: float) -> Optional[CLU11_Speed]:
        """
        Parse CLU11 CAN message according to DBC specification
        
        CLU11 Signal Layout (32-bit little-endian):
        - CF_Clu_CruiseSwState: bits [2:0] - 3-bit cruise switch state
        - CF_Clu_CruiseSwMain: bit [3] - cruise main switch
        - CF_Clu_SldMainSW: bit [4] - slide main switch
        - CF_Clu_ParityBit1: bit [5] - parity bit
        - CF_Clu_VanzDecimal: bits [7:6] - 2-bit decimal part
        - CF_Clu_Vanz: bits [16:8] - 9-bit vehicle speed
        - CF_Clu_SPEED_UNIT: bit [17] - speed unit (0=km/h, 1=MPH)
        - CF_Clu_DetentOut: bit [18] - detent output
        - CF_Clu_RheostatLevel: bits [23:19] - 5-bit rheostat level
        - CF_Clu_AliveCnt1: bits [31:28] - 4-bit alive counter
        """
        
        if len(data) != 4:
            self.log_message(f"CLU11 Invalid message length: {len(data)} bytes")
            return None
        
        try:
            # Extend to 8 bytes for easier processing
            data_extended = data + b'\x00\x00\x00\x00'
            frame_uint64 = struct.unpack('<Q', data_extended)[0]
            
            # Extract signals
            cruise_switch_state = frame_uint64 & 0x7
            cruise_main_switch = bool((frame_uint64 >> 3) & 0x1)
            speed_decimal_raw = (frame_uint64 >> 6) & 0x3
            speed_decimal = speed_decimal_raw * self.CLU_DECIMAL_SCALE
            
            # Extract vehicle speed (9-bit)
            vehicle_speed_raw = (frame_uint64 >> 8) & 0x1FF
            vehicle_speed = vehicle_speed_raw * self.CLU_SPEED_SCALE
            
            # Extract unit and other signals
            speed_unit = bool((frame_uint64 >> 17) & 0x1)  # 0=km/h, 1=MPH
            detent_output = bool((frame_uint64 >> 18) & 0x1)
            rheostat_level = (frame_uint64 >> 19) & 0x1F
            alive_counter = (frame_uint64 >> 28) & 0xF
            
            # Add decimal precision to speed
            total_speed = vehicle_speed + speed_decimal
            
            # Convert to km/h if in MPH mode
            if speed_unit:  # MPH mode
                total_speed_kmh = total_speed * self.MPH_TO_KMH
                self.stats['mph_mode_detected'] += 1
            else:  # km/h mode
                total_speed_kmh = total_speed
                self.stats['kmh_mode_detected'] += 1
            
            # Create signal structure
            signals = CLU11_Speed(
                vehicle_speed_kmh=total_speed_kmh,
                speed_decimal=speed_decimal,
                speed_unit=speed_unit,
                cruise_switch_state=cruise_switch_state,
                cruise_main_switch=cruise_main_switch,
                detent_output=detent_output,
                rheostat_level=rheostat_level,
                alive_counter=alive_counter,
                timestamp=timestamp,
                can_id=self.CLU11_CAN_ID,
                raw_data=data
            )
            
            # Validate signal ranges
            if self.validate_clu11_signals(signals):
                self.stats['clu_valid_messages'] += 1
                self.update_clu_statistics(signals)
                return signals
            else:
                return None
            
        except Exception as e:
            self.log_message(f"CLU11 parsing error: {e}")
            return None
    
    def validate_ems11_signals(self, signals: EMS11_Speed) -> bool:
        """Validate EMS11 signal values"""
        validation_errors = []
        
        if not (self.SPEED_MIN <= signals.vehicle_speed_kmh <= self.SPEED_MAX):
            validation_errors.append(f"EMS speed out of range: {signals.vehicle_speed_kmh:.1f}")
        
        if not (0 <= signals.engine_rpm <= self.RPM_MAX):
            validation_errors.append(f"RPM out of range: {signals.engine_rpm:.1f}")
        
        if not (0 <= signals.torque_indicated <= 100):
            validation_errors.append(f"Torque out of range: {signals.torque_indicated:.1f}")
        
        if validation_errors:
            self.log_message(f"EMS11 validation errors: {', '.join(validation_errors)}")
            return False
            
        return True
    
    def validate_clu11_signals(self, signals: CLU11_Speed) -> bool:
        """Validate CLU11 signal values"""
        validation_errors = []
        
        if not (self.SPEED_MIN <= signals.vehicle_speed_kmh <= self.SPEED_MAX):
            validation_errors.append(f"CLU speed out of range: {signals.vehicle_speed_kmh:.1f}")
        
        if validation_errors:
            self.log_message(f"CLU11 validation errors: {', '.join(validation_errors)}")
            return False
            
        return True
    
    def update_ems_statistics(self, signals: EMS11_Speed):
        """Update EMS11 statistics"""
        self.stats['max_ems_speed'] = max(self.stats['max_ems_speed'], 
                                         signals.vehicle_speed_kmh)
        self.stats['max_engine_rpm'] = max(self.stats['max_engine_rpm'], 
                                          signals.engine_rpm)
    
    def update_clu_statistics(self, signals: CLU11_Speed):
        """Update CLU11 statistics"""
        self.stats['max_clu_speed'] = max(self.stats['max_clu_speed'], 
                                         signals.vehicle_speed_kmh)
    
    def analyze_speed_correlation(self):
        """Analyze correlation between EMS and CLU speed sources"""
        if not (self.latest_ems and self.latest_clu):
            return None
        
        ems_speed = self.latest_ems.vehicle_speed_kmh
        clu_speed = self.latest_clu.vehicle_speed_kmh
        
        # Calculate deviation
        deviation_kmh = abs(ems_speed - clu_speed)
        
        # Calculate percentage deviation (avoid division by zero)
        avg_speed = (ems_speed + clu_speed) / 2
        if avg_speed > 1.0:  # Only calculate percentage if speeds are meaningful
            deviation_percent = (deviation_kmh / avg_speed) * 100
        else:
            deviation_percent = 0.0
        
        # Determine correlation quality
        if deviation_kmh <= 2.0:
            quality = "EXCELLENT"
            self.stats['good_correlations'] += 1
        elif deviation_kmh <= 5.0:
            quality = "GOOD"
            self.stats['good_correlations'] += 1
        elif deviation_kmh <= 10.0:
            quality = "FAIR"
        else:
            quality = "POOR"
            self.stats['poor_correlations'] += 1
            self.log_message(f"Poor speed correlation: EMS={ems_speed:.1f}, CLU={clu_speed:.1f}, Dev={deviation_kmh:.1f}")
        
        # Create correlation data
        correlation = SpeedCorrelation(
            ems_speed=ems_speed,
            clu_speed=clu_speed,
            deviation_kmh=deviation_kmh,
            deviation_percent=deviation_percent,
            timestamp=max(self.latest_ems.timestamp, self.latest_clu.timestamp),
            correlation_quality=quality
        )
        
        # Add to history
        self.speed_history.append(correlation)
        self.stats['correlations_analyzed'] += 1
        self.stats['max_deviation'] = max(self.stats['max_deviation'], deviation_kmh)
        
        return correlation
    
    def get_correlation_statistics(self) -> Dict:
        """Calculate correlation statistics from history"""
        if not self.speed_history:
            return {}
        
        deviations = [c.deviation_kmh for c in self.speed_history]
        
        return {
            'avg_deviation': statistics.mean(deviations),
            'max_deviation': max(deviations),
            'min_deviation': min(deviations),
            'std_deviation': statistics.stdev(deviations) if len(deviations) > 1 else 0.0,
            'samples': len(deviations)
        }
    
    def log_to_csv(self, signals, message_type: str):
        """Log signal data to appropriate CSV file"""
        if not self.csv_logging:
            return
            
        if message_type == 'EMS11' and isinstance(signals, EMS11_Speed):
            with open(self.ems_csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    signals.timestamp, signals.can_id, self.ems_message_count,
                    signals.vehicle_speed_kmh, signals.engine_running, signals.ignition_switch,
                    signals.power_unit_control, signals.ac_relay, signals.engine_rpm,
                    signals.torque_indicated, signals.torque_friction, signals.throttle_position,
                    signals.accelerator_pedal, signals.raw_data.hex().upper()
                ])
        
        elif message_type == 'CLU11' and isinstance(signals, CLU11_Speed):
            with open(self.clu_csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    signals.timestamp, signals.can_id, self.clu_message_count,
                    signals.vehicle_speed_kmh, signals.speed_decimal, signals.speed_unit,
                    signals.cruise_switch_state, signals.cruise_main_switch, signals.detent_output,
                    signals.rheostat_level, signals.alive_counter, signals.raw_data.hex().upper()
                ])
    
    def log_correlation_to_csv(self, correlation: SpeedCorrelation):
        """Log correlation data to CSV"""
        if self.csv_logging:
            with open(self.correlation_csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    correlation.timestamp, correlation.ems_speed, correlation.clu_speed,
                    correlation.deviation_kmh, correlation.deviation_percent, correlation.correlation_quality
                ])
    
    def display_speed_data(self):
        """Display comprehensive speed monitoring data"""
        
        print(f"\033[2J\033[H")  # Clear screen
        print("=" * 120)
        print(f"VEHICLE SPEED RESOURCES MONITOR - {self.interface}")
        print("=" * 120)
        
        print(f"EMS11: {self.ems_message_count} | CLU11: {self.clu_message_count}")
        print(f"Runtime: {time.time() - self.start_time:.1f}s")
        print()
        
        # EMS11 Data (Engine Management Speed)
        print("EMS11 - ENGINE MANAGEMENT SPEED:")
        print("-" * 120)
        if self.latest_ems:
            print(f"CAN ID: 0x{self.latest_ems.can_id:03X} | Raw: {self.latest_ems.raw_data.hex().upper()}")
            
            engine_status = "🟢 RUNNING" if self.latest_ems.engine_running else "🔴 STOPPED"
            ignition_status = "🟢 ON" if self.latest_ems.ignition_switch else "🔴 OFF"
            
            print(f"Vehicle Speed:    {self.latest_ems.vehicle_speed_kmh:6.1f} km/h ({self.latest_ems.vehicle_speed_kmh * self.KMH_TO_MPH:5.1f} mph)")
            print(f"Engine:           {engine_status} | RPM: {self.latest_ems.engine_rpm:6.0f} | Ignition: {ignition_status}")
            print(f"Torque Indicated: {self.latest_ems.torque_indicated:5.1f}% | Friction: {self.latest_ems.torque_friction:5.1f}%")
            print(f"Throttle:         {self.latest_ems.throttle_position:5.1f}% | Accelerator: {self.latest_ems.accelerator_pedal:5.1f}%")
        else:
            print("No EMS11 data received")
        print()
        
        # CLU11 Data (Cluster Speed)
        print("CLU11 - CLUSTER SPEED:")
        print("-" * 120)
        if self.latest_clu:
            print(f"CAN ID: 0x{self.latest_clu.can_id:03X} | Raw: {self.latest_clu.raw_data.hex().upper()}")
            
            unit_str = "MPH" if self.latest_clu.speed_unit else "km/h"
            original_speed = self.latest_clu.vehicle_speed_kmh / self.MPH_TO_KMH if self.latest_clu.speed_unit else self.latest_clu.vehicle_speed_kmh
            
            print(f"Vehicle Speed:    {self.latest_clu.vehicle_speed_kmh:6.1f} km/h ({original_speed:5.1f} {unit_str} display)")
            print(f"Precision:        {self.latest_clu.speed_decimal:5.3f} decimal | Unit: {unit_str}")
            print(f"Cruise Control:   State={self.latest_clu.cruise_switch_state} | Main={self.latest_clu.cruise_main_switch}")
            print(f"Alive Counter:    {self.latest_clu.alive_counter} | Rheostat: {self.latest_clu.rheostat_level}")
        else:
            print("No CLU11 data received")
        print()
        
        # Speed Correlation Analysis
        print("SPEED CORRELATION ANALYSIS:")
        print("-" * 120)
        current_correlation = self.analyze_speed_correlation()
        if current_correlation:
            status_icon = {"EXCELLENT": "🟢", "GOOD": "🟡", "FAIR": "🟠", "POOR": "🔴"}.get(
                current_correlation.correlation_quality, "❓")
            
            print(f"EMS Speed:        {current_correlation.ems_speed:6.1f} km/h")
            print(f"CLU Speed:        {current_correlation.clu_speed:6.1f} km/h")
            print(f"Deviation:        {current_correlation.deviation_kmh:6.2f} km/h ({current_correlation.deviation_percent:5.1f}%)")
            print(f"Correlation:      {status_icon} {current_correlation.correlation_quality}")
            
            # Add correlation to CSV
            self.log_correlation_to_csv(current_correlation)
            
            # Historical correlation statistics
            corr_stats = self.get_correlation_statistics()
            if corr_stats:
                print(f"Average Deviation: {corr_stats['avg_deviation']:.2f} km/h (±{corr_stats['std_deviation']:.2f})")
                print(f"Peak Deviation:   {corr_stats['max_deviation']:.2f} km/h | Samples: {corr_stats['samples']}")
        else:
            print("Insufficient data for correlation analysis")
        print()
        
        # Unit Detection Analysis
        print("UNIT DETECTION ANALYSIS:")
        print("-" * 120)
        total_detections = self.stats['mph_mode_detected'] + self.stats['kmh_mode_detected']
        if total_detections > 0:
            mph_percent = (self.stats['mph_mode_detected'] / total_detections) * 100
            kmh_percent = (self.stats['kmh_mode_detected'] / total_detections) * 100
            primary_unit = "MPH" if mph_percent > 50 else "km/h"
            
            print(f"Primary Unit:     {primary_unit}")
            print(f"km/h Mode:        {self.stats['kmh_mode_detected']} messages ({kmh_percent:.1f}%)")
            print(f"MPH Mode:         {self.stats['mph_mode_detected']} messages ({mph_percent:.1f}%)")
        else:
            print("No unit detection data available")
        print()
        
        # Session Statistics
        print("SESSION STATISTICS:")
        print("-" * 120)
        print(f"EMS11 - Total: {self.stats['ems_total_messages']} | Valid: {self.stats['ems_valid_messages']}")
        print(f"CLU11 - Total: {self.stats['clu_total_messages']} | Valid: {self.stats['clu_valid_messages']}")
        print(f"Correlations: {self.stats['correlations_analyzed']} | Good: {self.stats['good_correlations']} | Poor: {self.stats['poor_correlations']}")
        print(f"Peak EMS Speed: {self.stats['max_ems_speed']:.1f} km/h | Peak CLU Speed: {self.stats['max_clu_speed']:.1f} km/h")
        print(f"Peak Engine RPM: {self.stats['max_engine_rpm']:.0f} | Max Deviation: {self.stats['max_deviation']:.2f} km/h")
        
        if self.csv_logging:
            print(f"Data Logging: {self.ems_csv_filename}, {self.clu_csv_filename}, {self.correlation_csv_filename}")
        
        print("\nPress Ctrl+C to stop...")
    
    def monitor(self):
        """Main monitoring loop"""
        
        print(f"Starting Vehicle Speed Resources Monitor")
        print(f"Interface: {self.interface}")
        print(f"Targets: EMS11 (0x{self.EMS11_CAN_ID:03X}), CLU11 (0x{self.CLU11_CAN_ID:03X})")
        print(f"Multi-Source Speed Validation with Unit Detection")
        print("=" * 120)
        
        last_display_time = 0
        display_interval = 0.3  # Update display every 300ms
        
        try:
            while True:
                # Receive CAN message
                message = self.bus.recv(timeout=0.1)
                
                if message is None:
                    continue
                
                # Check for EMS11 message
                if message.arbitration_id == self.EMS11_CAN_ID:
                    self.ems_message_count += 1
                    self.stats['ems_total_messages'] += 1
                    
                    signals = self.parse_ems11_signals(message.data, message.timestamp)
                    if signals:
                        self.latest_ems = signals
                        self.log_to_csv(signals, 'EMS11')
                
                # Check for CLU11 message
                elif message.arbitration_id == self.CLU11_CAN_ID:
                    self.clu_message_count += 1
                    self.stats['clu_total_messages'] += 1
                    
                    signals = self.parse_clu11_signals(message.data, message.timestamp)
                    if signals:
                        self.latest_clu = signals
                        self.log_to_csv(signals, 'CLU11')
                
                # Update display periodically
                current_time = time.time()
                if current_time - last_display_time >= display_interval:
                    self.display_speed_data()
                    last_display_time = current_time
                    
        except KeyboardInterrupt:
            print(f"\nSpeed Resources monitoring stopped")
            self.log_message(f"Monitoring stopped - EMS11: {self.ems_message_count}, CLU11: {self.clu_message_count}")
            self.print_final_statistics()
        finally:
            if self.bus:
                self.bus.shutdown()
    
    def print_final_statistics(self):
        """Print final session statistics"""
        print(f"\nFINAL SPEED RESOURCES STATISTICS:")
        print("=" * 80)
        print(f"Runtime: {time.time() - self.start_time:.1f} seconds")
        print(f"EMS11 Messages: {self.stats['ems_total_messages']} (Valid: {self.stats['ems_valid_messages']})")
        print(f"CLU11 Messages: {self.stats['clu_total_messages']} (Valid: {self.stats['clu_valid_messages']})")
        print()
        
        # Correlation quality assessment
        if self.stats['correlations_analyzed'] > 0:
            good_percent = (self.stats['good_correlations'] / self.stats['correlations_analyzed']) * 100
            print(f"Speed Correlation Quality: {good_percent:.1f}% good ({self.stats['good_correlations']}/{self.stats['correlations_analyzed']})")
            
            corr_stats = self.get_correlation_statistics()
            if corr_stats:
                print(f"Average Speed Deviation: {corr_stats['avg_deviation']:.2f} ± {corr_stats['std_deviation']:.2f} km/h")
                print(f"Maximum Speed Deviation: {corr_stats['max_deviation']:.2f} km/h")
        
        print(f"Peak Speeds - EMS11: {self.stats['max_ems_speed']:.1f} km/h | CLU11: {self.stats['max_clu_speed']:.1f} km/h")
        print(f"Peak Engine RPM: {self.stats['max_engine_rpm']:.0f}")
        
        # Unit mode analysis
        total_detections = self.stats['mph_mode_detected'] + self.stats['kmh_mode_detected']
        if total_detections > 0:
            mph_percent = (self.stats['mph_mode_detected'] / total_detections) * 100
            primary_unit = "MPH" if mph_percent > 50 else "km/h"
            print(f"Primary Display Unit: {primary_unit} ({mph_percent:.1f}% MPH, {100-mph_percent:.1f}% km/h)")
        
        if self.csv_logging:
            print(f"\nEMS11 Data: {self.ems_csv_filename}")
            print(f"CLU11 Data: {self.clu_csv_filename}")
            print(f"Correlation Data: {self.correlation_csv_filename}")
        if self.log_to_file:
            print(f"Log: {self.log_filename}")

def main():
    parser = argparse.ArgumentParser(
        description='Vehicle Speed Resources Monitor - Multi-Source Validation',
        epilog='''
Monitors and cross-validates multiple vehicle speed sources:
- EMS11: Engine management system speed reference
- CLU11: Cluster display speed with unit detection
- Real-time correlation analysis and deviation detection
- Unit conversion handling (km/h ↔ MPH)

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
    
    print("Vehicle Speed Resources Monitor")
    print("Multi-Source Speed Validation & Unit Detection")
    print("DBC-Compliant with Correlation Analysis")
    print()
    
    monitor = SpeedResourcesMonitor(
        interface=args.interface,
        log_to_file=not args.no_log,
        csv_logging=not args.no_csv
    )
    monitor.monitor()

if __name__ == "__main__":
    main()
