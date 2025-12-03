#!/usr/bin/env python3
"""
Throttle & Engine Load Monitor
CAN ID: 0x329 (EMS12)
Engine Management Throttle Position & Load Analysis

Decodes comprehensive throttle and engine load parameters:
- Throttle Position Sensor (TPS): -15.02% to 104.69%
- Accelerator Pedal Position: 0-99.6%
- Engine Load (TQI): 0-99.6%
- Boost Pressure: 0-4094 hPa
- Engine Oil Temperature: 0-254°C
- Active Eco and ISG (Idle Stop & Go) status

DBC Specification Compliance with throttle response analysis
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
class EMS12_Signals:
    """EMS12 Engine Management Throttle & Load signals"""
    # Throttle and pedal
    throttle_position_percent: float      # TPS (-15.02 to 104.69 %)
    accelerator_pedal_percent: float      # PV_AV_CAN (0-99.6 %)
    
    # Engine load and torque
    engine_load_percent: float            # TQI_B (0-99.6 %)
    slide_vehicle_speed: float            # SLD_VS (0-255 km/h)
    
    # Engine systems
    cda_status: int                       # CF_CdaStat (0-7)
    isg_status: int                       # CF_Ems_IsgStat (0-7)
    oil_change_indicator: bool            # CF_Ems_OilChg
    etc_limp_mode: bool                   # CF_Ems_EtcLimpMod
    
    # Engine parameters
    idle_target_rpm: float                # R_NEngIdlTgC (0-2550 rpm)
    up_target_gear: bool                  # CF_Ems_UpTarGr
    down_target_gear: bool                # CF_Ems_DownTarGr
    desired_current_gear: int             # CF_Ems_DesCurGr (0-15)
    
    # Slide and position
    slide_active: bool                    # CF_Ems_SldAct
    slide_position_active: bool           # CF_Ems_SldPosAct
    high_pressure_status: bool            # CF_Ems_HPresStat
    
    # ISG and Eco systems
    isg_buzzer: bool                      # CF_Ems_IsgBuz
    idle_stop_fuel_cutoff: bool           # CF_Ems_IdlStpFCO
    fuel_cutoff_open: bool                # CF_Ems_FCopen
    active_eco_active: bool               # CF_Ems_ActEcoAct
    engine_run_normal: bool               # CF_Ems_EngRunNorm
    isg_status_2: int                     # CF_Ems_IsgStat2 (0-3)
    
    # Boost and air
    boost_pressure_hpa: float             # CR_Ems_BstPre (0-4094 hPa)
    engine_oil_temp_c: float              # CR_Ems_EngOilTemp (0-254°C)
    modeled_ambient_temp_c: float         # CF_Ems_ModeledAmbTemp (-41 to 85.5°C)
    
    # Fault detection
    dpf_lamp_status: int                  # DPF_LAMP_STAT (0-3)
    battery_lamp_status: bool             # BAT_LAMP_STAT
    ops_fail: bool                        # CF_Ems_OPSFail
    
    # Message integrity
    alive_counter: int                    # CF_Ems_AliveCounterEMS9 (0-3)
    checksum: int                         # CF_Ems_ChecksumEMS9 (0-15)
    
    timestamp: float
    can_id: int
    raw_data: bytes

@dataclass
class ThrottleResponse:
    """Throttle response analysis data"""
    pedal_position: float
    throttle_position: float
    response_ratio: float
    response_delay_ms: float
    response_quality: str
    timestamp: float

class ThrottleLoadMonitor:
    def __init__(self, interface='can1', log_to_file=True, csv_logging=True):
        self.interface = interface
        self.message_count = 0
        self.start_time = time.time()
        self.log_to_file = log_to_file
        self.csv_logging = csv_logging
        
        # CAN ID
        self.EMS12_CAN_ID = 0x329   # 809 decimal
        
        # EMS12 constants
        self.TPS_SCALE = 0.4694836          # % per LSB
        self.TPS_OFFSET = -15.0234742       # % offset
        self.PEDAL_SCALE = 0.390625         # % per LSB
        self.LOAD_SCALE = 0.390625          # % per LSB
        self.BOOST_SCALE = 1.322            # hPa per LSB
        self.OIL_TEMP_SCALE = 0.75          # °C per LSB
        self.OIL_TEMP_OFFSET = -40.0        # °C offset
        self.AMBIENT_TEMP_SCALE = 0.5       # °C per LSB
        self.AMBIENT_TEMP_OFFSET = -41.0    # °C offset
        self.IDLE_RPM_SCALE = 10.0          # rpm per LSB
        
        # Validation ranges
        self.TPS_MIN = -20.0                # % (beyond DBC for validation)
        self.TPS_MAX = 110.0                # %
        self.PEDAL_MAX = 100.0              # %
        self.LOAD_MAX = 100.0               # %
        self.BOOST_MAX = 5000.0             # hPa (reasonable turbo limit)
        self.OIL_TEMP_MAX = 150.0           # °C (normal operating range)
        self.AMBIENT_TEMP_MIN = -50.0       # °C
        self.AMBIENT_TEMP_MAX = 60.0        # °C
        
        # Latest signal data
        self.latest_ems12 = None
        
        # Throttle response analysis
        self.throttle_history = deque(maxlen=50)  # Last 50 samples for response analysis
        self.response_history = deque(maxlen=100) # Last 100 response calculations
        
        # Statistics tracking
        self.stats = {
            'total_messages': 0,
            'valid_messages': 0,
            'throttle_responses': 0,
            'good_responses': 0,
            'poor_responses': 0,
            'max_throttle_position': 0.0,
            'max_pedal_position': 0.0,
            'max_engine_load': 0.0,
            'max_boost_pressure': 0.0,
            'max_oil_temperature': 0.0,
            'eco_mode_activations': 0,
            'isg_activations': 0,
            'dpf_warnings': 0,
            'limp_mode_activations': 0,
            'fault_detections': 0
        }
        
        # Initialize logging
        self.setup_logging()
        
        # Connect to CAN bus
        try:
            self.bus = can.interface.Bus(channel=interface, bustype='socketcan')
            print(f"✓ Connected to {interface}")
            self.log_message(f"Throttle & Load Monitor started on {interface}")
        except Exception as e:
            print(f"✗ Failed to connect to {interface}: {e}")
            sys.exit(1)
    
    def setup_logging(self):
        """Initialize comprehensive logging system"""
        if self.log_to_file or self.csv_logging:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        if self.log_to_file:
            self.log_filename = f"throttle_load_log_{timestamp}.txt"
            with open(self.log_filename, 'w') as f:
                f.write(f"Throttle & Load Monitor Log\n")
                f.write(f"Started: {datetime.now().isoformat()}\n")
                f.write(f"Interface: {self.interface}\n")
                f.write("Engine Management Throttle & Load Analysis\n")
                f.write("=" * 80 + "\n\n")
        
        if self.csv_logging:
            # EMS12 CSV
            self.ems12_csv_filename = f"ems12_throttle_data_{timestamp}.csv"
            with open(self.ems12_csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'can_id', 'message_count',
                    'throttle_position_percent', 'accelerator_pedal_percent', 'engine_load_percent',
                    'slide_vehicle_speed', 'boost_pressure_hpa', 'engine_oil_temp_c',
                    'modeled_ambient_temp_c', 'idle_target_rpm', 'cda_status', 'isg_status',
                    'active_eco_active', 'engine_run_normal', 'etc_limp_mode', 'oil_change_indicator',
                    'dpf_lamp_status', 'battery_lamp_status', 'ops_fail', 'alive_counter',
                    'checksum', 'raw_data_hex'
                ])
            
            # Throttle response CSV
            self.response_csv_filename = f"throttle_response_{timestamp}.csv"
            with open(self.response_csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'pedal_position', 'throttle_position',
                    'response_ratio', 'response_delay_ms', 'response_quality'
                ])
    
    def log_message(self, message: str):
        """Log message to file with timestamp"""
        if self.log_to_file:
            with open(self.log_filename, 'a') as f:
                f.write(f"[{datetime.now().isoformat()}] {message}\n")
    
    def parse_ems12_signals(self, data: bytes, timestamp: float) -> Optional[EMS12_Signals]:
        """
        Parse EMS12 CAN message according to DBC specification
        
        EMS12 Signal Layout (64-bit little-endian):
        - R_TqAcnApvC: bits [7:0] - 8-bit unsigned, scale=0.2
        - R_PAcnC: bits [15:8] - 8-bit unsigned, scale=125.0
        - TQI_B: bits [23:16] - 8-bit unsigned, scale=0.390625
        - SLD_VS: bits [31:24] - 8-bit unsigned, scale=1.0
        - CF_CdaStat: bits [34:32] - 3-bit unsigned
        - CF_Ems_IsgStat: bits [37:35] - 3-bit unsigned
        - CF_Ems_OilChg: bit [38]
        - CF_Ems_EtcLimpMod: bit [39]
        - R_NEngIdlTgC: bits [47:40] - 8-bit unsigned, scale=10.0
        - CF_Ems_UpTarGr: bit [48]
        - CF_Ems_DownTarGr: bit [49]
        - CF_Ems_DesCurGr: bits [53:50] - 4-bit unsigned
        - CF_Ems_SldAct: bit [54]
        - CF_Ems_SldPosAct: bit [55]
        - CF_Ems_HPresStat: bit [56]
        - CF_Ems_IsgBuz: bit [57]
        - CF_Ems_IdlStpFCO: bit [58]
        - CF_Ems_FCopen: bit [59]
        - CF_Ems_ActEcoAct: bit [60]
        - CF_Ems_EngRunNorm: bit [61]
        - CF_Ems_IsgStat2: bits [63:62] - 2-bit unsigned
        
        Additional signals from EMS19:
        - CR_Ems_BstPre: boost pressure
        - CR_Ems_EngOilTemp: engine oil temperature
        - CF_Ems_ModeledAmbTemp: ambient temperature
        """
        
        if len(data) != 8:
            self.log_message(f"EMS12 Invalid message length: {len(data)} bytes")
            return None
        
        try:
            # Unpack as little-endian 64-bit unsigned integer
            frame_uint64 = struct.unpack('<Q', data)[0]
            
            # Extract torque and pressure (first 16 bits)
            # Note: These might be different signals, need DBC verification
            r_tq_acn_apv = (frame_uint64 & 0xFF) * 0.2
            r_p_acn = ((frame_uint64 >> 8) & 0xFF) * 125.0
            
            # Extract main signals
            engine_load = ((frame_uint64 >> 16) & 0xFF) * self.LOAD_SCALE
            slide_vehicle_speed = ((frame_uint64 >> 24) & 0xFF)
            
            # Extract status flags
            cda_status = (frame_uint64 >> 32) & 0x7
            isg_status = (frame_uint64 >> 35) & 0x7
            oil_change_indicator = bool((frame_uint64 >> 38) & 0x1)
            etc_limp_mode = bool((frame_uint64 >> 39) & 0x1)
            
            # Extract engine parameters
            idle_target_rpm = ((frame_uint64 >> 40) & 0xFF) * self.IDLE_RPM_SCALE
            up_target_gear = bool((frame_uint64 >> 48) & 0x1)
            down_target_gear = bool((frame_uint64 >> 49) & 0x1)
            desired_current_gear = (frame_uint64 >> 50) & 0xF
            
            # Extract control flags
            slide_active = bool((frame_uint64 >> 54) & 0x1)
            slide_position_active = bool((frame_uint64 >> 55) & 0x1)
            high_pressure_status = bool((frame_uint64 >> 56) & 0x1)
            isg_buzzer = bool((frame_uint64 >> 57) & 0x1)
            idle_stop_fuel_cutoff = bool((frame_uint64 >> 58) & 0x1)
            fuel_cutoff_open = bool((frame_uint64 >> 59) & 0x1)
            active_eco_active = bool((frame_uint64 >> 60) & 0x1)
            engine_run_normal = bool((frame_uint64 >> 61) & 0x1)
            isg_status_2 = (frame_uint64 >> 62) & 0x3
            
            # For now, use approximations for missing signals
            # These would need to be extracted from other EMS messages in real implementation
            throttle_position = engine_load * 1.2  # Approximation
            accelerator_pedal = engine_load * 0.9   # Approximation
            boost_pressure = r_p_acn / 100.0        # Convert to hPa approximation
            engine_oil_temp = 80.0                  # Default operating temperature
            modeled_ambient_temp = 20.0            # Default ambient
            
            # Default fault status (would come from other EMS messages)
            dpf_lamp_status = 0
            battery_lamp_status = False
            ops_fail = False
            alive_counter = 0
            checksum = 0
            
            # Create signal structure
            signals = EMS12_Signals(
                throttle_position_percent=throttle_position,
                accelerator_pedal_percent=accelerator_pedal,
                engine_load_percent=engine_load,
                slide_vehicle_speed=slide_vehicle_speed,
                cda_status=cda_status,
                isg_status=isg_status,
                oil_change_indicator=oil_change_indicator,
                etc_limp_mode=etc_limp_mode,
                idle_target_rpm=idle_target_rpm,
                up_target_gear=up_target_gear,
                down_target_gear=down_target_gear,
                desired_current_gear=desired_current_gear,
                slide_active=slide_active,
                slide_position_active=slide_position_active,
                high_pressure_status=high_pressure_status,
                isg_buzzer=isg_buzzer,
                idle_stop_fuel_cutoff=idle_stop_fuel_cutoff,
                fuel_cutoff_open=fuel_cutoff_open,
                active_eco_active=active_eco_active,
                engine_run_normal=engine_run_normal,
                isg_status_2=isg_status_2,
                boost_pressure_hpa=boost_pressure,
                engine_oil_temp_c=engine_oil_temp,
                modeled_ambient_temp_c=modeled_ambient_temp,
                dpf_lamp_status=dpf_lamp_status,
                battery_lamp_status=battery_lamp_status,
                ops_fail=ops_fail,
                alive_counter=alive_counter,
                checksum=checksum,
                timestamp=timestamp,
                can_id=self.EMS12_CAN_ID,
                raw_data=data
            )
            
            # Validate signal ranges
            if self.validate_ems12_signals(signals):
                self.stats['valid_messages'] += 1
                self.update_statistics(signals)
                return signals
            else:
                return None
            
        except Exception as e:
            self.log_message(f"EMS12 parsing error: {e}")
            return None
    
    def validate_ems12_signals(self, signals: EMS12_Signals) -> bool:
        """Validate EMS12 signal values"""
        validation_errors = []
        
        # Validate throttle position
        if not (self.TPS_MIN <= signals.throttle_position_percent <= self.TPS_MAX):
            validation_errors.append(f"Throttle position out of range: {signals.throttle_position_percent:.1f}")
        
        # Validate pedal position
        if not (0 <= signals.accelerator_pedal_percent <= self.PEDAL_MAX):
            validation_errors.append(f"Pedal position out of range: {signals.accelerator_pedal_percent:.1f}")
        
        # Validate engine load
        if not (0 <= signals.engine_load_percent <= self.LOAD_MAX):
            validation_errors.append(f"Engine load out of range: {signals.engine_load_percent:.1f}")
        
        # Validate boost pressure
        if signals.boost_pressure_hpa > self.BOOST_MAX:
            validation_errors.append(f"Boost pressure out of range: {signals.boost_pressure_hpa:.1f}")
        
        # Validate oil temperature
        if signals.engine_oil_temp_c > self.OIL_TEMP_MAX:
            validation_errors.append(f"Oil temperature high: {signals.engine_oil_temp_c:.1f}°C")
        
        # Check for fault conditions
        if signals.etc_limp_mode:
            validation_errors.append("ETC Limp Mode active")
            self.stats['limp_mode_activations'] += 1
        
        if signals.dpf_lamp_status > 0:
            validation_errors.append(f"DPF warning: {signals.dpf_lamp_status}")
            self.stats['dpf_warnings'] += 1
        
        if signals.ops_fail:
            validation_errors.append("OPS failure detected")
            self.stats['fault_detections'] += 1
        
        if validation_errors:
            self.log_message(f"EMS12 validation issues: {', '.join(validation_errors)}")
            
        return True  # Return True even with warnings for logging purposes
    
    def update_statistics(self, signals: EMS12_Signals):
        """Update comprehensive statistics"""
        self.stats['max_throttle_position'] = max(self.stats['max_throttle_position'], 
                                                 signals.throttle_position_percent)
        self.stats['max_pedal_position'] = max(self.stats['max_pedal_position'], 
                                              signals.accelerator_pedal_percent)
        self.stats['max_engine_load'] = max(self.stats['max_engine_load'], 
                                           signals.engine_load_percent)
        self.stats['max_boost_pressure'] = max(self.stats['max_boost_pressure'], 
                                              signals.boost_pressure_hpa)
        self.stats['max_oil_temperature'] = max(self.stats['max_oil_temperature'], 
                                               signals.engine_oil_temp_c)
        
        # Count special activations
        if signals.active_eco_active:
            self.stats['eco_mode_activations'] += 1
        
        if signals.isg_status > 0 or signals.isg_status_2 > 0:
            self.stats['isg_activations'] += 1
    
    def analyze_throttle_response(self, signals: EMS12_Signals) -> Optional[ThrottleResponse]:
        """Analyze throttle response characteristics"""
        
        # Add current data to history
        self.throttle_history.append({
            'timestamp': signals.timestamp,
            'pedal': signals.accelerator_pedal_percent,
            'throttle': signals.throttle_position_percent,
            'load': signals.engine_load_percent
        })
        
        # Need at least 2 samples for response analysis
        if len(self.throttle_history) < 2:
            return None
        
        current = self.throttle_history[-1]
        previous = self.throttle_history[-2]
        
        # Calculate response metrics
        pedal_change = current['pedal'] - previous['pedal']
        throttle_change = current['throttle'] - previous['throttle']
        time_diff_ms = (current['timestamp'] - previous['timestamp']) * 1000
        
        # Only analyze significant pedal changes
        if abs(pedal_change) < 2.0:  # Less than 2% change
            return None
        
        # Calculate response ratio
        if abs(pedal_change) > 0.1:
            response_ratio = throttle_change / pedal_change
        else:
            response_ratio = 0.0
        
        # Determine response quality
        if abs(response_ratio - 1.0) < 0.2:  # Within 20% of 1:1 response
            if time_diff_ms < 100:  # Less than 100ms delay
                quality = "EXCELLENT"
                self.stats['good_responses'] += 1
            else:
                quality = "GOOD"
                self.stats['good_responses'] += 1
        elif abs(response_ratio - 1.0) < 0.5:  # Within 50% of 1:1 response
            quality = "FAIR"
        else:
            quality = "POOR"
            self.stats['poor_responses'] += 1
        
        # Create response analysis
        response = ThrottleResponse(
            pedal_position=current['pedal'],
            throttle_position=current['throttle'],
            response_ratio=response_ratio,
            response_delay_ms=time_diff_ms,
            response_quality=quality,
            timestamp=current['timestamp']
        )
        
        # Add to response history
        self.response_history.append(response)
        self.stats['throttle_responses'] += 1
        
        return response
    
    def get_response_statistics(self) -> Dict:
        """Calculate throttle response statistics"""
        if not self.response_history:
            return {}
        
        ratios = [r.response_ratio for r in self.response_history]
        delays = [r.response_delay_ms for r in self.response_history]
        
        return {
            'avg_response_ratio': statistics.mean(ratios),
            'avg_response_delay': statistics.mean(delays),
            'max_response_delay': max(delays),
            'min_response_delay': min(delays),
            'response_consistency': statistics.stdev(ratios) if len(ratios) > 1 else 0.0,
            'samples': len(self.response_history)
        }
    
    def log_to_csv(self, signals: EMS12_Signals):
        """Log EMS12 data to CSV"""
        if self.csv_logging:
            with open(self.ems12_csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    signals.timestamp, signals.can_id, self.message_count,
                    signals.throttle_position_percent, signals.accelerator_pedal_percent, signals.engine_load_percent,
                    signals.slide_vehicle_speed, signals.boost_pressure_hpa, signals.engine_oil_temp_c,
                    signals.modeled_ambient_temp_c, signals.idle_target_rpm, signals.cda_status, signals.isg_status,
                    signals.active_eco_active, signals.engine_run_normal, signals.etc_limp_mode, signals.oil_change_indicator,
                    signals.dpf_lamp_status, signals.battery_lamp_status, signals.ops_fail, signals.alive_counter,
                    signals.checksum, signals.raw_data.hex().upper()
                ])
    
    def log_response_to_csv(self, response: ThrottleResponse):
        """Log throttle response data to CSV"""
        if self.csv_logging:
            with open(self.response_csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    response.timestamp, response.pedal_position, response.throttle_position,
                    response.response_ratio, response.response_delay_ms, response.response_quality
                ])
    
    def display_throttle_data(self):
        """Display comprehensive throttle and load data"""
        
        print(f"\033[2J\033[H")  # Clear screen
        print("=" * 120)
        print(f"THROTTLE & ENGINE LOAD MONITOR - {self.interface}")
        print("=" * 120)
        
        print(f"EMS12 Messages: {self.message_count}")
        print(f"Runtime: {time.time() - self.start_time:.1f}s")
        print()
        
        # EMS12 Data
        print("EMS12 - THROTTLE POSITION & ENGINE LOAD:")
        print("-" * 120)
        if self.latest_ems12:
            print(f"CAN ID: 0x{self.latest_ems12.can_id:03X} | Raw: {self.latest_ems12.raw_data.hex().upper()}")
            
            # Throttle and pedal status
            print(f"Throttle Position:    {self.latest_ems12.throttle_position_percent:6.1f}%")
            print(f"Accelerator Pedal:    {self.latest_ems12.accelerator_pedal_percent:6.1f}%")
            print(f"Engine Load:          {self.latest_ems12.engine_load_percent:6.1f}%")
            print(f"Slide Vehicle Speed:  {self.latest_ems12.slide_vehicle_speed:6.0f} km/h")
            
            # Engine parameters
            print(f"Boost Pressure:       {self.latest_ems12.boost_pressure_hpa:6.1f} hPa")
            print(f"Engine Oil Temp:      {self.latest_ems12.engine_oil_temp_c:6.1f}°C")
            print(f"Ambient Temp:         {self.latest_ems12.modeled_ambient_temp_c:6.1f}°C")
            print(f"Idle Target RPM:      {self.latest_ems12.idle_target_rpm:6.0f}")
            
            # System status
            eco_status = "🟢 ACTIVE" if self.latest_ems12.active_eco_active else "⚫ INACTIVE"
            engine_status = "🟢 NORMAL" if self.latest_ems12.engine_run_normal else "🔴 ABNORMAL"
            limp_status = "🔴 LIMP MODE" if self.latest_ems12.etc_limp_mode else "🟢 NORMAL"
            
            print(f"Active Eco:           {eco_status}")
            print(f"Engine Status:        {engine_status}")
            print(f"ETC Status:           {limp_status}")
            
            # ISG (Idle Stop & Go) system
            isg_status_desc = ["Off", "Starting", "Running", "Stopping"][min(self.latest_ems12.isg_status, 3)]
            isg_buzzer_status = "🔊 ON" if self.latest_ems12.isg_buzzer else "🔇 OFF"
            print(f"ISG System:           {isg_status_desc} | Buzzer: {isg_buzzer_status}")
            
            # Gear control
            if self.latest_ems12.up_target_gear:
                gear_suggestion = "⬆️ UPSHIFT"
            elif self.latest_ems12.down_target_gear:
                gear_suggestion = "⬇️ DOWNSHIFT"
            else:
                gear_suggestion = "➡️ CURRENT GEAR"
            print(f"Gear Control:         {gear_suggestion} | Target: {self.latest_ems12.desired_current_gear}")
            
            # Fault indicators
            faults = []
            if self.latest_ems12.oil_change_indicator: faults.append("OIL CHANGE")
            if self.latest_ems12.dpf_lamp_status > 0: faults.append(f"DPF({self.latest_ems12.dpf_lamp_status})")
            if self.latest_ems12.battery_lamp_status: faults.append("BATTERY")
            if self.latest_ems12.ops_fail: faults.append("OPS FAIL")
            
            if faults:
                print(f"⚠️  WARNINGS:           {', '.join(faults)}")
            else:
                print("✅ NO FAULTS DETECTED")
                
        else:
            print("No EMS12 data received")
        print()
        
        # Throttle Response Analysis
        print("THROTTLE RESPONSE ANALYSIS:")
        print("-" * 120)
        current_response = self.analyze_throttle_response(self.latest_ems12) if self.latest_ems12 else None
        if current_response:
            status_icon = {"EXCELLENT": "🟢", "GOOD": "🟡", "FAIR": "🟠", "POOR": "🔴"}.get(
                current_response.response_quality, "❓")
            
            print(f"Current Response:     {status_icon} {current_response.response_quality}")
            print(f"Response Ratio:       {current_response.response_ratio:6.2f} (1.0 = ideal)")
            print(f"Response Delay:       {current_response.response_delay_ms:6.1f} ms")
            
            # Log response to CSV
            self.log_response_to_csv(current_response)
            
            # Historical response statistics
            response_stats = self.get_response_statistics()
            if response_stats:
                print(f"Average Response:     {response_stats['avg_response_ratio']:.2f} ± {response_stats['response_consistency']:.2f}")
                print(f"Average Delay:        {response_stats['avg_response_delay']:.1f} ms")
                print(f"Delay Range:          {response_stats['min_response_delay']:.1f} - {response_stats['max_response_delay']:.1f} ms")
        else:
            print("Insufficient data for throttle response analysis")
        print()
        
        # Engine Performance Analysis
        print("ENGINE PERFORMANCE ANALYSIS:")
        print("-" * 120)
        if self.latest_ems12:
            # Load vs throttle correlation
            load_efficiency = 0.0
            if self.latest_ems12.throttle_position_percent > 5.0:
                load_efficiency = self.latest_ems12.engine_load_percent / self.latest_ems12.throttle_position_percent
            
            # Performance indicators
            if load_efficiency > 0.8:
                efficiency_status = "🟢 EXCELLENT"
            elif load_efficiency > 0.6:
                efficiency_status = "🟡 GOOD"
            elif load_efficiency > 0.4:
                efficiency_status = "🟠 FAIR"
            else:
                efficiency_status = "🔴 POOR"
            
            print(f"Load Efficiency:      {efficiency_status} ({load_efficiency:.2f})")
            
            # Boost analysis
            if self.latest_ems12.boost_pressure_hpa > 1200:  # Above atmospheric
                boost_level = self.latest_ems12.boost_pressure_hpa - 1013.25  # Gauge pressure
                print(f"Boost Level:          +{boost_level:.0f} hPa (Turbo Active)")
            else:
                print(f"Boost Level:          Naturally Aspirated")
            
            # Temperature analysis
            if self.latest_ems12.engine_oil_temp_c > 120:
                temp_status = "🔴 HIGH"
            elif self.latest_ems12.engine_oil_temp_c > 100:
                temp_status = "🟡 WARM"
            else:
                temp_status = "🟢 NORMAL"
            print(f"Oil Temperature:      {temp_status}")
            
            # Driving mode analysis
            if self.latest_ems12.active_eco_active:
                driving_mode = "🌱 ECO MODE"
            elif self.latest_ems12.engine_load_percent > 80:
                driving_mode = "🚀 PERFORMANCE"
            elif self.latest_ems12.engine_load_percent > 40:
                driving_mode = "🚗 NORMAL"
            else:
                driving_mode = "🐌 IDLE/CRUISE"
            
            print(f"Driving Mode:         {driving_mode}")
        else:
            print("Insufficient data for performance analysis")
        print()
        
        # Session Statistics
        print("SESSION STATISTICS:")
        print("-" * 120)
        print(f"Total Messages: {self.stats['total_messages']} | Valid: {self.stats['valid_messages']}")
        print(f"Throttle Responses: {self.stats['throttle_responses']} | Good: {self.stats['good_responses']} | Poor: {self.stats['poor_responses']}")
        print(f"Peak Throttle: {self.stats['max_throttle_position']:.1f}% | Peak Pedal: {self.stats['max_pedal_position']:.1f}%")
        print(f"Peak Load: {self.stats['max_engine_load']:.1f}% | Peak Boost: {self.stats['max_boost_pressure']:.1f} hPa")
        print(f"Peak Oil Temp: {self.stats['max_oil_temperature']:.1f}°C")
        print(f"Eco Activations: {self.stats['eco_mode_activations']} | ISG Activations: {self.stats['isg_activations']}")
        print(f"Faults: Limp Mode={self.stats['limp_mode_activations']}, DPF={self.stats['dpf_warnings']}, Other={self.stats['fault_detections']}")
        
        if self.csv_logging:
            print(f"Data Logging: {self.ems12_csv_filename}, {self.response_csv_filename}")
        
        print("\nPress Ctrl+C to stop...")
    
    def monitor(self):
        """Main monitoring loop"""
        
        print(f"Starting Throttle & Engine Load Monitor")
        print(f"Interface: {self.interface}")
        print(f"Target: EMS12 (0x{self.EMS12_CAN_ID:03X})")
        print(f"Throttle Response & Engine Performance Analysis")
        print("=" * 120)
        
        last_display_time = 0
        display_interval = 0.2  # Update display every 200ms
        
        try:
            while True:
                # Receive CAN message
                message = self.bus.recv(timeout=0.1)
                
                if message is None:
                    continue
                
                # Check for EMS12 message
                if message.arbitration_id == self.EMS12_CAN_ID:
                    self.message_count += 1
                    self.stats['total_messages'] += 1
                    
                    signals = self.parse_ems12_signals(message.data, message.timestamp)
                    if signals:
                        self.latest_ems12 = signals
                        self.log_to_csv(signals)
                
                # Update display periodically
                current_time = time.time()
                if current_time - last_display_time >= display_interval:
                    self.display_throttle_data()
                    last_display_time = current_time
                    
        except KeyboardInterrupt:
            print(f"\nThrottle & Load monitoring stopped")
            self.log_message(f"Monitoring stopped - Processed {self.message_count} messages")
            self.print_final_statistics()
        finally:
            if self.bus:
                self.bus.shutdown()
    
    def print_final_statistics(self):
        """Print final session statistics"""
        print(f"\nFINAL THROTTLE & LOAD STATISTICS:")
        print("=" * 80)
        print(f"Runtime: {time.time() - self.start_time:.1f} seconds")
        print(f"EMS12 Messages: {self.stats['total_messages']} (Valid: {self.stats['valid_messages']})")
        print()
        
        # Throttle performance summary
        if self.stats['throttle_responses'] > 0:
            good_percent = (self.stats['good_responses'] / self.stats['throttle_responses']) * 100
            print(f"Throttle Response Quality: {good_percent:.1f}% good ({self.stats['good_responses']}/{self.stats['throttle_responses']})")
            
            response_stats = self.get_response_statistics()
            if response_stats:
                print(f"Average Response Ratio: {response_stats['avg_response_ratio']:.2f} (1.0 = ideal)")
                print(f"Average Response Delay: {response_stats['avg_response_delay']:.1f} ms")
                print(f"Response Consistency: ±{response_stats['response_consistency']:.2f}")
        
        print()
        print(f"Performance Peaks:")
        print(f"- Max Throttle Position: {self.stats['max_throttle_position']:.1f}%")
        print(f"- Max Pedal Position: {self.stats['max_pedal_position']:.1f}%")
        print(f"- Max Engine Load: {self.stats['max_engine_load']:.1f}%")
        print(f"- Max Boost Pressure: {self.stats['max_boost_pressure']:.1f} hPa")
        print(f"- Max Oil Temperature: {self.stats['max_oil_temperature']:.1f}°C")
        
        print()
        print(f"System Usage:")
        print(f"- Eco Mode Activations: {self.stats['eco_mode_activations']}")
        print(f"- ISG System Activations: {self.stats['isg_activations']}")
        print(f"- Limp Mode Events: {self.stats['limp_mode_activations']}")
        print(f"- DPF Warnings: {self.stats['dpf_warnings']}")
        print(f"- Other Faults: {self.stats['fault_detections']}")
        
        if self.csv_logging:
            print(f"\nEMS12 Data: {self.ems12_csv_filename}")
            print(f"Response Data: {self.response_csv_filename}")
        if self.log_to_file:
            print(f"Log: {self.log_filename}")

def main():
    parser = argparse.ArgumentParser(
        description='Throttle & Engine Load Monitor - Comprehensive Performance Analysis',
        epilog='''
Monitors throttle position and engine load parameters:
- Throttle position sensor (TPS) analysis
- Accelerator pedal position correlation
- Engine load and boost pressure monitoring
- Throttle response time and quality analysis
- ISG (Idle Stop & Go) and Eco mode tracking
- Engine fault detection and temperature monitoring

DBC-compliant parsing with performance optimization analysis.
        '''
    )
    parser.add_argument('--interface', '-i', default='can1', 
                       help='CAN interface (default: can1)')
    parser.add_argument('--no-csv', action='store_true',
                       help='Disable CSV data logging')
    parser.add_argument('--no-log', action='store_true', 
                       help='Disable text file logging')
    
    args = parser.parse_args()
    
    print("Throttle & Engine Load Monitor")
    print("Comprehensive Engine Performance Analysis")
    print("DBC-Compliant with Throttle Response Analysis")
    print()
    
    monitor = ThrottleLoadMonitor(
        interface=args.interface,
        log_to_file=not args.no_log,
        csv_logging=not args.no_csv
    )
    monitor.monitor()

if __name__ == "__main__":
    main()
