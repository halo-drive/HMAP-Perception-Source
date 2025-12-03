#!/usr/bin/env python3
"""
Braking System Monitor
CAN IDs: 0x153 (TCS11), 0x394 (TCS13), 0x4FF (TCS15)
Electronic Stability & Brake Control Systems

Decodes comprehensive braking system parameters:
- ABS/ESP/TCS system states and control actions
- Brake light status and driver braking detection
- Longitudinal acceleration references (±10.24 m/s²)
- Hill-start assist and electronic brake distribution
- System fault detection and lamp status

DBC Specification Compliance with brake system safety validation
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
class TCS11_Signals:
    """TCS11 Traction Control System signals"""
    # Control system states
    tcs_request: bool                     # TCS_REQ
    msr_c_request: bool                   # MSR_C_REQ (Motor Schleppmoment Regelung)
    tcs_passive: bool                     # TCS_PAS
    tcs_gear_shift_control: bool          # TCS_GSC
    esp_limiter_info: int                 # CF_Esc_LimoInfo (0-3)
    
    # Fault detection
    abs_diagnostic: bool                  # ABS_DIAG
    abs_defect: bool                      # ABS_DEF
    tcs_defect: bool                      # TCS_DEF
    tcs_control_active: bool              # TCS_CTL
    abs_active: bool                      # ABS_ACT
    ebd_defect: bool                      # EBD_DEF (Electronic Brake Distribution)
    
    # ESP system
    esp_passive: bool                     # ESP_PAS
    esp_defect: bool                      # ESP_DEF
    esp_control_active: bool              # ESP_CTL
    tcs_malfunction: bool                 # TCS_MFRN
    
    # Additional brake systems
    dbc_control_active: bool              # DBC_CTL (Dynamic Brake Control)
    dbc_passive: bool                     # DBC_PAS
    dbc_defect: bool                      # DBC_DEF
    hac_control_active: bool              # HAC_CTL (Hill-start Assist)
    hac_passive: bool                     # HAC_PAS
    hac_defect: bool                      # HAC_DEF
    ess_status: int                       # ESS_STAT (Emergency Stop Signal)
    
    # Torque interventions
    tqi_tcs: float                        # TQI_TCS (0-99.6 %)
    tqi_msr: float                        # TQI_MSR (0-99.6 %)
    tqi_slow_tcs: float                   # TQI_SLW_TCS (0-99.6 %)
    
    # System status
    brake_control: bool                   # CF_Esc_BrkCtl
    brake_light_assistant: int            # BLA_CTL (0-3)
    alive_counter: int                    # AliveCounter_TCS1 (0-14)
    checksum: int                         # CheckSum_TCS1 (0-255)
    
    timestamp: float
    can_id: int
    raw_data: bytes

@dataclass
class TCS13_Signals:
    """TCS13 Brake Control and Acceleration Reference signals"""
    # Acceleration references
    acceleration_basis_ms2: float         # aBasis (-10.23 to 10.24 m/s²)
    acceleration_ref_acc_ms2: float       # ACCEL_REF_ACC (-10.23 to 10.24 m/s²)
    
    # Brake status
    brake_light_active: bool              # BrakeLight
    driver_braking: bool                  # DriverBraking
    parking_brake_active: bool            # PBRAKE_ACT
    
    # Control system states
    dc_enable: bool                       # DCEnable
    acc_enable: int                       # ACCEnable (0-3)
    driver_override: int                  # DriverOverride (0-3)
    standstill: bool                      # StandStill
    acc_equipped: bool                    # ACC_EQUIP
    acc_request: bool                     # ACC_REQ
    aeb_equipped: bool                    # AEB_EQUIP
    
    # VSM system
    vsm_coded: bool                       # CF_VSM_Coded
    vsm_available: int                    # CF_VSM_Avail (0-3)
    vsm_handshake: bool                   # CF_VSM_Handshake
    vsm_brake_status: bool                # CF_DriBkeStat
    vsm_confirmation_switch: int          # CF_VSM_ConfSwi (0-3)
    
    # Torque control
    tqi_scc: float                        # TQI_SCC (0-199.6 %)
    
    # Message integrity
    alive_counter: int                    # AliveCounterTCS (0-7)
    checksum: int                         # CheckSum_TCS3 (0-15)
    
    timestamp: float
    can_id: int
    raw_data: bytes

@dataclass
class TCS15_Signals:
    """TCS15 Warning Lamps and Status signals"""
    # Warning lamps
    abs_warning_lamp: bool                # ABS_W_LAMP
    tcs_off_lamp: int                     # TCS_OFF_LAMP (0-1)
    tcs_lamp: int                         # TCS_LAMP (0-3)
    dbc_warning_lamp: bool                # DBC_W_LAMP
    dbc_fault_lamp: int                   # DBC_F_LAMP (0-3)
    ebd_warning_lamp: bool                # EBD_W_LAMP
    
    # ESP system status
    esp_off_step: int                     # ESC_Off_Step (0-3)
    
    # AVH system (Auto Vehicle Hold)
    avh_cluster: int                      # AVH_CLU (0-255)
    avh_indicator_lamp: int               # AVH_I_LAMP (0-3)
    avh_alarm: int                        # AVH_ALARM (0-3)
    avh_lamp: int                         # AVH_LAMP (0-7)
    
    timestamp: float
    can_id: int
    raw_data: bytes

class BrakingSystemMonitor:
    def __init__(self, interface='can1', log_to_file=True, csv_logging=True):
        self.interface = interface
        self.tcs11_message_count = 0
        self.tcs13_message_count = 0
        self.tcs15_message_count = 0
        self.start_time = time.time()
        self.log_to_file = log_to_file
        self.csv_logging = csv_logging
        
        # CAN IDs
        self.TCS11_CAN_ID = 0x153   # 339 decimal
        self.TCS13_CAN_ID = 0x394   # 916 decimal
        self.TCS15_CAN_ID = 0x4FF   # 1287 decimal
        
        # TCS13 constants
        self.ACCEL_SCALE = 0.01             # m/s² per LSB
        self.ACCEL_OFFSET = -10.23          # m/s² offset
        self.TQI_SCALE = 0.390625           # % per LSB
        self.TQI_SCC_SCALE = 0.390625       # % per LSB
        
        # Validation ranges
        self.ACCEL_MIN = -15.0              # m/s² (beyond DBC for validation)
        self.ACCEL_MAX = 15.0               # m/s²
        self.TQI_MAX = 100.0                # %
        
        # Latest signal data
        self.latest_tcs11 = None
        self.latest_tcs13 = None
        self.latest_tcs15 = None
        
        # Statistics tracking
        self.stats = {
            'tcs11_total_messages': 0,
            'tcs11_valid_messages': 0,
            'tcs13_total_messages': 0,
            'tcs13_valid_messages': 0,
            'tcs15_total_messages': 0,
            'tcs15_valid_messages': 0,
            'abs_activations': 0,
            'esp_activations': 0,
            'tcs_activations': 0,
            'brake_light_activations': 0,
            'driver_braking_detections': 0,
            'system_faults': 0,
            'max_deceleration': 0.0,
            'max_acceleration': 0.0,
            'checksum_errors': 0
        }
        
        # Initialize logging
        self.setup_logging()
        
        # Connect to CAN bus
        try:
            self.bus = can.interface.Bus(channel=interface, bustype='socketcan')
            print(f"✓ Connected to {interface}")
            self.log_message(f"Braking System Monitor started on {interface}")
        except Exception as e:
            print(f"✗ Failed to connect to {interface}: {e}")
            sys.exit(1)
    
    def setup_logging(self):
        """Initialize comprehensive logging system"""
        if self.log_to_file or self.csv_logging:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
        if self.log_to_file:
            self.log_filename = f"braking_system_log_{timestamp}.txt"
            with open(self.log_filename, 'w') as f:
                f.write(f"Braking System Monitor Log\n")
                f.write(f"Started: {datetime.now().isoformat()}\n")
                f.write(f"Interface: {self.interface}\n")
                f.write("Electronic Stability & Brake Control Analysis\n")
                f.write("=" * 80 + "\n\n")
        
        if self.csv_logging:
            # TCS11 CSV
            self.tcs11_csv_filename = f"tcs11_data_{timestamp}.csv"
            with open(self.tcs11_csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'can_id', 'message_count',
                    'tcs_request', 'msr_c_request', 'tcs_passive', 'abs_active',
                    'esp_control_active', 'tcs_control_active', 'abs_defect', 'esp_defect',
                    'tcs_defect', 'driver_braking', 'brake_light_active', 'hac_control_active',
                    'tqi_tcs', 'tqi_msr', 'alive_counter', 'checksum', 'raw_data_hex'
                ])
            
            # TCS13 CSV
            self.tcs13_csv_filename = f"tcs13_data_{timestamp}.csv"
            with open(self.tcs13_csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'can_id', 'message_count',
                    'acceleration_basis_ms2', 'acceleration_ref_acc_ms2', 'brake_light_active',
                    'driver_braking', 'dc_enable', 'acc_enable', 'driver_override',
                    'standstill', 'acc_equipped', 'parking_brake_active', 'tqi_scc',
                    'alive_counter', 'checksum', 'raw_data_hex'
                ])
            
            # TCS15 CSV
            self.tcs15_csv_filename = f"tcs15_data_{timestamp}.csv"
            with open(self.tcs15_csv_filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'timestamp', 'can_id', 'message_count',
                    'abs_warning_lamp', 'tcs_off_lamp', 'tcs_lamp', 'esp_off_step',
                    'avh_cluster', 'avh_lamp', 'dbc_warning_lamp', 'ebd_warning_lamp',
                    'raw_data_hex'
                ])
    
    def log_message(self, message: str):
        """Log message to file with timestamp"""
        if self.log_to_file:
            with open(self.log_filename, 'a') as f:
                f.write(f"[{datetime.now().isoformat()}] {message}\n")
    
    def parse_tcs11_signals(self, data: bytes, timestamp: float) -> Optional[TCS11_Signals]:
        """
        Parse TCS11 CAN message according to DBC specification
        
        TCS11 Signal Layout (64-bit little-endian):
        - TCS_REQ: bit [0]
        - MSR_C_REQ: bit [1] 
        - TCS_PAS: bit [2]
        - TCS_GSC: bit [3]
        - CF_Esc_LimoInfo: bits [5:4]
        - ABS_DIAG: bit [6]
        - ABS_DEF: bit [7]
        - TCS_DEF: bit [8]
        - TCS_CTL: bit [9]
        - ABS_ACT: bit [10]
        - EBD_DEF: bit [11]
        - ESP_PAS: bit [12]
        - ESP_DEF: bit [13]
        - ESP_CTL: bit [14]
        - TCS_MFRN: bit [15]
        - DBC_CTL: bit [16]
        - DBC_PAS: bit [17]
        - DBC_DEF: bit [18]
        - HAC_CTL: bit [19]
        - HAC_PAS: bit [20]
        - HAC_DEF: bit [21]
        - ESS_STAT: bits [23:22]
        - TQI_TCS: bits [31:24] - 8-bit unsigned, scale=0.390625
        - TQI_MSR: bits [39:32] - 8-bit unsigned, scale=0.390625
        - TQI_SLW_TCS: bits [47:40] - 8-bit unsigned, scale=0.390625
        - CF_Esc_BrkCtl: bit [48]
        - BLA_CTL: bits [50:49]
        - AliveCounter_TCS1: bits [55:52] - 4-bit counter
        - CheckSum_TCS1: bits [63:56] - 8-bit checksum
        """
        
        if len(data) != 8:
            self.log_message(f"TCS11 Invalid message length: {len(data)} bytes")
            return None
        
        try:
            # Unpack as little-endian 64-bit unsigned integer
            frame_uint64 = struct.unpack('<Q', data)[0]
            
            # Extract control system states
            tcs_request = bool(frame_uint64 & 0x1)
            msr_c_request = bool((frame_uint64 >> 1) & 0x1)
            tcs_passive = bool((frame_uint64 >> 2) & 0x1)
            tcs_gear_shift_control = bool((frame_uint64 >> 3) & 0x1)
            esp_limiter_info = (frame_uint64 >> 4) & 0x3
            
            # Extract fault detection flags
            abs_diagnostic = bool((frame_uint64 >> 6) & 0x1)
            abs_defect = bool((frame_uint64 >> 7) & 0x1)
            tcs_defect = bool((frame_uint64 >> 8) & 0x1)
            tcs_control_active = bool((frame_uint64 >> 9) & 0x1)
            abs_active = bool((frame_uint64 >> 10) & 0x1)
            ebd_defect = bool((frame_uint64 >> 11) & 0x1)
            
            # Extract ESP system flags
            esp_passive = bool((frame_uint64 >> 12) & 0x1)
            esp_defect = bool((frame_uint64 >> 13) & 0x1)
            esp_control_active = bool((frame_uint64 >> 14) & 0x1)
            tcs_malfunction = bool((frame_uint64 >> 15) & 0x1)
            
            # Extract additional brake systems
            dbc_control_active = bool((frame_uint64 >> 16) & 0x1)
            dbc_passive = bool((frame_uint64 >> 17) & 0x1)
            dbc_defect = bool((frame_uint64 >> 18) & 0x1)
            hac_control_active = bool((frame_uint64 >> 19) & 0x1)
            hac_passive = bool((frame_uint64 >> 20) & 0x1)
            hac_defect = bool((frame_uint64 >> 21) & 0x1)
            ess_status = (frame_uint64 >> 22) & 0x3
            
            # Extract torque interventions
            tqi_tcs = ((frame_uint64 >> 24) & 0xFF) * self.TQI_SCALE
            tqi_msr = ((frame_uint64 >> 32) & 0xFF) * self.TQI_SCALE
            tqi_slow_tcs = ((frame_uint64 >> 40) & 0xFF) * self.TQI_SCALE
            
            # Extract system status
            brake_control = bool((frame_uint64 >> 48) & 0x1)
            brake_light_assistant = (frame_uint64 >> 49) & 0x3
            alive_counter = (frame_uint64 >> 52) & 0xF
            checksum = (frame_uint64 >> 56) & 0xFF
            
            # Create signal structure
            signals = TCS11_Signals(
                tcs_request=tcs_request,
                msr_c_request=msr_c_request,
                tcs_passive=tcs_passive,
                tcs_gear_shift_control=tcs_gear_shift_control,
                esp_limiter_info=esp_limiter_info,
                abs_diagnostic=abs_diagnostic,
                abs_defect=abs_defect,
                tcs_defect=tcs_defect,
                tcs_control_active=tcs_control_active,
                abs_active=abs_active,
                ebd_defect=ebd_defect,
                esp_passive=esp_passive,
                esp_defect=esp_defect,
                esp_control_active=esp_control_active,
                tcs_malfunction=tcs_malfunction,
                dbc_control_active=dbc_control_active,
                dbc_passive=dbc_passive,
                dbc_defect=dbc_defect,
                hac_control_active=hac_control_active,
                hac_passive=hac_passive,
                hac_defect=hac_defect,
                ess_status=ess_status,
                tqi_tcs=tqi_tcs,
                tqi_msr=tqi_msr,
                tqi_slow_tcs=tqi_slow_tcs,
                brake_control=brake_control,
                brake_light_assistant=brake_light_assistant,
                alive_counter=alive_counter,
                checksum=checksum,
                timestamp=timestamp,
                can_id=self.TCS11_CAN_ID,
                raw_data=data
            )
            
            # Validate and update statistics
            if self.validate_tcs11_signals(signals):
                self.stats['tcs11_valid_messages'] += 1
                self.update_tcs11_statistics(signals)
                return signals
            else:
                return None
            
        except Exception as e:
            self.log_message(f"TCS11 parsing error: {e}")
            return None
    
    def parse_tcs13_signals(self, data: bytes, timestamp: float) -> Optional[TCS13_Signals]:
        """
        Parse TCS13 CAN message according to DBC specification
        
        TCS13 Signal Layout (64-bit little-endian):
        - aBasis: bits [10:0] - 11-bit signed, scale=0.01, offset=-10.23
        - BrakeLight: bit [11]
        - DCEnable: bit [12] 
        - AliveCounterTCS: bits [15:13] - 3-bit counter
        - Pre_TCS_CTL: bit [16]
        - EBA_ACK: bit [17]
        - FCA_ACK: bit [18]
        - DF_BF_STAT: bits [20:19]
        - SCCReqLim: bits [22:21]
        - TQI_SCC: bits [31:23] - 9-bit unsigned, scale=0.390625
        - ACCEL_REF_ACC: bits [42:32] - 11-bit signed, scale=0.01, offset=-10.23
        - ACCEnable: bits [44:43]
        - DriverOverride: bits [46:45]
        - StandStill: bit [47]
        - CheckSum_TCS3: bits [51:48] - 4-bit checksum
        - ACC_EQUIP: bit [52]
        - PBRAKE_ACT: bit [53]
        - ACC_REQ: bit [54]
        - DriverBraking: bit [55]
        - CF_VSM_Coded: bit [56]
        - CF_VSM_Avail: bits [58:57]
        - CF_VSM_Handshake: bit [59]
        - CF_DriBkeStat: bit [60]
        - CF_VSM_ConfSwi: bits [62:61]
        - AEB_EQUIP: bit [63]
        """
        
        if len(data) != 8:
            self.log_message(f"TCS13 Invalid message length: {len(data)} bytes")
            return None
        
        try:
            # Unpack as little-endian 64-bit unsigned integer
            frame_uint64 = struct.unpack('<Q', data)[0]
            
            # Extract acceleration basis (11-bit signed)
            accel_basis_raw = frame_uint64 & 0x7FF
            if accel_basis_raw & 0x400:  # Sign extend
                accel_basis_raw |= 0xFFFFF800
                accel_basis_raw = struct.unpack('<i', struct.pack('<I', accel_basis_raw & 0xFFFFFFFF))[0]
            acceleration_basis = accel_basis_raw * self.ACCEL_SCALE + self.ACCEL_OFFSET
            
            # Extract control flags
            brake_light_active = bool((frame_uint64 >> 11) & 0x1)
            dc_enable = bool((frame_uint64 >> 12) & 0x1)
            alive_counter_tcs = (frame_uint64 >> 13) & 0x7
            
            # Extract TQI_SCC (9-bit unsigned)
            tqi_scc = ((frame_uint64 >> 23) & 0x1FF) * self.TQI_SCC_SCALE
            
            # Extract acceleration reference (11-bit signed)
            accel_ref_raw = (frame_uint64 >> 32) & 0x7FF
            if accel_ref_raw & 0x400:  # Sign extend
                accel_ref_raw |= 0xFFFFF800
                accel_ref_raw = struct.unpack('<i', struct.pack('<I', accel_ref_raw & 0xFFFFFFFF))[0]
            acceleration_ref_acc = accel_ref_raw * self.ACCEL_SCALE + self.ACCEL_OFFSET
            
            # Extract control system states
            acc_enable = (frame_uint64 >> 43) & 0x3
            driver_override = (frame_uint64 >> 45) & 0x3
            standstill = bool((frame_uint64 >> 47) & 0x1)
            checksum = (frame_uint64 >> 48) & 0xF
            acc_equipped = bool((frame_uint64 >> 52) & 0x1)
            parking_brake_active = bool((frame_uint64 >> 53) & 0x1)
            acc_request = bool((frame_uint64 >> 54) & 0x1)
            driver_braking = bool((frame_uint64 >> 55) & 0x1)
            
            # Extract VSM system
            vsm_coded = bool((frame_uint64 >> 56) & 0x1)
            vsm_available = (frame_uint64 >> 57) & 0x3
            vsm_handshake = bool((frame_uint64 >> 59) & 0x1)
            vsm_brake_status = bool((frame_uint64 >> 60) & 0x1)
            vsm_confirmation_switch = (frame_uint64 >> 61) & 0x3
            aeb_equipped = bool((frame_uint64 >> 63) & 0x1)
            
            # Create signal structure
            signals = TCS13_Signals(
                acceleration_basis_ms2=acceleration_basis,
                acceleration_ref_acc_ms2=acceleration_ref_acc,
                brake_light_active=brake_light_active,
                driver_braking=driver_braking,
                dc_enable=dc_enable,
                acc_enable=acc_enable,
                driver_override=driver_override,
                standstill=standstill,
                acc_equipped=acc_equipped,
                acc_request=acc_request,
                parking_brake_active=parking_brake_active,
                aeb_equipped=aeb_equipped,
                vsm_coded=vsm_coded,
                vsm_available=vsm_available,
                vsm_handshake=vsm_handshake,
                vsm_brake_status=vsm_brake_status,
                vsm_confirmation_switch=vsm_confirmation_switch,
                tqi_scc=tqi_scc,
                alive_counter=alive_counter_tcs,
                checksum=checksum,
                timestamp=timestamp,
                can_id=self.TCS13_CAN_ID,
                raw_data=data
            )
            
            # Validate and update statistics
            if self.validate_tcs13_signals(signals):
                self.stats['tcs13_valid_messages'] += 1
                self.update_tcs13_statistics(signals)
                return signals
            else:
                return None
            
        except Exception as e:
            self.log_message(f"TCS13 parsing error: {e}")
            return None
    
    def parse_tcs15_signals(self, data: bytes, timestamp: float) -> Optional[TCS15_Signals]:
        """Parse TCS15 Warning Lamps message"""
        
        if len(data) != 4:
            self.log_message(f"TCS15 Invalid message length: {len(data)} bytes")
            return None
        
        try:
            # Extend to 8 bytes for easier processing
            data_extended = data + b'\x00\x00\x00\x00'
            frame_uint64 = struct.unpack('<Q', data_extended)[0]
            
            # Extract warning lamps
            abs_warning_lamp = bool(frame_uint64 & 0x1)
            tcs_off_lamp = (frame_uint64 >> 1) & 0x3
            tcs_lamp = (frame_uint64 >> 3) & 0x3
            dbc_warning_lamp = bool((frame_uint64 >> 5) & 0x1)
            dbc_fault_lamp = (frame_uint64 >> 6) & 0x3
            esp_off_step = (frame_uint64 >> 8) & 0x3
            ebd_warning_lamp = bool((frame_uint64 >> 26) & 0x1)
            
            # Extract AVH system
            avh_cluster = (frame_uint64 >> 16) & 0xFF
            avh_indicator_lamp = (frame_uint64 >> 24) & 0x3
            avh_alarm = (frame_uint64 >> 27) & 0x3
            avh_lamp = (frame_uint64 >> 29) & 0x7
            
            # Create signal structure
            signals = TCS15_Signals(
                abs_warning_lamp=abs_warning_lamp,
                tcs_off_lamp=tcs_off_lamp,
                tcs_lamp=tcs_lamp,
                dbc_warning_lamp=dbc_warning_lamp,
                dbc_fault_lamp=dbc_fault_lamp,
                ebd_warning_lamp=ebd_warning_lamp,
                esp_off_step=esp_off_step,
                avh_cluster=avh_cluster,
                avh_indicator_lamp=avh_indicator_lamp,
                avh_alarm=avh_alarm,
                avh_lamp=avh_lamp,
                timestamp=timestamp,
                can_id=self.TCS15_CAN_ID,
                raw_data=data
            )
            
            self.stats['tcs15_valid_messages'] += 1
            return signals
            
        except Exception as e:
            self.log_message(f"TCS15 parsing error: {e}")
            return None
    
    def validate_tcs11_signals(self, signals: TCS11_Signals) -> bool:
        """Validate TCS11 signal values and detect system faults"""
        validation_errors = []
        
        # Check for system faults
        if signals.abs_defect:
            validation_errors.append("ABS system defect")
            self.stats['system_faults'] += 1
        
        if signals.esp_defect:
            validation_errors.append("ESP system defect")
            self.stats['system_faults'] += 1
        
        if signals.tcs_defect:
            validation_errors.append("TCS system defect")
            self.stats['system_faults'] += 1
        
        # Validate torque intervention ranges
        if not (0 <= signals.tqi_tcs <= self.TQI_MAX):
            validation_errors.append(f"TQI_TCS out of range: {signals.tqi_tcs:.1f}")
        
        if validation_errors:
            self.log_message(f"TCS11 validation issues: {', '.join(validation_errors)}")
            
        return True  # Return True even with faults for logging purposes
    
    def validate_tcs13_signals(self, signals: TCS13_Signals) -> bool:
        """Validate TCS13 signal values"""
        validation_errors = []
        
        # Validate acceleration ranges
        if not (self.ACCEL_MIN <= signals.acceleration_basis_ms2 <= self.ACCEL_MAX):
            validation_errors.append(f"Acceleration basis out of range: {signals.acceleration_basis_ms2:.2f}")
        
        if not (self.ACCEL_MIN <= signals.acceleration_ref_acc_ms2 <= self.ACCEL_MAX):
            validation_errors.append(f"Acceleration reference out of range: {signals.acceleration_ref_acc_ms2:.2f}")
        
        if validation_errors:
            self.log_message(f"TCS13 validation errors: {', '.join(validation_errors)}")
            return False
            
        return True
    
    def update_tcs11_statistics(self, signals: TCS11_Signals):
        """Update TCS11 statistics and activity counters"""
        if signals.abs_active:
            self.stats['abs_activations'] += 1
        if signals.esp_control_active:
            self.stats['esp_activations'] += 1
        if signals.tcs_control_active:
            self.stats['tcs_activations'] += 1
    
    def update_tcs13_statistics(self, signals: TCS13_Signals):
        """Update TCS13 statistics"""
        if signals.brake_light_active:
            self.stats['brake_light_activations'] += 1
        if signals.driver_braking:
            self.stats['driver_braking_detections'] += 1
        
        # Track peak accelerations
        self.stats['max_acceleration'] = max(self.stats['max_acceleration'], 
                                           max(signals.acceleration_basis_ms2, signals.acceleration_ref_acc_ms2))
        if signals.acceleration_basis_ms2 < 0:  # Deceleration
            self.stats['max_deceleration'] = max(self.stats['max_deceleration'], 
                                               abs(signals.acceleration_basis_ms2))
    
    def log_to_csv(self, signals, message_type: str):
        """Log signal data to appropriate CSV file"""
        if not self.csv_logging:
            return
            
        if message_type == 'TCS11' and isinstance(signals, TCS11_Signals):
            with open(self.tcs11_csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    signals.timestamp, signals.can_id, self.tcs11_message_count,
                    signals.tcs_request, signals.msr_c_request, signals.tcs_passive, signals.abs_active,
                    signals.esp_control_active, signals.tcs_control_active, signals.abs_defect, signals.esp_defect,
                    signals.tcs_defect, False, False, signals.hac_control_active,  # Placeholders for brake_light, driver_braking
                    signals.tqi_tcs, signals.tqi_msr, signals.alive_counter, signals.checksum,
                    signals.raw_data.hex().upper()
                ])
        
        elif message_type == 'TCS13' and isinstance(signals, TCS13_Signals):
            with open(self.tcs13_csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    signals.timestamp, signals.can_id, self.tcs13_message_count,
                    signals.acceleration_basis_ms2, signals.acceleration_ref_acc_ms2, signals.brake_light_active,
                    signals.driver_braking, signals.dc_enable, signals.acc_enable, signals.driver_override,
                    signals.standstill, signals.acc_equipped, signals.parking_brake_active, signals.tqi_scc,
                    signals.alive_counter, signals.checksum, signals.raw_data.hex().upper()
                ])
        
        elif message_type == 'TCS15' and isinstance(signals, TCS15_Signals):
            with open(self.tcs15_csv_filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    signals.timestamp, signals.can_id, self.tcs15_message_count,
                    signals.abs_warning_lamp, signals.tcs_off_lamp, signals.tcs_lamp, signals.esp_off_step,
                    signals.avh_cluster, signals.avh_lamp, signals.dbc_warning_lamp, signals.ebd_warning_lamp,
                    signals.raw_data.hex().upper()
                ])
    
    def display_braking_data(self):
        """Display comprehensive braking system data"""
        
        print(f"\033[2J\033[H")  # Clear screen
        print("=" * 120)
        print(f"BRAKING SYSTEM MONITOR - {self.interface}")
        print("=" * 120)
        
        print(f"TCS11: {self.tcs11_message_count} | TCS13: {self.tcs13_message_count} | TCS15: {self.tcs15_message_count}")
        print(f"Runtime: {time.time() - self.start_time:.1f}s")
        print()
        
        # TCS11 Data (Electronic Stability Control)
        print("TCS11 - ELECTRONIC STABILITY CONTROL:")
        print("-" * 120)
        if self.latest_tcs11:
            print(f"CAN ID: 0x{self.latest_tcs11.can_id:03X} | Raw: {self.latest_tcs11.raw_data.hex().upper()}")
            
            # System status indicators
            abs_status = "🔴 DEFECT" if self.latest_tcs11.abs_defect else ("🟡 ACTIVE" if self.latest_tcs11.abs_active else "🟢 READY")
            esp_status = "🔴 DEFECT" if self.latest_tcs11.esp_defect else ("🟡 ACTIVE" if self.latest_tcs11.esp_control_active else "🟢 READY")
            tcs_status = "🔴 DEFECT" if self.latest_tcs11.tcs_defect else ("🟡 ACTIVE" if self.latest_tcs11.tcs_control_active else "🟢 READY")
            hac_status = "🟡 ACTIVE" if self.latest_tcs11.hac_control_active else "🟢 STANDBY"
            
            print(f"ABS System:       {abs_status}")
            print(f"ESP System:       {esp_status}")
            print(f"TCS System:       {tcs_status}")
            print(f"Hill Assist:      {hac_status}")
            print(f"Torque Control:   TCS={self.latest_tcs11.tqi_tcs:.1f}% | MSR={self.latest_tcs11.tqi_msr:.1f}%")
            print(f"Message Count:    {self.latest_tcs11.alive_counter} | Checksum: 0x{self.latest_tcs11.checksum:02X}")
        else:
            print("No TCS11 data received")
        print()
        
        # TCS13 Data (Brake Control & Acceleration)
        print("TCS13 - BRAKE CONTROL & ACCELERATION REFERENCE:")
        print("-" * 120)
        if self.latest_tcs13:
            print(f"CAN ID: 0x{self.latest_tcs13.can_id:03X} | Raw: {self.latest_tcs13.raw_data.hex().upper()}")
            
            # Brake status indicators
            brake_light_status = "🔴 ON" if self.latest_tcs13.brake_light_active else "⚫ OFF"
            driver_braking_status = "🔴 BRAKING" if self.latest_tcs13.driver_braking else "⚫ NO INPUT"
            parking_brake_status = "🔴 ENGAGED" if self.latest_tcs13.parking_brake_active else "🟢 RELEASED"
            standstill_status = "🟡 STANDSTILL" if self.latest_tcs13.standstill else "🟢 MOVING"
            
            # Convert to g-force for better understanding
            accel_basis_g = self.latest_tcs13.acceleration_basis_ms2 / 9.81
            accel_ref_g = self.latest_tcs13.acceleration_ref_acc_ms2 / 9.81
            
            print(f"Acceleration Basis: {self.latest_tcs13.acceleration_basis_ms2:+6.2f} m/s² ({accel_basis_g:+5.2f}g)")
            print(f"Acceleration Ref:   {self.latest_tcs13.acceleration_ref_acc_ms2:+6.2f} m/s² ({accel_ref_g:+5.2f}g)")
            print(f"Brake Light:      {brake_light_status} | Driver Braking: {driver_braking_status}")
            print(f"Parking Brake:    {parking_brake_status} | Vehicle State: {standstill_status}")
            
            # ACC/VSM system status
            acc_status = "🟢 EQUIPPED" if self.latest_tcs13.acc_equipped else "⚫ NOT EQUIPPED"
            aeb_status = "🟢 EQUIPPED" if self.latest_tcs13.aeb_equipped else "⚫ NOT EQUIPPED"
            print(f"ACC System:       {acc_status} | AEB System: {aeb_status}")
            print(f"VSM Available:    {self.latest_tcs13.vsm_available} | TQI_SCC: {self.latest_tcs13.tqi_scc:.1f}%")
        else:
            print("No TCS13 data received")
        print()
        
        # TCS15 Data (Warning Lamps)
        print("TCS15 - WARNING LAMPS & INDICATORS:")
        print("-" * 120)
        if self.latest_tcs15:
            print(f"CAN ID: 0x{self.latest_tcs15.can_id:03X} | Raw: {self.latest_tcs15.raw_data.hex().upper()}")
            
            # Warning lamp status
            abs_lamp = "🔴 ON" if self.latest_tcs15.abs_warning_lamp else "⚫ OFF"
            tcs_lamp = "🔴 ON" if self.latest_tcs15.tcs_lamp > 0 else "⚫ OFF"
            esp_off = "🟡 OFF" if self.latest_tcs15.esp_off_step > 0 else "🟢 ACTIVE"
            avh_lamp = "🟡 ON" if self.latest_tcs15.avh_lamp > 0 else "⚫ OFF"
            
            print(f"ABS Warning:      {abs_lamp}")
            print(f"TCS Lamp:         {tcs_lamp} (Level: {self.latest_tcs15.tcs_lamp})")
            print(f"ESP Status:       {esp_off} (Step: {self.latest_tcs15.esp_off_step})")
            print(f"AVH System:       {avh_lamp} (Cluster: {self.latest_tcs15.avh_cluster})")
        else:
            print("No TCS15 data received")
        print()
        
        # Brake System Analysis
        print("BRAKE SYSTEM ANALYSIS:")
        print("-" * 120)
        if self.latest_tcs11 and self.latest_tcs13:
            # Overall system health
            system_faults = []
            if self.latest_tcs11.abs_defect: system_faults.append("ABS")
            if self.latest_tcs11.esp_defect: system_faults.append("ESP")
            if self.latest_tcs11.tcs_defect: system_faults.append("TCS")
            if self.latest_tcs11.ebd_defect: system_faults.append("EBD")
            
            if system_faults:
                print(f"⚠️  SYSTEM FAULTS: {', '.join(system_faults)}")
            else:
                print("✅ ALL BRAKE SYSTEMS OPERATIONAL")
            
            # Active interventions
            active_systems = []
            if self.latest_tcs11.abs_active: active_systems.append("ABS")
            if self.latest_tcs11.esp_control_active: active_systems.append("ESP")
            if self.latest_tcs11.tcs_control_active: active_systems.append("TCS")
            if self.latest_tcs11.hac_control_active: active_systems.append("HAC")
            
            if active_systems:
                print(f"🔄 ACTIVE INTERVENTIONS: {', '.join(active_systems)}")
            else:
                print("⏸️  NO ACTIVE INTERVENTIONS")
            
            # Braking state analysis
            if self.latest_tcs13.driver_braking and self.latest_tcs13.brake_light_active:
                decel_magnitude = abs(self.latest_tcs13.acceleration_basis_ms2) if self.latest_tcs13.acceleration_basis_ms2 < 0 else 0
                if decel_magnitude > 6.0:
                    print(f"🚨 EMERGENCY BRAKING: {decel_magnitude:.2f} m/s² ({decel_magnitude/9.81:.2f}g)")
                elif decel_magnitude > 3.0:
                    print(f"🟡 HARD BRAKING: {decel_magnitude:.2f} m/s² ({decel_magnitude/9.81:.2f}g)")
                else:
                    print(f"🟢 NORMAL BRAKING: {decel_magnitude:.2f} m/s² ({decel_magnitude/9.81:.2f}g)")
            elif self.latest_tcs13.standstill:
                print("🟡 VEHICLE AT STANDSTILL")
            else:
                print("🟢 NO BRAKING DETECTED")
        else:
            print("Insufficient data for brake system analysis")
        print()
        
        # Session Statistics
        print("SESSION STATISTICS:")
        print("-" * 120)
        print(f"Messages - TCS11: {self.stats['tcs11_total_messages']} | TCS13: {self.stats['tcs13_total_messages']} | TCS15: {self.stats['tcs15_total_messages']}")
        print(f"System Activations - ABS: {self.stats['abs_activations']} | ESP: {self.stats['esp_activations']} | TCS: {self.stats['tcs_activations']}")
        print(f"Brake Events - Light: {self.stats['brake_light_activations']} | Driver: {self.stats['driver_braking_detections']}")
        print(f"Peak Deceleration: {self.stats['max_deceleration']:.2f} m/s² ({self.stats['max_deceleration']/9.81:.2f}g)")
        print(f"System Faults: {self.stats['system_faults']}")
        
        if self.csv_logging:
            print(f"Data Logging: {self.tcs11_csv_filename}, {self.tcs13_csv_filename}, {self.tcs15_csv_filename}")
        
        print("\nPress Ctrl+C to stop...")
    
    def monitor(self):
        """Main monitoring loop"""
        
        print(f"Starting Braking System Monitor")
        print(f"Interface: {self.interface}")
        print(f"Targets: TCS11 (0x{self.TCS11_CAN_ID:03X}), TCS13 (0x{self.TCS13_CAN_ID:03X}), TCS15 (0x{self.TCS15_CAN_ID:03X})")
        print(f"Electronic Stability & Brake Control Analysis")
        print("=" * 120)
        
        last_display_time = 0
        display_interval = 0.2  # Update display every 200ms
        
        try:
            while True:
                # Receive CAN message
                message = self.bus.recv(timeout=0.1)
                
                if message is None:
                    continue
                
                # Check for TCS11 message
                if message.arbitration_id == self.TCS11_CAN_ID:
                    self.tcs11_message_count += 1
                    self.stats['tcs11_total_messages'] += 1
                    
                    signals = self.parse_tcs11_signals(message.data, message.timestamp)
                    if signals:
                        self.latest_tcs11 = signals
                        self.log_to_csv(signals, 'TCS11')
                
                # Check for TCS13 message
                elif message.arbitration_id == self.TCS13_CAN_ID:
                    self.tcs13_message_count += 1
                    self.stats['tcs13_total_messages'] += 1
                    
                    signals = self.parse_tcs13_signals(message.data, message.timestamp)
                    if signals:
                        self.latest_tcs13 = signals
                        self.log_to_csv(signals, 'TCS13')
                
                # Check for TCS15 message
                elif message.arbitration_id == self.TCS15_CAN_ID:
                    self.tcs15_message_count += 1
                    self.stats['tcs15_total_messages'] += 1
                    
                    signals = self.parse_tcs15_signals(message.data, message.timestamp)
                    if signals:
                        self.latest_tcs15 = signals
                        self.log_to_csv(signals, 'TCS15')
                
                # Update display periodically
                current_time = time.time()
                if current_time - last_display_time >= display_interval:
                    self.display_braking_data()
                    last_display_time = current_time
                    
        except KeyboardInterrupt:
            print(f"\nBraking System monitoring stopped")
            self.log_message(f"Monitoring stopped - TCS11: {self.tcs11_message_count}, TCS13: {self.tcs13_message_count}")
            self.print_final_statistics()
        finally:
            if self.bus:
                self.bus.shutdown()
    
    def print_final_statistics(self):
        """Print final session statistics"""
        print(f"\nFINAL BRAKING SYSTEM STATISTICS:")
        print("=" * 80)
        print(f"Runtime: {time.time() - self.start_time:.1f} seconds")
        print(f"TCS11 Messages: {self.stats['tcs11_total_messages']} (Valid: {self.stats['tcs11_valid_messages']})")
        print(f"TCS13 Messages: {self.stats['tcs13_total_messages']} (Valid: {self.stats['tcs13_valid_messages']})")
        print(f"TCS15 Messages: {self.stats['tcs15_total_messages']} (Valid: {self.stats['tcs15_valid_messages']})")
        print()
        print(f"System Safety Analysis:")
        print(f"- ABS Activations: {self.stats['abs_activations']}")
        print(f"- ESP Interventions: {self.stats['esp_activations']}")
        print(f"- TCS Interventions: {self.stats['tcs_activations']}")
        print(f"- System Faults Detected: {self.stats['system_faults']}")
        print()
        print(f"Braking Performance:")
        print(f"- Peak Deceleration: {self.stats['max_deceleration']:.2f} m/s² ({self.stats['max_deceleration']/9.81:.2f}g)")
        print(f"- Brake Light Activations: {self.stats['brake_light_activations']}")
        print(f"- Driver Braking Events: {self.stats['driver_braking_detections']}")
        
        if self.csv_logging:
            print(f"\nTCS11 Data: {self.tcs11_csv_filename}")
            print(f"TCS13 Data: {self.tcs13_csv_filename}")
            print(f"TCS15 Data: {self.tcs15_csv_filename}")
        if self.log_to_file:
            print(f"Log: {self.log_filename}")

def main():
    parser = argparse.ArgumentParser(
        description='Braking System Monitor - Electronic Stability & Brake Control',
        epilog='''
Comprehensive brake system monitoring:
- TCS11: ABS/ESP/TCS system states and control
- TCS13: Brake control and acceleration references  
- TCS15: Warning lamps and system indicators
- Real-time safety analysis and fault detection

DBC-compliant parsing with brake system safety validation.
        '''
    )
    parser.add_argument('--interface', '-i', default='can1', 
                       help='CAN interface (default: can1)')
    parser.add_argument('--no-csv', action='store_true',
                       help='Disable CSV data logging')
    parser.add_argument('--no-log', action='store_true', 
                       help='Disable text file logging')
    
    args = parser.parse_args()
    
    print("Braking System Monitor")
    print("Electronic Stability & Brake Control Analysis")
    print("DBC-Compliant with Safety Validation")
    print()
    
    monitor = BrakingSystemMonitor(
        interface=args.interface,
        log_to_file=not args.no_log,
        csv_logging=not args.no_csv
    )
    monitor.monitor()

if __name__ == "__main__":
    main()
