#!/usr/bin/env python3
"""
FINAL Corrected Wheel Speed Monitor - DBC 14-bit Method
Based on comparison with actual vehicle odometer at 5 mph

CONFIRMED: Method 4 (DBC 14-bit) gives correct readings:
- Shows 0.00 km/h when vehicle is idle (CORRECT)
- Shows ~7 km/h when odometer reads 5 mph (CORRECT)
- Odd bytes method was giving false low readings
"""

import can
import struct
import time
import argparse
import sys

class WheelSpeedMonitor:
    def __init__(self, interface='can1', wheel_radius=0.310):
        self.interface = interface
        self.wheel_radius = wheel_radius
        self.message_count = 0
        self.start_time = time.time()
        
        # Physical constants
        self.KMH_TO_MPH = 1.0 / 1.609344
        self.RAD_TO_RPM = 60.0 / (2.0 * 3.14159)
        
        # Connect to CAN bus
        try:
            self.bus = can.interface.Bus(channel=interface, bustype='socketcan')
            print(f"Connected to {interface}")
        except Exception as e:
            print(f"Failed to connect to {interface}: {e}")
            sys.exit(1)
    
    def parse_wheel_speeds_dbc_method(self, data):
        """
        FINAL CORRECT METHOD: DBC 14-bit extraction
        Confirmed by odometer comparison:
        - Vehicle odometer: 5 mph
        - This method: ~7 km/h = ~4.3 mph (realistic)
        - Odd bytes method: ~2 mph (wrong)
        """
        
        if len(data) != 8:
            return None
        
        try:
            # Unpack as little-endian 64-bit unsigned integer
            frame_uint64 = struct.unpack('<Q', data)[0]
            
            # Extract 14-bit wheel speed values using DBC specification
            wheel_speeds_raw = {
                'FL': (frame_uint64 >> 0) & 0x3FFF,   # Bits [13:0]
                'FR': (frame_uint64 >> 16) & 0x3FFF,  # Bits [29:16]
                'RL': (frame_uint64 >> 32) & 0x3FFF,  # Bits [45:32] 
                'RR': (frame_uint64 >> 48) & 0x3FFF   # Bits [61:48]
            }
            
            # Apply DBC scaling: 0.03125 km/h per LSB
            wheel_speeds_kmh = {wheel: raw * 0.03125 for wheel, raw in wheel_speeds_raw.items()}
            
            # Convert to mph
            wheel_speeds_mph = {wheel: kmh * self.KMH_TO_MPH for wheel, kmh in wheel_speeds_kmh.items()}
            
            # Convert to rpm
            wheel_speeds_rpm = {}
            for wheel, kmh in wheel_speeds_kmh.items():
                ms = kmh * (1000.0 / 3600.0)  # km/h to m/s
                rads = ms / self.wheel_radius  # rad/s
                rpm = rads * self.RAD_TO_RPM  # RPM
                wheel_speeds_rpm[wheel] = rpm
            
            return wheel_speeds_kmh, wheel_speeds_mph, wheel_speeds_rpm, wheel_speeds_raw
            
        except Exception as e:
            print(f"DBC parsing error: {e}")
            return None
    
    def display_speeds(self, kmh_data, mph_data, rpm_data, raw_data, can_raw):
        """Display speed data with validation info"""
        
        print(f"\033[2J\033[H")  # Clear screen
        print("=" * 80)
        print(f"WHEEL SPEED MONITOR - {self.interface} - Method: DBC 14-bit")
        print("=" * 80)
        
        print(f"Raw CAN Data: {can_raw}")
        print(f"14-bit Values: FL={raw_data['FL']}, FR={raw_data['FR']}, RL={raw_data['RL']}, RR={raw_data['RR']}")
        print(f"Messages: {self.message_count} | Runtime: {time.time() - self.start_time:.1f}s")
        print()
        
        print(f"{'Wheel':<8} {'Raw':<6} {'km/h':<8} {'mph':<8} {'RPM':<10}")
        print("-" * 80)
        
        for wheel in ['FL', 'FR', 'RL', 'RR']:
            raw = raw_data[wheel]
            kmh = kmh_data[wheel]
            mph = mph_data[wheel]
            rpm = rpm_data[wheel]
            print(f"{wheel:<8} {raw:<6} {kmh:<8.2f} {mph:<8.2f} {rpm:<10.1f}")
        
        avg_mph = sum(mph_data.values()) / 4
        avg_kmh = sum(kmh_data.values()) / 4
        max_speed = max(mph_data.values())
        
        print("-" * 80)
        print(f"Average Speed: {avg_mph:.2f} mph ({avg_kmh:.2f} km/h) | Max: {max_speed:.2f} mph")
        print(f"Method: DBC 14-bit extraction (CONFIRMED CORRECT)")
        print(f"Scale: 0.03125 km/h per LSB | Wheel radius: {self.wheel_radius:.3f}m")
        
        # Validation indicators
        if 0.1 <= avg_mph <= 100:
            print(f"Status: REALISTIC SPEEDS - Compare with odometer")
        else:
            print(f"Status: Check readings - unusual speeds detected")
            
        print("Press Ctrl+C to stop...")
    
    def monitor(self):
        """Main monitoring loop"""
        
        print(f"Starting FINAL Wheel Speed Monitor")
        print(f"Interface: {self.interface}")
        print(f"Target: CAN ID 0x386 (WHL_SPD11)")
        print(f"Method: DBC 14-bit extraction (validated against odometer)")
        print("=" * 80)
        
        last_display_time = 0
        display_interval = 0.3  # Update display every 300ms
        
        try:
            while True:
                # Receive CAN message
                message = self.bus.recv(timeout=0.1)
                
                if message is None:
                    continue
                
                # Check for wheel speed message
                if message.arbitration_id == 0x386:
                    self.message_count += 1
                    
                    # Parse wheel speeds using DBC method
                    result = self.parse_wheel_speeds_dbc_method(message.data)
                    if result:
                        kmh_data, mph_data, rpm_data, raw_data = result
                        
                        # Update display periodically
                        current_time = time.time()
                        if current_time - last_display_time >= display_interval:
                            self.display_speeds(kmh_data, mph_data, rpm_data, raw_data, message.data.hex().upper())
                            last_display_time = current_time
                    
        except KeyboardInterrupt:
            print(f"\nMonitoring stopped")
            print(f"Processed {self.message_count} messages")
        finally:
            if self.bus:
                self.bus.shutdown()

def main():
    parser = argparse.ArgumentParser(
        description='FINAL Wheel Speed Monitor - DBC Method (Validated)',
        epilog='''
VALIDATION RESULTS:
- Vehicle odometer: 5 mph
- DBC method: ~7 km/h = ~4.3 mph (REALISTIC)
- Odd bytes method: ~2 mph (WRONG)

DBC 14-bit extraction is the CORRECT method for this vehicle.
        '''
    )
    parser.add_argument('--interface', '-i', default='can1', 
                       help='CAN interface (default: can1)')
    parser.add_argument('--radius', '-r', type=float, default=0.310,
                       help='Wheel radius in meters (default: 0.310)')
    
    args = parser.parse_args()
    
    print("Starting FINAL Corrected Wheel Speed Monitor")
    print("Using DBC 14-bit extraction method")
    print("VALIDATED against actual vehicle odometer")
    print()
    
    monitor = WheelSpeedMonitor(args.interface, args.radius)
    monitor.monitor()

if __name__ == "__main__":
    main()
