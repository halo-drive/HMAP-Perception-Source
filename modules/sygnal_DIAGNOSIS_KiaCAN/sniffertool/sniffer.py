#!/usr/bin/env python3
"""
CAN Bus Signal Discovery Tool
Monitors CAN interface to capture unique frames and analyze signal patterns
for matching against DBC reference files.
"""

import can
import sys
import time
import signal
import argparse
import threading
from collections import defaultdict, deque
from datetime import datetime
import json

class CANSignalDiscovery:
    def __init__(self, interface='can0', bustype='socketcan'):
        self.interface = interface
        self.bustype = bustype
        self.bus = None
        self.running = False
        
        # Data structures for analysis
        self.unique_frames = {}  # frame_id -> latest frame data
        self.frame_stats = defaultdict(lambda: {
            'count': 0,
            'first_seen': None,
            'last_seen': None,
            'data_variants': set(),
            'period_estimates': deque(maxlen=10)
        })
        self.signal_patterns = defaultdict(lambda: defaultdict(set))  # frame_id -> byte_pos -> unique_values
        
        # Threading
        self.lock = threading.Lock()
        self.display_thread = None
        
        # Configuration
        self.display_interval = 1.0  # seconds
        self.max_data_variants = 50  # limit memory usage
        
    def setup_can_interface(self):
        """Initialize CAN bus connection"""
        try:
            self.bus = can.interface.Bus(
                channel=self.interface,
                bustype=self.bustype,
                receive_own_messages=False
            )
            print(f"[INFO] Connected to CAN interface: {self.interface} ({self.bustype})")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to connect to CAN interface {self.interface}: {e}")
            return False
    
    def signal_handler(self, sig, frame):
        """Handle Ctrl+C gracefully"""
        print(f"\n[INFO] Received signal {sig}, shutting down...")
        self.stop_monitoring()
    
    def analyze_frame(self, msg):
        """Analyze incoming CAN frame for patterns"""
        frame_id = msg.arbitration_id
        timestamp = time.time()
        data_bytes = tuple(msg.data)
        
        with self.lock:
            # Update unique frames
            self.unique_frames[frame_id] = {
                'id': frame_id,
                'dlc': msg.dlc,
                'data': data_bytes,
                'timestamp': timestamp,
                'is_extended': msg.is_extended_id,
                'is_error': msg.is_error_frame,
                'is_remote': msg.is_remote_frame
            }
            
            # Update statistics
            stats = self.frame_stats[frame_id]
            stats['count'] += 1
            
            if stats['first_seen'] is None:
                stats['first_seen'] = timestamp
            
            # Calculate period estimate
            if stats['last_seen'] is not None:
                period = timestamp - stats['last_seen']
                if 0.001 <= period <= 10.0:  # reasonable period range
                    stats['period_estimates'].append(period)
            
            stats['last_seen'] = timestamp
            
            # Track data variants (limit to prevent memory bloat)
            if len(stats['data_variants']) < self.max_data_variants:
                stats['data_variants'].add(data_bytes)
            
            # Analyze byte-level signal patterns
            for byte_pos, byte_val in enumerate(data_bytes):
                self.signal_patterns[frame_id][byte_pos].add(byte_val)
    
    def get_frame_period_estimate(self, frame_id):
        """Estimate transmission period for a frame"""
        periods = self.frame_stats[frame_id]['period_estimates']
        if len(periods) >= 3:
            # Use median for better stability
            sorted_periods = sorted(periods)
            median_idx = len(sorted_periods) // 2
            return sorted_periods[median_idx]
        return None
    
    def classify_signal_type(self, frame_id, byte_pos):
        """Classify signal type based on observed patterns"""
        values = self.signal_patterns[frame_id][byte_pos]
        unique_count = len(values)
        
        if unique_count <= 1:
            return "STATIC"
        elif unique_count == 2:
            return "BINARY"
        elif unique_count <= 16:
            return "ENUM"
        elif unique_count <= 64:
            return "COUNTER/MULTI"
        else:
            return "ANALOG"
    
    def display_status(self):
        """Display current monitoring status"""
        while self.running:
            try:
                time.sleep(self.display_interval)
                self.print_current_status()
            except Exception as e:
                print(f"[ERROR] Display thread error: {e}")
    
    def print_current_status(self):
        """Print formatted status to console"""
        with self.lock:
            if not self.unique_frames:
                return
            
            # Clear screen and print header
            print("\033[2J\033[H")  # Clear screen, move cursor to top
            print("=" * 100)
            print(f"CAN Signal Discovery - Interface: {self.interface}")
            print(f"Monitoring started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Total unique frames: {len(self.unique_frames)}")
            print("=" * 100)
            
            # Sort frames by ID for consistent display
            sorted_frames = sorted(self.unique_frames.items())
            
            for frame_id, frame_data in sorted_frames:
                stats = self.frame_stats[frame_id]
                period = self.get_frame_period_estimate(frame_id)
                
                # Frame header
                id_str = f"0x{frame_id:03X}" if frame_id <= 0x7FF else f"0x{frame_id:08X}"
                period_str = f"{period*1000:.1f}ms" if period else "Unknown"
                
                print(f"\nFrame ID: {id_str} | Count: {stats['count']:6d} | "
                      f"Period: {period_str:8s} | DLC: {frame_data['dlc']}")
                
                # Latest data
                data_hex = " ".join(f"{b:02X}" for b in frame_data['data'])
                print(f"  Data: [{data_hex}]")
                
                # Signal analysis
                if len(frame_data['data']) > 0:
                    signal_info = []
                    for byte_pos in range(len(frame_data['data'])):
                        signal_type = self.classify_signal_type(frame_id, byte_pos)
                        unique_vals = len(self.signal_patterns[frame_id][byte_pos])
                        signal_info.append(f"B{byte_pos}:{signal_type}({unique_vals})")
                    
                    print(f"  Signals: {' | '.join(signal_info)}")
                
                # Data variants summary
                variant_count = len(stats['data_variants'])
                if variant_count > 1:
                    print(f"  Variants: {variant_count} different data patterns observed")
    
    def export_results(self, filename=None):
        """Export discovered signals to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"can_signals_{self.interface}_{timestamp}.json"
        
        export_data = {
            'interface': self.interface,
            'timestamp': datetime.now().isoformat(),
            'frames': {}
        }
        
        with self.lock:
            for frame_id, frame_data in self.unique_frames.items():
                stats = self.frame_stats[frame_id]
                period = self.get_frame_period_estimate(frame_id)
                
                # Convert sets to lists for JSON serialization
                data_variants = [list(variant) for variant in stats['data_variants']]
                
                signal_analysis = {}
                for byte_pos in range(len(frame_data['data'])):
                    values = list(self.signal_patterns[frame_id][byte_pos])
                    signal_analysis[f"byte_{byte_pos}"] = {
                        'unique_values': values,
                        'value_count': len(values),
                        'classification': self.classify_signal_type(frame_id, byte_pos)
                    }
                
                export_data['frames'][f"0x{frame_id:X}"] = {
                    'frame_id': frame_id,
                    'dlc': frame_data['dlc'],
                    'count': stats['count'],
                    'period_ms': period * 1000 if period else None,
                    'data_variants': data_variants,
                    'signal_analysis': signal_analysis,
                    'flags': {
                        'extended_id': frame_data['is_extended'],
                        'error_frame': frame_data['is_error'],
                        'remote_frame': frame_data['is_remote']
                    }
                }
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            print(f"\n[INFO] Results exported to: {filename}")
        except Exception as e:
            print(f"[ERROR] Failed to export results: {e}")
    
    def start_monitoring(self):
        """Start CAN monitoring"""
        if not self.setup_can_interface():
            return False
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.running = True
        
        # Start display thread
        self.display_thread = threading.Thread(target=self.display_status, daemon=True)
        self.display_thread.start()
        
        print(f"[INFO] Starting CAN monitoring on {self.interface}")
        print("[INFO] Press Ctrl+C to stop and export results")
        
        try:
            # Main monitoring loop
            while self.running:
                msg = self.bus.recv(timeout=1.0)
                if msg is not None:
                    self.analyze_frame(msg)
        
        except Exception as e:
            print(f"[ERROR] Monitoring error: {e}")
        
        finally:
            self.stop_monitoring()
        
        return True
    
    def stop_monitoring(self):
        """Stop monitoring and cleanup"""
        self.running = False
        
        if self.bus:
            self.bus.shutdown()
        
        # Wait for display thread to finish
        if self.display_thread and self.display_thread.is_alive():
            self.display_thread.join(timeout=2.0)
        
        # Export results
        self.export_results()
        print("\n[INFO] Monitoring stopped")

def main():
    parser = argparse.ArgumentParser(
        description="CAN Bus Signal Discovery Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                           # Monitor can0 with socketcan
  %(prog)s -i can1                   # Monitor can1
  %(prog)s -i vcan0 -b socketcan     # Monitor virtual CAN
  %(prog)s -i can0 -b pcan           # Use PCAN interface
        """
    )
    
    parser.add_argument('-i', '--interface', default='can0',
                        help='CAN interface name (default: can0)')
    parser.add_argument('-b', '--bustype', default='socketcan',
                        choices=['socketcan', 'pcan', 'ixxat', 'vector', 'serial'],
                        help='CAN bus type (default: socketcan)')
    parser.add_argument('--display-interval', type=float, default=1.0,
                        help='Display update interval in seconds (default: 1.0)')
    
    args = parser.parse_args()
    
    # Create and start monitor
    monitor = CANSignalDiscovery(
        interface=args.interface,
        bustype=args.bustype
    )
    monitor.display_interval = args.display_interval
    
    print("CAN Signal Discovery Tool")
    print("=" * 50)
    print(f"Interface: {args.interface}")
    print(f"Bus Type:  {args.bustype}")
    print("=" * 50)
    
    # Check if we can run candump in parallel
    print("\nTip: Run 'candump -t A {}' in another terminal for raw frame comparison".format(args.interface))
    print("Tip: Ensure your CAN interface is up: 'sudo ip link set {} up type can bitrate 500000'".format(args.interface))
    
    success = monitor.start_monitoring()
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
