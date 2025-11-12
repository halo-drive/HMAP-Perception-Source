import can
import cantools
import crc8
import asyncio
import time
from datetime import datetime


class InterfaceDiscovery:
    def __init__(self, channel='can2'):
        self.db = cantools.database.Database()
        self.db.add_dbc_file('./sygnal_dbc/mcm/Control.dbc')
        
        try:
            self.bus = can.Bus(channel=channel, bustype='socketcan', bitrate=500000)
            print(f"Connected to {channel}")
        except Exception as e:
            print(f"Failed to connect: {e}")
            return

        self.bus_address = 1
        self.control_count = 0
        self.responses = {}  # interface_id -> response_data
        
    def calc_crc8(self, data):
        hash = crc8.crc8()
        hash.update(data[:-1])
        return hash.digest()[0]

    async def enable_interface(self, interface_id):
        """Enable control for specific interface"""
        msg = self.db.get_message_by_name('ControlEnable')
        data = bytearray(msg.encode({
            'BusAddress': self.bus_address,
            'InterfaceID': interface_id,
            'Enable': 1,
            'CRC': 0
        }))
        data[7] = self.calc_crc8(data)

        self.bus.send(can.Message(
            arbitration_id=msg.frame_id,
            is_extended_id=False,
            data=data
        ))
        print(f"Enable sent for InterfaceID {interface_id}")
        await asyncio.sleep(0.1)

    async def test_interface(self, interface_id, test_value=0.5):
        """Test specific interface with command"""
        int_value = int(test_value * 2147483647)
        
        msg = self.db.get_message_by_name('ControlCommand')
        data = bytearray(msg.encode({
            'BusAddress': self.bus_address,
            'InterfaceID': interface_id,
            'Count8': self.control_count,
            'Value': int_value,
            'CRC': 0
        }))
        data[7] = self.calc_crc8(data)

        timestamp = time.time()
        self.bus.send(can.Message(
            arbitration_id=msg.frame_id,
            is_extended_id=False,
            data=data
        ))
        
        self.control_count = (self.control_count + 1) % 256
        print(f"Command sent to Interface {interface_id}: {test_value} at {timestamp:.3f}")
        return timestamp

    async def monitor_responses(self, duration=2.0):
        """Monitor for responses from all interfaces"""
        enable_response_id = self.db.get_message_by_name('ControlEnableResponse').frame_id
        command_response_id = self.db.get_message_by_name('ControlCommandResponse').frame_id
        
        start_time = time.time()
        
        while time.time() - start_time < duration:
            try:
                msg = self.bus.recv(timeout=0.01)
                if msg:
                    if msg.arbitration_id == enable_response_id:
                        decoded = self.db.decode_message(msg.arbitration_id, msg.data)
                        interface_id = decoded.get('InterfaceID')
                        bus_addr = decoded.get('BusAddress')
                        
                        if bus_addr == self.bus_address:
                            enable_status = decoded.get('Enable')
                            print(f"EnableResponse - Interface {interface_id}: Enable={enable_status}")
                            
                    elif msg.arbitration_id == command_response_id:
                        decoded = self.db.decode_message(msg.arbitration_id, msg.data)
                        interface_id = decoded.get('InterfaceID')
                        bus_addr = decoded.get('BusAddress')
                        
                        if bus_addr == self.bus_address:
                            value = decoded.get('Value')
                            count = decoded.get('Count8')
                            float_val = value / 2147483647.0 if value else 0
                            
                            print(f"CommandResponse - Interface {interface_id}: Value={float_val:.3f} (raw={value}), Count={count}")
                            
                            if interface_id not in self.responses:
                                self.responses[interface_id] = []
                            self.responses[interface_id].append({
                                'timestamp': time.time(),
                                'value': float_val,
                                'raw_value': value,
                                'count': count
                            })
                            
            except can.CanOperationError:
                pass
            await asyncio.sleep(0.001)

    async def discover_interfaces(self):
        """Test all possible InterfaceID values (0-7)"""
        
        print("=== MCM Interface Discovery ===")
        print("Testing InterfaceID values 0-7...")
        
        # Test each interface
        for interface_id in range(8):  # 3-bit field allows 0-7
            print(f"\n--- Testing Interface {interface_id} ---")
            
            # Enable interface
            await self.enable_interface(interface_id)
            
            # Send test commands
            await self.test_interface(interface_id, 0.3)   # 30%
            await asyncio.sleep(0.2)
            await self.test_interface(interface_id, -0.3)  # -30%
            await asyncio.sleep(0.2)
            await self.test_interface(interface_id, 0.0)   # Center
            
            # Monitor responses
            await self.monitor_responses(duration=1.0)
            
        print(f"\n=== Discovery Results ===")
        print(f"Active interfaces found: {len(self.responses)}")
        
        for interface_id, responses in self.responses.items():
            print(f"Interface {interface_id}: {len(responses)} responses")
            if responses:
                values = [r['value'] for r in responses]
                print(f"  Value range: {min(values):.3f} to {max(values):.3f}")

async def main():
    discovery = InterfaceDiscovery()
    await discovery.discover_interfaces()

if __name__ == "__main__":
    asyncio.run(main())
