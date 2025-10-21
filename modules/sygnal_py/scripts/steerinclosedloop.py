import can
import cantools
import crc8
import asyncio
import signal
import sys
import struct
from datetime import datetime


class ClosedLoopSteeringController:
    def __init__(self, channel='can2'):
        self.db = cantools.database.Database()
        self.db.add_dbc_file('./sygnal_dbc/mcm/Heartbeat.dbc')
        self.db.add_dbc_file('./sygnal_dbc/mcm/Control.dbc')
        
        try:
            self.bus = can.Bus(channel=channel, bustype='socketcan', bitrate=500000)
            print(f"Connected to {channel}")
        except Exception as e:
            print(f"Failed to connect: {e}")
            sys.exit(1)

        self.control_count = 0
        self.bus_address = 1
        self.interface_id = 2
        self.control_enabled = False
        self.current_steering_value = 0
        self.target_steering_value = 0
        
        # Response tracking
        self.enable_confirmed = False
        self.command_confirmed = False
        self.last_response_value = 0
        
    def calc_crc8(self, data):
        hash = crc8.crc8()
        hash.update(data[:-1])
        return hash.digest()[0]
    
    def float_to_int32(self, value):
        """Convert float (-1.0 to +1.0) to int32 range for CAN transmission"""
        return int(value * 2147483647)
    
    def int32_to_float(self, value):
        """Convert int32 value back to float (-1.0 to +1.0) range"""
        return value / 2147483647.0

    async def enable_control_with_response(self):
        """Send enable command and wait for response"""
        msg = self.db.get_message_by_name('ControlEnable')
        data = bytearray(msg.encode({
            'BusAddress': self.bus_address,
            'InterfaceID': self.interface_id,
            'Enable': 1,
            'CRC': 0
        }))
        data[7] = self.calc_crc8(data)

        self.enable_confirmed = False
        
        try:
            self.bus.send(can.Message(
                arbitration_id=msg.frame_id,
                is_extended_id=False,
                data=data
            ))
            print("Enable command sent, waiting for response...")
            
            # Wait for enable response
            timeout_count = 0
            while not self.enable_confirmed and timeout_count < 10:
                await asyncio.sleep(0.01)
                timeout_count += 1
                
            if self.enable_confirmed:
                print("Control enabled successfully")
                self.control_enabled = True
                return True
            else:
                print("Enable response timeout")
                return False
                
        except Exception as e:
            print(f"Failed to enable control: {e}")
            return False

    async def send_command_with_response(self, target_value):
        """Send steering command and wait for response confirmation"""
        int_value = self.float_to_int32(target_value)
        
        msg = self.db.get_message_by_name('ControlCommand')
        data = bytearray(msg.encode({
            'BusAddress': self.bus_address,
            'InterfaceID': self.interface_id,
            'Count8': self.control_count,
            'Value': int_value,
            'CRC': 0
        }))
        data[7] = self.calc_crc8(data)

        self.command_confirmed = False
        
        try:
            self.bus.send(can.Message(
                arbitration_id=msg.frame_id,
                is_extended_id=False,
                data=data
            ))
            
            print(f"Command sent: {target_value:.3f} (int32: {int_value})")
            self.control_count = (self.control_count + 1) % 256
            
            # Wait for command response
            timeout_count = 0
            while not self.command_confirmed and timeout_count < 20:
                await asyncio.sleep(0.01)
                timeout_count += 1
                
            if self.command_confirmed:
                response_float = self.int32_to_float(self.last_response_value)
                error = abs(target_value - response_float)
                print(f"Response received: {response_float:.3f} (int32: {self.last_response_value})")
                print(f"Command error: {error:.6f}")
                self.current_steering_value = response_float
                return True, error
            else:
                print("Command response timeout")
                return False, 1.0
                
        except Exception as e:
            print(f"Failed to send command: {e}")
            return False, 1.0

    async def step_to_target(self, target_percentage):
        """Step gradually to target with closed-loop verification"""
        target = target_percentage / 100.0
        current = self.current_steering_value
        
        print(f"\nStepping from {current:.3f} to {target:.3f}")
        
        if not self.control_enabled:
            if not await self.enable_control_with_response():
                print("Failed to enable control")
                return False
        
        # Calculate stepping parameters
        total_distance = abs(target - current)
        step_size = 0.05
        num_steps = max(int(total_distance / step_size), 1)
        actual_step = (target - current) / num_steps
        
        step_value = current
        for step in range(num_steps):
            step_value += actual_step
            step_value = max(min(step_value, 1.0), -1.0)
            
            success, error = await self.send_command_with_response(step_value)
            if not success:
                print(f"Step {step+1} failed")
                return False
                
            if error > 0.01:  # 1% tolerance
                print(f"WARNING: High error on step {step+1}: {error:.6f}")
            
            await asyncio.sleep(0.05)  # Allow system to settle
        
        # Final target command
        if abs(step_value - target) > 0.001:
            success, error = await self.send_command_with_response(target)
            if success:
                print(f"Final positioning complete, error: {error:.6f}")
            else:
                print("Final positioning failed")
                return False
        
        self.target_steering_value = target
        return True

    async def monitor_responses(self):
        """Monitor CAN bus for response messages"""
        enable_response_id = self.db.get_message_by_name('ControlEnableResponse').frame_id
        command_response_id = self.db.get_message_by_name('ControlCommandResponse').frame_id
        
        while True:
            try:
                msg = self.bus.recv(timeout=0.01)
                if msg:
                    if msg.arbitration_id == enable_response_id:
                        decoded = self.db.decode_message(msg.arbitration_id, msg.data)
                        if (decoded.get('BusAddress') == self.bus_address and 
                            decoded.get('InterfaceID') == self.interface_id):
                            self.enable_confirmed = decoded.get('Enable') == 1
                    
                    elif msg.arbitration_id == command_response_id:
                        decoded = self.db.decode_message(msg.arbitration_id, msg.data)
                        if (decoded.get('BusAddress') == self.bus_address and 
                            decoded.get('InterfaceID') == self.interface_id):
                            self.last_response_value = decoded.get('Value')
                            self.command_confirmed = True
                            
            except can.CanOperationError:
                pass
            await asyncio.sleep(0.001)


async def main():
    controller = ClosedLoopSteeringController()
    
    # Start response monitoring task
    monitor_task = asyncio.create_task(controller.monitor_responses())
    
    def cleanup(sig=None, frame=None):
        print("\nStopping...")
        monitor_task.cancel()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, cleanup)
    
    try:
        while True:
            try:
                user_input = float(input("\nEnter steering percentage (-100 to 100): "))
                if -100 <= user_input <= 100:
                    success = await controller.step_to_target(user_input)
                    if success:
                        print(f"Successfully positioned to {user_input}%")
                    else:
                        print(f"Failed to reach target {user_input}%")
                else:
                    print("Value must be between -100 and 100")
            except ValueError:
                print("Invalid input. Please enter a number.")
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    asyncio.run(main())
