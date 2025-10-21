import can
import cantools
import crc8
import asyncio
import signal
import sys
from datetime import datetime


class Controller:
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
        self.last_steer_value = 0
        self.control_enabled = False

    def calc_crc8(self, data):
        hash = crc8.crc8()
        hash.update(data[:-1])
        return hash.digest()[0]

    async def enable_control(self):
        msg = self.db.get_message_by_name('ControlEnable')
        data = bytearray(msg.encode({
            'BusAddress': self.bus_address,
            'InterfaceID': 2,
            'Enable': 1,
            'CRC': 0
        }))
        data[7] = self.calc_crc8(data)

        try:
            self.bus.send(can.Message(
                arbitration_id=msg.frame_id,
                is_extended_id=False,
                data=data
            ))
            self.control_enabled = True
            await asyncio.sleep(0.02)
        except Exception as e:
            print(f"Failed to enable control: {e}")

    async def send_command(self, percentage):
        # Convert input percentage (-100 to 100) to value range (-1 to 1)
        target = percentage / 100.0
        current = self.last_steer_value

        # Calculate number of steps needed
        total_distance = abs(target - current)
        step_size = 0.05
        num_steps = max(int(total_distance / step_size), 1)  # Ensure at least one step

        # Calculate precise step size to reach target
        actual_step = (target - current) / num_steps

        # Always perform stepping, even if we're close to target
        for _ in range(num_steps):
            current += actual_step
            # Ensure we stay within the exact -1 to 1 bounds
            current = max(min(current, 1.0), -1.0)
            print(f"Step value: {current:.3f}, Target: {target:.3f}")
            await self.send_single_command(current)
            # Delay to allow system to stabilize
            await asyncio.sleep(0.08)

        # Send final target value to ensure precise positioning
        if abs(current - target) > 0.001:  # Handle any floating point imprecision
            await self.send_single_command(target)
        self.last_steer_value = target

    async def send_single_command(self, value):
        msg = self.db.get_message_by_name('ControlCommand')
        data = bytearray(msg.encode({
            'BusAddress': self.bus_address,
            'InterfaceID': 2,
            'Count8': self.control_count,
            'Value': value,  # Now using full -1 to 1 range
            'CRC': 0
        }))
        data[7] = self.calc_crc8(data)

        try:
            await self.enable_control()
            self.bus.send(can.Message(
                arbitration_id=msg.frame_id,
                is_extended_id=False,
                data=data
            ))
            print(f"Steering value: {value:.3f}")
            print(f"Encoded CAN data: {data.hex()}")
            self.control_count = (self.control_count + 1) % 256
            print(f"Set steer to value {value:.3f} (Input percentage: {value * 100:.1f}%)")
        except Exception as e:
            print(f"Failed to send command: {e}")

    async def maintain_control(self):
        while True:
            await self.enable_control()
            await asyncio.sleep(0.1)

    async def monitor_state(self):
        last_state = None
        while True:
            try:
                msg = self.bus.recv(timeout=0.1)
                if msg and msg.arbitration_id == self.db.get_message_by_name('Heartbeat').frame_id:
                    decoded = self.db.decode_message(msg.arbitration_id, msg.data)
                    state = (decoded.get('SubsystemID'), decoded.get('SystemState'))
                    if state != last_state:
                        print(f"System {state[0]} State: {state[1]}")
                        last_state = state
            except:
                pass
            await asyncio.sleep(0.02)


async def main():
    controller = Controller()

    monitor_task = asyncio.create_task(controller.monitor_state())
    control_task = asyncio.create_task(controller.maintain_control())

    def cleanup(sig=None, frame=None):
        print("\nStopping...")
        monitor_task.cancel()
        control_task.cancel()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)

    try:
        while True:
            try:
                user_input = float(input("\nEnter steering percentage (-100 to 100): "))
                if -100 <= user_input <= 100:
                    await controller.send_command(user_input)
                else:
                    print("Value must be between -100 and 100")
            except ValueError:
                print("Invalid input. Please enter a number.")
    except KeyboardInterrupt:
        cleanup()


if __name__ == "__main__":
    asyncio.run(main())
