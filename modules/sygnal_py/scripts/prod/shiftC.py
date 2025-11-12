import asyncio
import can
import cantools
import crc8
import sys


sys.path.append('/usr/local/lib/python3.10/dist-packages/gamepadcontroller')
from config import get_config

class MCM:
    def __init__(self, channel):
        # Load DBC files
        self.db = cantools.database.Database()
        try:
            print("Loading DBC files...")
            self.db.add_dbc_file('./sygnal_dbc/mcm/Control.dbc')
            self.db.add_dbc_file('./sygnal_dbc/vehicles/vehicle.dbc')
            print("DBC files loaded successfully.")
        except Exception as e:
            print(f"Error loading DBC files: {e}")
        
        self.bus = can.Bus(bustype='socketcan', channel=channel, bitrate=500000)
        self.bus_address_mcm = 1  # Bus address for braking (MCM)
        self.bus_address_cb1 = 3   # Bus address for gear shifting (CB-1)
        self.config = get_config(None)  # Load default config

        # Map values from the configuration to values compatible with the DBC file (0-7)
        self.gear_config = {
            'P': 0,    # Park
            'R': 1,    # Reverse
            'N': 2,    # Neutral
            'D': 3     # Drive
        }

    def calc_crc8(self, data):
        hash = crc8.crc8()
        hash.update(data[:-1])  # Exclude CRC byte itself
        return hash.digest()[0]

    async def enable_control(self, bus_address, interface):
        try:
            control_enable_msg = self.db.get_message_by_name('ControlEnable')

            data = bytearray(control_enable_msg.encode({
                'BusAddress': bus_address,
                'InterfaceID': interface,
                'Enable': 1,
                'CRC': 0
            }))
            data[-1] = self.calc_crc8(data)
            msg = can.Message(arbitration_id=control_enable_msg.frame_id, data=data, is_extended_id=False)
            print(f"Sending enable control message: {data.hex()}")
            self.bus.send(msg)
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error enabling control on bus address {bus_address} for interface {interface}: {e}")

    async def apply_brakes(self, brake_percentage, duration):
        try:
            # Enable braking control on MCM (Bus address 1, interface 0)
            await self.enable_control(self.bus_address_mcm, 0)

            # Apply brakes using brake percentage
            control_cmd_msg = self.db.get_message_by_name('ControlCommand')
            data = bytearray(control_cmd_msg.encode({
                'BusAddress': self.bus_address_mcm,
                'InterfaceID': 0,  # Brake interface
                'Count8': 0,
                'Value': brake_percentage,
                'CRC': 0
            }))
            data[-1] = self.calc_crc8(data)
            msg = can.Message(arbitration_id=control_cmd_msg.frame_id, data=data, is_extended_id=False)
            print(f"Sending brake command: {data.hex()}")
            self.bus.send(msg)
            await asyncio.sleep(duration)  # Apply brake for the specified duration

            # Release brakes by sending 0% brake command
            data = bytearray(control_cmd_msg.encode({
                'BusAddress': self.bus_address_mcm,
                'InterfaceID': 0,
                'Count8': 0,
                'Value': 0,  # Set brake percentage to 0 to release
                'CRC': 0
            }))
            data[-1] = self.calc_crc8(data)
            msg = can.Message(arbitration_id=control_cmd_msg.frame_id, data=data, is_extended_id=False)
            self.bus.send(msg)
            print("Brake released.")
        except Exception as e:
            print(f"Error applying brakes: {e}")

    async def change_gear(self, gear_option):
        try:
            # Check if the gear_option is valid
            gear_position = self.gear_config.get(gear_option)
            if gear_position is None:
                print(f"Invalid gear option: {gear_option}")
                return

            print(f"Gear position selected: {gear_position}")

            # Enable gear control on CB-1 (Bus address 3, interface 0)
            await self.enable_control(self.bus_address_cb1, 0)

            # Delay after enabling gear control
            await asyncio.sleep(0.5)  # Short delay to ensure the system registers control

            # Send the gear change command
            gear_cmd_msg = self.db.get_message_by_name('ELECT_GEAR')
            encoded_data = {
                'Elect_Gear_Shifter': gear_position
            }

            # Set default values for other signals in the message (if any)
            for signal in gear_cmd_msg.signals:
                if signal.name not in encoded_data:
                    encoded_data[signal.name] = 0  # Default value

            data = bytearray(gear_cmd_msg.encode(encoded_data))
            print(f"Sending gear update message for {gear_option}: {data.hex()}")
            msg = can.Message(arbitration_id=gear_cmd_msg.frame_id, data=data, is_extended_id=False)
            self.bus.send(msg)
            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error changing gear: {e}")

# Run the braking and gear shifting code
async def main():
    mcm = MCM(channel='can0')

    # User inputs for braking and gear shifting
    brake_percentage = float(input("Enter brake percentage (0.0 - 1.0): "))
    brake_duration = float(input("Enter brake application time period (in seconds): "))
    gear_option = input("Enter gear option (P for Park, R for Reverse, D for Drive): ").upper()

    # Apply brakes and shift gear concurrently within the brake duration
    await asyncio.gather(
        mcm.apply_brakes(brake_percentage, brake_duration),
        mcm.change_gear(gear_option)
    )

# Run the main function
asyncio.run(main())
