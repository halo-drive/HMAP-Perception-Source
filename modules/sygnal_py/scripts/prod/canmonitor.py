# can_monitor.py
import asyncio
import can
import cantools
import signal
import sys
from datetime import datetime

class CANMonitor:
    def __init__(self, channel='can0'):
        self.db = cantools.database.Database()
        try:
            self.db.add_dbc_file('./sygnal_dbc/mcm/Heartbeat.dbc')
            self.bus = can.Bus(channel=channel, bustype='socketcan', bitrate=500000)
            print(f"Connected to {channel}")
        except Exception as e:
            print(f"Failed to initialize: {e}")
            sys.exit(1)

    def decode_message(self, msg):
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        try:
            if msg.arbitration_id == self.db.get_message_by_name('Heartbeat').frame_id:
                decoded = self.db.decode_message(msg.arbitration_id, msg.data)
                return (f"[{timestamp}] System: {decoded.get('SubsystemID')} "
                       f"State: {decoded.get('SystemState', 0)}")
        except Exception as e:
            return None

    async def monitor(self):
        print("Monitoring system states... Press Ctrl+C to exit")
        print("Timestamp          System  State")
        print("-" * 40)

        while True:
            try:
                msg = await asyncio.get_event_loop().run_in_executor(None, self.bus.recv)
                if msg:
                    decoded = self.decode_message(msg)
                    if decoded:
                        print(decoded)
            except asyncio.CancelledError:
                break
            except Exception as e:
                await asyncio.sleep(0.1)

    def cleanup(self):
        if hasattr(self, 'bus'):
            self.bus.shutdown()

async def main():
    monitor = CANMonitor()
    signal.signal(signal.SIGINT, lambda sig, frame: (monitor.cleanup(), sys.exit(0)))
    try:
        await monitor.monitor()
    finally:
        monitor.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
