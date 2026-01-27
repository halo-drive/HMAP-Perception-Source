download drivers for vlp 16 hr x86 or aarch from here: https://drive.google.com/drive/folders/10MwLDHvEQ9OHVYcmvruuNwIJ63O5UmGe?usp=drive_link


# 1) Copy the patched library
sudo cp /path/to/libdw_sensors.so.5.20 /usr/local/driverworks/lib/libdw_sensors.so.5.20

# 2) Fix ownership and permissions (root, 755)
sudo chown root:root /usr/local/driverworks/lib/libdw_sensors.so.5.20
sudo chmod 755 /usr/local/driveworks/lib/libdw_sensors.so.5.20

# 3) Recreate symlinks in the lib folder 
sudo ln -sf libdw_sensors.so.5.20 libdw_sensors.so.5
sudo ln -sf libdw_sensors.so.5 libdw_sensors.so

# 4) Refresh the dynamic linker cache
sudo ldconfig

# 5) To verify, run sample lidar replay on live data on your platform (x86 or aarch)

./sample_lidar_replay --protocol=lidar.socket --params=device=VELO_VLP16,ip=192.168.1.201,port=2368,scan-frequency=10 --show-intensity=true

./sample_lidar_replay --protocol=lidar.socket --params=device=VELO_VLP16HR,ip=192.168.1.202,port=2369,scan-frequency=10 --show-intensity=true