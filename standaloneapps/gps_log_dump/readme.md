the default nvidia driveowrks application dumps dw_imuframes stdout instead of separate csv. this is a customized version to dump data in the csv file. separating logs with frame data.

./sample_imu_logger --driver=imu.novatel --params=file=/usr/local/imu_data.csv,connection-type=ethernet,ip=192.168.1.203,port=2000  --enable-dump-all=true
