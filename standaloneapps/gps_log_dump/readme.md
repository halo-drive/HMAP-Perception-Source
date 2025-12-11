the default nvidia driveowrks application dumps dw_imuframes in stdout instead of separate csv. this is a customized version to dump data in the csv file, thus separating logs from frame data.

./sample_imu_logger --driver=imu.novatel --params=file=/usr/local/imu_data.csv,connection-type=ethernet,ip=192.168.1.203,port=2000  --enable-dump-all=true

downloadable binary:
https://drive.google.com/drive/folders/16zdHh_V0FWUmlCNRHh-0TX0WWNMh2q5J?usp=drive_link