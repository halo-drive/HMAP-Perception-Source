# Lidar recording visualization Tool [ ]

renders recordings from sample_lidar_replay [ individual lidars ] and stitches them.

## Purpose

This tool:
1. Loads sensor correction from rig file and applies them 
2. performs stitching
4. future work: dumps frame by frame lidar dumps for nvidia recorder application

## Output

The tool visualizes combined point clouds.


issues 
as data was recorded with some time delay, correct synchronizaton is under progress. one point cloud can be seen lagging behind.

TODOs:
 - dump frames from recordings by nvidia recorder.
 - generate bin files for stitched point clouds. [ update according to nvidia recorder format]
