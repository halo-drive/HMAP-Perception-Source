### Overview

This is a “from zero” guide for teammates to:

- **Set up an Orin** with all libraries needed for Fast‑LIO + DriveWorks.
- **Export aarch64 libraries and headers from Orin** into tarballs.
- **Set up an x86 DriveWorks SDK container** for cross‑compiling `sample_dw_fastlio_slam` for Orin.
- **Deploy and run** the sample on Orin.
- Replace usernames/IPs as appropriate.

---

## 1. Prepare the Orin (target)

### 1.1. Basic packages

On Orin:

```bash
sudo apt-get update

# Build tools
sudo apt-get install -y \
  build-essential cmake git pkg-config python3 python3-pip python3-setuptools

# Core libs
sudo apt-get install -y \
  libprotobuf-dev protobuf-compiler \
  libnlopt-dev \
  libopencv-dev python3-opencv \
  libboost-system-dev libboost-filesystem-dev \
  libsuitesparse-dev \
  libeigen3-dev \
  libpcl-dev \
  libtbb-dev \
  libhdf5-serial-dev \
  libgdal-dev \
  libdc1394-22-dev \
  libtesseract-dev libleptonica-dev

# Optional tools
sudo apt-get install -y tree default-jdk
```

TensorRT headers/libs must match the SDK; normally they’re already installed via the DRIVE SDK. Verify (no need to change anything):

```bash
ls /usr/include/aarch64-linux-gnu/NvInfer*.h
ls /usr/lib/aarch64-linux-gnu/libnvinfer*.so*
```

If they exist, you’re good.

---

### 1.2. Build and install ZCM (if not available via apt)

On Orin:

```bash
cd ~
git clone https://github.com/ZeroCM/zcm.git
cd zcm

# Build and install
python3 ./waf configure --prefix=/usr
python3 ./waf
sudo python3 ./waf install
```

Verify:

```bash
ls /usr/include/zcm
ls /usr/lib/libzcm*.so*
```

---

### 1.3. Build and install NLOPT (if your apt version is too old)

If `/usr/include/nlopt.h` and `/usr/lib/aarch64-linux-gnu/libnlopt.so` already exist from `libnlopt-dev`, you can **skip** this.

Otherwise, use an older NLOPT tag compatible with CMake 3.16 (e.g. `v2.7.1`):

```bash
cd ~
git clone https://github.com/stevengj/nlopt.git
cd nlopt
git checkout v2.7.1

mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr ..
make -j$(nproc)
sudo make install
```

---

### 1.4. Build and install G2O with CHOLMOD

1. **Get g2o** on Orin:

```bash
cd ~
git clone https://github.com/RainerKuemmerle/g2o.git
cd g2o
```

2. **Add custom `FindCHOLMOD.cmake`** so g2o finds SuiteSparse / CHOLMOD. Create:

`~/g2o/cmake_modules/FindCHOLMOD.cmake` with something like:

```cmake
find_path(CHOLMOD_INCLUDE_DIR
  NAMES cholmod.h
  PATHS /usr/include /usr/include/suitesparse
)

find_library(CHOLMOD_LIBRARY
  NAMES cholmod
  PATHS /usr/lib/aarch64-linux-gnu /usr/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CHOLMOD
  REQUIRED_VARS CHOLMOD_INCLUDE_DIR CHOLMOD_LIBRARY
)

if(CHOLMOD_FOUND)
  add_library(SuiteSparse::CHOLMOD INTERFACE IMPORTED)
  set_target_properties(SuiteSparse::CHOLMOD PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CHOLMOD_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${CHOLMOD_LIBRARY}"
  )
endif()
```

3. **Build g2o**:

```bash
cd ~/g2o
mkdir build && cd build

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/usr/local \
  -DG2O_USE_CHOLMOD=ON \
  -DG2O_USE_CSPARSE=ON \
  -DG2O_BUILD_APPS=OFF \
  -DG2O_BUILD_EXAMPLES=OFF \
  -DG2O_BUILD_UNITTESTS=OFF \
  -DG2O_BUILD_SLAM3D_TYPES=ON \
  -DG2O_BUILD_SPARSE_FEATURE_VO=OFF \
  -DG2O_BUILD_STRUCTURE_ONLY=OFF \
  -DCMAKE_MODULE_PATH="${CMAKE_MODULE_PATH};${HOME}/g2o/cmake_modules" \
  ..

make -j$(nproc)
sudo make install
```

Verify:

```bash
ls /usr/local/include/g2o
ls /usr/local/lib/libg2o_*.so*
ls /usr/local/lib/cmake/g2o
```

---

### 1.5. Build and install `lidar-slam-detection` perception egg

Clone (or reuse the existing checkout) on Orin, then:

```bash
cd /usr/local/lidar-slam-detection   # adjust if different
sudo python3 setup.py install
```

This should install:

`/usr/local/lib/python3.8/dist-packages/perception-1.0.0-py3.8-linux-aarch64.egg/`

Check libs inside:

```bash
ls /usr/local/lib/python3.8/dist-packages/perception-1.0.0-py3.8-linux-aarch64.egg
# expect: libfast_lio.so, libg2o_backend.so, libmap_common.so, libfast_gicp.so, ...
```

---

### 1.6. Verify PCL, Boost, OpenCV on Orin

Just ensure dev packages are present:

```bash
ls /usr/include/pcl-1.10
ls /usr/lib/aarch64-linux-gnu/libpcl_common.so*
ls /usr/lib/aarch64-linux-gnu/libpcl_io.so*
ls /usr/lib/aarch64-linux-gnu/libpcl_filters.so*
ls /usr/lib/aarch64-linux-gnu/libpcl_search.so*
ls /usr/lib/aarch64-linux-gnu/libpcl_registration.so*
ls /usr/lib/aarch64-linux-gnu/libpcl_kdtree.so*
ls /usr/lib/aarch64-linux-gnu/libpcl_sample_consensus.so*

ls /usr/include/boost
ls /usr/lib/aarch64-linux-gnu/libboost_system.so*
ls /usr/lib/aarch64-linux-gnu/libboost_filesystem.so*

ls /usr/include/opencv4
ls /usr/lib/aarch64-linux-gnu/libopencv_core.so*
ls /usr/lib/aarch64-linux-gnu/cmake/opencv4
```

---

## 2. Create aarch64 tarballs on Orin

Create everything once on Orin; teammates can reuse the tarballs.

From **Orin** home dir (`~`):

```bash
cd ~
mkdir -p export-aarch64 && cd export-aarch64
```

### 2.1. Perception egg

```bash
tar -czf perception-aarch64-egg.tar.gz \
  -C /usr/local/lib/python3.8/dist-packages \
  perception-1.0.0-py3.8-linux-aarch64.egg
```

### 2.2. Boost (system, filesystem)

```bash
tar -czf boost-aarch64.tar.gz \
  -C /usr/lib/aarch64-linux-gnu \
  libboost_system.so* libboost_filesystem.so*
```

### 2.3. OpenCV

```bash
tar -czf opencv-aarch64.tar.gz \
  -C / \
  usr/include/opencv4 \
  usr/lib/aarch64-linux-gnu/libopencv*.so* \
  usr/lib/aarch64-linux-gnu/cmake/opencv4
```

### 2.4. G2O

```bash
tar -czf g2o-aarch64.tar.gz \
  -C / \
  usr/local/include/g2o \
  usr/local/lib/libg2o_*.so* \
  usr/local/lib/cmake/g2o
```

### 2.5. TensorRT

```bash
tar -czf tensorrt-aarch64.tar.gz \
  -C / \
  usr/include/aarch64-linux-gnu/NvInfer*.h \
  usr/lib/aarch64-linux-gnu/libnvinfer*.so* \
  usr/lib/aarch64-linux-gnu/libnvinfer_plugin*.so* \
  usr/lib/aarch64-linux-gnu/libnvonnxparser*.so* \
  usr/lib/aarch64-linux-gnu/stubs
```

### 2.6. PCL

```bash
tar -czf pcl-aarch64.tar.gz \
  -C / \
  usr/include/pcl-1.10 \
  usr/lib/aarch64-linux-gnu/libpcl_common.so* \
  usr/lib/aarch64-linux-gnu/libpcl_io.so* \
  usr/lib/aarch64-linux-gnu/libpcl_filters.so* \
  usr/lib/aarch64-linux-gnu/libpcl_search.so* \
  usr/lib/aarch64-linux-gnu/libpcl_registration.so* \
  usr/lib/aarch64-linux-gnu/libpcl_kdtree.so* \
  usr/lib/aarch64-linux-gnu/libpcl_sample_consensus.so*
```

### 2.7. Lidar‑slam‑detection headers

```bash
tar -czf lidar-slam-detection-headers.tar.gz \
  -C /usr/local \
  lidar-slam-detection
```

You should now have:

```bash
ls ~/export-aarch64
# boost-aarch64.tar.gz
# opencv-aarch64.tar.gz
# g2o-aarch64.tar.gz
# tensorrt-aarch64.tar.gz
# pcl-aarch64.tar.gz
# perception-aarch64-egg.tar.gz
# lidar-slam-detection-headers.tar.gz
```

---

## 3. Copy tarballs to x86 SDK container

From your **x86 host** (where you can reach Orin via SSH):

```bash
scp benchdev0@<ORIN_IP>:~/export-aarch64/*.tar.gz /usr/local/
```

(Adjust username/IP and destination path if needed.)

You should see them in the x86 container:

```bash
ls /usr/local/*.tar.gz
```

---

## 4. Prepare the x86 DriveWorks SDK container

### 4.1. Extract aarch64 tarballs into the cross sysroot

Inside the **x86 DriveWorks SDK container**:

```bash
cd /

sudo tar -xzf /usr/local/perception-aarch64-egg.tar.gz
sudo tar -xzf /usr/local/boost-aarch64.tar.gz
sudo tar -xzf /usr/local/opencv-aarch64.tar.gz
sudo tar -xzf /usr/local/g2o-aarch64.tar.gz
sudo tar -xzf /usr/local/tensorrt-aarch64.tar.gz
sudo tar -xzf /usr/local/pcl-aarch64.tar.gz
sudo tar -xzf /usr/local/lidar-slam-detection-headers.tar.gz
```

This will populate:

- `/usr/local/lib/python3.8/dist-packages/perception-1.0.0-py3.8-linux-aarch64.egg`
- `/usr/include/opencv4`, `/usr/lib/aarch64-linux-gnu/libopencv*.so*`, `/usr/lib/aarch64-linux-gnu/cmake/opencv4`
- `/usr/local/include/g2o`, `/usr/local/lib/libg2o_*.so*`, `/usr/local/lib/cmake/g2o`
- `/usr/include/pcl-1.10`, `/usr/lib/aarch64-linux-gnu/libpcl_*.so*`
- `/usr/include/aarch64-linux-gnu/NvInfer*.h`, `/usr/lib/aarch64-linux-gnu/libnvinfer*.so*`
- `/usr/local/lidar-slam-detection` headers
- `/usr/lib/aarch64-linux-gnu/libboost_system.so*`, `libboost_filesystem.so*`

Optionally, ensure Boost headers are visible under an aarch64-prefixed include dir (if your toolchain expects that):

```bash
sudo mkdir -p /usr/include/aarch64-linux-gnu/boost
sudo rsync -a /usr/include/boost/ /usr/include/aarch64-linux-gnu/boost/
```

---

### 4.2. Configure CMake for cross‑compile

In the SDK container, create a aarch64 build dir (you already use `/home/nvidia/build-aarch64-linux-gnu`):

```bash
mkdir -p /home/nvidia/build-aarch64-linux-gnu
cd /home/nvidia/build-aarch64-linux-gnu

cmake \
  -B /home/nvidia/build-aarch64-linux-gnu \
  -DCMAKE_TOOLCHAIN_FILE=/usr/local/driveworks/samples/cmake/Toolchain-V5L.cmake \
  -DBOOST_ROOT=/usr \
  -DBOOST_LIBRARYDIR=/usr/lib/aarch64-linux-gnu \
  -DOpenCV_DIR=/usr/lib/aarch64-linux-gnu/cmake/opencv4 \
  -S /usr/local/driveworks-5.20/samples
```

Notes:

- `dw_fastlio/CMakeLists.txt` is already customized to:
  - Add include dirs for PCL, Eigen, VTK, system headers.
  - Use **absolute PCL library paths** for aarch64.
  - Link TensorRT, G2O, Boost, OpenCV, perception libs.
  - Add `-Wl,--allow-shlib-undefined` for aarch64.

---

### 4.3. Build the Fast‑LIO sample only

Still in the build dir:

```bash
cd /home/nvidia/build-aarch64-linux-gnu
make sample_dw_fastlio_slam -j$(nproc)
```

If it succeeds, the cross‑compiled binary will be in the install tree indicated by the SDK CMake (usually under `install/usr/local/driveworks/samples/bin`).

Check:

```bash
ls /home/nvidia/build-aarch64-linux-gnu/install/usr/local/driveworks/samples/bin/sample_dw_fastlio_slam
```

---

## 5. Deploy binary to Orin and run

### 5.1. Copy binary to Orin

From x86 host:

```bash
scp /home/nvidia/build-aarch64-linux-gnu/install/usr/local/driveworks/samples/bin/sample_dw_fastlio_slam \
    benchdev0@<ORIN_IP>:/usr/local/driveworks/bin/
```

On Orin:

```bash
chmod +x /usr/local/driveworks/bin/sample_dw_fastlio_slam
```

### 5.2. Ensure Orin can find the perception libs

On Orin, add the egg directory to the dynamic linker config **once**:

```bash
echo "/usr/local/lib/python3.8/dist-packages/perception-1.0.0-py3.8-linux-aarch64.egg" | \
  sudo tee /etc/ld.so.conf.d/perception.conf

sudo ldconfig
```

(If you don’t want to touch `ld.so.conf`, you can instead set `LD_LIBRARY_PATH` each time you run.)

### 5.3. Run headless (no X11)

If you’re SSH’d into Orin with no display, use `--offscreen`:

```bash
cd /usr/local/driveworks/bin

sudo ./sample_dw_fastlio_slam \
  --offscreen \
  --rig-file=/usr/local/driveworks-5.20/samples/src/sensors/slam/dw_fastlio/rig.json \
  --voxel-size=0.2
```

To capture logs:

```bash
sudo ./sample_dw_fastlio_slam \
  --offscreen \
  --rig-file=/usr/local/driveworks-5.20/samples/src/sensors/slam/dw_fastlio/rig.json \
  --voxel-size=0.2 \
  2>&1 | grep "\[DWFastLIO\]" > /usr/local/slam_logs.txt
```

If you have a display attached and X11 working, you can omit `--offscreen` and run the standard command from `command.txt`.

---

If you’d like, I can turn this into a `README_fastlio_cross_compile.md` file in the repo with your exact usernames and IPs filled in.