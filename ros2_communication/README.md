# Differential Drive Wheel RPM Pipeline

A ROS 2-based system that processes velocity commands, computes wheel speeds, and visualizes results using shared memory and REST API communication.

## Overview

This project implements three programs that work together to:
1. Subscribe to ROS 2 `cmd_vel` messages
2. Compute differential-drive wheel RPMs
3. Exchange data via shared memory and REST API (no ROS middleware)
4. Visualize wheel RPMs in real-time

## Components

1. **`script_a.cpp`** - ROS 2 C++ node that subscribes to `/cmd_vel`, computes wheel RPMs, writes to shared memory
2. **`script_b.cpp`** - ROS-independent C++ HTTP server that reads shared memory and exposes REST API
3. **`script_c.py`** - Python client that polls REST API and plots wheel RPMs over time

## Robot Parameters

1. Wheel-to-wheel distance (track width):** 0.443 m
2. Wheel diameter:** 0.181 m (radius: 0.0905 m)

## Prerequisites

1. ROS 2 Humble (or compatible)
2. C++17 compiler
3. Python 3 with `matplotlib` and `requests`
4. POSIX environment (Linux)

## Build

```bash
cd /home/ros_master/mukus_robotics
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

## Quick Start

### 1. Start REST Server (Terminal 1)
```bash
./install/wheel_rpm_pipeline/lib/wheel_rpm_pipeline/script_b
```

### 2. Start ROS 2 Node (Terminal 2)
```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run wheel_rpm_pipeline script_a
```

### 3. Publish Velocity Commands (Terminal 3)

**Option A: Play Bag File (Recommended)**
```bash
source /opt/ros/humble/setup.bash
ros2 bag play bag/cmd_vel.db3
```

**Option B: Manual Publishing**
```bash
source /opt/ros/humble/setup.bash
ros2 topic pub -r 10 /cmd_vel geometry_msgs/msg/Twist \
  "{linear: {x: 0.2, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.1}}"
```

### 4. Run Plotter (Terminal 4)
```bash
python3 script_c.py
```

## API Documentation

### REST Endpoint

- **URL:** `http://localhost:8080/wheel_rpm`
- **Method:** `GET`
- **Response:** JSON
  ```json
  {
    "seq": 123,
    "stamp": 1765539366.123,
    "rpm_left": 18.77,
    "rpm_right": 23.44
  }
  ```

### Shared Memory

- **Name:** `/wheel_rpm_shm`
- **Structure:** See `shared/shared_defs.hpp`
  ```cpp
  pthread_mutex_t mutex;
  uint64_t seq;
  double stamp_sec;
  double rpm_left;
  double rpm_right;
  ```

## Testing

### Verify Components

**Check Script A is receiving messages:**
```bash
ros2 topic info /cmd_vel
# Should show: Subscription count: 1
```

**Test REST API:**
```bash
curl http://localhost:8080/wheel_rpm
```

**Check shared memory:**
```bash
ls -l /dev/shm/wheel_rpm_shm
```

### Expected Behavior

- **Script A:** Logs RPM values when `cmd_vel` messages arrive
- **Script B:** Serves HTTP requests on port 8080
- **Script C:** Displays real-time graph updating at 5 Hz





