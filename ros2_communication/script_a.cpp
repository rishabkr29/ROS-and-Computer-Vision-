#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>

#include "shared/shared_defs.hpp"

namespace
{
using shared_memory::SharedWheelRpm;

// Robot parameters (meters)
constexpr double kTrackWidth = 0.443;   // wheel-to-wheel distance
constexpr double kWheelDiameter = 0.181;
constexpr double kWheelRadius = kWheelDiameter / 2.0;

constexpr double kTwoPi = 2.0 * M_PI;

double linear_to_rpm(double linear_mps)
{
  // Convert linear velocity at wheel circumference to signed RPM.
  const double rad_per_sec = linear_mps / kWheelRadius;
  return (rad_per_sec * 60.0) / kTwoPi;
}

class ShmWriter
{
public:
  ShmWriter()
  {
    fd_ = shm_open(shared_memory::kShmName, O_CREAT | O_RDWR, 0666);
    if (fd_ < 0)
    {
      throw std::runtime_error("Failed to open shared memory");
    }
    if (ftruncate(fd_, static_cast<off_t>(shared_memory::shm_size())) != 0)
    {
      throw std::runtime_error("Failed to size shared memory");
    }

    void *addr = mmap(nullptr, shared_memory::shm_size(), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (addr == MAP_FAILED)
    {
      throw std::runtime_error("Failed to mmap shared memory");
    }

    data_ = static_cast<SharedWheelRpm *>(addr);
    initialize_if_needed();
  }

  ~ShmWriter()
  {
    if (data_)
    {
      munmap(data_, shared_memory::shm_size());
    }
    if (fd_ >= 0)
    {
      close(fd_);
    }
  }

  void write(uint64_t seq, double stamp_sec, double rpm_left, double rpm_right)
  {
    pthread_mutex_lock(&data_->mutex);
    data_->seq = seq;
    data_->stamp_sec = stamp_sec;
    data_->rpm_left = rpm_left;
    data_->rpm_right = rpm_right;
    pthread_mutex_unlock(&data_->mutex);
  }

private:
  void initialize_if_needed()
  {
    // Attempt to detect uninitialized mutex by checking pthread magic via trylock.
    pthread_mutexattr_t attr;
    pthread_mutexattr_init(&attr);
    pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);

    // Reinitialize mutex unconditionally to be safe.
    pthread_mutex_init(&data_->mutex, &attr);
    data_->seq = 0;
    data_->stamp_sec = 0.0;
    data_->rpm_left = 0.0;
    data_->rpm_right = 0.0;

    pthread_mutexattr_destroy(&attr);
  }

  int fd_{-1};
  SharedWheelRpm *data_{nullptr};
};

class RpmNode : public rclcpp::Node
{
public:
  explicit RpmNode() : Node("wheel_rpm_writer"), writer_()
  {
    sub_ = this->create_subscription<geometry_msgs::msg::Twist>(
        "cmd_vel", 10, std::bind(&RpmNode::twistCallback, this, std::placeholders::_1));
  }

private:
  void twistCallback(const geometry_msgs::msg::Twist::SharedPtr msg)
  {
    const double v = msg->linear.x;
    const double w = msg->angular.z;

    const double v_left = v - (w * kTrackWidth * 0.5);
    const double v_right = v + (w * kTrackWidth * 0.5);

    const double rpm_left = linear_to_rpm(v_left);
    const double rpm_right = linear_to_rpm(v_right);

    const double stamp = this->now().seconds();
    const uint64_t seq = ++seq_;

    writer_.write(seq, stamp, rpm_left, rpm_right);

    RCLCPP_INFO_THROTTLE(this->get_logger(), *this->get_clock(), 1000,
                        "seq=%lu v=%.3f w=%.3f rpm_left=%.2f rpm_right=%.2f",
                        static_cast<unsigned long>(seq), v, w, rpm_left, rpm_right);
  }

  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr sub_;
  ShmWriter writer_;
  uint64_t seq_{0};
};

} // namespace

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);

  try
  {
    auto node = std::make_shared<RpmNode>();
    rclcpp::spin(node);
  }
  catch (const std::exception &e)
  {
    RCLCPP_ERROR(rclcpp::get_logger("wheel_rpm_writer"), "Fatal error: %s", e.what());
    rclcpp::shutdown();
    return 1;
  }

  rclcpp::shutdown();
  return 0;
}


