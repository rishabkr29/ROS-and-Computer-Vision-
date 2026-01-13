#pragma once

#include <cstdint>
#include <cstddef>
#include <pthread.h>

namespace shared_memory
{
constexpr const char *kShmName = "/wheel_rpm_shm";

struct SharedWheelRpm
{
  pthread_mutex_t mutex;
  uint64_t seq;
  double stamp_sec;
  double rpm_left;
  double rpm_right;
};

inline std::size_t shm_size()
{
  return sizeof(SharedWheelRpm);
}
} // namespace shared_memory



