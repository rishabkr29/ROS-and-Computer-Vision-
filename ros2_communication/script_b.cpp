#include <arpa/inet.h>
#include <cerrno>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <netinet/in.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>

#include "shared/shared_defs.hpp"

namespace
{
using shared_memory::SharedWheelRpm;

class ShmReader
{
public:
  ShmReader()
  {
    // Wait for shared memory to be created by script_a
    const int max_retries = 50;  // 5 seconds total (50 * 100ms)
    const int retry_delay_ms = 100;
    
    for (int i = 0; i < max_retries; ++i)
    {
      fd_ = shm_open(shared_memory::kShmName, O_RDWR, 0666);
      if (fd_ >= 0)
      {
        break;  // Successfully opened
      }
      
      if (i == 0)
      {
        std::cout << "Waiting for shared memory to be created by script_a..." << std::endl;
      }
      
      std::this_thread::sleep_for(std::chrono::milliseconds(retry_delay_ms));
    }
    
    if (fd_ < 0)
    {
      throw std::runtime_error("Shared memory not found after waiting. Start script_a first.");
    }
    
    void *addr = mmap(nullptr, shared_memory::shm_size(), PROT_READ | PROT_WRITE, MAP_SHARED, fd_, 0);
    if (addr == MAP_FAILED)
    {
      throw std::runtime_error("Failed to mmap shared memory");
    }
    data_ = static_cast<SharedWheelRpm *>(addr);
    std::cout << "Connected to shared memory successfully." << std::endl;
  }

  ~ShmReader()
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

  SharedWheelRpm snapshot()
  {
    SharedWheelRpm copy;
    pthread_mutex_lock(&data_->mutex);
    copy = *data_;
    pthread_mutex_unlock(&data_->mutex);
    return copy;
  }

private:
  int fd_{-1};
  SharedWheelRpm *data_{nullptr};
};

std::string build_http_response(const SharedWheelRpm &data)
{
  std::ostringstream body;
  body << "{"
       << "\"seq\":" << data.seq << ","
       << "\"stamp\":" << data.stamp_sec << ","
       << "\"rpm_left\":" << data.rpm_left << ","
       << "\"rpm_right\":" << data.rpm_right
       << "}";
  const std::string body_str = body.str();

  std::ostringstream resp;
  resp << "HTTP/1.1 200 OK\r\n"
       << "Content-Type: application/json\r\n"
       << "Content-Length: " << body_str.size() << "\r\n"
       << "Connection: close\r\n"
       << "\r\n"
       << body_str;
  return resp.str();
}

bool is_get_wheel_rpm(const std::string &req)
{
  // Very small parser: expects "GET /wheel_rpm"
  return req.rfind("GET /wheel_rpm", 0) == 0;
}

class HttpServer
{
public:
  HttpServer(std::string host, uint16_t port, ShmReader &reader)
      : host_(std::move(host)), port_(port), reader_(reader) {}

  void run()
  {
    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0)
    {
      throw std::runtime_error("Failed to create socket");
    }

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port_);
    addr.sin_addr.s_addr = inet_addr(host_.c_str());

    if (bind(server_fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0)
    {
      throw std::runtime_error("Bind failed: ensure host/port are free");
    }

    if (listen(server_fd, 10) < 0)
    {
      throw std::runtime_error("Listen failed");
    }

    std::cout << "HTTP server listening on " << host_ << ":" << port_ << std::endl;

    while (true)
    {
      sockaddr_in client{};
      socklen_t len = sizeof(client);
      int client_fd = accept(server_fd, reinterpret_cast<sockaddr *>(&client), &len);
      if (client_fd < 0)
      {
        if (errno == EINTR)
        {
          continue;
        }
        std::perror("accept");
        break;
      }
      handle_client(client_fd);
      close(client_fd);
    }

    close(server_fd);
  }

private:
  void handle_client(int client_fd)
  {
    char buf[1024];
    const ssize_t n = recv(client_fd, buf, sizeof(buf) - 1, 0);
    if (n <= 0)
    {
      return;
    }
    buf[n] = '\0';
    const std::string req(buf);

    if (!is_get_wheel_rpm(req))
    {
      const std::string resp = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
      send(client_fd, resp.c_str(), resp.size(), 0);
      return;
    }

    const auto data = reader_.snapshot();
    const std::string resp = build_http_response(data);
    send(client_fd, resp.c_str(), resp.size(), 0);
  }

  std::string host_;
  uint16_t port_;
  ShmReader &reader_;
};

volatile std::sig_atomic_t stop_flag = 0;

void signal_handler(int)
{
  stop_flag = 1;
}

} // namespace

int main()
{
  try
  {
    std::signal(SIGINT, signal_handler);
    ShmReader reader;
    HttpServer server("0.0.0.0", 8080, reader);
    server.run();
  }
  catch (const std::exception &e)
  {
    std::cerr << "Fatal error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}



