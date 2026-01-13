cmake_minimum_required(VERSION 3.10)
project(wheel_rpm_pipeline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(Threads REQUIRED)

add_executable(script_a script_a.cpp)
target_include_directories(script_a PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
ament_target_dependencies(script_a rclcpp geometry_msgs)
target_link_libraries(script_a Threads::Threads)

add_executable(script_b script_b.cpp)
target_include_directories(script_b PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(script_b Threads::Threads)

add_executable(test_cmd_vel_publisher test_cmd_vel_publisher.cpp)
target_include_directories(test_cmd_vel_publisher PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
ament_target_dependencies(test_cmd_vel_publisher rclcpp geometry_msgs)

install(TARGETS script_a script_b test_cmd_vel_publisher
  DESTINATION lib/${PROJECT_NAME})

ament_package()


