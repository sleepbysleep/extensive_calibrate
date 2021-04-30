cmake_minimum_required(VERSION 3.0)

project(extensive_calibrate)
set(CMAKE_CXX_STANDARD 11)

set(Boost_USE_STATIC_LIBS ON)
#add_definitions(-DBOOST_LOG_DYN_LINK)
find_package(Boost REQUIRED system log log_setup)
find_package(Threads)
find_package(OpenCV REQUIRED)

include_directories(
  ${Boost_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}  
  )

add_executable(extensive_calibrate
  main.cpp
  )

target_link_libraries(extensive_calibrate
  ${Boost_LIBRARIES}
  ${Boost_LOG_LIBRARY}
  ${Boost_LOG_SETUP_LIBRARY}
  ${CMAKE_THREAD_LIBS_INIT}
  ${OpenCV_LIBS}
  )

add_executable(imagelist_creator
  imagelist_creator.cpp
  )

target_link_libraries(imagelist_creator
  ${Boost_LIBRARIES}
  ${Boost_LOG_LIBRARY}
  ${Boost_LOG_SETUP_LIBRARY}
  ${CMAKE_THREAD_LIBS_INIT}
  ${OpenCV_LIBS}
  )
