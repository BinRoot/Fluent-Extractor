cmake_minimum_required(VERSION 2.8.3)
project(fluent_extractor)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  genmsg
  pcl_ros
)

find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS filesystem system)
find_package(PCL 1.7 REQUIRED)

generate_messages(DEPENDENCIES std_msgs)

catkin_package()

include_directories(include ${catkin_INCLUDE_DIRS})

include_directories(include)
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})

link_directories(${Boost_LIBRARY_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})
set(CMAKE_CXX_FLAGS "-std=c++11 -O1")

### Vision Buffer
set(SOURCE_FILES ${SOURCE_FILES} src/Seg2D.cpp src/Seg2D.h)
set(SOURCE_FILES ${SOURCE_FILES} src/FileFrameScanner.cpp src/FileFrameScanner.h)
set(SOURCE_FILES ${SOURCE_FILES} src/CommonTools.h)

add_executable(vision_buffer src/vision_buffer.cpp ${SOURCE_FILES})
target_link_libraries(vision_buffer ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} mgl boost_graph)

### Fluent Extractor
add_executable(fluent_extractor src/fluent_extractor.cpp ${SOURCE_FILES})
target_link_libraries(fluent_extractor ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} mgl boost_graph)

### Camera Pointer
add_executable(cam_ui src/camera_pointer.cpp ${SOURCE_FILES})
target_link_libraries(cam_ui ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} mgl boost_graph)
