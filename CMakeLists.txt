cmake_minimum_required(VERSION 2.8.3)
project(fluent_extractor)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  genmsg
  cv_bridge
  pcl_ros
  message_generation
)


find_package(OpenCV REQUIRED)
find_package(Boost COMPONENTS filesystem system)
find_package(PCL 1.7 REQUIRED)

add_message_files(
        FILES
        ClothSegment.msg
        PgFragment.msg
)

generate_messages(DEPENDENCIES std_msgs sensor_msgs)

catkin_package(
        CATKIN_DEPENDS message_runtime
)


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
set(SOURCE_FILES ${SOURCE_FILES} src/FluentCalc.cpp src/FluentCalc.h)
set(SOURCE_FILES ${SOURCE_FILES} src/FoldSimulator.cpp src/FoldSimulator.h)

add_executable(vision_buffer src/vision_buffer.cpp ${SOURCE_FILES})
target_link_libraries(vision_buffer ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} boost_graph)
add_dependencies(vision_buffer ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

### Fluent Extractor (learning)
add_executable(fluent_extractor src/fluent_extractor.cpp ${SOURCE_FILES})
target_link_libraries(fluent_extractor ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} boost_graph)
add_dependencies(fluent_extractor ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

### Camera Pointer
add_executable(cam_ui src/camera_pointer.cpp ${SOURCE_FILES})
target_link_libraries(cam_ui ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} boost_graph)
add_dependencies(cam_ui ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

### Action Simulator (inference)
add_executable(action_inferer src/action_inferer.cpp ${SOURCE_FILES})
target_link_libraries(action_inferer ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} boost_graph)
add_dependencies(action_inferer ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

### Fluent Calculator Tester
add_executable(fluent_calc_tester src/fluent_calc_tester.cpp ${SOURCE_FILES})
target_link_libraries(fluent_calc_tester ${catkin_LIBRARIES} ${Boost_LIBRARIES} ${OpenCV_LIBS} ${PCL_LIBRARIES} boost_graph)
add_dependencies(fluent_calc_tester ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
