#include "ros/ros.h"

using namespace std;

int main(int argc, char **argv) {
  cout << "Vision Buffer" << endl;

  ros::init(argc, argv, "vision_buffer");

  ros::spin();

  return 0;
}

