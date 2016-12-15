#include "ros/ros.h"

using namespace std;

int main(int argc, char **argv) {
  cout << "Fluent Extractor" << endl;

  ros::init(argc, argv, "fluent_extractor");

  ros::spin();

  return 0;
}

