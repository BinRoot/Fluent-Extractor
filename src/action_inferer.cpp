#include "ros/ros.h"
#include <opencv2/opencv.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/io/pcd_io.h>


#include "CommonTools.h"
#include "FluentCalc.h"


using namespace std;
using namespace cv;

class CloudAnalyzer {
public:
  CloudAnalyzer() {
  }

  void callback(const CloudConstPtr& cloud_const_ptr) {
    // Get the table normal and vid_idx, which is saved in the header
    string payload = cloud_const_ptr->header.frame_id;
    int vid_idx;
    float table_normal_x, table_normal_y, table_normal_z;
    float table_midpoint_x, table_midpoint_y, table_midpoint_z;
    sscanf(payload.c_str(), "%u %f %f %f %f %f %f", &vid_idx,
           &table_normal_x, &table_normal_y, &table_normal_z,
           &table_midpoint_x, &table_midpoint_y, &table_midpoint_z);
    PointT table_normal, table_midpoint;
    table_normal.x = table_normal_x;
    table_normal.y = table_normal_y;
    table_normal.z = table_normal_z;
    table_midpoint.x = table_midpoint_x;
    table_midpoint.y = table_midpoint_y;
    table_midpoint.z = table_midpoint_z;


    vector<float> fluent_vector;
    
    // Compute width and height fluents
    vector<float> width_height_fluents = m_fluent_calc.calc_width_and_height(cloud_const_ptr->makeShared(), table_normal);
    fluent_vector.insert(fluent_vector.end(), width_height_fluents.begin(), width_height_fluents.end());

    // Compute thickness fluents
    vector<float> thickness_fluents = m_fluent_calc.calc_thickness(cloud_const_ptr->makeShared(), table_normal, table_midpoint);
    fluent_vector.insert(fluent_vector.end(), thickness_fluents.begin(), thickness_fluents.end());
    
    print_fluent_vector(fluent_vector);

    ros::shutdown();
  }

private:
  FluentCalc m_fluent_calc;
  
  void print_fluent_vector(std::vector<float> fluent_vector) {
    for (int i = 0; i < fluent_vector.size(); i++) {
      cout << fluent_vector[i] << "    ";
    }
    cout << endl;
  }

  float compute_fluent_dist(std::vector<float> f1, std::vector<float> f2) {
    if (f1.size() != f2.size()) {
      return std::numeric_limits<float>::infinity();
    }
    float sum_of_squares = 0;
    for (int i = 0; i < f1.size(); i++) {
      sum_of_squares += pow(f1[i] - f2[i], 2.0f);
    }
    return pow(sum_of_squares, 0.5f);
  }

};



int main(int argc, char **argv) {
  ros::init(argc, argv, "action_inferer");
  
  ros::NodeHandle node_handle;
  CloudAnalyzer cloud_analyzer;
  ros::Subscriber sub = node_handle.subscribe<Cloud>("vision_buffer_pcl", 1, &CloudAnalyzer::callback, &cloud_analyzer);
  
  ros::spin();
  return 0;
}

