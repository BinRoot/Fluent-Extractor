#include "ros/ros.h"
#include <opencv2/opencv.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>

#include "CommonTools.h"
#include "FluentCalc.h"


using namespace std;
using namespace cv;

class CloudAnalyzer {
public:
  CloudAnalyzer() {
    m_outfile.open("train.dat", std::ios_base::app);
    m_step_number = 1;
    m_pcd_filename_idx = 1;
  }

  void callback(const CloudConstPtr& cloud_const_ptr) {
    
    std::vector<float> fluent_vector;

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
    if (m_vid_idx != vid_idx) {
      m_step_number = 1;
    }
    m_vid_idx = vid_idx;

    // Compute width and height fluents
    vector<float> width_height_fluents = m_fluent_calc.calc_width_and_height(cloud_const_ptr->makeShared(), table_normal);
    fluent_vector.insert(fluent_vector.end(), width_height_fluents.begin(), width_height_fluents.end());

    // Compute thickness fluents
    vector<float> thickness_fluents = m_fluent_calc.calc_thickness(cloud_const_ptr->makeShared(), table_normal, table_midpoint);
    fluent_vector.insert(fluent_vector.end(), thickness_fluents.begin(), thickness_fluents.end());


    float dist = compute_fluent_dist(fluent_vector, m_prev_fluent_vector);
//    cout << "dist from prev fluent: " << dist << endl;
    if (dist > 0.05) {
      cout << "STATE DETECTED: " << m_pcd_filename_idx << endl;
      //-- save pcd
      stringstream pcd_filename;
      pcd_filename << "out_" << m_step_number << ".pcd";
      pcl::io::savePCDFile(pcd_filename.str(), *(cloud_const_ptr->makeShared()));
      //--

      save_fluent_vector(fluent_vector);
      print_fluent_vector(fluent_vector);

      m_prev_fluent_vector = fluent_vector;
    }
    m_pcd_filename_idx++;
  }

private:
  FluentCalc m_fluent_calc;
  std::vector<float> m_prev_fluent_vector;
  std::ofstream m_outfile;
  int m_step_number;
  int m_vid_idx;
  int m_pcd_filename_idx;
  
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

  void save_fluent_vector(std::vector<float> fluent_vector) {
    m_outfile << m_step_number++ << " qid:" << m_vid_idx << " ";
    for (int i = 0; i < fluent_vector.size(); i++) {
      m_outfile << (i+1) << ":" << fluent_vector[i];
      if (i < fluent_vector.size() - 1) m_outfile << " ";
    }
    m_outfile << endl;
    m_outfile.flush();
  }
};



int main(int argc, char **argv) {
  ros::init(argc, argv, "fluent_extractor");

  CloudAnalyzer cloud_analyzer;

  ros::NodeHandle node_handle;
  ros::Subscriber sub = node_handle.subscribe<Cloud>("vision_buffer_pcl", 1, &CloudAnalyzer::callback, &cloud_analyzer);


  // outfile << step_number++ << " qid:" << json["vid_idx"].GetInt() << " 1:" << cloth_feature[0] << " 2:" << cloth_feature[1] << endl;


  ros::spin();

  return 0;
}

