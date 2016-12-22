#include "ros/ros.h"
#include <opencv2/opencv.hpp>
#include <pcl/visualization/cloud_viewer.h>

#include "CommonTools.h"


using namespace std;
using namespace cv;

class CloudAnalyzer {
public:
  CloudAnalyzer() {
    m_outfile.open("train.dat", std::ios_base::app);
    m_step_number = 1;
  }

  void callback(const CloudConstPtr& cloud_const_ptr) {
    std::vector<float> m_fluent_vector;

    // Get the table normal, which is saved in the header
    string payload = cloud_const_ptr->header.frame_id;
    int vid_idx;
    float table_normal_x, table_normal_y, table_normal_z;
    sscanf(payload.c_str(), "%u %f %f %f", &vid_idx, &table_normal_x, &table_normal_y, &table_normal_z);
    PointT table_normal;
    table_normal.x = table_normal_x;
    table_normal.y = table_normal_y;
    table_normal.z = table_normal_z;
    if (m_vid_idx != vid_idx) {
      m_step_number = 1;
    }
    m_vid_idx = vid_idx;


    double min_z = std::numeric_limits<double>::infinity();
    double max_z = -std::numeric_limits<double>::infinity();
    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();;

    PointT min_z_point, max_z_point, min_x_point, max_x_point;
    for (int i = 0; i < cloud_const_ptr->size(); i++) {
      PointT p = cloud_const_ptr->at(i);
      if (p.z > max_z) {
        max_z = p.z;
        max_z_point = p;
      }
      if (p.z < min_z) {
        min_z = p.z;
        min_z_point = p;
      }
      if (p.x > max_x) {
        max_x = p.x;
        max_x_point = p;
      }
      if (p.x < min_x) {
        min_x = p.x;
        min_x_point = p;
      }
    }

    float width_of_cloth = pcl::euclideanDistance(min_x_point, max_x_point);
    float length_of_cloth = pcl::euclideanDistance(min_z_point, max_z_point);

    m_fluent_vector.push_back(width_of_cloth);
    m_fluent_vector.push_back(length_of_cloth);

    float dist = compute_fluent_dist(m_fluent_vector, m_prev_fluent_vector);
//    cout << "dist from prev fluent: " << dist << endl;
    if (dist > 0.1) {
      save_fluent_vector(m_fluent_vector);
      cout << width_of_cloth << " x " << length_of_cloth << endl;
      m_prev_fluent_vector = m_fluent_vector;
    }
  }

private:

  std::vector<float> m_prev_fluent_vector;
  std::ofstream m_outfile;
  int m_step_number;
  int m_vid_idx;

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

