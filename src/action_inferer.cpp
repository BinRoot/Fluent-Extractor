#include "ros/ros.h"
#include <opencv2/opencv.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/io/pcd_io.h>


#include "CommonTools.h"
#include "FluentCalc.h"
#include "FoldSimulator.h"


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


    Mat cloud_mask;
    get_cloud_mask(cloud_const_ptr, cloud_mask);
    imshow("cloud_mask", cloud_mask);
    waitKey(30);


    // fold the cloth
    infer_best_fold(cloud_mask);


    vector<float> fluent_vector;
    
    // Compute width and height fluents
    vector<float> width_height_fluents = m_fluent_calc.calc_width_and_height(cloud_const_ptr->makeShared(), table_normal);
    fluent_vector.insert(fluent_vector.end(), width_height_fluents.begin(), width_height_fluents.end());

    // Compute thickness fluents
    vector<float> thickness_fluents = m_fluent_calc.calc_thickness(cloud_const_ptr->makeShared(), table_normal, table_midpoint);
    fluent_vector.insert(fluent_vector.end(), thickness_fluents.begin(), thickness_fluents.end());
    
    print_fluent_vector(fluent_vector);

//    ros::shutdown();
  }

private:
  FluentCalc m_fluent_calc;

  void infer_best_fold(Mat& cloth_mask) {
    // find grip candidates
    // find release candidates for each grip candidate
    // compute score
    // select grip and release points of max score

    FoldSimulator simulator(cloth_mask);
    simulator.run_gui();

  }


  void get_cloud_mask(const CloudConstPtr& cloud_const_ptr, Mat& cloud_mask) {
    // collect 2d points
    vector<cv::Point2f> points;
    float x_min = std::numeric_limits<float>::infinity();
    float y_min = std::numeric_limits<float>::infinity();
    float x_max = 0;
    float y_max = 0;


    for (int i = 0; i < cloud_const_ptr->size(); i++) {
      PointT p = cloud_const_ptr->at(i);
      if (p.x < x_min) x_min = p.x;
      if (p.x > x_max) x_max = p.x;
      if (p.y < y_min) y_min = p.y;
      if (p.y > y_max) y_max = p.y;

      cv::Point2f p2;
      p2.x = p.x;
      p2.y = p.y;
      points.push_back(p2);
    }

    float scale_x = x_max - x_min;
    float scale_y = y_max - y_min;
    cout << "scale x: " << scale_x << ", scale_y: " << scale_y << endl;

    float mask_width = 256;
    float mask_height = mask_width * (scale_y / scale_x);
    cout << "mask width: " << mask_width << ", mask_height: " << mask_height << endl;

    Mat cloud_mask_raw = Mat::zeros(mask_height, mask_width, CV_8U);
    cloud_mask = Mat::zeros(mask_height, mask_width, CV_8U);
    for (int i = 0; i < points.size(); i++) {
      int x = mask_width * (points[i].x - x_min) / scale_x;
      clip_to_bounds(x, 0, int(mask_width) - 1);
      int y = mask_height * (points[i].y - y_min) / scale_y;
      clip_to_bounds(y, 0, int(mask_height) - 1);
      cloud_mask_raw.at<uchar>(y, x) = 255;
    }

    CommonTools::dilate_erode(cloud_mask_raw, 1);
    CommonTools::dilate_erode(cloud_mask_raw, 2);
    CommonTools::dilate_erode(cloud_mask_raw, 3);

    CommonTools::draw_contour(cloud_mask, cloud_mask_raw.clone(), cv::Scalar(255));
  }

  void clip_to_bounds(int& x, int min_val, int max_val) {
    if (x < min_val) {
      x = min_val;
    } else if (x > max_val) {
      x = max_val;
    }
  }

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

