#include "ros/ros.h"
#include <opencv2/opencv.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <std_msgs/String.h>
#include "ros/package.h"
#include <fluent_extractor/ClothSegment.h>
#include <cv_bridge/cv_bridge.h>

#include "CommonTools.h"
#include "FluentCalc.h"


using namespace std;
using namespace cv;

class CloudAnalyzer {
public:
  CloudAnalyzer(ros::Publisher& pub, ros::Publisher& pub_ui) {
    m_pub = pub;
    m_pub_ui = pub_ui;
    m_outfile.open("train.dat", std::ios_base::app);
    m_step_number = 1;
    m_pcd_filename_idx = 1;
    m_compute_fold = false;
//    m_viz = new pcl::visualization::PCLVisualizer("CloudViewer");
  }

  void callback_hoi(const std_msgs::String::ConstPtr& msg) {
    std::string str = msg->data;
    float grip_x, grip_y, release_x, release_y;
    sscanf(str.c_str(), "%f,%f,%f,%f", &grip_x, &grip_y, &release_x, &release_y);
    m_grip.x = grip_x;
    m_grip.y = grip_y;
    m_release.x = release_x;
    m_release.y = release_y;
    m_compute_fold = true;
  }

  void callback_cloth_segment(fluent_extractor::ClothSegment cloth_segment) {
    cout << "in cloth_segment callback: " << cloth_segment.vid_idx << " " << cloth_segment.img_idx << endl;
    sensor_msgs::PointCloud2 ros_cloud = cloth_segment.cloud;
    Cloud cloud;
    pcl::fromROSMsg(ros_cloud, cloud);
    cout << "cloud size: " << cloud.size() << endl;
    sensor_msgs::Image ros_mask = cloth_segment.mask;
    sensor_msgs::Image ros_img = cloth_segment.img;
    Mat mask = cv_bridge::toCvCopy(ros_mask, "mono8")->image;
    Mat img = cv_bridge::toCvCopy(ros_img, "bgr8")->image;

    int* pixel2voxel = cloth_segment.pixel2voxel.data();
    imshow("fluent_extractor mask", mask);
    imshow("fluent_extractor img", img);
    waitKey(20);
    CloudPtr cloud_ptr = CloudPtr(new Cloud(cloud));
    CloudPtr cloth_cloud_ptr = CommonTools::get_pointcloud_from_mask(cloud_ptr, pixel2voxel, mask);

    PointT table_normal;
    table_normal.x = cloth_segment.table_normal[0];
    table_normal.y = cloth_segment.table_normal[1];
    table_normal.z = cloth_segment.table_normal[2];
    PointT table_midpoint;
    table_midpoint.x = cloth_segment.table_midpoint[0];
    table_midpoint.y = cloth_segment.table_midpoint[1];
    table_midpoint.z = cloth_segment.table_midpoint[2];

    int vid_idx = cloth_segment.vid_idx;
    int img_idx = cloth_segment.img_idx;

    compute_fluents(cloth_cloud_ptr, table_normal, table_midpoint, vid_idx, img_idx);

    if (m_prev_mask.size().area() > 0) {
        // compare mask with m_prev_mask
        Mat debug_mask_img = Mat::zeros(mask.size(), CV_8UC3);
//        debug_mask_img = CommonTools::draw_mask(debug_mask_img, m_prev_mask, Scalar(0, 255, 0));
        debug_mask_img = CommonTools::draw_mask(debug_mask_img, mask, Scalar(255, 0, 0));
        Mat child_moved_mask = m_prev_mask - mask;
        vector<cv::Point> pts;
        CommonTools::max_contour(child_moved_mask, pts, child_moved_mask);
        CommonTools::erode_dilate(child_moved_mask, 2);
        debug_mask_img = CommonTools::draw_mask(debug_mask_img, child_moved_mask, Scalar(0, 0, 255));
        imshow("debug_mask_img", debug_mask_img);
        waitKey(20);

        // [cloth_cloud_ptr] pcl of current cloth
        // pcl of moved prev cloth-part (from prev_mask - mask)
        // pcl of stationary prev cloth-part (from mask)
        CloudPtr cloth_moved_cloud = CommonTools::get_pointcloud_from_mask(m_prev_cloud_ptr, m_prev_pixel2voxel, child_moved_mask);
        Mat child_remained_mask = m_prev_mask - child_moved_mask;
        CommonTools::erode_dilate(child_remained_mask, 2);
        CloudPtr cloth_remained_cloud = CommonTools::get_pointcloud_from_mask(m_prev_cloud_ptr, m_prev_pixel2voxel, child_remained_mask);
        cout << "child cloud (moved) size: " << cloth_moved_cloud->size() << endl;
        cout << "child cloud (remained) size: " << cloth_remained_cloud->size() << endl;
        cout << "total current cloth cloud size: " << cloth_cloud_ptr->size() << endl;

        stringstream pcd_moved_filename;
        pcd_moved_filename << "cloth_moved_" << vid_idx << "_" << img_idx << ".pcd";
        pcl::io::savePCDFile(pcd_moved_filename.str(), *cloth_moved_cloud);
        stringstream pcd_remained_filename;
        pcd_remained_filename << "cloth_remained_" << vid_idx << "_" << img_idx << ".pcd";
        pcl::io::savePCDFile(pcd_remained_filename.str(), *cloth_remained_cloud);
    }

    m_prev_mask = mask.clone();
    m_prev_cloud_ptr = CloudPtr(cloud_ptr);
    delete m_prev_pixel2voxel;
    m_prev_pixel2voxel = new int[img.size().area()];
    std::memcpy(m_prev_pixel2voxel, pixel2voxel, sizeof(int) * img.size().area());
  }

  void compute_fluents(CloudConstPtr cloth_cloud, PointT table_normal, PointT table_midpoint, int vid_idx, int img_idx) {
      std::vector<float> fluent_vector;

      if (m_vid_idx != vid_idx) {
          m_step_number = 1;
      }
      m_vid_idx = vid_idx;

      Eigen::Matrix4f transform = CommonTools::get_projection_transform(cloth_cloud->makeShared());
      CloudPtr aligned_cloud = CommonTools::transform3d(cloth_cloud->makeShared(), transform);

      float x_min, y_min, z_min;
      float scale_x, scale_y, scale_z;
      Mat debug_img;
      Rect outer_bbox;
//    vector<float> bbox_fluents = m_fluent_calc.calc_inner_outer_bbox(aligned_cloud->makeShared(), debug_img,
//                                                                     x_min, y_min, z_min, scale_x, scale_y, scale_z, outer_bbox);
//    fluent_vector.insert(fluent_vector.end(), bbox_fluents.begin(), bbox_fluents.end());

      if (m_compute_fold) {
          m_grip.x = m_grip.x * outer_bbox.height;
          m_grip.y = m_grip.y * outer_bbox.height;
          m_release.x = m_release.x * outer_bbox.height;
          m_release.y = m_release.y * outer_bbox.height;

          cout << "\t\t" << "GRIP" << m_grip << " --> RELEASE " << m_release << endl;

          // draw these grip/release circle on debug_img
          int debug_grip_col = m_grip.x + outer_bbox.x;
          int debug_grip_row = m_grip.y + outer_bbox.y;
          int debug_release_col = m_release.x + outer_bbox.x;
          int debug_release_row = m_release.y + outer_bbox.y;
          Mat debug_img_col;
          cvtColor(debug_img, debug_img_col, CV_GRAY2BGR);
          circle(debug_img_col, cv::Point(debug_grip_col, debug_grip_row), 10, cv::Scalar(0, 0, 255), -1);
          circle(debug_img_col, cv::Point(debug_release_col, debug_release_row), 10, cv::Scalar(255, 0, 0), -1);
          imshow("debug points", debug_img_col);
          waitKey(40);

          float x3d_grip = (m_grip.x * scale_y) / debug_img.cols + y_min;
          float y3d_grip = (m_grip.y * scale_z) / debug_img.rows + z_min;
          float x3d_release = (m_release.x * scale_y) / debug_img.cols + y_min;
          float y3d_release = (m_release.y * scale_z) / debug_img.rows + z_min;

          Point2f target_release_point(x3d_release, y3d_release);
          Point2f target_grip_point(x3d_grip, y3d_grip);
          float min_dist_release = std::numeric_limits<float>::infinity();
          float min_dist_grip = std::numeric_limits<float>::infinity();
          PointT closest_3d_release;
          PointT closest_3d_grip;
          for (int i = 0; i < aligned_cloud->size(); i++) {
              PointT p3d = aligned_cloud->at(i);
              Point2f p3d_proj(p3d.x, p3d.y);
              float dist_release = cv::norm(p3d_proj - target_release_point);
              float dist_grip = cv::norm(p3d_proj - target_grip_point);
              if (dist_release < min_dist_release) {
                  min_dist_release = dist_release;
                  closest_3d_release.x = p3d.x;
                  closest_3d_release.y = p3d.y;
                  closest_3d_release.z = p3d.z;
              }
              if (dist_grip < min_dist_grip) {
                  min_dist_grip = dist_grip;
                  closest_3d_grip.x = p3d.x;
                  closest_3d_grip.y = p3d.y;
                  closest_3d_grip.z = p3d.z;
              }
          }

          cout << "FOUND 3D point in projected cloud: " << closest_3d_grip << " --> " << closest_3d_release << endl;
          CloudPtr release_cloud(new pcl::PointCloud<PointT>());
          release_cloud->push_back(closest_3d_release);
          CloudPtr release_cloud_orig = CommonTools::transform3d(release_cloud, transform.inverse());
          PointT true_release = release_cloud_orig->at(0);

          CloudPtr grip_cloud(new pcl::PointCloud<PointT>());
          grip_cloud->push_back(closest_3d_grip);
          CloudPtr grip_cloud_orig = CommonTools::transform3d(grip_cloud, transform.inverse());
          PointT true_grip = grip_cloud_orig->at(0);

          cout << "FOUND 3D point in original cloud: " << true_grip << " --> " << true_release << endl;
          stringstream pub_ui_msg;
          pub_ui_msg << true_grip.x << ","
                     << true_grip.y << ","
                     << true_grip.z << ","
                     << true_release.x << ","
                     << true_release.y << ","
                     << true_release.z << endl;
          std_msgs::String msg;
          msg.data = pub_ui_msg.str();
          m_pub_ui.publish(msg);

          m_compute_fold = false;
      }

      Mat aligned_mask = m_fluent_calc.get_mask_from_aligned_cloud(aligned_cloud->makeShared());

      // Compute thickness fluents
      vector<float> thickness_fluents = m_fluent_calc.calc_thickness(cloth_cloud->makeShared(), table_normal, table_midpoint);
      fluent_vector.insert(fluent_vector.end(), thickness_fluents.begin(), thickness_fluents.end());

      // Compute wrinkle fluents
      vector<float> wrinkle_fluents = m_fluent_calc.calc_wrinkles(cloth_cloud->makeShared(), table_normal, table_midpoint);
      fluent_vector.insert(fluent_vector.end(), wrinkle_fluents.begin(), wrinkle_fluents.end());

      // Compute bounding-box fluents
      vector<float> bbox3d_fluents = m_fluent_calc.calc_bbox(aligned_cloud->makeShared());
      fluent_vector.insert(fluent_vector.end(), bbox3d_fluents.begin(), bbox3d_fluents.end());

      // Compute symmetry fluents
      vector<float> symmetry_fluents = m_fluent_calc.calc_principal_symmetries(aligned_cloud->makeShared());
      fluent_vector.insert(fluent_vector.end(), symmetry_fluents.begin(), symmetry_fluents.end());

      // Compute moment flunets
      vector<float> moment_fluents = m_fluent_calc.calc_hu_moments(aligned_mask);
      fluent_vector.insert(fluent_vector.end(), moment_fluents.begin(), moment_fluents.end());

      // Compute squareness fluent
      vector<float> squareness_fluents = m_fluent_calc.calc_squareness(aligned_mask);
      fluent_vector.insert(fluent_vector.end(), squareness_fluents.begin(), squareness_fluents.end());

      float dist = compute_fluent_dist(fluent_vector, m_prev_fluent_vector);
      cout << "dist from prev fluent: " << dist << endl;
      publish_fluent_vector(fluent_vector);
      print_fluent_vector(fluent_vector);

      if (true || dist > 1) {
          cout << "STATE DETECTED: " << m_pcd_filename_idx << endl;
          // SAVE  state_img, debug_img, fluent vector

//      Mat state_img = CommonTools::get_image_from_cloud(aligned_cloud, "yz");
//      save_fluent_data(state_img, debug_img, fluent_vector);

          save_fluent_vector(fluent_vector, img_idx);
          print_fluent_vector(fluent_vector);

          m_prev_fluent_vector = fluent_vector;
      }
      m_pcd_filename_idx++;
  }

  void callback(const CloudConstPtr& cloud_const_ptr) {
    cout << "cloud size: " << cloud_const_ptr->size() << endl;

    if (cloud_const_ptr->size() < 100) {
      return;
    }

    // Get the table normal and vid_idx, which is saved in the header
    string payload = cloud_const_ptr->header.frame_id;
    int vid_idx, img_idx;
    float table_normal_x, table_normal_y, table_normal_z;
    float table_midpoint_x, table_midpoint_y, table_midpoint_z;
    sscanf(payload.c_str(), "%d %d %f %f %f %f %f %f", &vid_idx, &img_idx,
         &table_normal_x, &table_normal_y, &table_normal_z,
         &table_midpoint_x, &table_midpoint_y, &table_midpoint_z);
    PointT table_normal, table_midpoint;
    table_normal.x = table_normal_x;
    table_normal.y = table_normal_y;
    table_normal.z = table_normal_z;
    table_midpoint.x = table_midpoint_x;
    table_midpoint.y = table_midpoint_y;
    table_midpoint.z = table_midpoint_z;

    compute_fluents(cloud_const_ptr, table_normal, table_midpoint, vid_idx, img_idx);
  }

private:
  FluentCalc m_fluent_calc;
  std::vector<float> m_prev_fluent_vector;
  std::ofstream m_outfile;
  int m_step_number;
  int m_vid_idx;
  int m_pcd_filename_idx;
  ros::Publisher m_pub;
  ros::Publisher m_pub_ui;
  pcl::visualization::PCLVisualizer *m_viz;
  cv::Point2f m_grip;
  cv::Point2f m_release;
  bool m_compute_fold;
  Mat m_prev_mask;
  CloudPtr m_prev_cloud_ptr;
  int* m_prev_pixel2voxel;
  
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

  void save_fluent_data(cv::Mat state_img, cv::Mat debug_img, std::vector<float> fluent_vector) {
    String path = ros::package::getPath("fluent_extractor") + "/fluents/";
    long int timestamp = CommonTools::unix_timestamp();

    stringstream state_img_filename;
    state_img_filename << path << timestamp << "_state.png";
    imwrite(state_img_filename.str(), state_img);

    stringstream debug_img_filename;
    debug_img_filename << path << timestamp << "_debug.png";
    imwrite(debug_img_filename.str(), debug_img);

    stringstream fluent_mat_filename;
    fluent_mat_filename << path << timestamp << "_fluents";
    cv::Mat fluent_mat = Mat::zeros(fluent_vector.size(), 1, CV_32F);
    for (int i = 0; i < fluent_vector.size(); i++) {
      fluent_mat.at<float>(i) = fluent_vector[i];
    }
    FileStorage fs(fluent_mat_filename.str(), FileStorage::WRITE);
    fs << "fluents" << fluent_mat;
    fs.release();
  }

  void save_fluent_vector(std::vector<float> fluent_vector, int step_number) {
    if (step_number == -1) {
      step_number = m_step_number;
      m_step_number++;
    }
    m_outfile << step_number << " qid:" << m_vid_idx << " ";
    for (int i = 0; i < fluent_vector.size(); i++) {
      m_outfile << (i+1) << ":" << fluent_vector[i];
      if (i < fluent_vector.size() - 1) m_outfile << " ";
    }
    m_outfile << endl;
    m_outfile.flush();
  }

  void publish_fluent_vector(std::vector<float> fluent_vector) {
    stringstream fluents_str;
    for (int i = 0; i < fluent_vector.size(); i++) {
      fluents_str << (i + 1) << ":" << fluent_vector[i] << " ";
    }
    std_msgs::String msg;
    msg.data = fluents_str.str();
    m_pub.publish(msg);
  }
};



int main(int argc, char **argv) {
  ros::init(argc, argv, "fluent_extractor");

  ros::NodeHandle node_handle;
  ros::Publisher pub = node_handle.advertise<std_msgs::String>("/vcla/cloth_folding/fluent_vector", 1);
  ros::Publisher pub_ui = node_handle.advertise<std_msgs::String>("/vcla/cloth_folding/grip_release", 1);

  CloudAnalyzer cloud_analyzer(pub, pub_ui);

  ros::Subscriber sub = node_handle.subscribe<Cloud>("/vcla/cloth_folding/vision_buffer_pcl", 1, &CloudAnalyzer::callback, &cloud_analyzer);

  ros::Subscriber sub2 = node_handle.subscribe("/vcla/cloth_folding/hoi_action", 1, &CloudAnalyzer::callback_hoi, &cloud_analyzer);

  ros::Subscriber sub_cloth_segment = node_handle.subscribe("/vcla/cloth_folding/cloth_segment", 1, &CloudAnalyzer::callback_cloth_segment, &cloud_analyzer);


  // outfile << step_number++ << " qid:" << json["vid_idx"].GetInt() << " 1:" << cloth_feature[0] << " 2:" << cloth_feature[1] << endl;


  ros::spin();

  return 0;
}

