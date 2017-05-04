#include "ros/ros.h"
#include <opencv2/opencv.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>
#include <std_msgs/String.h>
#include "ros/package.h"
#include <fluent_extractor/ClothSegment.h>
#include <cv_bridge/cv_bridge.h>
#include <fluent_extractor/PgFragment.h>

#include "CommonTools.h"
#include "FluentCalc.h"


using namespace std;
using namespace cv;

const float ICON_SCALE = 0.1;

class CloudAnalyzer {
public:
  CloudAnalyzer(ros::Publisher& pub, ros::Publisher& pub_pg, ros::Publisher& pub_ui) {
    m_pub = pub;
    m_pub_pg = pub_pg;
    m_pub_ui = pub_ui;
    m_outfile.open("train.dat", std::ios_base::app);
    m_vid_idx = -1;
    m_step_number = 1;
    m_pcd_filename_idx = 1;
    m_compute_fold = false;
    m_prev_pixel2voxel = NULL;
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

    // if video_idx changed, then restart the step_number counter
    if (m_vid_idx != vid_idx) {
      m_step_number = 1;
      m_prev_mask = Mat();
      if (m_vid_idx >= 0) {
          fluent_extractor::PgFragment pg_fragment;
          stringstream pg_name;
          pg_name << "pg_" << m_vid_idx;
          pg_fragment.name = pg_name.str();
          m_pub_pg.publish(pg_fragment);
      }
    }
    m_vid_idx = vid_idx;

    vector<float> fluent_vector = compute_fluents(cloth_cloud_ptr, table_normal, table_midpoint);

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
      m_prev_fluent_vector = fluent_vector;
    }
    m_pcd_filename_idx++;


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
        vector<float> pcd_moved_fluent_vector = compute_fluents(cloth_moved_cloud, table_normal, table_midpoint);

        stringstream pcd_remained_filename;
        pcd_remained_filename << "cloth_remained_" << vid_idx << "_" << img_idx << ".pcd";
        pcl::io::savePCDFile(pcd_remained_filename.str(), *cloth_remained_cloud);
        vector<float> pcd_remained_fluent_vector = compute_fluents(cloth_remained_cloud, table_normal, table_midpoint);

        fluent_extractor::PgFragment pg_fragment;
        stringstream pg_name;
        pg_name << "pg_" << m_vid_idx;
        pg_fragment.name = pg_name.str();

        if (m_prev_pcd_remained_fluent_vector.size() == 0) {
            pg_fragment.a = fluent_vector;
            Mat prev_mask_small; cv::resize(m_prev_mask, prev_mask_small, Size(), ICON_SCALE, ICON_SCALE);
            sensor_msgs::ImagePtr ros_img_a = cv_bridge::CvImage(std_msgs::Header(), "mono8", prev_mask_small).toImageMsg();
            pg_fragment.a_icon = *ros_img_a;
        } else {
            pg_fragment.a = m_prev_pcd_remained_fluent_vector;
        }

        pg_fragment.b = pcd_moved_fluent_vector;
        pg_fragment.c = pcd_remained_fluent_vector;

        Mat mask_b = m_fluent_calc.get_mask_from_aligned_cloud(cloth_moved_cloud->makeShared());
        Mat mask_c = m_fluent_calc.get_mask_from_aligned_cloud(cloth_remained_cloud->makeShared());

        Mat child_moved_mask_small; resize(child_moved_mask, child_moved_mask_small, Size(), ICON_SCALE, ICON_SCALE);
        Mat child_remained_mask_small; resize(child_remained_mask, child_remained_mask_small, Size(), ICON_SCALE, ICON_SCALE);
        sensor_msgs::ImagePtr ros_img_b = cv_bridge::CvImage(std_msgs::Header(), "mono8", child_moved_mask_small).toImageMsg();
        sensor_msgs::ImagePtr ros_img_c = cv_bridge::CvImage(std_msgs::Header(), "mono8", child_remained_mask_small).toImageMsg();

        pg_fragment.b_icon = *ros_img_b;
        pg_fragment.c_icon = *ros_img_c;

        m_pub_pg.publish(pg_fragment);

        m_prev_pcd_remained_fluent_vector = pcd_remained_fluent_vector;
    }

    m_prev_mask = mask.clone();
    m_prev_cloud_ptr = CloudPtr(cloud_ptr);

      if (!m_prev_pixel2voxel) {
          cout << "allocating space for pixel2voxel" << endl;
          m_prev_pixel2voxel = new int[img.size().area()];
      }
      std::memcpy(m_prev_pixel2voxel, pixel2voxel, sizeof(int) * img.size().area());
  }

  vector<float> compute_fluents(CloudConstPtr cloth_cloud, PointT table_normal, PointT table_midpoint) {
      std::vector<float> fluent_vector;

      Eigen::Matrix4f transform = CommonTools::get_projection_transform(cloth_cloud->makeShared());
      CloudPtr aligned_cloud = CommonTools::transform3d(cloth_cloud->makeShared(), transform);

      float x_min, y_min, z_min;
      float scale_x, scale_y, scale_z;
      Mat debug_img;
      Rect outer_bbox;
//    vector<float> bbox_fluents = m_fluent_calc.calc_inner_outer_bbox(aligned_cloud->makeShared(), debug_img,
//                                                                     x_min, y_min, z_min, scale_x, scale_y, scale_z, outer_bbox);
//    fluent_vector.insert(fluent_vector.end(), bbox_fluents.begin(), bbox_fluents.end());

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

      return fluent_vector;
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
  ros::Publisher m_pub_pg;
  pcl::visualization::PCLVisualizer *m_viz;
  cv::Point2f m_grip;
  cv::Point2f m_release;
  bool m_compute_fold;
  Mat m_prev_mask;
  CloudPtr m_prev_cloud_ptr;
  int* m_prev_pixel2voxel;
  vector<float> m_prev_pcd_remained_fluent_vector;
  
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
  ros::Publisher pub = node_handle.advertise<std_msgs::String>("/vcla/cloth_folding/fluent_vector", 1000);
  ros::Publisher pub_ui = node_handle.advertise<std_msgs::String>("/vcla/cloth_folding/grip_release", 1000);
  ros::Publisher pub_pg = node_handle.advertise<fluent_extractor::PgFragment>("/aog_engine/pg_fragment", 1000);

  CloudAnalyzer cloud_analyzer(pub, pub_pg, pub_ui);
  ros::Subscriber sub2 = node_handle.subscribe("/vcla/cloth_folding/hoi_action", 1000, &CloudAnalyzer::callback_hoi, &cloud_analyzer);
  ros::Subscriber sub_cloth_segment = node_handle.subscribe("/vcla/cloth_folding/cloth_segment", 1000, &CloudAnalyzer::callback_cloth_segment, &cloud_analyzer);

  ros::spin();

  return 0;
}

