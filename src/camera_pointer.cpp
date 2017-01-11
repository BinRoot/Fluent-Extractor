#include "ros/ros.h"
#include "std_msgs/String.h"
#include <iostream>
#include <sstream>
#include <opencv2/opencv.hpp>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/point_cloud_image_extractors.h>
#include <pcl/io/pcd_io.h>
#include "CommonTools.h"
#include <pcl/common/transformation_from_correspondences.h>
#include <pcl/common/transforms.h>
#include <iostream>
#include <fstream>

#include <rapidjson/document.h>
#include "rapidjson/filereadstream.h"

#include <tf/transform_datatypes.h>

#include <pcl/common/transformation_from_correspondences.h>
#include <pcl/common/transforms.h>


using namespace std;
using namespace cv;
using namespace rapidjson;


class CameraPointer {
public:
  void grip_release_cb(const std_msgs::String::ConstPtr& msg) {
    std::string str = msg->data;
    float grip_x, grip_y, grip_z, release_x, release_y, release_z;
    sscanf(str.c_str(), "%f,%f,%f,%f,%f,%f", &grip_x, &grip_y, &grip_z, &release_x, &release_y, &release_z);
    m_use_3d_points = true;
    m_grip_3d.x = grip_x;
    m_grip_3d.y = grip_y;
    m_grip_3d.z = grip_z;
    m_release_3d.x = release_x;
    m_release_3d.y = release_y;
    m_release_3d.z = release_z;
  }

  void cloud_cb (const pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud_ptr_) {
    if (m_cloud_mutex.try_lock()) {
      pcl::PointCloud<pcl::PointXYZRGB> cloud;
      cloud = *cloud_ptr_;
      CloudPtr cloud_ptr = cloud.makeShared();

      int pixel2voxel[cloud_ptr->width * cloud_ptr->height];
      int voxel2pixel[cloud_ptr->width * cloud_ptr->height];
      CommonTools::get_cloud_info(*cloud_ptr, m_img, pixel2voxel, voxel2pixel);

      if (m_use_3d_points) {
        int closest_grip_id;
        int closest_release_id;
        float min_grip_dist = std::numeric_limits<float>::infinity();
        float min_release_dist = std::numeric_limits<float>::infinity();
        for (int i = 0; i < cloud_ptr->size(); i++) {
          PointT p = cloud_ptr->at(i);

          float dist_to_grip = pcl::euclideanDistance(p, m_grip_3d);
          if (dist_to_grip < min_grip_dist) {
            min_grip_dist = dist_to_grip;
            closest_grip_id = i;
          }
          float dist_to_release = pcl::euclideanDistance(p, m_release_3d);
          if (dist_to_release < min_release_dist) {
            min_release_dist = dist_to_release;
            closest_release_id = i;
          }
        }


        int grip_row, grip_col;
        int release_row, release_col;
        CommonTools::vox2pix(voxel2pixel, closest_grip_id, grip_row, grip_col, cloud_ptr->width);
        CommonTools::vox2pix(voxel2pixel, closest_release_id, release_row, release_col, cloud_ptr->width);

        m_grip_point.x = grip_col;
        m_grip_point.y = grip_row;
        m_release_point.x = release_col;
        m_release_point.y = release_row;

        Mat cloth_mask = imread("src/fluent_extractor/cloth.png", -1);

        Mat contour_mask = Mat::zeros(cloth_mask.size(), CV_8U);
        CommonTools::draw_contour(contour_mask, cloth_mask.clone(), Scalar(255), 1);

        CommonTools::attract_grip_point1(contour_mask, m_grip_point);

        Point2i center_point;
        CommonTools::center_of_mask(cloth_mask, center_point);

        if (cv::norm(m_grip_point - m_release_point) < cv::norm(m_grip_point - center_point)) {
          m_release_point.x = center_point.x;
          m_release_point.y = center_point.y;
        } else {
          m_release_point += (m_release_point - m_grip_point);
          CommonTools::attract_grip_point1(contour_mask, m_release_point);
        }

        circle(contour_mask, center_point, 10, Scalar(255), 4);

        imshow("cam_ui cloth mask", contour_mask);

        m_use_3d_points = false;
      }

      circle(m_img, m_grip_point, 10, Scalar(0, 0, 255), 4);
      circle(m_img, m_release_point, 10, Scalar(255, 0, 0), 4);
      imshow("Cam UI", m_img);
      char c = waitKey(50);

      if (m_do_fold) {
        PointT grip_3d = CommonTools::get_3d_approx(m_grip_point, m_img.size(), pixel2voxel, cloud_ptr);
        PointT release_3d = CommonTools::get_3d_approx(m_release_point, m_img.size(), pixel2voxel, cloud_ptr);

        cout << "grip: " << m_grip_point << " --> " << grip_3d << endl;
        cout << "release: " << m_release_point << " --> " << release_3d << endl;

        grip_3d = pcl::transformPoint(grip_3d, m_trans);
        release_3d = pcl::transformPoint(release_3d, m_trans);

//        tf::Vector3 obj_local_grip_pos(grip_3d.x, grip_3d.y, grip_3d.z);
//        tf::Vector3 baxter_grip_pos = m_kinect2_ex_tf.inverse() * obj_local_grip_pos;
//        tf::Vector3 obj_local_release_pos(grip_3d.x, grip_3d.y, grip_3d.z);
//        tf::Vector3 baxter_release_pos = m_kinect2_ex_tf.inverse() * obj_local_release_pos;

        std_msgs::String msg;
        msg.data = "1," +
            to_string(grip_3d.x) + "," +
            to_string(grip_3d.y) + "," +
            to_string(grip_3d.z) + "," +
            to_string(release_3d.x) + "," +
            to_string(release_3d.y) + "," +
            to_string(release_3d.z);
        m_publisher.publish(msg);
        ros::spinOnce();
        m_do_fold = false;
      }


      m_cloud_mutex.unlock();
    }
  }

  void fold() {
    m_do_fold = true;
  }

  void set_grip_point(cv::Point2i grip_point) {
    m_grip_point = grip_point;
  }

  void set_release_point(cv::Point2i release_point) {
    m_release_point = release_point;
  }

  static void mouse_callback(int event, int x, int y, int flags, void* param) {
    if  ( event == EVENT_LBUTTONDOWN ) {
      CameraPointer* camera_info = (CameraPointer*)(param);
      camera_info->set_grip_point(Point(x, y));
    }
    else if  ( event == EVENT_RBUTTONDOWN ) {
      cout << "right click" << endl;
      CameraPointer* camera_info = (CameraPointer*)(param);
      camera_info->set_release_point(Point2i(x, y));
    }
    else if  ( event == EVENT_MBUTTONDOWN ) {
      cout << "middle click" << endl;
      CameraPointer* camera_info = (CameraPointer*)(param);
      camera_info->fold();
    }
  }

  CameraPointer(string kinect_topic) {
    m_use_3d_points = false;
    m_subscriber = m_n.subscribe(m_n.resolveName(kinect_topic), 1, &CameraPointer::cloud_cb, this);
    m_subscriber2 = m_n.subscribe("/vcla/cloth_folding/grip_release", 1000, &CameraPointer::grip_release_cb, this);
    m_publisher = m_n.advertise<std_msgs::String>("/vcla/cloth_folding/action", 1000);

    namedWindow("Camera Live Stream", CV_WINDOW_AUTOSIZE);
    setMouseCallback("Camera Live Stream", &CameraPointer::mouse_callback, this);

    tf::Quaternion kinect2_ex_q(-0.007, -0.460, -0.041, 0.887);
    tf::Vector3 kinect2_ex_t(0.665, 0.134, -0.695);
    tf::Quaternion q0(0, -0.707, 0, 0.707);
    tf::Quaternion q1(0, 0, 0.707, 0.707);
    tf::Transform kinect2_ex(kinect2_ex_q, kinect2_ex_t);
    tf::Transform t0(q0, tf::Vector3(0, 0, 0));
    tf::Transform t1(q1, tf::Vector3(0, 0, 0));
    m_kinect2_ex_tf = t1 * t0 * kinect2_ex;

    Eigen::Vector3f cam_1(0.224488,0.302983,0.628);
    Eigen::Vector3f cam_2(0.0887969,0.303677,0.633);
    Eigen::Vector3f cam_3(0.225015,0.178694,0.777);
    Eigen::Vector3f cam_4(0.0876768,0.180338,0.775);
    Eigen::Vector3f bax_1(0.5, -0.3, 0.045);
    Eigen::Vector3f bax_2(0.5, -0.15, 0.045);
    Eigen::Vector3f bax_3(0.7, -0.3, 0.045);
    Eigen::Vector3f bax_4(0.7, -0.15, 0.045);
    m_correspondences.add(cam_1, bax_1, 1);
    m_correspondences.add(cam_2, bax_2, 1);
    m_correspondences.add(cam_3, bax_3, 1);
    m_correspondences.add(cam_4, bax_4, 1);
    m_trans = m_correspondences.getTransformation();
    ofstream myfile;
    myfile.open("cam_bax_trans.txt");
    for (int i = 0; i < 16; i++) {
      cout << m_trans.data()[i] << endl;
      myfile << m_trans.data()[i] << endl;
    }
    myfile.close();
  }

private:
  ros::NodeHandle m_n;
  ros::Publisher m_publisher;
  ros::Subscriber m_subscriber;
  ros::Subscriber m_subscriber2;
  boost::mutex m_cloud_mutex;
  Mat m_img;
  pcl::TransformationFromCorrespondences m_correspondences;
  Eigen::Affine3f m_trans;
  bool m_do_fold = false;
  cv::Point2i m_grip_point;
  cv::Point2i m_release_point;
  PointT m_grip_3d;
  PointT m_release_3d;
  bool m_use_3d_points;

  tf::Transform m_kinect2_ex_tf;
};

int main(int argc, char **argv) {
  ros::init(argc, argv, "camera_pointer");

  cout << "in main of camera_pointer" << endl;

  // Read the config file
  FILE* fp = fopen(argv[1], "rb");
  char readBuffer[65536];
  FileReadStream is(fp, readBuffer, sizeof(readBuffer));
  Document json;
  json.ParseStream(is);

  CameraPointer camera_tool(json["kinect_topic"].GetString());

  ros::spin();

  return 0;
}