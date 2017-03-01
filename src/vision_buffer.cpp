#include <opencv2/opencv.hpp>
#include <sstream>
#include <pcl/visualization/cloud_viewer.h>

#include "ros/ros.h"
#include "ros/package.h"

#include "FileFrameScanner.h"
#include "CommonTools.h"

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <iostream>
#include "rapidjson/filereadstream.h"
#include "Seg2D.h"

#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>

#include <cstdio>
#include "pcl_ros/point_cloud.h"

using namespace std;
using namespace cv;
using namespace rapidjson;


class BufferManager {
public:
  BufferManager(int vid_idx, ros::Publisher& pub, ros::Publisher& xdisplay_pub) {
    m_vid_idx = vid_idx;
    m_xdisplay_pub = xdisplay_pub;
    m_pub = pub;
  }

  void process(CloudPtr cloud_ptr, Mat img_bgr, int* pixel2voxel, int* voxel2pixel, int img_idx=-1) {
    if (m_table_mask.size().area() == 0) {
      CommonTools::find_biggest_plane(cloud_ptr->makeShared(), voxel2pixel, img_bgr.size(), m_table_mask, m_table_midpoint, m_table_normal);
      cout << "table midpoint is " << m_table_midpoint << endl;
      cout << "table normal is " << m_table_normal << endl;
      imshow("table", m_table_mask);
    }

    Mat cloth_mask;
    if (m_table_mask.size().area() > 0) {
      CloudPtr table_cloud_ptr = CommonTools::get_pointcloud_from_mask(cloud_ptr->makeShared(), pixel2voxel, m_table_mask, true);

      // no obstacles allowed
      double max_dist_from_table = 0;
      int max_dist_vox_idx = -1;
      for (int i = 0; i < table_cloud_ptr->size(); i++) {
        PointT p = table_cloud_ptr->at(i);
        double d = -m_table_normal.x * m_table_midpoint.x -
            m_table_normal.y * m_table_midpoint.y -
            m_table_normal.z * m_table_midpoint.z;
        double dist_from_table = m_table_normal.x * p.x + m_table_normal.y * p.y + m_table_normal.z * p.z + d;
        if (dist_from_table > 50) continue;  // unreasonable
        if (dist_from_table > max_dist_from_table) {
          max_dist_from_table = dist_from_table;
          max_dist_vox_idx = i;
        }
      }
      if (max_dist_from_table > 0.15) {
        cout << "detected obstacle at distance " << max_dist_from_table << endl;
        int obstacle_row, obstacle_col;
        CommonTools::vox2pix(voxel2pixel, max_dist_vox_idx, obstacle_row, obstacle_col, cloud_ptr->width);
        Mat img_with_circle = img_bgr.clone();
        Point p;
        p.x = obstacle_col;
        p.y = obstacle_row;
        circle(img_with_circle, p, 10, Scalar(0, 0, 255), 3);
        imshow("Obstacle detected", img_with_circle);
        resize(img_with_circle, img_with_circle, cv::Size(), 1.5, 1.5);
        Mat xdisplay_img = Mat::zeros(800, 1024, CV_8UC3);

        img_with_circle.copyTo(xdisplay_img(cv::Rect(128, 0, img_with_circle.cols, img_with_circle.rows)));

        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", xdisplay_img).toImageMsg();
        m_xdisplay_pub.publish(*msg);
        waitKey(100);
        return;
      } else {
        cout << "no obstacle" << endl;
      }


      Mat img_masked;
      img_bgr.copyTo(img_masked, m_table_mask);
      Mat img_seg = m_seg2D.seg(img_masked, 0.5, 5000, 100);


      imshow("img_seg", img_seg);
      vector<vector<cv::Point2i> > components;
      CommonTools::get_components(img_seg, components);


      Mat max_component_img;
      int max_component_size = 0;
      Mat table_mask_eroded = m_table_mask.clone();
      CommonTools::erode(table_mask_eroded, 15);


      int table_2d_area = cv::countNonZero(m_table_mask);

      for (int i = 0; i < components.size(); i++) {
//          cout << "component " << i << " / " << components.size() << endl;

        // don't allow small components
//          cout << "don't allow small components..." << endl;
        if (components[i].size() < 200) continue;


        // midpoint of component should be in table_mask
//          cout << "midpoint of component should be in table_mask..." << endl;
        Mat component_img = Mat::zeros(img_bgr.size(), CV_8U);
        Point2i midpoint(0, 0);
        for (cv::Point2i p : components[i]) {
          component_img.at<uchar>(p.y, p.x) = 255;
          midpoint.x += p.x;
          midpoint.y += p.y;
        }
        midpoint.x /= components[i].size();
        midpoint.y /= components[i].size();

        if (table_mask_eroded.at<uchar>(midpoint.y, midpoint.x) == 0) continue;

        // component outline should not be too similar to table
//          cout << "component outline should not be too similar to table..." << endl;
        vector<Point> hull;
        convexHull(components[i], hull);
        vector<vector<Point>> hulls;
        hulls.push_back(hull);
        Mat component_img_hull = component_img.clone();
        drawContours(component_img_hull, hulls, 0, Scalar(255), -1);
        double semantic_dist_to_table = CommonTools::shape_dist(m_table_mask, component_img_hull);
        if (semantic_dist_to_table < 0.2) continue;

//        cout << "semantic dist to table is " << semantic_dist_to_table << endl;
//        Mat img_with_table_and_component = CommonTools::draw_mask(img_bgr, m_table_mask, Scalar(0, 0, 255));
//        img_with_table_and_component = CommonTools::draw_mask(img_with_table_and_component, component_img_hull, Scalar(0, 255, 0));
//        imshow("table_and_component", img_with_table_and_component);
        // UPLOAD TO BAXTER

//        Mat xdisplay_img = CommonTools::xdisplay(img_with_table_and_component);
//        sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", xdisplay_img).toImageMsg();
//        m_xdisplay_pub.publish(*msg);

//        imshow("cloth component", component_img);
//        waitKey(100);

        // component size should not be bigger than table
//          cout << "component size should not be bigger than table" << endl;
        if (components[i].size() > 0.95 * table_2d_area) {
          continue;
        }

        // eroding it shouldn't make it disappear
//          cout << "eroding it shouldn't make it disappear..." << endl;
        Mat eroded_component = component_img.clone();
        CommonTools::erode(eroded_component, 10);
        if (cv::countNonZero(eroded_component) < 10) {
          continue;
        }

        // save the biggest component
//          cout << "save the biggest component..." << endl;
        if (components[i].size() > max_component_size) {
          max_component_size = components[i].size();
          max_component_img = component_img.clone();
        }
      }


      if (max_component_size > 0) {
        stringstream cloth_filename;
        cloth_filename << "out/cloth_mask_" << img_idx << ".png";
//          cout << "writing " << cloth_filename.str() << " " << max_component_img.size() << endl;
        imwrite(cloth_filename.str(), max_component_img);

//        stringstream rgb_filename;
//        rgb_filename << "out/img_" << img_idx << ".png";
//        imwrite(rgb_filename.str(), img_bgr);

        cloth_mask = max_component_img.clone();
      } else {
        cloth_mask = Mat();
      }
    } else {
      cout << "no table found" << endl;
    }

    if (cloth_mask.size().area() > 0) {
      // try grabcut on cloth_mask
      cloth_mask = CommonTools::grab_cut_segmentation(img_bgr, cloth_mask);

//      CommonTools::dilate_erode(cloth_mask, 1);
//      CommonTools::dilate_erode(cloth_mask, 2);
//      CommonTools::dilate_erode(cloth_mask, 3);
//      CommonTools::draw_contour(cloth_mask, cloth_mask.clone(), cv::Scalar(255));


      // find keypoints of cloth
      int min_x = std::numeric_limits<float>::infinity();
      int max_x = 0;
      Point left_p2d(0, 0);
      Point right_p2d(0, 0);
      for (int row = 0; row < cloth_mask.rows; row++) {
          for (int col = 0; col < cloth_mask.cols; col++) {
              if (cloth_mask.at<uchar>(row, col) == 255) {
                  if (col < min_x) {
                      min_x = col;
                      left_p2d.x = col;
                      left_p2d.y = row;
                  }
                  if (col > max_x) {
                      max_x = col;
                      right_p2d.x = col;
                      right_p2d.y = row;
                  }
              }
          }
      }
      cout << "left: " << left_p2d << ", right: " << right_p2d << endl;

      Mat cloth_mask_col;
      cvtColor(cloth_mask, cloth_mask_col, CV_GRAY2RGB);

      circle(cloth_mask_col, left_p2d, 15, Scalar(90, 128, 220), 4);
      circle(cloth_mask_col, right_p2d, 15, Scalar(90, 128, 220), 4);
      imshow("cloth", cloth_mask_col);

      Mat xdisplay_img = CommonTools::xdisplay(cloth_mask_col);
      sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", xdisplay_img).toImageMsg();
      m_xdisplay_pub.publish(*msg);


      // save cloth_mask to file
      String path = ros::package::getPath("fluent_extractor") + "/cloth.png";
      imwrite(path, cloth_mask);

      CloudPtr cloth_cloud_ptr = CommonTools::get_pointcloud_from_mask(cloud_ptr, pixel2voxel, cloth_mask);

      stringstream payload;

      payload << m_vid_idx << " "
              << img_idx << " "
              << m_table_normal.x << " "
              << m_table_normal.y << " "
              << m_table_normal.z << " "
              << m_table_midpoint.x << " "
              << m_table_midpoint.y << " "
              << m_table_midpoint.z;
      cloth_cloud_ptr->header.frame_id = payload.str();

      m_pub.publish(cloth_cloud_ptr);

//      stringstream pcl_filename;
//      pcl_filename << img_idx << ".pcd";
//      pcl::io::savePCDFileASCII(pcl_filename.str(), *cloth_cloud_ptr);

//      waitKey(0);
    } else {
      cout << "no cloth found" << endl;
    }

    waitKey(20);
   
  }

  void kinect_callback(const CloudConstPtr& cloud_ptr) {
    Mat img;
    int pixel2voxel[cloud_ptr->width * cloud_ptr->height];
    int voxel2pixel[cloud_ptr->width * cloud_ptr->height];
    Cloud cld = *cloud_ptr->makeShared();
    CommonTools::get_cloud_info(cld, img, pixel2voxel, voxel2pixel);
    imshow("Camera Live Stream", img);
    waitKey(40);
    process(cloud_ptr->makeShared(), img, pixel2voxel, voxel2pixel);
  }
private:
  ros::Publisher m_pub;
  ros::Publisher m_xdisplay_pub;
  int m_vid_idx;
  Mat m_table_mask;
  PointT m_table_midpoint, m_table_normal;
  Seg2D m_seg2D;
  vector<double> m_cloth_feature;
};


int main(int argc, char **argv) {
  ros::init(argc, argv, "vision_buffer");
  ros::NodeHandle node_handle;


  // Read the config file
  FILE* fp = fopen(argv[1], "rb");
  char readBuffer[65536];
  FileReadStream is(fp, readBuffer, sizeof(readBuffer));
  Document json;
  json.ParseStream(is);


  ros::Publisher pub = node_handle.advertise<Cloud>("/vcla/cloth_folding/vision_buffer_pcl", 1);
  ros::Publisher xdisplay_pub = node_handle.advertise<sensor_msgs::Image>("/vcla/cloth_folding/vision_buffer", 1);

  if (json["use_kinect"].GetBool()) {
    BufferManager buffer_manager(json["vid_idx"].GetInt(), pub, xdisplay_pub);
    std::string kinect_topic = node_handle.resolveName(json["kinect_topic"].GetString());
    ros::Subscriber sub = node_handle.subscribe(kinect_topic, 1, &BufferManager::kinect_callback, &buffer_manager);
    ros::spin();
  } else if (json["use_recording"].GetBool()) {
    BufferManager buffer_manager(0, pub, xdisplay_pub);

    string record_dir = ros::package::getPath("fluent_extractor") + "/" + json["record_dir"].GetString() + "/";
    FileFrameScanner scanner(record_dir);

    Mat img_bgr, x, y, z;
    PointT left_hand, right_hand;
    int img_idx = json["start_frame_idx"].GetInt();

    while(scanner.get(img_idx++, img_bgr, x, y, z, left_hand, right_hand)) {
      if (!ros::ok()) break;
      int pixel2voxel[img_bgr.size().area()];
      int voxel2pixel[img_bgr.size().area()];
      CloudPtr cloud_ptr = CommonTools::make_cloud_ptr(img_bgr, x, y, z, pixel2voxel, voxel2pixel);

      buffer_manager.process(cloud_ptr, img_bgr, pixel2voxel, voxel2pixel, img_idx - 1);
      if (json["loop_mode"].GetBool()) img_idx--;
    }
  } else {
    for (int vid_idx = json["vid_idx_start"].GetInt(); vid_idx <= json["vid_idx_end"].GetInt(); vid_idx++) {
      BufferManager buffer_manager(vid_idx, pub, xdisplay_pub);

      // Define the video directory
      stringstream vids_directory;
      vids_directory << json["vids_directory"].GetString();
      vids_directory << vid_idx << "/";
      FileFrameScanner scanner(vids_directory.str());

      Mat img_bgr, x, y, z;
      PointT left_hand, right_hand;
      int img_idx = json["start_frame_idx"].GetInt();

      while(scanner.get(img_idx++, img_bgr, x, y, z, left_hand, right_hand)) {
        if (!ros::ok()) break;
        int pixel2voxel[img_bgr.size().area()];
        int voxel2pixel[img_bgr.size().area()];
        CloudPtr cloud_ptr = CommonTools::make_cloud_ptr(img_bgr, x, y, z, pixel2voxel, voxel2pixel);

        buffer_manager.process(cloud_ptr, img_bgr, pixel2voxel, voxel2pixel, img_idx - 1);
        if (json["loop_mode"].GetBool()) img_idx--;
      }
    }

  }

  ros::spin();
  return 0;
}

