#include "ros/ros.h"
#include "FileFrameScanner.h"
#include "CommonTools.h"
#include <opencv2/opencv.hpp>
#include <sstream>
#include <pcl/visualization/cloud_viewer.h>

#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <iostream>
#include "rapidjson/filereadstream.h"
#include <cstdio>

using namespace std;
using namespace cv;
using namespace rapidjson;

int main(int argc, char **argv) {
  ros::init(argc, argv, "vision_buffer");

  // Read the config file
  FILE* fp = fopen(argv[1], "rb");
  char readBuffer[65536];
  FileReadStream is(fp, readBuffer, sizeof(readBuffer));
  Document json;
  json.ParseStream(is);

  // Define the video directory
  stringstream vids_directory;
  vids_directory << json["vids_directory"].GetString();
  vids_directory << json["vid_idx"].GetInt() << "/";
  FileFrameScanner scanner(vids_directory.str());

  pcl::visualization::CloudViewer viewer("Cloud Viewer");

  Mat img_bgr, x, y, z;
  PointT left_hand, right_hand;
  int img_idx = json["start_frame_idx"].GetInt();

  Mat table_mask;

  while(scanner.get(img_idx++, img_bgr, x, y, z, left_hand, right_hand)) {
    int pixel2voxel[img_bgr.size().area()];
    int voxel2pixel[img_bgr.size().area()];
    CloudPtr cloud_ptr = CommonTools::make_cloud_ptr(img_bgr, x, y, z, pixel2voxel, voxel2pixel);

    if (table_mask.size().area() == 0) {
      CommonTools::find_biggest_plane(cloud_ptr->makeShared(), voxel2pixel, table_mask);
    }

    Mat img_with_mask = CommonTools::draw_mask(img_bgr, table_mask, Scalar(0, 0, 255));
    imshow("img", img_with_mask);

    waitKey(30);
  }

  ros::spin();
  return 0;
}

