#include "ros/ros.h"
#include "ros/package.h"

#include <opencv2/opencv.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/io/pcd_io.h>

#include "CommonTools.h"
#include "FluentCalc.h"
#include "FoldSimulator.h"

#include <boost/filesystem.hpp>

using namespace std;
using namespace cv;
namespace fs = boost::filesystem;

int main(int argc, char **argv) {
  ros::init(argc, argv, "action_inferer");

  string fluents_path = ros::package::getPath("fluent_extractor") + "/fluents/";
  fs::path directory(fluents_path);
  fs::directory_iterator iter(directory), end;

  std::map <unsigned int, string> mymap;
  for(;iter != end; ++iter) {
    string filepath = iter->path().string();
    string filename = iter->path().filename().string();
    if (boost::algorithm::ends_with(filename, "_state.png")) {
      unsigned int timestamp;
      sscanf(filename.c_str(), "%u_state.png", &timestamp);
      mymap[timestamp] = filepath;
    }
  }

  std::ofstream outfile;
  outfile.open(fluents_path + "hoi.csv", std::ios_base::app);
  Mat prev_state;
  unsigned int prev_timestamp;
  for (std::map<unsigned int, string>::iterator i = mymap.begin(); i != mymap.end(); i++) {
    unsigned int curr_timestamp = i->first;
    string state_filename = i->second;

    Mat curr_state = imread(state_filename, CV_LOAD_IMAGE_UNCHANGED);

    cout << state_filename << endl;

    if (prev_state.size().area() > 0 && curr_timestamp - prev_timestamp < 50) {
      cout << "time diff: " << curr_timestamp - prev_timestamp << endl;

      imshow("prev state", prev_state);
      imshow("curr state", curr_state);

      FoldSimulator fold(prev_state);
      fold.run_gui();
      imshow("cloth_sim", fold.visualize());
      waitKey(0);
      Point2i grip_point = fold.get_grip_point();
      Point2i release_point = fold.get_release_point();
      cout << "fold at " << grip_point << ", release at " << release_point << endl;
      Mat prev_state_col;
      cvtColor(prev_state, prev_state_col, CV_GRAY2BGR);
      circle(prev_state_col, grip_point, 10, Scalar(0, 0, 255));
      circle(prev_state_col, release_point, 10, Scalar(255, 0, 0));
      imshow("prev state", prev_state_col);
      waitKey(0);

//      stringstream fluents_filepath;
//      fluents_filepath << fluents_path << prev_timestamp << "_fluents";
//      FileStorage fs2(fluents_filepath.str(), FileStorage::READ);
//      Mat fluent_vals;
//      fs2["fluents"] >> fluent_vals;
//
//      cout << "prev fluents "

      float grip_x = grip_point.x / float(prev_state.rows);
      float grip_y = grip_point.y / float(prev_state.rows);
      float release_x = release_point.x / float(prev_state.rows);
      float release_y = release_point.y / float(prev_state.rows);

      cout << "writing data to hoi.csv" << endl;

      // save fa, fb, action in csv
      outfile << prev_timestamp << "_fluents" << ","
              << curr_timestamp << "_fluents" << ","
              << grip_x << "," << grip_y << ","
              << release_x << "," << release_y << endl;
      outfile.flush();

    }


    prev_state = curr_state;
    prev_timestamp = curr_timestamp;
  }
  outfile.close();
  return 0;
}

