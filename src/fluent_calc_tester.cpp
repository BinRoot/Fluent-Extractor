#include "ros/ros.h"
#include "ros/package.h"
#include <opencv2/opencv.hpp>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>

#include "CommonTools.h"
#include "FluentCalc.h"

using namespace std;
using namespace cv;

int main() {

    CloudPtr cloud(new pcl::PointCloud<PointT>);
    String path = ros::package::getPath("fluent_extractor") + "/test_pcd.pcd";

    if (pcl::io::loadPCDFile<PointT> (path, *cloud) == -1) //* load the file
    {
        PCL_ERROR ("Couldn't read file test_pcd.pcd \n");
        return (-1);
    }

	FluentCalc fluentCalc;
	vector<float> fluents;
    fluents = fluentCalc.x_and_y_symmetry(cloud);
    cout << "x_symmetry: " << fluents[0] << endl;
    cout << "y_symmetry: " << fluents[1] << endl;
    fluents = fluentCalc.calc_bbox(cloud);
    cout << "outer_bounding_box: ";
    for (auto &t : fluents)
        cout << t << " ";
    cout << endl;

    return 0;
}
