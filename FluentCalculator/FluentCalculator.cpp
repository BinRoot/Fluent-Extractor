#include "FluentCalculator.h"

using namespace std;
using namespace pcl;

// Incorrect: need to use table normal and a point on table to compute table plane.
//            then enumerate all points to get the highest one above the plane.
float FluentCalculator::thickness(PointCloud<PointXYZ>::Ptr cloud) {
    float diff = 0;
    for (int i=0; i<cloud->points.size(); i++) {
        for (int j=0; j<cloud->points.size(); j++) {
            diff = max(diff, abs(cloud->points[i].z - cloud->points[j].z));
        }
    }
    return diff;
}

float FluentCalculator::x_symmetry(PointCloud<PointXYZ>::Ptr cloud) {
    // Normalize the X, Y coordinates
    float min_x = cloud->points[0].x, min_y = cloud->points[0].y;
    float max_x = cloud->points[0].x, max_y = cloud->points[0].y;
    for (int i=0; i<cloud->points.size(); i++) {
        min_x = min(min_x, cloud->points[i].x);
        min_y = min(min_y, cloud->points[i].y);
        max_x = max(max_x, cloud->points[i].x);
        max_y = max(max_y, cloud->points[i].y);
    }
    vector<PointXYZ> normalized_points;
    for (int i=0; i<cloud->points.size(); i++) {
        normalized_points.push_back(PointXYZ(
            (cloud->points[i].x - min_x) / (max_x - min_x),
            (cloud->points[i].y - min_y) / (max_y - min_y),
            cloud->points[i].z
        ));
    }

    // Drop Z coordinate and project onto X-Y plain
    Eigen::MatrixXd proj(100, 100);
    proj.setZero();
    for (int i=0; i<normalized_points.size(); i++) {
        proj(int(normalized_points[i].y * 99), int(normalized_points[i].x * 99)) = 1;
    }

    // Apply symmetry filter
    Eigen::MatrixXd filter(100, 100);
    filter.setZero();
    filter.block(0, 0, 100, 50) = Eigen::MatrixXd::Constant(100, 50, 1);
    filter.block(0, 50, 100, 50) = Eigen::MatrixXd::Constant(100, 50, -1);
    Eigen::MatrixXd sym = filter * proj;

    return abs(sym.sum()) / (100*100);
}

float FluentCalculator::y_symmetry(PointCloud<PointXYZ>::Ptr cloud) {
    // Normalize the X, Y coordinates
    float min_x = cloud->points[0].x, min_y = cloud->points[0].y;
    float max_x = cloud->points[0].x, max_y = cloud->points[0].y;
    for (int i=0; i<cloud->points.size(); i++) {
        min_x = min(min_x, cloud->points[i].x);
        min_y = min(min_y, cloud->points[i].y);
        max_x = max(max_x, cloud->points[i].x);
        max_y = max(max_y, cloud->points[i].y);
    }
    vector<PointXYZ> normalized_points;
    for (int i=0; i<cloud->points.size(); i++) {
        normalized_points.push_back(PointXYZ(
            (cloud->points[i].x - min_x) / (max_x - min_x),
            (cloud->points[i].y - min_y) / (max_y - min_y),
            cloud->points[i].z
        ));
    }

    // Drop Z coordinate and project onto X-Y plain
    Eigen::MatrixXd proj(100, 100);
    proj.setZero();
    for (int i=0; i<normalized_points.size(); i++) {
        proj(int(normalized_points[i].y * 99), int(normalized_points[i].x * 99)) = 1;
    }

    // Apply symmetry filter
    Eigen::MatrixXd filter(100, 100);
    filter.setZero();
    filter.block(0, 0, 50, 100) = Eigen::MatrixXd::Constant(50, 100, 1);
    filter.block(50, 0, 50, 100) = Eigen::MatrixXd::Constant(50, 100, -1);
    Eigen::MatrixXd sym = filter * proj;

    return abs(sym.sum()) / (100*100);
}


void FluentCalculator::visualize(PointCloud<PointXYZ>::Ptr cloud) {
    visualization::CloudViewer viewer ("Simple Cloud Viewer");
    viewer.showCloud (cloud);
    while (!viewer.wasStopped ()) {}
}
