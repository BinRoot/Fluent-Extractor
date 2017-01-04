#ifndef FLUENT_CALCULATOR_H
#define FLUENT_CALCULATOR_H

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <vector>

class FluentCalculator {
public:
    static float thickness(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    static float x_symmetry(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    static float y_symmetry(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    static float height(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    static float width(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    static std::vector<pcl::PointXYZ> outerBoundingBox(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    static void visualize(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
};

#endif //FLUENT_CALCULATOR_H
