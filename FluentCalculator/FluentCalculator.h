#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

class FluentCalculator {
public:
    static float thickness(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    static float x_symmetry(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    static float y_symmetry(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
    static void visualize(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud);
};
