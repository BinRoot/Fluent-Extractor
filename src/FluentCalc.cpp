#include "FluentCalc.h"

using namespace std;

vector<float> FluentCalc::calc_thickness(CloudPtr cloud, PointT table_normal, PointT table_midpoint) {
    double d = -table_normal.x * table_midpoint.x - table_normal.y * table_midpoint.y - table_normal.z * table_midpoint.z;
    float max_dist = 0;
    for (int i = 0; i < cloud->size(); i++) {
        PointT p = cloud->at(i);
        float dist = table_normal.x * p.x + table_normal.y * p.y + table_normal.z * p.z + d;
        if (dist > max_dist) {
            max_dist = dist;
        }
    }
    vector<float> fluents(1);
    fluents[0] = max_dist;
    return fluents;
}


vector<float> FluentCalc::calc_width_and_height(CloudPtr cloud, PointT normal) {
    double min_z = std::numeric_limits<double>::infinity();
    double max_z = -std::numeric_limits<double>::infinity();
    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();;

    PointT min_z_point, max_z_point, min_x_point, max_x_point;
    for (int i = 0; i < cloud->size(); i++) {
        PointT p = cloud->at(i);
        if (p.z > max_z) {
            max_z = p.z;
            max_z_point = p;
        }
        if (p.z < min_z) {
            min_z = p.z;
            min_z_point = p;
        }
        if (p.x > max_x) {
            max_x = p.x;
            max_x_point = p;
        }
        if (p.x < min_x) {
            min_x = p.x;
            min_x_point = p;
        }
    }

    float width_of_cloth = pcl::euclideanDistance(min_x_point, max_x_point);
    float length_of_cloth = pcl::euclideanDistance(min_z_point, max_z_point);
    vector<float> fluents(2);
    fluents[0] = width_of_cloth;
    fluents[1] = length_of_cloth;
    return fluents;
}


vector<float> FluentCalc::x_and_y_symmetry(CloudPtr cloud) {
    // Normalize X, Y coordinates  
    float min_x = cloud->points[0].x, min_y = cloud->points[0].y;
    float max_x = cloud->points[0].x, max_y = cloud->points[0].y;
    for (int i=0; i<cloud->points.size(); i++) {
        min_x = min(min_x, cloud->points[i].x);
        min_y = min(min_y, cloud->points[i].y);
        max_x = max(max_x, cloud->points[i].x);
        max_y = max(max_y, cloud->points[i].y);
    }
    vector<PointT> normalized_points;
    for (int i=0; i<cloud->points.size(); i++) {
        PointT norm_point = PointT(cloud->points[i]);
        norm_point.x = (cloud->points[i].x - min_x) / (max_x - min_x);
        norm_point.y = (cloud->points[i].y - min_y) / (max_y - min_y);
        normalized_points.push_back(norm_point);
    }

    // Drop Z coordinate and project onto X-Y plain
    Eigen::MatrixXd proj(100, 100);
    proj.setZero();
    for (int i=0; i<normalized_points.size(); i++) {
        proj(int(normalized_points[i].y * 99), int(normalized_points[i].x * 99)) = 1;
    }

    // Compute symmetry measures by pixel-wise comparision
    float x_sym_measure = 0; // x_axis symmetry
    for (int i=0; i<100; i++) {
        for (int j=0; j<50; j++) {
            x_sym_measure += int(proj(j, i) == proj(i, 99-j));
        }
    }
    float y_sym_measure = 0; // y_axis symmetry
    for (int i=0; i<100; i++) {
        for (int j=0; j<50; j++) {
            y_sym_measure += int(proj(i, i) == proj(i, 99-j));
        }
    }
    x_sym_measure /= 100*100;
    y_sym_measure /= 100*100;

    vector<float> fluents;
    fluents.push_back(x_sym_measure);
    fluents.push_back(y_sym_measure);
    return fluents;
}
