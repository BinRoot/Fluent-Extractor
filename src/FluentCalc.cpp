#include "FluentCalc.h"
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <boost/format.hpp>

using namespace cv;
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

Mat FluentCalc::get_mask_from_aligned_cloud(CloudPtr aligned_cloud) {
    // Normalize X, Y coordinates
    float min_x = aligned_cloud->points[0].x, min_y = aligned_cloud->points[0].y;
    float max_x = aligned_cloud->points[0].x, max_y = aligned_cloud->points[0].y;
    for (int i=0; i<aligned_cloud->points.size(); i++) {
        min_x = min(min_x, aligned_cloud->points[i].x);
        min_y = min(min_y, aligned_cloud->points[i].y);
        max_x = max(max_x, aligned_cloud->points[i].x);
        max_y = max(max_y, aligned_cloud->points[i].y);
    }

    Mat img = cv::Mat::zeros(100, 100, CV_8U);
    for (int i=0; i<aligned_cloud->points.size(); i++) {
        float norm_x = (aligned_cloud->points[i].x - min_x) / (max_x - min_x);
        float norm_y = (aligned_cloud->points[i].y - min_y) / (max_y - min_y);
        img.at<uchar>(int(norm_y * 99), int(norm_x * 99)) = 255;
    }

    cv::GaussianBlur(img, img, cv::Size(15, 15), 1, 1);
    cv::threshold(img, img, 10, 255, CV_THRESH_BINARY);
    CommonTools::draw_contour(img, img.clone(), cv::Scalar(255));

    return img;
}

vector<float> FluentCalc::calc_x_and_y_symmetries(CloudPtr cloud, Mat& img) {

    img = FluentCalc::get_mask_from_aligned_cloud(cloud);



    // Compute symmetry measures by pixel-wise comparision

    float x_sym_measure = 0; // x_axis symmetry
    for (int row = 0; row < 50; row++) {
        for (int col = 0; col < 100; col++) {
            x_sym_measure += int(img.at<uchar>(row, col) == img.at<uchar>(100 - row - 1, col));
        }
    }

    float y_sym_measure = 0; // y_axis symmetry
    for (int row = 0; row < 100; row++) {
        for (int col = 0; col < 50; col++) {
            y_sym_measure += int(img.at<uchar>(row, col) == img.at<uchar>(row, 100 - col - 1));
        }
    }
    x_sym_measure /= 100*100;
    y_sym_measure /= 100*100;

    vector<float> fluents(2);
    fluents[0] = x_sym_measure;
    fluents[1] = y_sym_measure;

    return fluents;
}

// Computes a orinted outer-bounding box from a point cloud.
// It returns the length of the diagonal
vector<float> FluentCalc::calc_bbox(CloudPtr cloud) {
    PointT minPoint, maxPoint;
    getMinMax3D(*cloud, minPoint, maxPoint);

    vector<float> fluents;
    fluents.push_back(maxPoint.x - minPoint.x);
    fluents.push_back(maxPoint.y - minPoint.y);
    return fluents;
}

vector<float> FluentCalc::calc_wrinkles(CloudPtr cloud, PointT table_normal, PointT table_midpoint) {
    double d = -table_normal.x * table_midpoint.x - table_normal.y * table_midpoint.y - table_normal.z * table_midpoint.z;

    std::map<string, int> frequencies;
    vector<string> teststring;
    for (int i = 0; i < cloud->size(); i++) {
        PointT p = cloud->at(i);
        float dist = table_normal.x * p.x + table_normal.y * p.y + table_normal.z * p.z + d;
        dist = floor( dist * 100.00 + 0.5 ) / 100.00;
        string dist_str = std::to_string(dist);
        frequencies[dist_str]++;
        teststring.push_back(dist_str);
    }

    int numlen = teststring.size();
    double infocontent = 0 ;
    for ( std::pair<string , int> p : frequencies ) {
        double freq = static_cast<double>( p.second ) / numlen ;
        infocontent += freq * log2( freq ) ;
    }
    infocontent *= -1 ;
//    cout << "entropy: " << infocontent << endl;

    vector<float> fluents(1);
    fluents[0] = infocontent;
    return fluents;
}

template <typename T> int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

vector<float> FluentCalc::calc_hu_moments(Mat mask) {
    imshow("hu moment of", mask);
    waitKey(20);
    // get hu moments
    double vals[7];
    cv::HuMoments(cv::moments(mask), vals);

    // store in fluent vector
    vector<float> fluents(7);
    for (int i = 0; i < 7; i++) {
        if (i >= 4) {
            fluents[i] = log10(abs(vals[i]));
        } else {
            fluents[i] = -sgn(vals[i]) * log10(abs(vals[i]));
        }
    }

    return fluents;
}

vector<float> FluentCalc::calc_squareness(Mat mask) {
    vector<float> fluents(1);
    int nonZeros = cv::countNonZero(mask);
    int area = mask.size().area();
    fluents[0] = ((float) nonZeros / (float) area);
    return fluents;
}

vector<float> FluentCalc::calc_principal_symmetries(CloudPtr aligned_cloud) {

    // Find symmetry measure by searching for best axis of symmetry
    int rotationSteps = 180;
    vector<float> bestSym;
    bestSym.push_back(0);
    bestSym.push_back(0);

    Mat best_img;
    for (int step = 0; step < rotationSteps; step++) {
        // Rotate the point cloud to find optimal axis of symmetry
        float theta = (2 * M_PI) * (float(step) / rotationSteps);
        Eigen::Rotation2D<float> rot2(theta);
        Eigen::Matrix4f rotation;
        rotation.setZero();
        rotation.block<2,2>(0,0) = rot2.toRotationMatrix();
        CloudPtr rotatedPointCloud = CommonTools::transform3d(aligned_cloud, rotation);

        // Re-center point cloud
        PointT minPoint, maxPoint;
        getMinMax3D(*rotatedPointCloud, minPoint, maxPoint);
        for (int i=0; i<rotatedPointCloud->points.size(); i++) {
            rotatedPointCloud->points[i].x -= (abs(maxPoint.x) - abs(minPoint.x))/2;
            rotatedPointCloud->points[i].y -= (abs(maxPoint.y) - abs(minPoint.y))/2;
        }

        // Uncomment this code to see the axises
//        pcl::visualization::PCLVisualizer *visu;
//        visu = new pcl::visualization::PCLVisualizer("PlyViewer");
//        visu->addPointCloud(rotatedPointCloud, "bboxedCloud");
//        visu->addCoordinateSystem(0.5);
//        while (!visu->wasStopped ()) {
//            visu->spinOnce(100);
//        }
        Mat debug_img;
        vector<float> sym = FluentCalc::calc_x_and_y_symmetries(rotatedPointCloud, debug_img);
        if (sym[0]+sym[1] > bestSym[0]+bestSym[1]) {// a heurestic
            bestSym = sym;
            best_img = debug_img.clone();
        }
    }

    // put larger symmetry score first
    if (bestSym[1] > bestSym[0]) {
        float tmp = bestSym[0];
        bestSym[0] = bestSym[1];
        bestSym[1] = tmp;
    }

    return bestSym;
}

vector<float> FluentCalc::calc_inner_outer_bbox(CloudPtr cloud, cv::Mat& debug_img,
                                                float& x_min, float& y_min, float& z_min,
                                                float& scale_x, float& scale_y, float& scale_z,
                                                cv::Rect& outer_bbox) {
    cv::Mat mask = CommonTools::get_image_from_cloud(cloud, x_min, y_min, z_min, scale_x, scale_y, scale_z, "xy");

    outer_bbox = CommonTools::get_outer_rect(mask);
    cv::Rect inner_bbox = CommonTools::get_inner_rect(mask);

    debug_img = mask.clone();
    cv::rectangle(debug_img, outer_bbox, cv::Scalar(0, 0, 255), 3);
    cv::rectangle(debug_img, inner_bbox, cv::Scalar(0, 255, 0), 3);

    cv::imshow("fluents", debug_img);
    cv::waitKey(100);

    float bbox_height = outer_bbox.height;
    float h1 = 1.0;
    float w1 = outer_bbox.width / float(outer_bbox.height);
    float h2 = inner_bbox.height / float(outer_bbox.height);
    float w2 = inner_bbox.width / float(outer_bbox.height);
    float dx = (inner_bbox.x - outer_bbox.x) / float(outer_bbox.height);
    float dy = (inner_bbox.y - outer_bbox.y) / float(outer_bbox.height);

    vector<float> fluents(5);
    fluents[0] = w1;
    fluents[1] = h2;
    fluents[2] = w2;
    fluents[3] = dx;
    fluents[4] = dy;
    return fluents;
}

// Intrinsic shape signature keypoints. TO-DO: normalize it into real fluents
vector<float> FluentCalc::calc_keypoints(CloudPtr cloud) {
    CloudPtr keypoints(new pcl::PointCloud<PointT> ());

    double cloud_resolution = 0.0058329;

    pcl::ISSKeypoint3D<PointT, PointT> iss_detector;
    pcl::search::KdTree<PointT>::Ptr tree(new pcl::search::KdTree<PointT> ());

    iss_detector.setSearchMethod(tree);
    iss_detector.setSalientRadius (6 * cloud_resolution);
    iss_detector.setNonMaxRadius (4 * cloud_resolution);
    iss_detector.setThreshold21(0.975);
    iss_detector.setThreshold32(0.975);
    iss_detector.setMinNeighbors(5);
    iss_detector.setNumberOfThreads(1);
    iss_detector.setInputCloud(cloud);
    iss_detector.compute(*keypoints);

    vector<float> fluents;

    for (int i=0; i<keypoints->points.size(); i++) {
        fluents.push_back(keypoints->points[i].x);
        fluents.push_back(keypoints->points[i].y);
        fluents.push_back(keypoints->points[i].z);
    }

    return fluents;
}
