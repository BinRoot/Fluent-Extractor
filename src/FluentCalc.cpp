#include "FluentCalc.h"
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>
#include <pcl/visualization/pcl_visualizer.h>

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


vector<float> FluentCalc::x_and_y_symmetries(CloudPtr cloud, Mat& img) {
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
    img = cv::Mat::zeros(100, 100, CV_8U);
    proj.setZero();
    for (int i=0; i<normalized_points.size(); i++) {
        proj(int(normalized_points[i].y * 99), int(normalized_points[i].x * 99)) = 1;
        img.at<uchar>(int(normalized_points[i].y * 99), int(normalized_points[i].x * 99)) = 255;
    }


    cv::GaussianBlur(img, img, cv::Size(15, 15), 1, 1);
    cv::threshold(img, img, 10, 255, CV_THRESH_BINARY);
    CommonTools::draw_contour(img, img.clone(), cv::Scalar(255));

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

    vector<float> fluents;
    fluents.push_back(x_sym_measure);
    fluents.push_back(y_sym_measure);

    return fluents;
}

// Computes a orinted outer-bounding bax from a point cloud.
// It returns the length of the diagonal
vector<float> FluentCalc::calc_bbox(CloudPtr cloud) {
    Eigen::Vector4f pcaCentroid;
    compute3DCentroid(*cloud, pcaCentroid);
    Eigen::Matrix3f covariance;
    computeCovarianceMatrixNormalized(*cloud, pcaCentroid, covariance);
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
    Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
    eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));

    // Transform original cloud to the origin where the principal components correspond to the axes.
    Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
    projectionTransform.block<3,3>(0,0) = eigenVectorsPCA.transpose();
    projectionTransform.block<3,1>(0,3) = -1.f * (projectionTransform.block<3,3>(0,0) * pcaCentroid.head<3>());
    CloudPtr cloudPointsProjected(new pcl::PointCloud<PointT>());
    transformPointCloud(*cloud, *cloudPointsProjected, projectionTransform);

    // Get the minimum and maximum points of the transformed cloud.
    PointT minPoint, maxPoint;
    getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);
    const Eigen::Vector3f meanDiagonal = 0.5f*(maxPoint.getVector3fMap() + minPoint.getVector3fMap());

    // Final transform
    const Eigen::Quaternionf bboxQuaternion(eigenVectorsPCA);
    const Eigen::Vector3f bboxTransform = eigenVectorsPCA * meanDiagonal + pcaCentroid.head<3>();

    float xSize  = maxPoint.x-minPoint.x, ySize = maxPoint.y-minPoint.y, zSize = maxPoint.z-minPoint.z;
    Eigen::Vector3f boxCorner_1(-xSize/2, -ySize/2, -zSize/2);
    Eigen::Vector3f boxCorner_2(xSize/2, ySize/2, zSize/2);
    boxCorner_1 = bboxQuaternion.toRotationMatrix() * boxCorner_1 + bboxTransform;
    boxCorner_2 = bboxQuaternion.toRotationMatrix() * boxCorner_2 + bboxTransform;

    vector<float> fluents;

//    for (int i=0; i<3; i++) {
//        fluents.push_back(boxCorner_1[i]);
//    }
//    for (int i=0; i<3; i++) {
//        fluents.push_back(boxCorner_2[i]);
//    }

    float diag_length = (boxCorner_1 - boxCorner_2).norm();
    fluents.push_back(diag_length);

    // Uncomment this code to see how well the bounding box fits.

//    pcl::visualization::PCLVisualizer *visu;
//    visu = new pcl::visualization::PCLVisualizer("PlyViewer");
//
//    CloudPtr corners(new pcl::PointCloud<PointT>());
//    PointT corner_1, corner_2;
//    corner_1.x = boxCorner_1[0]; corner_1.y = boxCorner_1[1]; corner_1.z = boxCorner_1[2];
//    corner_2.x = boxCorner_2[0]; corner_2.y = boxCorner_2[1]; corner_2.z = boxCorner_2[2];
//    corners->push_back(corner_1);
//    corners->push_back(corner_2);
//    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZRGB> c1 (cloud, 0, 255, 0);
//    pcl::visualization::PointCloudColorHandlerCustom<PointT> c2 (corners, 255, 0, 0);
//    visu->addPointCloud(cloud, c1, "bboxedCloud");
//    visu->addPointCloud(corners, c2, "corners");
//    visu->addCube(bboxTransform, bboxQuaternion, xSize, ySize, zSize, "bbox");
//
//    stringstream filename;
//    filename << "out/fluent_" << time(0) << ".png";
//
//    while (!visu->wasStopped ()) {
//        visu->spinOnce(200);
//    }

    return fluents;
}

vector<float> FluentCalc::principal_symmetries(CloudPtr cloud) {

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
        CloudPtr rotatedPointCloud = CommonTools::transform3d(cloud, rotation);

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
        vector<float> sym = FluentCalc::x_and_y_symmetries(rotatedPointCloud, debug_img);
        if (sym[0]+sym[1] > bestSym[0]+bestSym[1]) {// a heurestic
            bestSym = sym;
            best_img = debug_img.clone();
        }
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
