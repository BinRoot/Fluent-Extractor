#include "FluentCalculator.h"
#include <pcl/common/transforms.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h>

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


// Compute a orinted outer-bounding bax from a point cloud, and returns the upper left and lower right point.
// Check http://codextechnicanum.blogspot.com/2015/04/find-minimum-oriented-bounding-box-of.html for reference
vector<PointXYZ> FluentCalculator::outerBoundingBox(PointCloud<PointXYZ>::Ptr cloudSegmented) {
	Eigen::Vector4f pcaCentroid;
	compute3DCentroid(*cloudSegmented, pcaCentroid);
	Eigen::Matrix3f covariance;
	computeCovarianceMatrixNormalized(*cloudSegmented, pcaCentroid, covariance);
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
	Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
	eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1)); 

	// Transform the original cloud to the origin where the principal components correspond to the axes.
	Eigen::Matrix4f projectionTransform(Eigen::Matrix4f::Identity());
	projectionTransform.block<3,3>(0,0) = eigenVectorsPCA.transpose();
	projectionTransform.block<3,1>(0,3) = -1.f * (projectionTransform.block<3,3>(0,0) * pcaCentroid.head<3>());
	PointCloud<PointXYZ>::Ptr cloudPointsProjected (new PointCloud<PointXYZ>);
	transformPointCloud(*cloudSegmented, *cloudPointsProjected, projectionTransform);
	// Get the minimum and maximum points of the transformed cloud.
	PointXYZ minPoint, maxPoint;
	getMinMax3D(*cloudPointsProjected, minPoint, maxPoint);
	const Eigen::Vector3f meanDiagonal = 0.5f*(maxPoint.getVector3fMap() + minPoint.getVector3fMap());

	// Final transform
	const Eigen::Quaternionf bboxQuaternion(eigenVectorsPCA);
	const Eigen::Vector3f bboxTransform = eigenVectorsPCA * meanDiagonal + pcaCentroid.head<3>();

	Eigen::Vector3f upperLeft(minPoint.x, minPoint.y, minPoint.z);
	Eigen::Vector3f lowerRight(maxPoint.x, maxPoint.y, maxPoint.z);
	upperLeft = bboxQuaternion * (upperLeft + bboxTransform);
	lowerRight = bboxQuaternion * (lowerRight + bboxTransform);

	vector<PointXYZ> box;
	box.push_back(PointXYZ(upperLeft[0], upperLeft[1], upperLeft[2]));
	box.push_back(PointXYZ(lowerRight[0], lowerRight[1], lowerRight[2]));

	return box;
}


// Relative height and width - this is correct assume fixed camera position and fixed cloth distance.
// to camera. If the assumption is not true, then need to compensate the size change due to distance difference to the camera.

float FluentCalculator::height(PointCloud<PointXYZ>::Ptr cloud) {
	PointXYZ minPoint, maxPoint;
	getMinMax3D(cloud, minPoint, maxPoint);
	return maxPoint.x - minPoint.x;
}

float FluentCalculator::width(PointCloud<PointXYZ>::Ptr cloud) {
	PointXYZ minPoint, maxPoint;
	getMinMax3D(cloud, minPoint, maxPoint);
	return maxPoint.y - minPoint.y;
}

void FluentCalculator::visualize(PointCloud<PointXYZ>::Ptr cloud) {
    visualization::CloudViewer viewer ("Simple Cloud Viewer");
    viewer.showCloud (cloud);
    while (!viewer.wasStopped ()) {}
}
