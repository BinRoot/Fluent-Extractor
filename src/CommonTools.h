//
// Created by binroot on 6/30/15.
//

#ifndef ROBOTCLOTHFOLDING_COMMONTOOLS_H
#define ROBOTCLOTHFOLDING_COMMONTOOLS_H

//#include <pcl/common/common_headers.h>
#include <pcl_ros/point_cloud.h>
#include <opencv2/opencv.hpp>
// #include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/common/distances.h>
#include <pcl/common/centroid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/segmentation/region_growing_rgb.h>
#include <pcl/segmentation/min_cut_segmentation.h>
#include <pcl/common/transforms.h>

#include <ctime>

#define BOOST_NO_CXX11_SCOPED_ENUMS
#include <boost/filesystem.hpp>
#undef BOOST_NO_CXX11_SCOPED_ENUMS

#define DEBUG false

using namespace boost::filesystem;

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> Cloud;
typedef Cloud::Ptr CloudPtr;
typedef Cloud::ConstPtr CloudConstPtr;
typedef pcl::search::KdTree<PointT>::Ptr TreePtr;

const static std::string ROS_PATH = "src/mind_graph/src/";

class CommonTools {
public:
    static cv::Mat get_mask_from_points(std::vector<cv::Point>& points) {
        cv::Rect outer_rect = boundingRect(points);
        cv::Mat mask(outer_rect.size(), CV_8U);
        std::vector<std::vector<cv::Point> > pointss(1, points);
        cv::drawContours(mask, pointss, 0, cv::Scalar::all(255));
        return mask;
    }

    static cv::Mat to_mat(std::vector<float> vals) {
        cv::Mat m(vals.size(), 1, CV_32F);
        for (int i = 0; i < vals.size(); i++)
            m.at<float>(i) = vals[i];
        return m;
    }

    static cv::Mat draw_mask(cv::Mat img, cv::Mat mask, cv::Scalar color) {
        double color_strength = 0.5;
        cv::Mat display = img.clone();
        for (int row = 0; row < img.rows; row++) {
            for (int col = 0; col < img.cols; col++) {
                if (is_row_col_in_mask(row, col, mask)) {
                    display.at<cv::Vec3b>(row, col)[0] =
                        img.at<cv::Vec3b>(row, col)[0]*(1-color_strength) + color[0]*color_strength;
                    display.at<cv::Vec3b>(row, col)[1] =
                        img.at<cv::Vec3b>(row, col)[1]*(1-color_strength) + color[1]*color_strength;
                    display.at<cv::Vec3b>(row, col)[2] =
                        img.at<cv::Vec3b>(row, col)[2]*(1-color_strength) + color[2]*color_strength;
                }
            }
        }
        return display;
    }

    static cv::Point reflect(cv::Point line_p1, cv::Point line_p2, cv::Point p) {
        double m = double(line_p2.y - line_p1.y) / (double(line_p2.x - line_p1.x)+0.0001);
        double b = line_p2.y - m * line_p2.x;
        double d = (p.x + (p.y - b) * m) / (1.0 + m*m);
        double xp = 2 * d - p.x;
        double yp = 2 * d * m - p.y + 2 * b;
        return cv::Point(xp, yp);
    }

    static cv::Mat get_rotation_matrix(cv::Mat& src, double angle_degrees) {
        cv::Point2f pt(src.cols/2., src.rows/2.);
        return getRotationMatrix2D(pt, angle_degrees, 1.0);
    }

    static void rotate(cv::Mat& src, double angle_degrees, cv::Mat& dst) {
        cv::Point2f pt(src.cols/2., src.rows/2.);
        cv::Mat r = getRotationMatrix2D(pt, angle_degrees, 1.0);
        warpAffine(src, dst, r, cv::Size(src.cols, src.rows));
    }

    static cv::Rect get_outer_rect(cv::Mat& mask) {
        std::vector<std::vector<cv::Point> > contours; std::vector<cv::Vec4i> hierarchy;
        findContours(mask.clone(),
                     contours,
                     hierarchy,
                     cv::RETR_TREE,
                     cv::CHAIN_APPROX_SIMPLE,
                     cv::Point(0, 0));
        std::vector<cv::Point> pts;
        for (auto& contour : contours) {
            for (auto& p : contour) {
                pts.push_back(p);
            }
        }
        return boundingRect(pts);
    }

    static bool is_rect_inside(const cv::Rect& r, const cv::Mat& mask) {
        int count = 0;
        int max_count = r.area() * 0.005;
        for (int row = r.y; row < r.y + r.height; row++) {
            for (int col = r.x; col < r.x + r.width; col++) {
                if (mask.at<uchar>(row, col) == 0) {
                    if (count > max_count) return false;
                    count++;
                }
            }
        }
        return true;
    }

    static long int unix_timestamp() {
      time_t t = std::time(0);
      long int now = static_cast<long int> (t);
      return now;
    }

    static cv::Rect get_inner_rect(const cv::Mat& mask_) {
        std::vector<std::vector<cv::Point> > contours; std::vector<cv::Vec4i> hierarchy;
        cv::Mat mask = mask_.clone();
        findContours(mask,
                     contours,
                     hierarchy,
                     cv::RETR_TREE,
                     cv::CHAIN_APPROX_SIMPLE,
                     cv::Point(0, 0));
        cv::Rect best_rect(0,0,0,0);
        unsigned long max_contour_size = 0;
        int contour_idx = 0;
        for (int i = 0; i < contours.size(); i++) {
            if (contours[i].size() > max_contour_size) {
                max_contour_size = contours[i].size();
                contour_idx = i;
            }
        }

        std::vector<cv::Point> pts;
        for (int i = 0; i < contours[contour_idx].size(); i++) {
            if (i % 2 == 0) {
                pts.push_back(contours[contour_idx][i]);
            }
        }

        for (auto& p1 : pts) {
            for (auto& p2 : pts) {
                if (p1.x == p2.x && p1.y == p2.y) continue;
                cv::Rect r(p1, p2);
                if (is_rect_inside(r, mask)) {
                    if (r.area() > best_rect.area()) {
                        best_rect.x = r.x;
                        best_rect.y = r.y;
                        best_rect.width = r.width;
                        best_rect.height = r.height;
                    }
                }
            }
        }
        return best_rect;
    }

    static double cross_product(const cv::Point& a, const cv::Point& b ){
        return a.x*b.y - a.y*b.x;
    }

    static double distance_to_line(cv::Point begin_, cv::Point end_, cv::Point x_) {
        cv::Point begin(begin_.x, begin_.y);
        cv::Point end(end_.x, end_.y);
        cv::Point x(x_.x, x_.y);
        end = end - begin;
        //cout<<"end :"<<end.x<<" "<<end.y<<endl;
        x = x - begin;
        double area = cross_product(x, end);
       // cout << "area :"<<area<<endl;
        return area / ((double)(norm(end))); //+0.0000001
    }

    static void max_contour(const cv::Mat& mask, std::vector<cv::Point>& pts) {
        std::vector<std::vector<cv::Point> > contours; std::vector<cv::Vec4i> hierarchy;
        findContours(mask.clone(),
                     contours,
                     hierarchy,
                     CV_RETR_TREE,
                     CV_CHAIN_APPROX_SIMPLE,
                     cv::Point(0, 0) );
        if (contours.empty()) return;
        unsigned long max_contour_size = 0;
        int contour_idx = -1;
        for (int i = 0; i < contours.size(); i++) {
            if (contours[i].size() > max_contour_size) {
                max_contour_size = contours[i].size();
                contour_idx = i;
            }
        }
        for (auto& p : contours[contour_idx]) {
            pts.push_back(p);
        }
    }

    static void max_contour(const cv::Mat& mask, std::vector<cv::Point>& pts, cv::Mat& display) {
        std::vector<std::vector<cv::Point> > contours; std::vector<cv::Vec4i> hierarchy;
        findContours(mask.clone(),
                     contours,
                     hierarchy,
                     CV_RETR_TREE,
                     CV_CHAIN_APPROX_SIMPLE,
                     cv::Point(0, 0) );
        if (contours.empty()) return;
        
        unsigned long max_contour_size = 0;
        int contour_idx = -1;
        for (int i = 0; i < contours.size(); i++) {
            if (contours[i].size() > max_contour_size) {
                max_contour_size = contours[i].size();
                contour_idx = i;
            }
        }
        for (auto& p : contours[contour_idx]) {
            pts.push_back(p);
        }
        display = cv::Mat::zeros(mask.size(), CV_8U);
        drawContours(display, contours, contour_idx, cv::Scalar(255), -1);
    }

    static cv::Mat icon_from_bgr(cv::Mat& img) {
        cv::Mat out;
        double scale = 100.0 / img.rows;
        cv::Size size(img.size().width * scale, img.size().height * scale);
        resize(img, out, size);

        int left = out.size().width + 1;
        int right = -1;
        int top = out.size().height + 1;
        int bot = -1;
        for (int row = 0; row < out.rows; row++) {
            for (int col = 0; col < out.cols; col++) {
                cv::Vec3b pix = out.at<cv::Vec3b>(row, col);
                if (!(pix[0] == 0 && pix[1] == 0 && pix[2] == 0)) {
                    if (col < left) left = col;
                    if (col > right) right = col;
                    if (row < top) top = row;
                    if (row > bot) bot = row;
                }
            }
        }
        cv::Rect myROI(left, top, right - left, bot - top);
        out = out(myROI);
        return out;
    }

    static cv::Mat icon_from_mask(cv::Mat& mask) {
        cv::Mat out;
        double scale = 100.0 / mask.rows;
        cv::Size size(mask.size().width * scale, mask.size().height * scale);
        resize(mask, out, size);

        int left = out.size().width + 1;
        int right = -1;
        int top = out.size().height + 1;
        int bot = -1;
        for (int row = 0; row < out.rows; row++) {
            for (int col = 0; col < out.cols; col++) {
                if (out.at<uchar>(row, col) > 0) {
                    if (col < left) left = col;
                    if (col > right) right = col;
                    if (row < top) top = row;
                    if (row > bot) bot = row;
                }
            }
        }
        cv::Rect myROI(left, top, right - left, bot - top);
        out = out(myROI);
        return out;
    }

    static std::vector<float> get_rect_feature(cv::Mat& mask, cv::Rect outer_rect, cv::Rect inner_rect, double mask_scale) {
        std::vector<float> feature(6);
        feature[0] = outer_rect.width / double(outer_rect.height);
        feature[1] = inner_rect.width / double(outer_rect.height);
        feature[2] = inner_rect.height / double(outer_rect.height);
        feature[3] = ((outer_rect.x + outer_rect.width/2) - (inner_rect.x + inner_rect.width/2)) / double(outer_rect.height);
        feature[4] = ((outer_rect.y + outer_rect.height/2) - (inner_rect.y + inner_rect.height/2)) / double(outer_rect.height);
        feature[5] = mask_scale;

        // normalize features
        normalize_spatial_features(feature);
        return feature;
    }

    static void normalize_spatial_features(std::vector<float>& feature) {
        feature[0] = (feature[0] - 1.01145) / 0.153941;
        feature[1] = (feature[1] - 0.641966) / 0.236256;
        feature[2] = (feature[2] - 0.624886) / 0.175061;
        feature[3] = (feature[3] - 0.00103533) / 0.0490741;
        feature[4] = (feature[4] - 0.00158592) / 0.0489192;
        feature[5] = (feature[5] - 0.5) / 0.2;
    }

    static void unnormalize_spatial_features(std::vector<float>& feature) {
        feature[0] = feature[0] * 0.153941 + 1.01145;
        feature[1] = feature[1] * 0.236256 + 0.641966;
        feature[2] = feature[2] * 0.175061 + 0.624886 ;
        feature[3] = feature[3] * 0.0490741 + 0.00103533;
        feature[4] = feature[4] * 0.0489192 + 0.00158592;
        feature[5] = feature[5] * 0.2 + 0.5;
    }

    static std::vector<float> get_rect_feature(cv::Mat& mask, double mask_scale) {
        cv::Rect outer_rect = get_outer_rect(mask);
        cv::Rect inner_rect = get_inner_rect(mask);
        return get_rect_feature(mask, outer_rect, inner_rect, mask_scale);
    }

    static float l2_dist(std::vector<float>& feature1, std::vector<float>& feature2) {
        double sum = 0;
        for (int j = 0; j < feature1.size(); j++) {
            sum += (feature1[j] - feature2[j]) * (feature1[j] - feature2[j]);
        }
        return sqrt(sum);
    }

    static void rotate_features_clockwise(std::vector<float>& features) {
        unnormalize_spatial_features(features);

        double old_w1 = features[0];
        double old_w2 = features[1];
        double old_h1 = features[2];
        double old_dx = features[3];
        double old_dy = features[4];
        double new_w1 = 1 / old_w1;
        double new_w2 = old_h1 / old_w1;
        double new_h1 = old_w2 / old_w1;
        double new_dx = old_dy;
        double new_dy = -old_dx;


        features[0] = new_w1;
        features[1] = new_w2;
        features[2] = new_h1;
        features[3] = new_dx;
        features[4] = new_dy;
        features[5] = features[5];

        normalize_spatial_features(features);

    }

    static cv::Mat grab_cut_segmentation(cv::Mat& img, cv::Mat& mask) {
      cv::Mat bg_model, fg_model;
      cv::Mat result = mask.clone();
      CommonTools::erode(result, 10);
      result = (result / 255) * cv::GC_PR_FGD;

      cv::Mat background = mask.clone();
      CommonTools::dilate(background, 5);
      cv::threshold(background,background,1, cv::GC_PR_BGD,cv::THRESH_BINARY_INV);
      result += background;

      /* set more probable foreground */
      for (int i = 0; i < result.size().area(); i++) {
        if (result.at<uchar>(i) == 0) {
          result.at<uchar>(i) = cv::GC_PR_BGD;
        }
      }

      /* set confident background on corners */
      int rs = 50;
      result(cv::Rect(0, 0, rs, rs)).setTo(cv::Scalar(cv::GC_BGD));
      result(cv::Rect(0, result.rows-rs, rs, rs)).setTo(cv::Scalar(cv::GC_BGD));
      result(cv::Rect(result.cols-rs, 0, rs, rs)).setTo(cv::Scalar(cv::GC_BGD));
      result(cv::Rect(result.cols-rs, result.rows-rs, rs, rs)).setTo(cv::Scalar(cv::GC_BGD));

      cv::imshow("img with probable FG, probable BG, & confident BG", result*80);
      cv::waitKey(20);


      cv::Rect box = CommonTools::get_outer_rect(mask);
      /* GrabCut Segmentation */
      grabCut(img,                  // input image
              result,               // segmentation result
              box,                  // rectangle containing foreground
              bg_model, fg_model,   // models
              1,                    // number of iterations
              cv::GC_INIT_WITH_MASK);   // use mask

      /* get the pixels marked as likely foreground */
      cv::Mat result_pf;
      compare(result, cv::GC_PR_FGD, result_pf, cv::CMP_EQ);

      /* generate output image */
//      cv::imshow("grabcut result_pf", result_pf);
//      cv::waitKey(20);
      std::vector<cv::Point> pts;
      CommonTools::max_contour(result_pf, pts, result_pf);

      return result_pf;
    }

    static void repel_grip_point(cv::Point line_p1, cv::Point line_p2, cv::Mat cloth_mask, cv::Point& grip_point) {
        double dist1 = distance_to_line(line_p1, line_p2, grip_point);
        double max_dist = 0;
        for (int row = 0; row < cloth_mask.rows; row++) {
            for (int col = 0; col < cloth_mask.cols; col++) {
                if (is_row_col_in_mask(row, col, cloth_mask)) {
                    double dist2 = distance_to_line(line_p1, line_p2, cv::Point(col, row));
                    if (dist1 * dist2 > 0) {
                        if (abs(dist2) > max_dist) {
                            max_dist = abs(dist2);
                            grip_point.x = col;
                            grip_point.y = row;
                        }
                    }
                }
            }
        }
    }


    static Eigen::Matrix4f get_projection_transform(CloudPtr cloud) {
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

      // Add a rotation s.t. the first three principal components lies on x, y, z respectively
      Eigen::Matrix4f rotation;
      rotation.setZero();
      rotation(2,0) = 1; rotation(1,1) = 1; rotation(0,2) = 1;
      projectionTransform  = rotation * projectionTransform;

      return projectionTransform;
    }

    static cv::Mat xdisplay(cv::Mat img, double scale=1.5) {
        cv::Mat xdisplay_img = cv::Mat::zeros(800, 1024, CV_8UC3);
        resize(img, img, cv::Size(), scale, scale);
        int x_offset = (800 - img.cols) / 2;
        img.copyTo(xdisplay_img(cv::Rect(x_offset, 0, img.cols, img.rows)));
        return xdisplay_img;
    }

    static CloudPtr transform3d(CloudPtr cloud, Eigen::Matrix4f transform) {
        CloudPtr cloudPointsProjected(new pcl::PointCloud<PointT>());
        transformPointCloud(*cloud, *cloudPointsProjected, transform);
        return cloudPointsProjected;
    }

    static void attract_grip_point(cv::Point line_p1, cv::Point line_p2, cv::Mat cloth_mask, cv::Point& grip_point) {
        double dist1 = distance_to_line(line_p1, line_p2, grip_point);
        double min_dist = 99999;
        cv::Point best_grip_point;
        for (int row = 0; row < cloth_mask.rows; row++) {
            for (int col = 0; col < cloth_mask.cols; col++) {
                if (is_row_col_in_mask(row, col, cloth_mask)) {
                    cv::Point test_point(col, row);
                    double dist2 = distance_to_line(line_p1, line_p2, test_point);
                    if (dist1 * dist2 > 0) {
                        double grip_dist = cv::norm(grip_point - test_point);
                        if (grip_dist < min_dist) {
                            min_dist = grip_dist;
                            best_grip_point.x = test_point.x;
                            best_grip_point.y = test_point.y;
                        }
                    }
                }
            }
        }
        if (min_dist < 99990 ) {
            grip_point.x = best_grip_point.x;
            grip_point.y = best_grip_point.y;
        }
    }

    static void understand_fold(cv::Mat mask1, cv::Mat mask2, cv::Vec4f& line_params, cv::Point& grip_point) {

        imshow("previous mask (understand_fold)", mask1);
        imshow("current mask (understand_fold)", mask2);
        cv::waitKey(200);

        cv::Mat diff = mask1 - mask2;
        erode_dilate(diff, 4);
        if (DEBUG) cv::imshow("diff", diff);
        std::vector<cv::Point> diff_pts;
        cv::Mat diff_blob;
        max_contour(diff, diff_pts, diff_blob);
        double diff_ratio = double(cv::countNonZero(diff)) / double(countNonZero(mask1));
        std::cout << "diff_ratio: " << diff_ratio << std::endl;

        if (diff_ratio < 0.02) return;
        if (DEBUG) imshow("diff_blob_orig", diff_blob);
        erode(diff_blob, 2);
        dilate(diff_blob, 3);
        if (DEBUG) imshow("diff_blob", diff_blob);
        std::vector<cv::Point> diff_blob_pts;
        max_contour(diff_blob, diff_blob_pts);

        cv::Mat fold_line_area = diff_blob & mask2;

        std::vector<cv::Point> fold_pts;
        max_contour(fold_line_area, fold_pts, fold_line_area);

        std::cout << fold_pts.size() << std::endl;
//        waitKey(0);
        if (DEBUG) imshow("fold_line_area", fold_line_area);
//        if (DEBUG) waitKey(0);
        cv::fitLine(fold_pts, line_params, CV_DIST_L2, 0, 0.01, 0.01);

        double line_scale = 70;
        double line_dx = line_params[0];
        double line_dy = line_params[1];

        if (line_dx != 0) {
            if (abs(line_dy / line_dx) < 0.35) {
                line_dy = 0;
                line_dx = 1;
            }
        }



        double line_x = line_params[2];
        double line_y = line_params[3];
        cv::Point line_p1 = cv::Point(line_x - line_scale * line_dx, line_y - line_scale * line_dy);
        cv::Point line_p2 = cv::Point(line_x + line_scale * line_dx, line_y + line_scale * line_dy);



        double max_dist = -1;
        cv::Point grip_point_dilate;
        for (auto& p : diff_blob_pts) {
            double dist = abs(distance_to_line(line_p1, line_p2, p));
            if (dist > max_dist) {
                max_dist = dist;
                grip_point_dilate.x = p.x;
                grip_point_dilate.y = p.y;
            }
        }
        double min_dist = 999999999;
        for (int row = 0; row < mask1.rows; row++) {
            for (int col = 0; col < mask1.cols; col++) {
                if (mask1.at<uchar>(row, col) > 0) {
                    double dist = cv::norm(grip_point_dilate - cv::Point(col, row));
                    if (dist < min_dist) {
                        min_dist = dist;
                        grip_point.x = col;
                        grip_point.y = row;
                    }
                }
            }
        }
    }

    static double shape_dist(cv::Mat& mask_a, cv::Mat& mask_b) {
        cv::Mat union_mask = mask_a | mask_b;
        double sym_diff = cv::countNonZero(union_mask - (mask_a & mask_b));
//        double area =  countNonZero(mask_a) + countNonZero(mask_b);
        double area = countNonZero(union_mask);
        return sym_diff / area;
    }

    static double shape_dist2(cv::Mat& mask_a, cv::Mat& mask_b) {
        double sym_diff = cv::countNonZero((mask_a & mask_b));
        double area = countNonZero(mask_b);
        return sym_diff / area;
    }

    static double shape_dist1(cv::Mat& mask_a, cv::Mat& mask_b) {
        cv::Mat union_mask = mask_a | mask_b;
        double sym_diff = cv::countNonZero(union_mask - (mask_a & mask_b));
//        double area =  countNonZero(mask_a) + countNonZero(mask_b);
        double area = countNonZero(union_mask);
        return sym_diff;
    }

    static void erode(cv::Mat& img, int size) {
        cv::Mat element = cv::getStructuringElement(
                cv::MORPH_ELLIPSE,
                cv::Size(2*size + 1, 2*size+1),
                cv::Point(size, size)
        );
        cv::erode(img, img, element);
    }

    static void dilate(cv::Mat& img, int size) {
        cv::Mat element = cv::getStructuringElement(
                cv::MORPH_ELLIPSE,
                cv::Size(2*size + 1, 2*size+1),
                cv::Point(size, size)
        );
        cv::dilate(img, img, element);
    }

    static void erode_dilate(cv::Mat& img, int size) {
        cv::Mat element = cv::getStructuringElement(
                cv::MORPH_ELLIPSE,
                cv::Size(2*size + 1, 2*size+1),
                cv::Point(size, size)
        );
        cv::erode(img, img, element);
        cv::dilate(img, img, element);
    }

    static void dilate_erode(cv::Mat& img, int size) {
        cv::Mat element = cv::getStructuringElement(
                cv::MORPH_ELLIPSE,
                cv::Size(2*size + 1, 2*size+1),
                cv::Point(size, size)
        );
        cv::dilate(img, img, element);
        cv::erode(img, img, element);
    }

    static void clip_to_bounds(int& x, int min_val, int max_val) {
        if (x < min_val) {
            x = min_val;
        } else if (x > max_val) {
            x = max_val;
        }
    }

      static void attract_grip_point1(cv::Mat cloth_mask, cv::Point& grip_point) {

          double min_dist = 99999;
          cv::Point best_grip_point;
          for (int row = 0; row < cloth_mask.rows; row++) {
              for (int col = 0; col < cloth_mask.cols; col++) {
                  if (is_row_col_in_mask(row, col, cloth_mask)) {
                      cv::Point test_point(col, row);
                      {
                          double grip_dist = cv::norm(grip_point - test_point);
                          if (grip_dist < min_dist) {
                              min_dist = grip_dist;
                              best_grip_point.x = test_point.x;
                              best_grip_point.y = test_point.y;
                          }
                      }
                  }
              }
          }
          if (min_dist < 99990 ) {
              grip_point.x = best_grip_point.x;
              grip_point.y = best_grip_point.y;
          }
      }

    static cv::Mat get_image_from_cloud(const CloudConstPtr& cloud_const_ptr,
                                        float& x_min, float& y_min, float& z_min,
                                        float& scale_x, float& scale_y, float& scale_z, std::string mode="xy") {
        std::vector<cv::Point2f> points;
        x_min = std::numeric_limits<float>::infinity();
        y_min = std::numeric_limits<float>::infinity();
        z_min = std::numeric_limits<float>::infinity();
        float x_max = 0;
        float y_max = 0;
        float z_max = 0;


        for (int i = 0; i < cloud_const_ptr->size(); i++) {
            PointT p = cloud_const_ptr->at(i);
            if (p.x < x_min) x_min = p.x;
            if (p.x > x_max) x_max = p.x;
            if (p.y < y_min) y_min = p.y;
            if (p.y > y_max) y_max = p.y;
            if (p.z < z_min) z_min = p.z;
            if (p.z > z_max) z_max = p.z;

            cv::Point2f p2;
            if (!mode.compare("xy")) {
                p2.x = p.x;
                p2.y = p.y;
            } else if (!mode.compare("yz")) {
                p2.x = p.y;
                p2.y = p.z;
            }

            points.push_back(p2);
        }

        scale_x = x_max - x_min;
        scale_y = y_max - y_min;
        scale_z = z_max - z_min;
        std::cout << "scale x: " << scale_x
             << ", scale y: " << scale_y
             << ", scale z: " << scale_z << std::endl;

        float mask_width = 256;

        float mask_height;
        if (!mode.compare("xy")) {
            mask_height = mask_width * (scale_y / scale_x);
        } else if (!mode.compare("yz")) {
            mask_height = mask_width * (scale_z / scale_y);
        }

        std::cout << "mask width: " << mask_width << ", mask_height: " << mask_height << std::endl;
        cv::Mat cloud_mask_raw = cv::Mat::zeros(mask_height, mask_width, CV_8U);
        for (int i = 0; i < points.size(); i++) {
            if (!mode.compare("xy")) {
                int x = mask_width * (points[i].x - x_min) / scale_x;
                clip_to_bounds(x, 0, int(mask_width) - 1);
                int y = mask_height * (points[i].y - y_min) / scale_y;
                clip_to_bounds(y, 0, int(mask_height) - 1);
                cloud_mask_raw.at<uchar>(y, x) = 255;
            } else if (!mode.compare("yz")) {
                int x = mask_width * (points[i].x - y_min) / scale_y;
                clip_to_bounds(x, 0, int(mask_width) - 1);
                int y = mask_height * (points[i].y - z_min) / scale_z;
                clip_to_bounds(y, 0, int(mask_height) - 1);
                cloud_mask_raw.at<uchar>(y, x) = 255;
            }
        }

//    CommonTools::dilate_erode(cloud_mask_raw, 1);
//    CommonTools::dilate_erode(cloud_mask_raw, 2);
//    CommonTools::dilate_erode(cloud_mask_raw, 3);
//    CommonTools::draw_contour(cloud_mask_raw, cloud_mask_raw.clone(), cv::Scalar(255));

        cv::GaussianBlur(cloud_mask_raw, cloud_mask_raw, cv::Size(15, 15), 10, 10);
        cv::threshold(cloud_mask_raw, cloud_mask_raw, 10, 255, CV_THRESH_BINARY);
        CommonTools::draw_contour(cloud_mask_raw, cloud_mask_raw.clone(), cv::Scalar(255));

        return cloud_mask_raw;
    }

    static void center_of_mask(cv::Mat mask, cv::Point& p) {
        p.x = 0;
        p.y = 0;
        int count = 0;
        for (int row = 0; row < mask.rows; row++) {
            for (int col = 0; col < mask.cols; col++) {
                if (mask.at<uchar>(row, col) == 255) {
                    p.x += col;
                    p.y += row;
                    count++;
                }
            }
        }

        if (count == 0) {
            count = 1;
        }

        p.x /= count;
        p.y /= count;
    }

    static void draw_contour(cv::Mat& img, const cv::Mat& mask, cv::Scalar color, int thickness=-1) {
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
        int max_contour_size = 0;
        int max_contour_idx = -1;
        for (int i = 0; i < contours.size(); i++) {
            if (contours[i].size() > max_contour_size) {
                max_contour_size = contours[i].size();
                max_contour_idx = i;
            }
        }
        cv::drawContours(img, contours, max_contour_idx, color, thickness);
    }

    static double color_dist(cv::Vec3b color, cv::Vec3b color1){
        return abs(color[0] - color1[0]) + abs(color[1] - color1[1]) + abs(color[2] - color1[2]);
    }

    static void get_components(cv::Mat& I, std::vector<std::vector<cv::Point2i>>& components, int maxGradient = 1) {
        cv::Mat bChecked = cv::Mat::zeros(I.rows, I.cols, CV_8U);
        for(int r = 0; r < I.rows; r++) {
            for (int c = 0; c < I.cols; c++) {
                if (bChecked.at<uchar>(r, c) > 0) {
                    continue;
                }
                std::queue<cv::Point2i> q;
                q.push(cv::Point2i(c,r));
                bChecked.at<uchar>(r,c) = 1;
                int dx[4] = {-1,0,1,0};
                int dy[4] = {0,1,0,-1};
                cv::Vec3b color = I.at<cv::Vec3b>(r,c);
                std::vector<cv::Point2i> b;
                cv::Point2i current;
                while (!q.empty()) {
                    current = q.front();
                    b.push_back(cv::Point2i(current.x,current.y));
                    q.pop();
                    for (int i = 0; i < 4; i++) {
                        int nx = current.x+dx[i];
                        int ny = current.y+dy[i];
                        if (nx>=0 && nx<I.cols && ny>=0 && ny<I.rows) {
                            if (color_dist(color, I.at<cv::Vec3b>(ny,nx)) < maxGradient) {
                                if (bChecked.at<uchar>(ny, nx) > 0) {
                                    continue;
                                }
                                bChecked.at<uchar>(ny, nx) = 1;
                                q.push(cv::Point2i(nx,ny));
                            }
                        }
                    }
                }
                components.push_back(b);
            }
        }
    }


    static void draw_block(CloudPtr cloud, PointT p, double size) {
        for (double x = p.x - size/2; x < p.x + size/2; x += 0.01) {
            for (double y = p.y - size/2; y < p.y + size/2; y += 0.01) {
                for (double z = p.z - size/2; z < p.z + size/2; z += 0.01) {
                    PointT q(255, 0, 0);
                    q.x = x; q.y = y; q.z = z;
                    cloud->push_back(q);
                }
            }
        }
    }

    static int pix2vox(int* pixel2voxel, int row, int col, int width) {
        return pixel2voxel[col + row * width];
    }

    static int pix2vox(int* pixel2voxel, cv::Point2i p, int width) {
        return pixel2voxel[p.x + p.y * width];
    }

    static void vox2pix(int* voxel2pixel, int vox_idx, int& row, int& col, int width) {
        row = voxel2pixel[vox_idx] / width;
        col = voxel2pixel[vox_idx] % width;
    }

    static cv::Point2i vox2pix(int* voxel2pixel, int vox_idx, int width) {
        int y = voxel2pixel[vox_idx] / width;
        int x = voxel2pixel[vox_idx] % width;
        return cv::Point2i(x, y);
    }

    static double estimate_avg_dist(std::vector<int>& indices, pcl::RegionGrowing<PointT, pcl::Normal>& reg, PointT& midpoint, int samples) {
        double avg_dist;
        for (int i = 0; i < samples; i++) {
            int rand_idx = rand() % indices.size();
            PointT p = reg[indices[rand_idx]];
            avg_dist += pcl::euclideanDistance(p, midpoint);
        }
        avg_dist /= double(samples);
        return avg_dist;
    }

    static PointT estimate_normal(std::vector<int>& indices, pcl::PointCloud<::pcl::Normal>::Ptr normals, int samples) {
        PointT normal;
        for (int i = 0; i < samples; i++) {
            int rand_idx = rand() % indices.size();
            pcl::Normal n = normals->at(indices[rand_idx]);
            normal.x += n.normal_x;
            normal.y += n.normal_y;
            normal.z += n.normal_z;
        }
        normal.x = normal.x / double(samples);
        normal.y = normal.y / double(samples);
        normal.z = normal.z / double(samples);
        return normal;
    }

    static PointT estimate_normal_and_averages(std::vector<int>& indices, pcl::PointCloud<::pcl::Normal>::Ptr normals,
                                  pcl::RegionGrowing<PointT, pcl::Normal>& reg, PointT& normal, PointT& midpoint,
                                  cv::Scalar& avg_color, cv::Mat& mask, int* voxel2pixel, cv::Size size, int samples) {
        mask = cv::Mat::zeros(size, CV_8U);
        double largest_z = -99999;
        double smallest_z = 99999;
        std::vector<cv::Point2i> points;
        for (int i = 0; i < indices.size(); i++) {
            int rand_idx = i;
            int row, col;
            vox2pix(voxel2pixel, indices[rand_idx], row, col, size.width);
            points.push_back(cv::Point2i(col, row));
            pcl::Normal n = normals->at(indices[rand_idx]);
            normal.x += n.normal_x;
            normal.y += n.normal_y;
            normal.z += n.normal_z;
            // TODO check normals point same way (by dist)
            PointT p = reg[indices[rand_idx]];
            cv::Scalar color(p.r, p.g, p.b);
            avg_color += color;
            midpoint.x += p.x;
            midpoint.y += p.y;
            midpoint.z += p.z;
            if (p.z > largest_z) largest_z = p.z;
            if (p.z < smallest_z) smallest_z = p.z;
        }
        midpoint.x /= double(samples);
        midpoint.y /= double(samples);
        midpoint.z /= double(samples);
        //double alpha = 0.9;
        //midpoint.z = midpoint.z * (1 - alpha) + alpha * (largest_z + smallest_z) / 2.0;
        normal.x = normal.x / double(samples);
        normal.y = normal.y / double(samples);
        normal.z = normal.z / double(samples);
        avg_color[0] = avg_color[0] / double(samples);
        avg_color[1] = avg_color[1] / double(samples);
        avg_color[2] = avg_color[2] / double(samples);

        std::vector<cv::Point2i> hull;
        cv::convexHull(points, hull, true);
        std::vector<std::vector<cv::Point2i>> hulls;
        hulls.push_back(hull);
        cv::drawContours(mask, hulls, 0, cv::Scalar(255, 255, 255), -1);
    }

    static int sgn(int val) {
        return (0 < val) - (val < 0);
    }

    static int determinant(cv::Point2i vec1, cv::Point2i vec2) {
        return vec1.x * vec2.y - vec1.y * vec2.x;
    }

    static bool is_point_in_mask(cv::Point2i p, const cv::Mat& mask) {
        return is_row_col_in_mask(p.y, p.x, mask);
    }

    static bool is_row_col_in_mask(int row, int col, const cv::Mat& mask) {
        return mask.at<uchar>(row, col) > 0;
    }

    static cv::Mat threshold(cv::Mat img) {
        cv::Mat shirt_mask;
        cv::cvtColor(img, shirt_mask, CV_RGB2GRAY);
        cv::threshold(shirt_mask, shirt_mask, 250, 255, CV_THRESH_BINARY);
        shirt_mask = 255 - shirt_mask;
        return shirt_mask;
    }

    static bool check_mkdir(std::string path) {
      return boost::filesystem::create_directory(path);
    }

    static CloudPtr get_keypoints(CloudPtr cloud) {
        double cloud_resolution (0.0058329);
        //TreePtr tree (new TreePtr());
        pcl::search::Search<PointT>::Ptr tree = boost::shared_ptr<pcl::search::Search<PointT> > (new pcl::search::KdTree<PointT>);
        Cloud keypoints;

        pcl::ISSKeypoint3D<PointT, PointT> iss_detector;
        iss_detector.setSearchMethod (tree);
        iss_detector.setSalientRadius (6 * cloud_resolution);
        iss_detector.setNonMaxRadius (4 * cloud_resolution);

        iss_detector.setThreshold21 (0.975);
        iss_detector.setThreshold32 (0.975);
        iss_detector.setMinNeighbors (5);
        iss_detector.setNumberOfThreads (1);
        iss_detector.setInputCloud (cloud);
        iss_detector.compute (keypoints);

        for (int i = 0; i < keypoints.points.size(); i++) {
            keypoints.points[i].r = 0;
            keypoints.points[i].g = 255;
            keypoints.points[i].b = 0;
        }

        CloudPtr keypoints_ptr = CloudPtr(new Cloud(keypoints));
        return keypoints_ptr;
    }

  static CloudPtr get_segments_mincut(CloudPtr cloud_ptr) {
      pcl::search::Search <PointT>::Ptr tree =
          boost::shared_ptr<pcl::search::Search<PointT> > (new pcl::search::KdTree<PointT>);

      pcl::IndicesPtr indices (new std::vector <int>);
      pcl::PassThrough<PointT> pass;
      pass.setInputCloud (cloud_ptr);
      pass.setFilterFieldName ("z");
      pass.setFilterLimits (0.0, 1.0);
      pass.filter (*indices);

      pcl::MinCutSegmentation<PointT> seg;
      seg.setInputCloud (cloud_ptr);
      seg.setIndices (indices);

      pcl::PointCloud<PointT>::Ptr foreground_points(new pcl::PointCloud<PointT> ());
      PointT point;
      pcl::computeCentroid<PointT, PointT>(*cloud_ptr, point);
      std::cout << "centroid: " << point << std::endl;
      foreground_points->points.push_back(point);
      seg.setForegroundPoints (foreground_points);

      seg.setSigma (0.25);
      seg.setRadius (3.0433856);
      seg.setNumberOfNeighbours (14);
      seg.setSourceWeight (0.8);

      std::vector <pcl::PointIndices> clusters;
      seg.extract (clusters);

      std::cout << "Maximum flow is " << seg.getMaxFlow () << std::endl;

      pcl::PointCloud <PointT>::Ptr colored_cloud = seg.getColoredCloud ();
      return colored_cloud;
  }

    static CloudPtr get_segments_color(CloudPtr cloud_ptr) {
        pcl::search::Search <PointT>::Ptr tree =
            boost::shared_ptr<pcl::search::Search<PointT> > (new pcl::search::KdTree<PointT>);

        pcl::IndicesPtr indices (new std::vector <int>);
        pcl::PassThrough<PointT> pass;
        pass.setInputCloud (cloud_ptr->makeShared());
        pass.setFilterFieldName ("z");
        pass.setFilterLimits (0.0, 1.0);
        pass.filter (*indices);

        pcl::RegionGrowingRGB<PointT> reg;
        reg.setInputCloud (cloud_ptr->makeShared());
        reg.setIndices (indices);
        reg.setSearchMethod (tree);
        reg.setDistanceThreshold (10);
        reg.setPointColorThreshold (6);
        reg.setRegionColorThreshold (5);
        reg.setMinClusterSize (600);

        std::vector <pcl::PointIndices> clusters;
        reg.extract (clusters);

        CloudPtr colored_cloud = reg.getColoredCloud ();
//        pcl::visualization::CloudViewer viewer ("Cluster viewer");
//        viewer.showCloud (colored_cloud);
//        while (!viewer.wasStopped ())
//        {
//          boost::this_thread::sleep (boost::posix_time::microseconds (100));
//        }
        return colored_cloud;
    }

    static CloudPtr remove_biggest_plane(CloudPtr cloud_ptr) {

        std::cout << "Finding biggest plane in the pointcloud..." << std::endl;
        CloudPtr remaining_cloud_ptr = CloudPtr(new Cloud);

        cv::Size size(cloud_ptr->width, cloud_ptr->height);

        pcl::search::Search<PointT>::Ptr tree =
            boost::shared_ptr<pcl::search::Search<PointT> >(new pcl::search::KdTree<PointT>);
        pcl::PointCloud <pcl::Normal>::Ptr normals(new pcl::PointCloud <pcl::Normal>);
        pcl::NormalEstimation<PointT, pcl::Normal> normal_estimator;
        normal_estimator.setSearchMethod(tree);
        normal_estimator.setInputCloud(cloud_ptr->makeShared());
        normal_estimator.setKSearch(50);
        normal_estimator.compute(*normals);

        pcl::IndicesPtr indices(new std::vector <int>);
        pcl::PassThrough<PointT> pass;
        pass.setInputCloud(cloud_ptr->makeShared());
        pass.setFilterFieldName("z");
        pass.setFilterLimits(0.0, 1.0);
        pass.filter(*indices);

        pcl::RegionGrowing<PointT, pcl::Normal> reg;
        reg.setMinClusterSize(50);
        reg.setMaxClusterSize(1000000);
        reg.setSearchMethod(tree);
        reg.setNumberOfNeighbours(60);
        reg.setInputCloud(cloud_ptr->makeShared());
        //reg.setIndices (indices);
        reg.setInputNormals (normals);
        reg.setSmoothnessThreshold(3.0 / 180.0 * M_PI);
        reg.setCurvatureThreshold(1.0);

        std::vector <pcl::PointIndices> clusters;
        reg.extract (clusters);

        int max_cluster_size = 0;
        if (clusters.size() > 0) {
            int max_cluster_idx = -1;
            for (int cluster_idx = 0; cluster_idx < clusters.size(); cluster_idx++) {
              // calculate average normal of each cluster, and make sure y is dominant
              PointT cluster_normal = CommonTools::estimate_normal(clusters[cluster_idx].indices, normals, 20);
              double y_ratio = fabs(cluster_normal.y) / (fabs(cluster_normal.x) + fabs(cluster_normal.y) + fabs(cluster_normal.z));
              if (clusters[cluster_idx].indices.size() > max_cluster_size && y_ratio > 0.3) {
                max_cluster_size = clusters[cluster_idx].indices.size();
                max_cluster_idx = cluster_idx;
              }
            }
            if (max_cluster_idx < 0) {
              std::cout << "No table found" << std::endl;
              return remaining_cloud_ptr;
            } else {
              // Add points to the cloud that are not on the plane
              for (int i, j = 0; i < cloud_ptr->size(); i++) {
                if (i == clusters[max_cluster_idx].indices[j]) {
                  j++;
                } else {
                  remaining_cloud_ptr->push_back(cloud_ptr->at(i));
                }
              }
            }
        }

        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud(remaining_cloud_ptr);
        sor.setMeanK(50);
        sor.setStddevMulThresh(0.01);
        sor.filter(*remaining_cloud_ptr);

        return remaining_cloud_ptr;
    }


    static bool find_biggest_plane(CloudPtr cloud_ptr,
                                   int* voxel2pixel,
				   cv::Size size,
                                   cv::Mat& mask,
                                   PointT& midpoint,
                                   PointT& normal) {
        std::cout << "Finding biggest plane in the pointcloud..." << std::endl;

        pcl::search::Search<PointT>::Ptr tree =
            boost::shared_ptr<pcl::search::Search<PointT> >(new pcl::search::KdTree<PointT>);
        pcl::PointCloud <pcl::Normal>::Ptr normals(new pcl::PointCloud <pcl::Normal>);
        pcl::NormalEstimation<PointT, pcl::Normal> normal_estimator;
        normal_estimator.setSearchMethod(tree);
        normal_estimator.setInputCloud(cloud_ptr->makeShared());
        normal_estimator.setKSearch(50);
        normal_estimator.compute(*normals);

        pcl::IndicesPtr indices(new std::vector <int>);
        pcl::PassThrough<PointT> pass;
        pass.setInputCloud(cloud_ptr->makeShared());
        pass.setFilterFieldName("z");
        pass.setFilterLimits(0.0, 1.0);
        pass.filter(*indices);

        pcl::RegionGrowing<PointT, pcl::Normal> reg;
        reg.setMinClusterSize(50);
        reg.setMaxClusterSize(1000000);
        reg.setSearchMethod(tree);
        reg.setNumberOfNeighbours(30);
        reg.setInputCloud(cloud_ptr->makeShared());
        //reg.setIndices (indices);
        reg.setInputNormals (normals);
        reg.setSmoothnessThreshold(3.0 / 180.0 * M_PI);
        reg.setCurvatureThreshold(1.0);

        std::vector <pcl::PointIndices> clusters;
        reg.extract (clusters);


        int max_cluster_size = 0;
        if (clusters.size() > 0) {
            int max_cluster_idx = -1;
            for (int cluster_idx = 0; cluster_idx < clusters.size(); cluster_idx++) {
                // calculate average normal of each cluster, and make sure y is dominant
                PointT cluster_normal = CommonTools::estimate_normal(clusters[cluster_idx].indices, normals, 20);
                double y_ratio = fabs(cluster_normal.y) / (fabs(cluster_normal.x) + fabs(cluster_normal.y) + fabs(cluster_normal.z));
                if (clusters[cluster_idx].indices.size() > max_cluster_size && y_ratio > 0.3) {
                    max_cluster_size = clusters[cluster_idx].indices.size();
                    max_cluster_idx = cluster_idx;
                    normal = cluster_normal;
                }
            }
            if (max_cluster_idx < 0) {
                std::cout << "No table found" << std::endl;
                return false;
            } else {
                std::vector<cv::Point2i> plane_points;
                midpoint.x = 0; midpoint.y = 0; midpoint.z = 0;
                for (int i = 0; i < clusters[max_cluster_idx].indices.size(); i++) {
                    int row, col;
                    vox2pix(voxel2pixel, clusters[max_cluster_idx].indices[i], row, col, cloud_ptr->width);
                    plane_points.push_back(cv::Point2i(col, row));
                    PointT p = cloud_ptr->at(clusters[max_cluster_idx].indices[i]);
                    midpoint.x += p.x;
                    midpoint.y += p.y;
                    midpoint.z += p.z;
//                    PointT p = cloud_ptr->at(clusters[max_cluster_idx].indices[i]);
//                    plane_cloud->push_back(p);
                }
                midpoint.x /= plane_points.size();
                midpoint.y /= plane_points.size();
                midpoint.z /= plane_points.size();
                std::vector<cv::Point2i> hull;
                cv::convexHull(plane_points, hull, true);
                std::vector<std::vector<cv::Point2i>> hulls;
                hulls.push_back(hull);
                mask = cv::Mat::zeros(size, CV_8U);
                cv::drawContours(mask, hulls, 0, cv::Scalar(255, 255, 255), -1);
                return true;
            }
        }
        std::cout << "No planes found" << std::endl;
        return false;
    }

    static PointT get_3d_approx(cv::Point p, cv::Size size, int* pixel2voxel, CloudPtr cloud_ptr) {
      int n = 5;  // neighborhood
      int start_col = cv::max(p.x - n, 0);
      int end_col = cv::min(p.x + n, size.width - 1);
      int start_row = cv::max(p.y - n, 0);
      int end_row = cv::min(p.y + n, size.height - 1);
      PointT pt3d;
      pt3d.x = -999;
      pt3d.y = -999;
      pt3d.z = -999;
      double closest_dist = 999999;
      for (int row = start_row; row < end_row; row++) {
        for (int col = start_col; col < end_col; col++) {
          int cloud_idx = CommonTools::pix2vox(pixel2voxel, cv::Point(col, row), size.width);
          if (cloud_idx >= 0) {
            cv::Point test_point(col, row);
            double dist = cv::norm(test_point - p);
            if (dist < closest_dist) {
              closest_dist = dist;
              pt3d = cloud_ptr->at(cloud_idx);
            }
          }
        }
      }
      return pt3d;
    }

    static bool get_cloud_info(pcl::PointCloud<pcl::PointXYZRGB>& cloud,
                               cv::Mat& img,
                               int* pixel2voxel,
                               int* voxel2pixel) {
        std::vector<pcl::PCLPointField> fields;
        int field_idx = pcl::getFieldIndex (cloud, "rgb", fields);
        if (field_idx == -1) {
            field_idx = pcl::getFieldIndex (cloud, "rgba", fields);
            if (field_idx == -1)
                return false;
        }
        const size_t offset = fields[field_idx].offset;

        img = cv::Mat(cloud.height, cloud.width, CV_8UC3);


        for (size_t i = 0; i < cloud.points.size (); ++i) {
            if (!pcl::isFinite(cloud[i])) {
                cv::Vec3b color_vec(0, 0, 0);
                img.at<cv::Vec3b>(i) = color_vec;
                pixel2voxel[i] = -1;
                voxel2pixel[i] = -1;
                cloud[i].x = 10;
                cloud[i].y = 10;
                cloud[i].z = 10;
            } else {
                uint32_t val;
                pcl::getFieldValue<PointT, uint32_t> (cloud.points[i], offset, val);
                cv::Vec3b color_vec((val) & 0x0000ff, (val >> 8) & 0x0000ff, (val >> 16) & 0x0000ff);
                img.at<cv::Vec3b>(i) = color_vec;
                pixel2voxel[i] = i;
                voxel2pixel[i] = i;
            }
        }

        return true;
    }

    static CloudPtr get_pointcloud_from_mask(CloudPtr cloud_ptr, int* pixel2voxel, cv::Mat mask, bool dense=false) {
        CloudPtr mask_cloud_ptr = CloudPtr(new Cloud);
        PointT dummy_point;
        dummy_point.x = 0;
        dummy_point.y = 0;
        dummy_point.z = -999;
        for (int i = 0; i < mask.size().area(); i++) {
            if (mask.at<uchar>(i) == 255) {
                if (pixel2voxel[i] >= 0) {
                  PointT p = cloud_ptr->at(pixel2voxel[i]);
                  mask_cloud_ptr->push_back(p);
                } else if (dense) mask_cloud_ptr->push_back(dummy_point);
            } else if (dense) mask_cloud_ptr->push_back(dummy_point);
        }
        return mask_cloud_ptr;
    }

    static void get_cloud_projections(CloudPtr cloud, int* voxel2pixel,
                                      cv::Mat& img_bgr, cv::Mat& img_x,
                                      cv::Mat& img_y, cv::Mat& img_z) {
      img_bgr = cv::Mat::zeros(cloud->height, cloud->width, CV_8UC3);
      img_x = cv::Mat::zeros(cloud->height, cloud->width, CV_16U);
      img_y = cv::Mat::zeros(cloud->height, cloud->width, CV_16U);
      img_z = cv::Mat::zeros(cloud->height, cloud->width, CV_16U);
      for (int i = 0; i < cloud->size(); i++) {
        int row = i / cloud->width;
        int col = i % cloud->width;
        PointT p = cloud->at(i);
        img_x.at<ushort>(row, col) = ushort(p.x * 1000.0 + 10000.0);
        img_y.at<ushort>(row, col) = ushort(p.y * 1000.0 + 10000.0);
        img_z.at<ushort>(row, col) = ushort(p.z * 1000.0 + 10000.0);
        img_bgr.at<cv::Vec3b>(row, col) = cv::Vec3b(p.b, p.g, p.r);
      }
    }

    static CloudPtr make_cloud_ptr(cv::Mat img_bgr,
                                   cv::Mat img_x,
                                   cv::Mat img_y,
                                   cv::Mat img_z,
                                   int* pixel2voxel,
                                   int* voxel2pixel) {
        CloudPtr cloud_ptr = CloudPtr(new Cloud);
        cloud_ptr->header.frame_id = "robot";
        cloud_ptr->is_dense = true;
        uchar* img_bgr_ptr = img_bgr.data;
        uint16_t* x_ptr = (uint16_t*)img_x.data;
        uint16_t* y_ptr = (uint16_t*)img_y.data;
        uint16_t* z_ptr = (uint16_t*)img_z.data;

        float max_x = -1;
        float max_y = -1;
        float max_z = -1;
        float min_x = 999999;
        float min_y = 999999;
        float min_z = 999999;

        int img_idx = 0;
        for (int row = 0; row < img_bgr.rows; row++) {
            for (int col = 0; col < img_bgr.cols; col++) {
                int b = img_bgr_ptr[row*img_bgr.cols*3 + col*3 + 0];
                int g = img_bgr_ptr[row*img_bgr.cols*3 + col*3 + 1];
                int r = img_bgr_ptr[row*img_bgr.cols*3 + col*3 + 2];
                float x = x_ptr[row*img_bgr.cols + col];
                float y = y_ptr[row*img_bgr.cols + col];
                float z = z_ptr[row*img_bgr.cols + col];

                x = (x - 10000.0)/1000.0;
                y = (y - 10000.0)/1000.0;
                z = (z - 10000.0)/1000.0;


                if (x > max_x) max_x = x;
                if (y > max_y) max_y = y;
                if (z > max_z) max_z = z;
                if (x < min_x && x !=0) min_x = x;
                if (y < min_y && y != 0) min_y = y;
                if (z < min_z && z != 0) min_z = z;

                PointT p;
                uint32_t rgb = (static_cast<uint32_t>(r) << 16 |
                                static_cast<uint32_t>(g) << 8 |
                                static_cast<uint32_t>(b));
                p.rgb = *reinterpret_cast<float*>(&rgb);
                p.x = x;
                p.y = y;
                if (z > 1e-3 && z < 3) {
                    pixel2voxel[img_idx] = cloud_ptr->size();
                    voxel2pixel[cloud_ptr->size()] = img_idx;
                    p.z = z;
                    cloud_ptr->push_back(p);
                } else {
                    pixel2voxel[img_idx] = -1;
                    p.z = -10; //std::numeric_limits<float>::quiet_NaN();
                    //cloud_ptr->push_back(p);
                }
                img_idx++;
                //std::numeric_limits<float>::quiet_NaN();
            }
        }

//        cout << "x range: " << min_x << " --> " << max_x << endl;
//        cout << "y range: " << min_y << " --> " << max_y << endl;
//        cout << "z range: " << min_z << " --> " << max_z << endl;

        cloud_ptr->width = img_bgr.cols;
        cloud_ptr->height = img_bgr.rows;
//        cout << "cloud_ptr organized: " << cloud_ptr->isOrganized() << endl;
        return cloud_ptr;
    }
};

#endif //ROBOTCLOTHFOLDING_COMMONTOOLS_H


