//
// Created by binroot on 6/30/15.
//

#ifndef ROBOTCLOTHFOLDING_COMMONTOOLS_H
#define ROBOTCLOTHFOLDING_COMMONTOOLS_H

#include <pcl/common/common_headers.h>
// #include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/keypoints/iss_3d.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/common/distances.h>
#include "CurveCSS.h"
#include "CurveSignature.h"
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

    static PointT get_3d_approx(Point p, cv::Size size, int* pixel2voxel, CloudPtr cloud_ptr) {
        int n = 5;  // neighborhood
        int start_col = max(p.x - n, 0);
        int end_col = min(p.x + n, size.width - 1);
        int start_row = max(p.y - n, 0);
        int end_row = min(p.y + n, size.height - 1);
        PointT pt3d;
        pt3d.x = -999;
        pt3d.y = -999;
        pt3d.z = -999;
        double closest_dist = 999999;
        for (int row = start_row; row < end_row; row++) {
            for (int col = start_col; col < end_col; col++) {
                int cloud_idx = CommonTools::pix2vox(pixel2voxel, Point(col, row), size.width);
                if (cloud_idx >= 0) {
                    Point test_point(col, row);
                    double dist = norm(test_point - p);
                    if (dist < closest_dist) {
                        closest_dist = dist;
                        pt3d = cloud_ptr->at(cloud_idx);
                    }
                }
            }
        }
        return pt3d;
    }

    static int get_3d_approx(Point p, cv::Size size, int* pixel2voxel) {
        int start_col = max(p.x - 10, 0);
        int end_col = min(p.x + 10, size.width - 1);
        int start_row = max(p.y - 10, 0);
        int end_row = min(p.y + 10, size.height - 1);
        int best_cloud_idx = -1;
        double closest_dist = 999999;
        for (int row = start_row; row < end_row; row++) {
            for (int col = start_col; col < end_col; col++) {
                int cloud_idx = CommonTools::pix2vox(pixel2voxel, Point(col, row), size.width);
                if (cloud_idx >= 0) {
                    Point test_point(col, row);
                    double dist = norm(test_point - p);
                    if (dist < closest_dist) {
                        closest_dist = dist;
                        best_cloud_idx = cloud_idx;
                    }
                }
            }
        }
        return best_cloud_idx;
    }

    static cv::Mat get_mask_from_points(std::vector<cv::Point>& points) {
        cv::Rect outer_rect = boundingRect(points);
        cv::Mat mask(outer_rect.size(), CV_8U);
        std::vector<std::vector<cv::Point> > pointss(1, points);
        cv::drawContours(mask, pointss, 0, cv::Scalar::all(255));
        return mask;
    }

    static std::vector<cv::Point2f> rotate_and_transform_back(std::vector<cv::Point2f> _points, double rotation_angle, cv::Mat mask, cv::Mat inv_transform) {
        std::vector<cv::Point2f> points(_points.size());
        for (int i = 0; i < points.size(); i++)
            points[i] = _points[i];
        Mat rot_trans = get_rotation_matrix(mask, -rotation_angle);
        transform(points, points, rot_trans);
        perspectiveTransform(points, points, inv_transform);
        return points;
    }

    /**
     * rotate, then perspective transform
     */
    static std::vector<int> get_3d(std::vector<cv::Point2f> _points, double rotation_angle, cv::Mat mask, cv::Mat inv_transform, int* pixel2voxel) {
        std::vector<cv::Point2f> points(_points.size());
        for (int i = 0; i < points.size(); i++)
            points[i] = _points[i];
        Mat rot_trans = get_rotation_matrix(mask, -rotation_angle);
        cout << "rot trans: " << rot_trans << endl;
        transform(points, points, rot_trans);
        perspectiveTransform(points, points, inv_transform);
        std::vector<int> cloud_idxs(points.size());
        for (int i = 0; i < points.size(); i++) {
            cloud_idxs[i] = get_3d_approx(points[i], mask.size(), pixel2voxel);
        }
        return cloud_idxs;
    }

    static cv::Mat spatial_features_viz(cv::Mat shape_features_, cv::Mat line_params) {

        vector<float> features(shape_features_.rows * shape_features_.cols);
        for (int i = 0; i < features.size(); i++)
            features[i] = shape_features_.at<float>(i);
        unnormalize_spatial_features(features);

        double height = 200;

        double h1 = 100;
        double w1 = features[0] * h1;
        double w2 = features[1] * h1;
        double h2 = features[2] * h1;
        double dx = features[3] * h1;
        double dy = features[4] * h1;

        /*
         * feature[3] = ((outer_rect.x + outer_rect.width/2) - (inner_rect.x + inner_rect.width/2)) / double(outer_rect.height);
         *   feature[3] = outer_center_x - inner_center_x
         * feature[4] = ((outer_rect.y + outer_rect.height/2) - (inner_rect.y + inner_rect.height/2)) / double(outer_rect.height);
         */

        cv::Mat img = cv::Mat::zeros(height, w1*2, CV_8UC3);
        int center_x = img.cols / 2;
        int center_y = img.rows / 2;
        Rect outer_rect(center_x - w1/2, center_y - h1/2, w1, h1);
        cout << "drawing outer_rect" << endl;
        rectangle(img, outer_rect, Scalar(255, 255, 255), 2);
        Rect inner_rect(center_x - dx - w2/2, center_y - dy - h2/2, w2, h2);
        cout << "drawing inner_rect" << endl;
        rectangle(img, inner_rect, Scalar(0, 0, 255), 2);


        /*
         * line_mat.at<float>(2) = (m_line_x - inner_rect.x) / double(inner_rect.width);
         * line_mat.at<float>(3) = (m_line_y - inner_rect.y) / double(inner_rect.height);
         */

        double line_dx = line_params.at<float>(0);
        double line_dy = line_params.at<float>(1);
        double line_x = line_params.at<float>(2) * inner_rect.width + inner_rect.x;
        double line_y = line_params.at<float>(3) * inner_rect.height + inner_rect.y;
        double line_scale = 50;
        cv::Point line_p1(line_x - line_scale*line_dx, line_y - line_scale * line_dy);
        cv::Point line_p2(line_x + line_scale * line_dx, line_y + line_scale * line_dy);
        line(img, line_p1, line_p2, Scalar(0, 255, 0), 5);

        if (DEBUG) imshow("fold spatial viz 2", img);

        return img;
    }

    static cv::Mat spatial_features_viz(std::vector<float> features_) {

        vector<float> features(features_.size());
        for (int i = 0; i < features_.size(); i++)
            features[i] = features_[i];
        unnormalize_spatial_features(features);

        cout << "w1: " << features[0] << endl;
        cout << "w2: " << features[1] << endl;
        cout << "h2: " << features[2] << endl;
        cout << "dx: " << features[3] << endl;
        cout << "dy: " << features[4] << endl;

        double h1 = 100;
        double w1 = features[0] * h1;
        double w2 = features[1] * h1;
        double h2 = features[2] * h1;
        double dx = features[3] * h1;
        double dy = features[4] * h1;

        cv::Mat img = cv::Mat::zeros(h1*2, w1*2, CV_8UC3);
        int center_x = img.cols / 2;
        int center_y = img.rows / 2;
        Rect outer_rect(center_x - w1/2, center_y - h1/2, w1, h1);
        cout << "drawing outer_rect" << endl;
        rectangle(img, outer_rect, Scalar(255, 255, 255), 4);
        Rect inner_rect(center_x - dx - w2/2, center_y - dy - h2/2, w2, h2);
        cout << "drawing inner_rect" << endl;
        rectangle(img, inner_rect, Scalar(0, 0, 255), 4);

        return img;
    }

    static std::vector<float> get_spatial_features(cv::Mat& cloth_mask, cv::Mat& table_mask) {
        Rect outer_rect = CommonTools::get_outer_rect(cloth_mask);
        Rect inner_rect = CommonTools::get_inner_rect(cloth_mask);
        double table_area = cv::countNonZero(table_mask);
        double mat_scale = cv::countNonZero(cloth_mask) / table_area;
        return get_rect_feature(cloth_mask, outer_rect, inner_rect, mat_scale);

    }

    static std::vector<float> get_spatial_features(cv::Mat& cloth_mask, cv::Mat& table_mask, float ratio) {
        Rect outer_rect = CommonTools::get_outer_rect(cloth_mask);
        Rect inner_rect = CommonTools::get_inner_rect(cloth_mask);

        double table_area = cv::countNonZero(table_mask);
        double mat_scale = cv::countNonZero(cloth_mask) / table_area;
        return get_rect_feature(cloth_mask, outer_rect, inner_rect, mat_scale);

    }

    static cv::Mat to_mat(std::vector<float> vals) {
        cv::Mat m(vals.size(), 1, CV_32F);
        for (int i = 0; i < vals.size(); i++)
            m.at<float>(i) = vals[i];
        return m;
    }

    static void save_data(std::string dir, std::vector<cv::Mat> data) {
        if (!is_directory(dir)) {
            create_directory(dir);
        } else {
            remove_all(dir);
            create_directory(dir);
        }

        for (int i = 0; i < data.size(); i++) {
            cv::FileStorage storage(dir + to_string(i) + ".yml", cv::FileStorage::WRITE);
            storage << "data" << data[i];
            storage.release();
        }
    }

    static std::vector<cv::Mat> load_data(std::string dir) {
        std::vector<cv::Mat> data;
        if (!is_directory(dir)) {
            cout << "Err: " <<  dir << " not found" << endl;
            return data;
        }

        for (int i = 0; i >= 0; i++) {
            string data_filepath = dir + to_string(i) + ".yml";
            if (is_regular_file(data_filepath)) {
                Mat datum;
                cv::FileStorage storage(data_filepath, cv::FileStorage::READ);
                storage["data"] >> datum;
                storage.release();
                data.push_back(datum);
            } else break;
        }
        return data;
    }

    static Mat draw_mask(cv::Mat img, cv::Mat mask, Scalar color) {
        double color_strength = 0.9;
        cv::Mat display = img.clone();
        for (int row = 0; row < img.rows; row++) {
            for (int col = 0; col < img.cols; col++) {
                if (is_row_col_in_mask(row, col, mask)) {
                    display.at<Vec3b>(row, col)[0] = img.at<Vec3b>(row, col)[0]*(1-color_strength) + color[0]*color_strength;
                    display.at<Vec3b>(row, col)[1] = img.at<Vec3b>(row, col)[1]*(1-color_strength) + color[1]*color_strength;
                    display.at<Vec3b>(row, col)[2] = img.at<Vec3b>(row, col)[2]*(1-color_strength) + color[2]*color_strength;
                }
            }
        }
        return display;
    }

    static void detect_tshirt_keypoints(cv::Mat& img_, cv::Mat& mask_, double scale, double& score, double& orientation, std::vector<cv::Point>& keypoints) {
        Mat img = img_.clone();
        Mat mask = mask_.clone();
        orientation = 0;
        vector<Point> mask_pts;
        CommonTools::max_contour(mask, mask_pts);

        Rect outer_rect = boundingRect(mask_pts);
        double calculated_scale = max(outer_rect.width, outer_rect.height) / 99.0;
        cout << "calculated scale: " << calculated_scale << endl;

        double orientation_kp;
//        imshow("img_rot", img);
//        imshow("mask_rot", mask);

        string img_filename = ROS_PATH + "out/kp_img.png";
        string mask_filename = ROS_PATH + "out/kp_mask.png";
        imwrite(img_filename, img);
        imwrite(mask_filename, mask);

        string cmd = "./" + ROS_PATH + "keypoint/ClothesFolding " + img_filename + " " + mask_filename + " " + to_string(calculated_scale);
        cout << "running " << cmd << endl;
        FILE *lsofFile_p = popen(cmd.c_str(), "r");
        char buffer[2048];
        char *line_p = fgets(buffer, sizeof(buffer), lsofFile_p);
        pclose(lsofFile_p);

        std::ifstream data(ROS_PATH + "keypoints.csv");
        std::string line_str;
        int line_num = 0;
        while(std::getline(data, line_str)) {
            if (line_num == 0) {
                score = atof(line_str.c_str());
                cout << "score: " << score << endl;
            } else if (line_num == 1) {
                double o = atof(line_str.c_str());
                orientation += o * 360.0 / 16.0;
                orientation_kp = o * 360.0 / 16.0;
                cout << "orientation: " << o << endl;
            } else {
                std::stringstream  lineStream(line_str);
                std::string        cell;
                int cell_idx = 0;
                Point keypoint;
                while(std::getline(lineStream, cell, ',')) {
                    if (cell_idx == 0) keypoint.x = atof(cell.c_str());
                    else if (cell_idx == 1) keypoint.y = atof(cell.c_str());
                    cell_idx++;
                }
                keypoints.push_back(keypoint);
            }
            line_num++;
        }
//        for (auto& p : keypoints) {
//            cout << p << endl;
//            circle(img, p, 3, Scalar(0, 0, 255), 2);
//        }
//        imshow("keypoints output", img);
//        waitKey(0);

        // rotate all the keypoints too
        Mat rot_trans = CommonTools::get_rotation_matrix(mask, orientation_kp);
        transform(keypoints, keypoints, rot_trans);
    }


    static cv::Mat watershed_mask(Mat& img, vector<vector<cv::Point>> foreground_contours, Mat& confident_background_mask) {

        // Create the marker image for the watershed algorithm
        cv::Mat markers = cv::Mat::zeros(img.size(), CV_32SC1);
        cv::drawContours(markers, foreground_contours, 0, cv::Scalar::all(1), -1);

        // Draw the background marker
        cv::Mat outside_table_region = cv::Mat::zeros(img.size(), CV_8U);
        for (int row = 0; row < img.rows; row++) {
            for (int col = 0; col < img.cols; col++) {
                if (CommonTools::is_row_col_in_mask(row, col, confident_background_mask)) {
                    outside_table_region.at<uchar>(row, col) = 255;
                }
            }
        }
        CommonTools::dilate(outside_table_region, 2);
        for (int row = 0; row < img.rows; row++) {
            for (int col = 0; col < img.cols; col++) {
                if (CommonTools::is_row_col_in_mask(row, col, outside_table_region)) {
                    cv::circle(markers, cv::Point(col, row), 1, CV_RGB(255, 255, 255), -1);
                }
            }
        }
        //cv::imshow("markers", markers * 10000);

        cv::watershed(img, markers);
        cv::Mat new_cloth_mask = cv::Mat::zeros(img.size(), CV_8U);
        // Fill labeled objects with random colors
        for (int row = 0; row < markers.rows; row++) {
            for (int col = 0; col < markers.cols; col++) {
                int index = markers.at<int>(row, col);
                if (index == 1) {
                    new_cloth_mask.at<uchar>(row, col) = 255;
                } else {
                    new_cloth_mask.at<uchar>(row, col) = 0;
                }
            }
        }
        //imshow("watershed", new_cloth_mask);

        return new_cloth_mask;
    }

    static void match_curves(const cv::Mat& mask_a, const cv::Mat& mask_b, double& score, cv::Mat& mask_a_out) {
        std::vector<cv::Point> a, b;
        GetCurveForImage(mask_a, a, false);
        if (a.empty()) {
            cout << "curve for mask_a is empty" << endl;
            return;
        }
        ResampleCurve(a, a, 200, false);
        GetCurveForImage(mask_b, b, false);
        if (b.empty()) {
            cout << "curve for mask_b is empty" << endl;
            return;
        }
        ResampleCurve(b, b, 200, false);

        int a_len, a_off, b_len, b_off;
        CompareCurvesUsingSignatureDB(a,
                                      b,
                                      a_len,
                                      a_off,
                                      b_len,
                                      b_off,
                                      score
        );


        //Get matched subsets of curves
        std::vector<cv::Point> a_subset(a.begin() + a_off, a.begin() + a_off + a_len);
        std::vector<cv::Point> b_subset(b.begin() + b_off, b.begin() + b_off + b_len);

        //Normalize to equal length
        ResampleCurve(a_subset, a_subset, 200, true);
        ResampleCurve(b_subset, b_subset, 200, true);

        //Prepare the curves for finding the transformation
        std::vector<cv::Point2f> seq_a_32f, seq_b_32f, seq_a_32f_, seq_b_32f_;

        ConvertCurve(a_subset, seq_a_32f_);
        ConvertCurve(b_subset, seq_b_32f_);

        assert(seq_a_32f_.size() == seq_b_32f_.size());

        seq_a_32f.clear(); seq_b_32f.clear();
        for (int i=0; i<seq_a_32f_.size(); i++) {
            //		if(i%2 == 0) { // you can use only part of the points to find the transformation
            seq_a_32f.push_back(seq_a_32f_[i]);
            seq_b_32f.push_back(seq_b_32f_[i]);
            //		}
        }
        assert(seq_a_32f.size() == seq_b_32f.size()); //just making sure

        vector<Point2d> seq_a_trans(a.size());

        //Find the fitting transformation
        //	Mat affineT = estimateRigidTransform(seq_a_32f,seq_b_32f,false); //may wanna use Affine here..
        Mat trans = Find2DRigidTransform(seq_a_32f, seq_b_32f);
        std::vector<cv::Point2d> a_p2d;
        ConvertCurve(a, a_p2d);
        cv::transform(a_p2d, seq_a_trans, trans);

        std::vector<cv::Point2d> b_p2d;
        ConvertCurve(b, b_p2d);
        cv::warpAffine(mask_a, mask_a_out, trans, mask_a.size());

//        Mat visual = Mat::zeros(mask_a.size(), CV_8UC3);
//        drawOpenCurve(visual, seq_a_trans, Scalar(0, 255, 0), 1);
//        drawOpenCurve(visual, b_p2d, Scalar(255, 0, 0), 1);
//
//        cv::transform(seq_a_32f,  seq_a_32f, trans);
//        for (int i = 0; i < seq_a_32f.size(); i++) {
//            line(visual, seq_a_32f[i], seq_b_32f[i], Scalar(0,0,255), 1);
//        }
//        imshow("match_visual", visual);
    }

    static void normalize_features(std::vector<std::vector<float> >& features) {
        if (features.empty()) {
            return;
        }

        for (int d = 0; d < features[0].size(); d++) {
            vector<float> vals;
            for (int n = 0; n < features.size(); n++) {
                vals.push_back(features[n][d]);
            }
            double sum = std::accumulate(vals.begin(), vals.end(), 0.0);
            double mean = sum / vals.size();
            double sq_sum = std::inner_product(vals.begin(), vals.end(), vals.begin(), 0.0);
            double stdev = std::sqrt(sq_sum / vals.size() - mean * mean);
            cout << d << ":: " << "mean: " << mean << ", stdev: " << stdev << endl;
            for (int n = 0; n < features.size(); n++) {
                features[n][d] = (features[n][d] - mean) / stdev;
            }
        }
    }

    static void adjust_mask(cv::Mat& mask, std::vector<cv::Point>& mask_pts, double& angle) {
        Rect mask_rect = boundingRect(mask_pts);
        Mat mask_cropped = mask(mask_rect);
        double scale = 4;
        Size mask_small(mask.size().width / scale, mask.size().height / scale);
        resize(mask_cropped, mask_cropped, mask_small);

        double max_rect_score = -1;
        angle = -1;
        for (int i = 0; i < 180; i+=1) {
            Mat mask1_rot;
            rotate(mask_cropped, i, mask1_rot);
            Rect outer_rect_ = get_outer_rect(mask1_rot);
            if (outer_rect_.width < outer_rect_.height) continue;
            Rect inner_rect_ = get_inner_rect(mask1_rot);
            double rect_score = double(inner_rect_.area()) / double(outer_rect_.area());
            if (rect_score > max_rect_score) {
                max_rect_score = rect_score;
                angle = i;
            }
        }
        rotate(mask, angle, mask);
    }

    static cv::Point reflect(Point line_p1, Point line_p2, Point p) {
        double m = double(line_p2.y - line_p1.y) / (double(line_p2.x - line_p1.x)+0.0001);
        double b = line_p2.y - m * line_p2.x;
        double d = (p.x + (p.y - b) * m) / (1.0 + m*m);
        double xp = 2 * d - p.x;
        double yp = 2 * d * m - p.y + 2 * b;
        return Point(xp, yp);
    }

    static cv::Mat get_rotation_matrix(cv::Mat& src, double angle_degrees) {
        Point2f pt(src.cols/2., src.rows/2.);
        return getRotationMatrix2D(pt, angle_degrees, 1.0);
    }

    static void rotate(cv::Mat& src, double angle_degrees, cv::Mat& dst) {
        Point2f pt(src.cols/2., src.rows/2.);
        Mat r = getRotationMatrix2D(pt, angle_degrees, 1.0);
        warpAffine(src, dst, r, Size(src.cols, src.rows));
    }

    static Rect get_outer_rect(Mat& mask) {
        vector<vector<Point> > contours; vector<Vec4i> hierarchy;
        findContours(mask.clone(), contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
        vector<Point> pts;
        for (auto& contour : contours) {
            for (auto& p : contour) {
                pts.push_back(p);
            }
        }
        return boundingRect(pts);
    }

    static bool is_rect_inside(const Rect& r, const Mat& mask) {
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

    static Rect get_inner_rect(const Mat& mask_) {
        vector<vector<Point> > contours; vector<Vec4i> hierarchy;
        Mat mask = mask_.clone();
        findContours(mask, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
        Rect best_rect(0,0,0,0);
        unsigned long max_contour_size = 0;
        int contour_idx = 0;
        for (int i = 0; i < contours.size(); i++) {
            if (contours[i].size() > max_contour_size) {
                max_contour_size = contours[i].size();
                contour_idx = i;
            }
        }

        vector<Point> pts;
        for (int i = 0; i < contours[contour_idx].size(); i++) {
            if (i % 2 == 0) {
                pts.push_back(contours[contour_idx][i]);
            }
        }

        for (auto& p1 : pts) {
            for (auto& p2 : pts) {
                if (p1.x == p2.x && p1.y == p2.y) continue;
                Rect r(p1, p2);
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

    static double cross_product(const Point& a, const Point& b ){
        return a.x*b.y - a.y*b.x;
    }

    static double distance_to_line(Point begin_, Point end_, Point x_) {
        Point begin(begin_.x, begin_.y);
        Point end(end_.x, end_.y);
        Point x(x_.x, x_.y);
        end = end - begin;
        //cout<<"end :"<<end.x<<" "<<end.y<<endl;
        x = x - begin;
        double area = cross_product(x, end);
       // cout << "area :"<<area<<endl;
        return area / ((double)(norm(end))); //+0.0000001
    }

    static void max_contour(const cv::Mat& mask, std::vector<cv::Point>& pts) {
        vector<vector<Point> > contours; vector<Vec4i> hierarchy;
        findContours(mask.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
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
        vector<vector<Point> > contours; vector<Vec4i> hierarchy;
        findContours(mask.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
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
        display = Mat::zeros(mask.size(), CV_8U);
        drawContours(display, contours, contour_idx, Scalar(255), -1);
    }

    static cv::Mat icon_from_bgr(cv::Mat& img) {
        Mat out;
        double scale = 100.0 / img.rows;
        Size size(img.size().width * scale, img.size().height * scale);
        resize(img, out, size);

        int left = out.size().width + 1;
        int right = -1;
        int top = out.size().height + 1;
        int bot = -1;
        for (int row = 0; row < out.rows; row++) {
            for (int col = 0; col < out.cols; col++) {
                Vec3b pix = out.at<Vec3b>(row, col);
                if (!(pix[0] == 0 && pix[1] == 0 && pix[2] == 0)) {
                    if (col < left) left = col;
                    if (col > right) right = col;
                    if (row < top) top = row;
                    if (row > bot) bot = row;
                }
            }
        }
        Rect myROI(left, top, right - left, bot - top);
        out = out(myROI);
        return out;
    }

    static cv::Mat icon_from_mask(cv::Mat& mask) {
        Mat out;
        double scale = 100.0 / mask.rows;
        Size size(mask.size().width * scale, mask.size().height * scale);
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
        Rect myROI(left, top, right - left, bot - top);
        out = out(myROI);
        return out;
    }

    static cv::Mat get_xyz(cv::Mat& template_mask_new, cv::Mat& inv_trans, int* pixel2voxel, CloudPtr cloud) {
        Mat xyz = Mat::zeros(template_mask_new.size(), CV_64FC3);
        for (int row = 0; row < template_mask_new.rows; row++) {
            for (int col = 0; col < template_mask_new.cols; col++) {
                if (CommonTools::is_row_col_in_mask(row, col, template_mask_new)) {
                    Mat pm(3, 1, CV_64F);
                    pm.at<double>(0) = col;
                    pm.at<double>(1) = row;
                    pm.at<double>(2) = 1;
                    Mat qm = inv_trans * pm;
                    double t2 = qm.at<double>(2);
                    double x2 = qm.at<double>(0) / t2;
                    double y2 = qm.at<double>(1) / t2;
                    int cloud_idx = CommonTools::pix2vox(pixel2voxel, y2, x2, template_mask_new.size().width);
                    if (cloud_idx >= 0) {
                        PointT p3d = cloud->at(cloud_idx);
                        xyz.at<Vec3f>(row, col) = Vec3f(p3d.x, p3d.y, p3d.z);
                    }
                }
            }
        }
        return xyz;
    }

    static std::vector<float> get_rect_feature(cv::Mat& mask, cv::Rect outer_rect, cv::Rect inner_rect, double mask_scale) {
        vector<float> feature(6);
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
        Rect outer_rect = get_outer_rect(mask);
        Rect inner_rect = get_inner_rect(mask);
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

    static bool spatial_match(std::vector<float> feature1, std::vector<cv::Mat> mats, double& angle) {
        cv::Mat m = mats[0];
        std::vector<float> feature2 = from_mat(m);

        double min_dist = 9999999;
        for (int i = 1; i <= 4; i++) {
            cout << i << ", features pre: ";
            for (auto& f : feature2) cout << f << " ";
            cout << endl;
            rotate_features_clockwise(feature2);
            cout << "features post: ";
            for (auto& f : feature2) cout << f << " ";
            cout << endl;
            double dist = l2_dist(feature1, feature2);
            if (dist < min_dist) {
                angle = - i * 360 / 4;
                min_dist = dist;
            }
        }
        cout << "spatial match min dist: " << min_dist << endl;
        return min_dist < 1.3;
    }

    static void repel_grip_point(cv::Point line_p1, cv::Point line_p2, cv::Mat cloth_mask, cv::Point& grip_point) {
        double dist1 = distance_to_line(line_p1, line_p2, grip_point);
        double max_dist = 0;
        for (int row = 0; row < cloth_mask.rows; row++) {
            for (int col = 0; col < cloth_mask.cols; col++) {
                if (is_row_col_in_mask(row, col, cloth_mask)) {
                    double dist2 = distance_to_line(line_p1, line_p2, Point(col, row));
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

    static void get_corners_from_mask(Rect inner_rect, Point line_p1, Point line_p2, Point& inner_p1, Point& inner_p2) {
        Point p1(inner_rect.x, inner_rect.y);
        Point p2(inner_rect.x + inner_rect.width, inner_rect.y);
        Point p3(inner_rect.x + inner_rect.width, inner_rect.y + inner_rect.height);
        Point p4(inner_rect.x, inner_rect.y + inner_rect.height);
//        circle(fold_display, p1, 2, Scalar(0, 255, 0), -1);
//        circle(fold_display, p2, 2, Scalar(0, 255, 0), -1);
//        circle(fold_display, p3, 2, Scalar(0, 255, 0), -1);
//        circle(fold_display, p4, 2, Scalar(0, 255, 0), -1);

        vector<Point> inner_pts(4);
        inner_pts[0] = p1;
        inner_pts[1] = p2;
        inner_pts[2] = p3;
        inner_pts[3] = p4;
        double inner_max_dist1 = -1;
        int inner_max_dist1_idx = -1;
        double inner_max_dist2 = -1;
        int inner_max_dist2_idx = -1;
        for (int i = 0; i < inner_pts.size(); i++) {
            double dist = abs(distance_to_line(line_p1, line_p2, inner_pts[i]));
            if (dist > inner_max_dist1) {
                inner_max_dist2 = inner_max_dist1;
                inner_max_dist2_idx = inner_max_dist1_idx;
                inner_max_dist1 = dist;
                inner_max_dist1_idx = i;
            } else if (dist > inner_max_dist2) {
                inner_max_dist2 = dist;
                inner_max_dist2_idx = i;
            }
        }
       // cout << "before"<<endl;
        inner_p1 = inner_pts[inner_max_dist1_idx];
        inner_p2 = inner_pts[inner_max_dist2_idx];
        cout << "inner p1: " << inner_p1 << endl;
        cout << "inner p2: " << inner_p2 << endl;
    }

    static bool get_secondary_grip(cv::Point line_p1, cv::Point line_p2, cv::Mat cloth_mask, cv::Point& grip_point, Point& second_grip_point) {
        double dist1 = distance_to_line(line_p1, line_p2, grip_point);
        Point best_grip_point;
        cv::Mat folded_part = cv::Mat::zeros(cloth_mask.size(), CV_8U);
        for (int row = 0; row < cloth_mask.rows; row++) {
            for (int col = 0; col < cloth_mask.cols; col++) {
                if (is_row_col_in_mask(row, col, cloth_mask)) {
                    cv::Point test_point(col, row);
                    double dist2 = distance_to_line(line_p1, line_p2, test_point);
                    if (dist1 * dist2 > 0) {
                        folded_part.at<uchar>(row, col) = 255;
                    }
                }
            }
        }
        //cout<<"1"<<endl;
        Mat fold_display = folded_part.clone();
        cvtColor(fold_display, fold_display, CV_GRAY2BGR);
        vector<Point> folded_pts;
        max_contour(folded_part, folded_pts);
       // cout<<"2.1"<<endl;
        //imshow("folded_part",folded_part);
       // waitKey(0);
        Rect inner_rect = get_inner_rect(folded_part);
        //cout<<"2.2"<<endl;
        Rect outer_rect = get_outer_rect(folded_part);
        //cout<<"2.3"<<endl;
        Point inner_p1, inner_p2;
        get_corners_from_mask(inner_rect, line_p1, line_p2, inner_p1, inner_p2);
        //cout<<"2"<<endl;
        circle(fold_display, inner_p1, 3, Scalar(0, 0, 255), 2);
        // this is the separation in pixels for two handed
        if (norm(inner_p1 - inner_p2) > 150) {
            circle(fold_display, inner_p2, 3, Scalar(255, 0, 0), 2);


            Point outer_p1, outer_p2;
            //cout<<"3"<<endl;
            get_corners_from_mask(outer_rect, line_p1, line_p2, outer_p1, outer_p2);
            attract_grip_point(line_p1, line_p2, cloth_mask, outer_p1);
            attract_grip_point(line_p1, line_p2, cloth_mask, outer_p2);

            second_grip_point.x = outer_p2.x;
            second_grip_point.y = outer_p2.y;
            grip_point.x = outer_p1.x;
            grip_point.y = outer_p1.y;

            return true;
        }
        //cout<<"4"<<endl;
        return false;
    }

    static void attract_grip_point(cv::Point line_p1, cv::Point line_p2, cv::Mat cloth_mask, cv::Point& grip_point) {
        double dist1 = distance_to_line(line_p1, line_p2, grip_point);
        double min_dist = 99999;
        Point best_grip_point;
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

    static void attract_grip_point1(cv::Mat cloth_mask, cv::Point& grip_point) {

            double min_dist = 99999;
            Point best_grip_point;
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


    static double spatial_match_dist(std::vector<float> feature1, std::vector<cv::Mat> mats, double& angle) {
        cv::Mat m = mats[0];
        std::vector<float> feature2 = from_mat(m);

        double min_dist = 9999999;
        for (int i = 1; i <= 4; i++) {
            rotate_features_clockwise(feature2);
            double dist = l2_dist(feature1, feature2);
            //cout << "i = " << i << ": " << dist << endl;
            if (dist < min_dist) {
                angle = - i * 360 / 4;
                min_dist = dist;
            }
        }
        return min_dist;
    }

    static std::vector<float> from_mat(cv::Mat m) {
        int size = m.rows * m.cols;
        std::vector<float> vals(size);
        for (int i = 0; i < size; i++)
            vals[i] = m.at<float>(i);
        return vals;
    }

    static bool spatial_match_rects(cv::Mat& mat_, cv::Mat& blob, cv::Mat& scale_mask, double& angle) {
        cv::Mat mat = mat_.clone();
        cv::Mat mat2 = blob.clone();

        double table_area = cv::countNonZero(scale_mask);
        double mat_scale = cv::countNonZero(mat) / table_area;
        double mat2_scale = cv::countNonZero(mat2) / table_area;
        cout << "mat scale: " << mat_scale << endl;
        cout << "mat2 scale: " << mat2_scale << endl;

        vector<Point> mat_pts;
        max_contour(mat, mat_pts);
        Rect mat_rect = boundingRect(mat_pts);
        Mat mat_cropped = mat(mat_rect);
        Size mat_small(mat.size().width / 4, mat.size().height / 4);
        resize(mat_cropped, mat_cropped, mat_small);

        vector<Point> mat2_pts;
        max_contour(mat2, mat2_pts);
        Rect mat2_rect = boundingRect(mat2_pts);
        Mat mat2_cropped = mat2(mat2_rect);
        Size mat2_small(mat2.size().width / 4, mat2.size().height / 4);
        resize(mat2_cropped, mat2_cropped, mat2_small);

        Rect outer_rect2 = get_outer_rect(mat2_cropped);
        Rect inner_rect2 = get_inner_rect(mat2_cropped);
        vector<float> feature2 = get_rect_feature(mat2_cropped, outer_rect2, inner_rect2, mat2_scale);
        rectangle(mat2_cropped, outer_rect2, Scalar(255));
        rectangle(mat2_cropped, inner_rect2, Scalar(0));

        double min_dist = 9999999999;
        Mat best_mat_rotated;
        for (int i = 0; i < 360; i+=3) {
            cv::Mat mat_rotated;
            rotate(mat_cropped, i, mat_rotated);

            Rect outer_rect = get_outer_rect(mat_rotated);
            Rect inner_rect = get_inner_rect(mat_rotated);

            vector<float> feature1 = get_rect_feature(mat_rotated, outer_rect, inner_rect, mat_scale);
            double dist = l2_dist(feature1, feature2);
            if (dist < min_dist) {
                min_dist = dist;
                angle = i;
                best_mat_rotated = mat_rotated.clone();
                rectangle(best_mat_rotated, outer_rect, Scalar(255));
                rectangle(best_mat_rotated, inner_rect, Scalar(0));
            }
        }
        cout << "dist: " << min_dist << ", angle: " << angle << endl;
//        imshow("match1: best_mat_rotated", best_mat_rotated);
//        imshow("match2: mat2", mat2_cropped);
//        waitKey(0);
        return min_dist < 1.3;
    }

    static bool spatial_match(cv::Mat& mat_, std::vector<cv::Mat> blobs) {
        cv::Mat mat = mat_.clone();
        cv::Mat mat2 = blobs[0].clone();
        double score;
        cv::Mat mat_trans;
        match_curves(mat, mat2, score, mat_trans);

//        waitKey(0);
        if (mat_trans.size().area() == 0) {
            return false;
        }

        vector<vector<Point> > contours1; vector<Vec4i> hierarchy1;
        cv::Mat match_img = cv::Mat::zeros(mat_trans.size(), CV_8UC3);
        draw_contour(match_img, mat_trans, Scalar(255, 0, 255));
        draw_contour(match_img, mat2, Scalar(0, 255, 0));
//        imshow("match_img", match_img);
        double dist = shape_dist(mat_trans, mat2);
        cout << dist << endl;
//        waitKey(0);
        return dist < 0.072;
    }

    static void understand_fold(cv::Mat mask1, cv::Mat mask2, cv::Vec4f& line_params, cv::Point& grip_point) {

        imshow("previous mask (understand_fold)", mask1);
        imshow("current mask (understand_fold)", mask2);
        waitKey(200);

        Mat diff = mask1 - mask2;
        erode_dilate(diff, 4);
        if (DEBUG) imshow("diff", diff);
        vector<Point> diff_pts;
        Mat diff_blob;
        max_contour(diff, diff_pts, diff_blob);
        double diff_ratio = double(countNonZero(diff)) / double(countNonZero(mask1));
        cout << "diff_ratio: " << diff_ratio << endl;

        if (diff_ratio < 0.02) return;
        if (DEBUG) imshow("diff_blob_orig", diff_blob);
        erode(diff_blob, 2);
        dilate(diff_blob, 3);
        if (DEBUG) imshow("diff_blob", diff_blob);
        vector<Point> diff_blob_pts;
        max_contour(diff_blob, diff_blob_pts);

        Mat fold_line_area = diff_blob & mask2;

        vector<Point> fold_pts;
        max_contour(fold_line_area, fold_pts, fold_line_area);

        cout << fold_pts.size() << endl;
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
        Point line_p1 = Point(line_x - line_scale * line_dx, line_y - line_scale * line_dy);
        Point line_p2 = Point(line_x + line_scale * line_dx, line_y + line_scale * line_dy);



        double max_dist = -1;
        Point grip_point_dilate;
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
                    double dist = norm(grip_point_dilate - Point(col, row));
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

    static void match_curves_for_trans(cv::Mat& mask_a, cv::Mat& mask_b, double& score, cv::Mat& trans) {
        std::vector<cv::Point2d> seqa, seqb;
        cv::Mat vis;
        return match_curves(mask_a, mask_b, score, seqa, seqb, vis, trans);
    }

    static void match_curves(cv::Mat& mask_a, cv::Mat& mask_b, double& score) {
        std::vector<cv::Point2d> seqa, seqb;
        cv::Mat vis;
        cv::Mat trans;
        return match_curves(mask_a, mask_b, score, seqa, seqb, vis, trans);
    }


    // TODO fix method
    static void match_curves(cv::Mat& mask_a, cv::Mat& mask_b, double& score, std::vector<cv::Point2d>& seq_a_trans, std::vector<cv::Point2d>& seq_b_trans, cv::Mat& visual, cv::Mat& trans) {
        vector<Point> mask_a_pts, mask_b_pts;
        max_contour(mask_a, mask_a_pts, mask_a);
        max_contour(mask_b, mask_b_pts, mask_b);
        std::vector<cv::Point> a, b;
        GetCurveForImage(mask_a, a, false);
        if (a.empty()) {
            cout << "curve a is empty" << endl;
            return;
        }
        ResampleCurve(a, a, 200, false);
        GetCurveForImage(mask_b, b, false);
        if (b.empty()) {
            cout << "curve b is empty" << endl;
            return;
        }
        ResampleCurve(b, b, 200, false);

        Mat outout = Mat::zeros(mask_a.size(), CV_8UC3);
        drawOpenCurve(outout, a, Scalar(255,0,0), 2);
        drawOpenCurve(outout, b, Scalar(0,0,255), 2);

        if (DEBUG) imshow("curves", outout);
//        waitKey(0);

        int a_len, a_off, b_len, b_off;
        CompareCurvesUsingSignatureDB(a,
                                      b,
                                      a_len,
                                      a_off,
                                      b_len,
                                      b_off,
                                      score
        );

        //Get matched subsets of curves
        std::vector<cv::Point> a_subset(a.begin() + a_off, a.begin() + a_off + a_len);
        std::vector<cv::Point> b_subset(b.begin() + b_off, b.begin() + b_off + b_len);

        //Normalize to equal length
        ResampleCurve(a_subset, a_subset, 200, true);
        ResampleCurve(b_subset, b_subset, 200, true);

        //Prepare the curves for finding the transformation
        std::vector<cv::Point2f> seq_a_32f, seq_b_32f, seq_a_32f_, seq_b_32f_;

        ConvertCurve(a_subset, seq_a_32f_);
        ConvertCurve(b_subset, seq_b_32f_);

        assert(seq_a_32f_.size() == seq_b_32f_.size());

        seq_a_32f.clear(); seq_b_32f.clear();
        for (int i=0; i<seq_a_32f_.size(); i++) {
            //		if(i%2 == 0) { // you can use only part of the points to find the transformation
            seq_a_32f.push_back(seq_a_32f_[i]);
            seq_b_32f.push_back(seq_b_32f_[i]);
            //		}
        }
        assert(seq_a_32f.size() == seq_b_32f.size()); //just making sure

        seq_a_trans.resize(a.size());

        //Find the fitting transformation
        //	Mat affineT = estimateRigidTransform(seq_a_32f,seq_b_32f,false); //may wanna use Affine here..
        trans = Find2DRigidTransform(seq_a_32f, seq_b_32f);
        cout << "trans: " << trans << endl;

        std::vector<cv::Point2d> a_p2d;
        ConvertCurve(a, a_p2d);
        cv::transform(a_p2d, seq_a_trans, trans);

        std::vector<cv::Point2d> b_p2d;
        ConvertCurve(b, b_p2d);

        //draw the result matching : the complete original curve as matched to the target
        visual = Mat::zeros(mask_a.size(), CV_8UC3);
        drawOpenCurve(visual, seq_a_trans, Scalar(0, 255, 0), 2);
        for (int i = 0; i < seq_a_32f.size(); i++) {
            line(visual, seq_a_32f[i], seq_b_32f[i], Scalar(0,0,255), 1);
        }
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

    static void draw_contour(cv::Mat& img, const cv::Mat& mask, cv::Scalar color) {
        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;
        cv::findContours(mask, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
        cv::drawContours(img, contours, 0, color, 2);
    }

    static void find_3d(PointT p, CloudPtr cloud, int* voxel2pixel, int width, int& row, int& col) {
        double min_dist = 999999;
        int min_idx = -1;
        std::cout << "cloud size is " << cloud->size() << std::endl;
        std::cout << "looking for " << p << std::endl;

        for (int i = 0; i < cloud->size(); i++) {
            double dist = pcl::euclideanDistance(cloud->at(i), p);
            if (dist < min_dist) {
                min_dist = dist;
                min_idx = i;
            }
        }
        std::cout << "min idx is " << min_idx << ", with point " << cloud->at(min_idx) << std::endl;
        cv::Point2i xy = vox2pix(voxel2pixel, min_idx, width);
        row = xy.y;
        col = xy.x;
    }

    static std::vector<cv::Point2i> find_3ds(const std::vector<PointT>& ps, CloudPtr cloud,
                         int* voxel2pixel, int width) {
        std::vector<double> min_dists(ps.size());
        std::vector<int> min_idxs(ps.size());
        for (int i = 0; i < ps.size(); i++) {
            min_dists[i] = 999999;
            min_idxs[i] = -1;
        }

        for (int i = 0; i < cloud->size(); i++) {
            for (int j = 0; j < ps.size(); j++) {
                double dist = pcl::euclideanDistance(cloud->at(i), ps[j]);
                if (dist < min_dists[j]) {
                    min_dists[j] = dist;
                    min_idxs[j] = i;
                }
            }
        }

        std::vector<cv::Point2i> xys;
        for (int j = 0; j < ps.size(); j++) {
            xys.push_back(vox2pix(voxel2pixel, min_idxs[j], width));
        }
        return xys;
    }

    // http://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    cv::Mat static get_transformation(cv::Point3d v, double s, double c) {
        cv::Mat vm = cv::Mat::zeros(3, 3, CV_32F);
        vm.at<float>(0, 0) = 0;
        vm.at<float>(0, 1) = -v.z;
        vm.at<float>(0, 2) = v.y;
        vm.at<float>(1, 0) = v.z;
        vm.at<float>(1, 1) = 0;
        vm.at<float>(1, 2) = -v.x;
        vm.at<float>(2, 0) = -v.y;
        vm.at<float>(2, 1) = v.x;
        vm.at<float>(2, 2) = 0;
        cv::Mat I = cv::Mat::eye(3, 3, CV_32F);
        return I + vm + (vm*vm) * (1 - c) / (s*s);
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

    static cv::Mat threshold(cv::Mat img) {
        cv::Mat shirt_mask;
        cv::cvtColor(img, shirt_mask, CV_RGB2GRAY);
        cv::threshold(shirt_mask, shirt_mask, 250, 255, CV_THRESH_BINARY);
        shirt_mask = 255 - shirt_mask;
        return shirt_mask;
    }

    static cv::Mat draw_towel() {
        cv::Mat towel_mask = cv::Mat::zeros(480, 640, CV_8U);
        for (int row = 40; row < 200; row++) {
            for (int col = 80; col < 300; col++) {
                towel_mask.at<uchar>(row, col) = 255;
            }
        }
        return towel_mask;
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

    /*
    static void filter_cloud(CloudPtr cloud) {
        pcl::StatisticalOutlierRemoval<PointT> sor;
        sor.setInputCloud(cloud);
        sor.setMeanK(50);
        sor.setStddevMulThresh (1.0);
        sor.filter(*cloud);
    }
    */

    static bool get_cloud_info(pcl::PointCloud<pcl::PointXYZRGB>& cloud, Mat& img, int* pixel2voxel, int* voxel2pixel) {
        std::vector<pcl::PCLPointField> fields;
        int field_idx = pcl::getFieldIndex (cloud, "rgb", fields);
        if (field_idx == -1) {
            field_idx = pcl::getFieldIndex (cloud, "rgba", fields);
            if (field_idx == -1)
                return false;
        }
        const size_t offset = fields[field_idx].offset;

        img = Mat(cloud.height, cloud.width, CV_8UC3);


        for (size_t i = 0; i < cloud.points.size (); ++i) {
            if (!pcl::isFinite(cloud[i])) {
                Vec3b color_vec(0, 0, 0);
                img.at<Vec3b>(i) = color_vec;
                pixel2voxel[i] = -1;
                voxel2pixel[i] = -1;
                cloud[i].x = 10;
                cloud[i].y = 10;
                cloud[i].z = 10;
            } else {
                uint32_t val;
                pcl::getFieldValue<PointT, uint32_t> (cloud.points[i], offset, val);
                Vec3b color_vec((val) & 0x0000ff, (val >> 8) & 0x0000ff, (val >> 16) & 0x0000ff);
                img.at<Vec3b>(i) = color_vec;
                pixel2voxel[i] = i;
                voxel2pixel[i] = i;
            }
        }

        return true;
    }

    static CloudPtr make_cloud_ptr(cv::Mat img_bgr, cv::Mat img_x, cv::Mat img_y, cv::Mat img_z, int* pixel2voxel, int* voxel2pixel) {
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
//                    std::cout << p << std::endl;
                    cloud_ptr->push_back(p);
                } else {
                    pixel2voxel[img_idx] = -1;
                    p.z = std::numeric_limits<float>::quiet_NaN();
                    cloud_ptr->push_back(p);
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


