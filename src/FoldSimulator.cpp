//
// Created by binroot on 7/7/15.
//

#include "FoldSimulator.h"


using namespace std;
using cv::Mat;
using cv::EVENT_LBUTTONDOWN;
using cv::EVENT_RBUTTONDOWN;
using cv::EVENT_MBUTTONDOWN;
using cv::Point2i;
using cv::Point;
using cv::namedWindow;
using cv::setMouseCallback;
using cv::waitKey;
using cv::Vec4i;
using cv::Scalar;
using cv::imwrite;

string FoldSimulator::get_name() {
    return m_name;
}

void FoldSimulator::set_grip_point(cv::Point2i grip_point) {
    m_grip_point = grip_point;
}

void FoldSimulator::set_line_point(cv::Point2i line_p) {
    *m_next_line_p = line_p;
    if (m_next_line_p == &m_line_p1) {
        m_next_line_p = &m_line_p2;
    } else {
        m_next_line_p = &m_line_p1;
    }
}

void CallBackFunc(int event, int x, int y, int flags, void* param) {
    if  ( event == EVENT_LBUTTONDOWN ) {
        FoldSimulator* cloth = (FoldSimulator*)(param);
        cloth->set_grip_point(Point2i(x, y));
    }
    else if  ( event == EVENT_RBUTTONDOWN ) {
        cout << "right click" << endl;
        FoldSimulator* cloth = (FoldSimulator*)(param);
        cloth->set_line_point(Point2i(x, y));
    }
    else if  ( event == EVENT_MBUTTONDOWN ) {
        FoldSimulator* cloth = (FoldSimulator*)(param);
        cloth->fold();
    }
}

void FoldSimulator::run_gui() {
    namedWindow(get_name(), CV_WINDOW_AUTOSIZE);
    setMouseCallback(get_name(), CallBackFunc, this);

    while (waitKey(20) != 'q') {
        imshow(get_name(), visualize());
    }
}

bool FoldSimulator::fold() {
    return fold(m_grip_point, m_line_p1, m_line_p2);
}

Mat FoldSimulator::flatten() {
    // flatten all the contours
    Mat flatten = Mat::zeros(m_contours[0].size(), CV_8U);
    for (auto& contour : m_contours) {
        flatten = flatten | contour;
    }
    return flatten;
}

bool FoldSimulator::fold(Point2i point, Point2i line_p1, Point2i line_p2) {
    Mat flat = flatten();
    m_contours.clear();
    m_contours.push_back(flat);

    // detect on which contour the point lies, starting from the top
    int contour_idx = -1;
    for (int i = m_contours.size() - 1; i >= 0; i--) {
        if (CommonTools::is_point_in_mask(point, m_contours[i])) {
            contour_idx = i;
            break;
        }
    }

//    if (contour_idx < 0) {
//        cout << "point not found on cloth" << endl;
//        return false;
//    }
    contour_idx = 0;

    // flip the cloth
    Point2i line_vec = line_p2 - line_p1;
    int point_det = CommonTools::determinant(point - line_p1, line_vec);
    if (point_det == 0) {
        cout << "line goes through point" << endl;
        return false;
    }

    double line_slope = double(line_p2.y - line_p1.y) / double(line_p2.x - line_p1.x);
    double line_y_intersect = line_p2.y - line_slope * line_p2.x;

    Mat fold_region = Mat::zeros(m_contours[contour_idx].size(), CV_8U);
    for (int row = 0; row < m_contours[contour_idx].rows; row++) {
        for (int col = 0; col < m_contours[contour_idx].cols; col++) {
            if (CommonTools::is_row_col_in_mask(row, col, m_contours[contour_idx])) {
                int det = CommonTools::determinant(Point2i(col, row) - line_p1, line_vec);
                if (det != 0 && CommonTools::sgn(det) == CommonTools::sgn(point_det)) {
                    fold_region.at<uchar>(row, col) = 255;
                }
            }
        }
    }
    vector<vector<Point> > fold_region_contours;
    vector<Vec4i> fold_region_hierarchy;
    findContours( fold_region, fold_region_contours, fold_region_hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );
    int fold_region_idx = 0;
    if (fold_region_contours.size() > 1) {
        double min_dist = numeric_limits<double>::infinity();
        for (int contour_idx = 0; contour_idx < fold_region_contours.size(); contour_idx++) {
            Point centroid;
            for (auto& p : fold_region_contours[contour_idx]) {
                centroid.x += p.x;
                centroid.y += p.y;
            }
            centroid.x /= fold_region_contours[contour_idx].size();
            centroid.y /= fold_region_contours[contour_idx].size();
            double dist = norm(centroid - m_grip_point);
            if (dist < min_dist) {
                min_dist = dist;
                fold_region_idx = contour_idx;
            }
        }
    }
    Mat fold_region_mask = Mat::zeros(fold_region.size(), CV_8U);
    drawContours(fold_region_mask, fold_region_contours, fold_region_idx, Scalar(255), -1);


    Mat point_region = Mat::zeros(m_contours[contour_idx].size(), m_contours[contour_idx].type());
    for (int row = 0; row < m_contours[contour_idx].rows; row++) {
        for (int col = 0; col < m_contours[contour_idx].cols; col++) {
//            if (CommonTools::is_row_col_in_mask(row, col, m_contours[contour_idx])) {
            if (CommonTools::is_row_col_in_mask(row, col, fold_region_mask)) {
                int det = CommonTools::determinant(Point2i(col, row) - line_p1, line_vec);
                if (det != 0 && CommonTools::sgn(det) == CommonTools::sgn(point_det)) {

                    // given (col, row), and line y = ax + c
                    double d = (col + (row - line_y_intersect) * line_slope) / (1.0 + line_slope*line_slope);
                    double xp = 2 * d - col;
                    double yp = 2 * d * line_slope - row + 2 * line_y_intersect;

                    // if this point exists in any of the contours above, flip those too
                    for (int higher_contour_idx = m_contours.size() - 1; higher_contour_idx > contour_idx; higher_contour_idx--) {
                        if (CommonTools::is_row_col_in_mask(row, col, m_contours[higher_contour_idx])) {
                            if (yp > 0 && xp > 0 && yp < m_contours[higher_contour_idx].cols && xp < m_contours[higher_contour_idx].rows)
                                point_region.at<uchar>(yp, xp) = 255;
                            m_contours[higher_contour_idx].at<uchar>(row, col) = 0;
                        }
                    }


                    if (yp > 0 && xp > 0 && yp < m_contours[contour_idx].cols && xp < m_contours[contour_idx].rows)
                        point_region.at<uchar>(yp, xp) = 255;
                    m_contours[contour_idx].at<uchar>(row, col) = 0;
                }
            }
        }
    }

    cout << "(sim) reflected point_region: " << countNonZero(point_region) << endl;

    imshow("simulation", visualize());
    waitKey(20);

    if (countNonZero(point_region) > 0) {
        vector<vector<Point>> hull(1);
        vector<vector<Point>> contours; vector<Vec4i> hierarchy;
        findContours(point_region, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
        vector<Point> all_contours;
        for (auto& contour : contours) {
            for (Point& p : contour) {
                all_contours.push_back(p);
            }
        }

        convexHull(all_contours, hull[0]);
        drawContours(point_region, hull, 0, Scalar(255, 255, 255), -1);
//    imshow("point_region", point_region);
        m_contours.push_back(point_region);
    }


    return true;
}

Mat FoldSimulator::visualize() {
    if (m_contours.size() > 0) {
        Mat cartoon = Mat::zeros(m_contours[0].size(), CV_8U);
        for (int i = 0; i < m_contours.size(); i++) {
            vector<vector<Point> > contours;
            vector<Vec4i> hierarchy;
            Mat tmp = m_contours[i].clone();
            findContours(tmp, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));
            for (int i = 0; i < contours.size(); i++) {
                drawContours(cartoon, contours, i, Scalar(255, 255, 255), 2, 8, hierarchy, 0, Point(0, 0));
            }
        }

        circle(cartoon, m_grip_point, 4, Scalar(128, 128, 128), 3);
        line(cartoon, m_line_p1, m_line_p2, Scalar(128, 128, 128), 2);
        circle(cartoon, m_line_p1, 3, Scalar(255, 0, 0), -1);
        circle(cartoon, m_line_p2, 3, Scalar(0, 0, 255), -1);

        return cartoon;
    } else {
        return Mat();
    }
}

void FoldSimulator::save(string filename) {
    imwrite(filename, visualize());
}
