//
// Created by binroot on 7/7/15.
//

#ifndef ROBOTCLOTHFOLDING_FOLDSIMULATOR_H
#define ROBOTCLOTHFOLDING_FOLDSIMULATOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "CommonTools.h"

class FoldSimulator {
public:
    FoldSimulator() {}
    FoldSimulator(cv::Mat mask) {
        m_next_line_p = &m_line_p1;
        m_name = "cloth_sim";
        m_contours.push_back(mask);
    }
    FoldSimulator(std::string name, std::string filepath) {
        m_next_line_p = &m_line_p1;
        m_name = name;
        cv::Mat img = cv::imread(filepath);
        cv::Mat cloth_mask = CommonTools::threshold(img);
        std::vector<cv::Mat> cloth_contours;
        cloth_contours.push_back(cloth_mask);
        for (cv::Mat contour : cloth_contours)
            m_contours.push_back(contour);
    }
    void run_gui();
    bool fold(cv::Point2i point, cv::Point2i line_p1, cv::Point2i line_p2);
    bool fold();
    cv::Mat visualize();
    void set_grip_point(cv::Point2i grip_point);
    void set_line_point(cv::Point2i line_p);
    cv::Point2i get_grip_point() { return m_grip_point; }
    cv::Point2i get_release_point() {
      double line_slope = double(m_line_p2.y - m_line_p1.y) / double(m_line_p2.x - m_line_p1.x);
      double line_y_intersect = m_line_p2.y - line_slope * m_line_p2.x;

      double d = (m_grip_point.x + (m_grip_point.y - line_y_intersect) * line_slope) / (1.0 + line_slope*line_slope);
      double xp = 2 * d - m_grip_point.x;
      double yp = 2 * d * line_slope - m_grip_point.y + 2 * line_y_intersect;
      return cv::Point2i(xp, yp);
    }
    std::string get_name();
    void save(std::string filename);
    cv::Mat flatten();
private:
    std::string m_name;
    std::vector<cv::Mat> m_contours;
    cv::Point2i m_grip_point;
    cv::Point2i m_line_p1;
    cv::Point2i m_line_p2;
    cv::Point2i* m_next_line_p;
};


#endif //ROBOTCLOTHFOLDING_FOLDSIMULATOR_H
