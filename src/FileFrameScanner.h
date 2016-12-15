//
// Created by binroot on 12/14/15.
//

#ifndef MIND_GRAPH_FILEFRAMESCANNER_H
#define MIND_GRAPH_FILEFRAMESCANNER_H

#include <pcl/common/common_headers.h>
#include <opencv2/opencv.hpp>

typedef pcl::PointXYZRGB PointT;
typedef pcl::PointCloud<PointT> Cloud;
typedef Cloud::Ptr CloudPtr;
typedef Cloud::ConstPtr CloudConstPtr;

class FileFrameScanner {
public:
    FileFrameScanner(std::string filepath);
    virtual bool get(int idx, cv::Mat& img_bgr, cv::Mat& x, cv::Mat& y, cv::Mat& z, PointT& left_hand, PointT& right_hand);
private:
    std::string m_rgb_file_prefix;
    std::string m_xyz_file_prefix;
    std::string m_skeleton_file_prefix;
    int m_file_num_length;
    std::string m_filepath;
    std::string rgb_filename(int idx);
    void xyz_filename(int idx, std::string& x_file, std::string& y_file, std::string& z_file);
    std::string skeleton_filename(int idx);
    void read_skeleton(std::string filename);
    PointT m_left_hand;
    PointT m_right_hand;
};


#endif //MIND_GRAPH_FILEFRAMESCANNER_H
