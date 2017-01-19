//
// Created by binroot on 12/14/15.
//

#include "FileFrameScanner.h"
#include <boost/filesystem.hpp>
#include <sstream>
#include <fstream>

using namespace std;
using cv::imread;


FileFrameScanner::FileFrameScanner(std::string filepath) {
    m_filepath = filepath;
    m_rgb_file_prefix = "aligned_rgb_";
    m_xyz_file_prefix = "raw_point_";
    m_skeleton_file_prefix = "skeleton_";
    m_file_num_length = 5;
}

std::string zero_pad(long num, int length) {
    std::string num_str = std::to_string(num);
    int zeros_to_prepend = length - num_str.length();
    num_str.insert(0, zeros_to_prepend, '0');
    return num_str;
}

string FileFrameScanner::rgb_filename(int idx) {
    return m_rgb_file_prefix + zero_pad(idx, m_file_num_length) + ".png";
}

void FileFrameScanner::xyz_filename(int idx, string& x_file, string& y_file, string& z_file) {
    x_file = m_xyz_file_prefix + zero_pad(idx, m_file_num_length) + "_X.png";
    y_file = m_xyz_file_prefix + zero_pad(idx, m_file_num_length) + "_Y.png";
    z_file = m_xyz_file_prefix + zero_pad(idx, m_file_num_length) + "_Z.png";
}

string FileFrameScanner::skeleton_filename(int idx) {
    return m_skeleton_file_prefix + zero_pad(idx, m_file_num_length) + ".txt";
}

bool FileFrameScanner::get(int idx, cv::Mat& img_bgr, cv::Mat& img_x, cv::Mat& img_y, cv::Mat& img_z, PointT& left_hand, PointT& right_hand) {
    string img_file = rgb_filename(idx);
    string x_file, y_file, z_file;
    xyz_filename(idx, x_file, y_file, z_file);
    string skeleton_file = skeleton_filename(idx);
    img_bgr = imread(m_filepath + img_file);
    img_x = imread(m_filepath + x_file, CV_LOAD_IMAGE_UNCHANGED);
    img_y = imread(m_filepath + y_file, CV_LOAD_IMAGE_UNCHANGED);
    img_z = imread(m_filepath + z_file, CV_LOAD_IMAGE_UNCHANGED);
    read_skeleton(m_filepath + skeleton_file);
    left_hand = m_left_hand;
    right_hand = m_right_hand;
    if (img_bgr.size().area() == 0) {
        cout << "ERR: " << img_file << " is not found in " << m_filepath << endl;
        return false;
    } else if (img_x.size().area() == 0) {
        cout << "ERR: " << x_file << " is malformed" << endl;
        exit(1);
    } else if (img_y.size().area() == 0) {
        cout << "ERR: " << y_file << " is malformed" << endl;
        exit(1);
    } else if (img_z.size().area() == 0) {
        cout << "ERR: " << z_file << " is malformed" << endl;
        exit(1);
    }
    return true;
}

void FileFrameScanner::read_skeleton(string filename) {
    if (boost::filesystem::is_regular_file(m_filepath + filename)) {
        std::ifstream infile(filename);
        std::string line;
        int line_idx = 0;
        while (getline(infile, line)) {
            if (line_idx == 7) {
                // left hand
                std::istringstream iss(line);
                double l, p3x, p3y, p3z, t, p2x, p2y;
                iss >> l >> p3x >> p3y >> p3z >> t >> p2x >> p2y;
                m_left_hand.x = p3x;
                m_left_hand.y = p3y;
                m_left_hand.z = p3z;
            } else if (line_idx == 11) {
                // right hand
                std::istringstream iss(line);
                double l, p3x, p3y, p3z, t, p2x, p2y;
                iss >> l >> p3x >> p3y >> p3z >> t >> p2x >> p2y;
                m_right_hand.x = p3x;
                m_right_hand.y = p3y;
                m_right_hand.z = p3z;
            } else if (line_idx > 11) {
                break;
            }
            line_idx++;
        }
    }
}