//
// Created by binroot on 6/1/15.
//

#ifndef STABLESEGMENT_SEG2D_H
#define STABLESEGMENT_SEG2D_H

#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"

class Seg2D {
public:
    Seg2D();
    cv::Mat seg(cv::Mat& img, double sigma, double k, double minSize);
};

#endif //STABLESEGMENT_SEG2D_H
