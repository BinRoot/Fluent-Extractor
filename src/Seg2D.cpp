//
// Created by binroot on 6/1/15.
//

#include "Seg2D.h"

#include "segment/segment-image.h"

using namespace std;
using namespace cv;

Seg2D::Seg2D() {
}

Mat Seg2D::seg(Mat& img, double sigma, double k, double minSize) {
    SegmentParams segParams;
    segParams.sigma = sigma;
    segParams.k = k;
    segParams.min_size = minSize;
    return segment(img, segParams);
}

