#ifndef FLUENT_EXTRACTOR_FLUENTCALC_H
#define FLUENT_EXTRACTOR_FLUENTCALC_H

#include <vector>
#include "CommonTools.h"

class FluentCalc {
public:
    static std::vector<float> calc_width_and_height(CloudPtr cloud, PointT normal);
    static std::vector<float> calc_thickness(CloudPtr cloud, PointT table_normal, PointT table_midpoint);
    static std::vector<float> x_and_y_symmetries(CloudPtr cloud);
    static std::vector<float> calc_bbox(CloudPtr cloud);
    static std::vector<float> principal_symmetries(CloudPtr cloud);
};


#endif //FLUENT_EXTRACTOR_FLUENTCALC_H
