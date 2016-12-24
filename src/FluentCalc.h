#ifndef FLUENT_EXTRACTOR_FLUENTCALC_H
#define FLUENT_EXTRACTOR_FLUENTCALC_H

#include <vector>
#include "CommonTools.h"

class FluentCalc {
public:
    static std::vector<float> calc_width_and_height(CloudPtr cloud, PointT normal);
    static std::vector<float> calc_thickness(CloudPtr cloud, PointT table_normal, PointT table_midpoint);
};


#endif //FLUENT_EXTRACTOR_FLUENTCALC_H
