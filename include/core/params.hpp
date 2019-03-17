#pragma once
#include "calibration/params_struct.hpp"
#include <opencv2/opencv.hpp>
#include <string>

namespace Params
{
extern Calibration::IntrinsicParams rgb_intrinsic;
extern Calibration::IntrinsicParams depth_intrinsic;
extern Calibration::ExtrinsicParams extrinsic;

void init(
    Calibration::IntrinsicParams rgb,
    Calibration::IntrinsicParams depth,
    Calibration::ExtrinsicParams ext);

}  // namespace Params