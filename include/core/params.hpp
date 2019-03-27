#pragma once
#include "calibration/params_struct.hpp"
#include <opencv2/opencv.hpp>
#include <string>

namespace Params
{
const Calibration::IntrinsicParams RGB();
const Calibration::IntrinsicParams DEPTH();
const Calibration::ExtrinsicParams EXT();

void init(const std::string& config_path);

}  // namespace Params