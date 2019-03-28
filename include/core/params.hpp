#pragma once
#include "calibration/params_struct.hpp"
#include <opencv2/opencv.hpp>
#include <string>

namespace Params
{
void init(const std::string& config_path);

const Calibration::IntrinsicParams RGB();
const Calibration::IntrinsicParams DEPTH();
const Calibration::ExtrinsicParams EXT();

}  // namespace Params