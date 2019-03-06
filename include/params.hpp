#pragma once
#include <opencv2/opencv.hpp>
#include <string>

namespace Params
{
extern const int MAX_FILE_NUM;

// 内部行列・外部行列・射影行列(左)・射影行列(右)
extern const cv::Size2i ZED_RESOLUTION;
extern const cv::Mat1f ZED_INTRINSIC;
extern const cv::Mat1f ZED_EXTRINSIC;
extern const cv::Mat1f ZED_PERSPECTIVE_LEFT;
extern const cv::Mat1f ZED_PERSPECTIVE_RIGHT;

}  // namespace Params