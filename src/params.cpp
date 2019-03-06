#include "params.hpp"

namespace Params
{

// 画像の最大数
const int MAX_FILE_NUM = 8;

const cv::Size2i ZED_RESOLUTION = cv::Size2i(672, 376);

// ZEDのパラメータ
const cv::Mat1f ZED_INTRINSIC
    = (cv::Mat_<float>(3, 3) << 350, 0, 336,
        0, 350, 336,
        0, 0, 1);
const cv::Mat1f ZED_EXTRINSIC
    = (cv::Mat_<float>(3, 4) << 1, 0, 0, 0.120,
        0, 1, 0, 0,
        0, 0, 1, 0);
const cv::Mat1f ZED_PERSPECTIVE_RIGHT
    = (cv::Mat_<float>(3, 4) << 350, 0, 336, 0,
        0, 350, 336, 0,
        0, 0, 1, 0);
const cv::Mat1f ZED_PERSPECTIVE_LEFT
    = static_cast<cv::Mat1f>(ZED_INTRINSIC * ZED_EXTRINSIC);

}  // namespace Params