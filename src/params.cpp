#include "params.hpp"
#include "converter.hpp"

namespace Params
{
const cv::Mat1f IDENTITY(cv::Mat1f::eye(4, 4));

// ZED
const cv::Size2i ZED_RESOLUTION = cv::Size2i(672, 376);
const cv::Mat1f ZED_INTRINSIC
    = (cv::Mat1f(3, 3) << 350, 0, 336,
        0, 350, 336,
        0, 0, 1);
const cv::Mat1f ZED_EXTRINSIC
    = (cv::Mat1f(4, 4) << 1, 0, 0, 0.120,
        0, 1, 0, 0,
        0, 0, 1, 0,
        0, 0, 0, 1);
const cv::Mat1f ZED_EXTRINSIC_INVERSE = Converter::inversePose(ZED_EXTRINSIC);

const cv::Mat1f ZED_PERSPECTIVE_RIGHT
    = static_cast<cv::Mat1f>(ZED_INTRINSIC * IDENTITY.rowRange(0, 3));
const cv::Mat1f ZED_PERSPECTIVE_LEFT
    = static_cast<cv::Mat1f>(ZED_INTRINSIC * ZED_EXTRINSIC.rowRange(0, 3));

// Kinect-v2
const cv::Size2i KINECTV2_RESOLUTION_RGB = cv::Size2i(1920, 1080);
const cv::Size2i KINECTV2_RESOLUTION_DEPTH = cv::Size2i(512, 424);
const cv::Mat1f KINECTV2_INTRINSIC_RGB
    = (cv::Mat1f(3, 3) << 1.0581373927643231e+03f, 0.0f, 9.4575585119795539e+02f,
        0.0f, 1.0610057107426685e+03f, 5.1298481972935826e+02f,
        0.0f, 0.0f, 1.0f);
const cv::Mat1f KINECTV2_INTRINSIC_DEPTH
    = (cv::Mat1f(3, 3) << 3.6338057175332000e+02f, 0.0f, 2.5642271789981970e+02f,
        0.0f, 3.6406922750437235e+02f, 2.0284938253328897e+02f,
        0.0f, 0.0f, 1.0f);
const cv::Mat1f KINECTV2_EXTRINSIC
    = (cv::Mat1f(4, 4) << 1, 0, 0, 5.6319455444830524e-02f,
        0, 1, 0, 1.2369380813873844e-02f,
        0, 0, 1, 3.1557970682765544e-03f,
        0, 0, 0, 1);
const cv::Mat1f KINECTV2_EXTRINSIC_INVERSE = Converter::inversePose(KINECTV2_EXTRINSIC);

const cv::Mat1f KINECTV2_PERSPECTIVE_RGB
    = static_cast<cv::Mat1f>(KINECTV2_INTRINSIC_RGB * IDENTITY.rowRange(0, 3));
const cv::Mat1f KINECTV2_PERSPECTIVE_DEPTH
    = static_cast<cv::Mat1f>(KINECTV2_INTRINSIC_DEPTH * KINECTV2_EXTRINSIC.rowRange(0, 3));


}  // namespace Params