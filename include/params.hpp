#pragma once
#include <opencv2/opencv.hpp>
#include <string>

namespace Params
{
// ZED
// 解像度・内部行列・外部行列(左->右)・射影行列・歪パラメータ
extern const cv::Size2i ZED_RESOLUTION;
extern const cv::Mat1f ZED_INTRINSIC;
extern const cv::Mat1f ZED_EXTRINSIC;
extern const cv::Mat1f ZED_EXTRINSIC_INVERSE;
extern const cv::Mat1f ZED_PERSPECTIVE_LEFT;
extern const cv::Mat1f ZED_PERSPECTIVE_RIGHT;
// extern const cv::Mat1f ZED_DISTORTION_LEFT;
// extern const cv::Mat1f ZED_DISTORTION_RIGHT;

// Kinect-v2
// 解像度・内部行列・外部行列(RGB->DEPTH)・射影行列・歪パラメータ
extern const cv::Size2i KINECTV2_RESOLUTION_RGB;
extern const cv::Size2i KINECTV2_RESOLUTION_DEPTH;
extern const cv::Mat1f KINECTV2_INTRINSIC_RGB;
extern const cv::Mat1f KINECTV2_INTRINSIC_DEPTH;
extern const cv::Mat1f KINECTV2_EXTRINSIC;
extern const cv::Mat1f KINECTV2_EXTRINSIC_INVERSE;
extern const cv::Mat1f KINECTV2_PERSPECTIVE_RGB;
extern const cv::Mat1f KINECTV2_PERSPECTIVE_DEPTH;
// extern const cv::Mat1f KINECTV2_DISTORTION_RGB;
// extern const cv::Mat1f KINECTV2_DISTORTION_DEPTH;
}  // namespace Params