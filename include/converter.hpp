#pragma once
#include <opencv2/opencv.hpp>

namespace Converter
{
constexpr float INVALID = -2.0f;

inline bool isValid(float num) { return INVALID < num; }
inline bool isInvalid(float num) { return num <= INVALID; }

cv::Mat1f toMat1f(float x, float y);
cv::Mat1f toMat1f(float x, float y, float z);

// 正規化
cv::Mat depthNormalize(const cv::Mat& depth_image);
cv::Mat colorNormalize(const cv::Mat& color_image);

// T(4x4) => T(4x4)
cv::Mat1f inversePose(const cv::Mat1f& T);

cv::Mat1f gradiate(const cv::Mat1f& gray_image, bool x);

float getColorSubpix(const cv::Mat1f& img, cv::Point2f pt);

cv::Mat cullImage(const cv::Mat& src_image);
}  // namespace Converter