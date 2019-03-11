#pragma once
#include <opencv2/opencv.hpp>

namespace Converter
{

float getColorSubpix(const cv::Mat1f& img, cv::Point2f pt);

cv::Mat1f toMat1f(float x, float y);

cv::Mat1f toMat1f(float x, float y, float z);

// T(4x4),x(3x1) => Rx+t(3x1)
cv::Mat1f transform(const cv::Mat1f T, const cv::Mat1f& x);
// x_c => x_i
cv::Mat1f project(const cv::Mat1f& intrinsic, const cv::Mat1f& point);

// T(4x4) => T(4x4)
cv::Mat1f inversePose(const cv::Mat1f& T);

// x_i => x_c
cv::Mat1f backProject(const cv::Mat1f& intrinsic, const cv::Mat1f& point, float depth);

cv::Mat mapDepthtoGray(const cv::Mat& depth_image, const cv::Mat& color_image);

cv::Mat depthNormalize(const cv::Mat& depth_image);
cv::Mat colorNormalize(const cv::Mat& color_image);

}  // namespace Converter