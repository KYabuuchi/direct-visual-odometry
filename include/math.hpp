#pragma once
#include <cassert>
#include <opencv2/opencv.hpp>

namespace math
{
// 3x1 => 3x3
cv::Mat1f hat(const cv::Mat1f& vec);

namespace so3
{

// 3x1 => 3x3
cv::Mat1f exp(const cv::Mat1f& twist);

// 3x3 => 3x1
cv::Mat1f log(const cv::Mat1f& R);

}  // namespace so3

namespace se3
{
// 6x1 => 4x4
cv::Mat1f exp(const cv::Mat1f& twist);

// 4x4 => 6x1
cv::Mat1f log(const cv::Mat1f& T);

// {6x1,6x1} => 6x1
cv::Mat1f concatenate(const cv::Mat1f& xi0, const cv::Mat1f& xi1);

}  // namespace se3

}  // namespace math
