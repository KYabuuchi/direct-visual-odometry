#pragma once
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

// 3x3
cv::Mat1f R(const std::array<float, 3> data);
cv::Mat1f R();

// 3x1
cv::Mat1f omega(const std::array<float, 3> data);
cv::Mat1f omega();
}  // namespace so3

namespace se3
{
// 6x1 => 4x4
cv::Mat1f exp(const cv::Mat1f& twist);

// 4x4 => 6x1
cv::Mat1f log(const cv::Mat1f& T);

// {6x1,6x1} => 6x1
cv::Mat1f concatenate(const cv::Mat1f& xi0, const cv::Mat1f& xi1);

// 4x4
cv::Mat1f T(const std::array<float, 6>& data);
cv::Mat1f T();

// 6x1
cv::Mat xi(const std::array<float, 6>& data);
cv::Mat xi();

}  // namespace se3

}  // namespace math