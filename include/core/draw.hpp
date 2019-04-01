#pragma once
#include <opencv2/opencv.hpp>

namespace Draw
{
// NOTE:無効画素は赤色

// 輝度
cv::Mat visiblizeGray(const cv::Mat& src_image);

// 深度
cv::Mat visiblizeDepth(const cv::Mat& src_image);

// 勾配
cv::Mat visiblizeGradient(const cv::Mat& x_image, const cv::Mat& y_image);

// TODO: より汎用的に
// window名,画像x5
void showImage(const std::string& window_name, const cv::Mat1f& pre_gray, const cv::Mat1f& pre_depth,
    const cv::Mat1f& warped_gray, const cv::Mat1f& cur_gray, const cv::Mat1f& cur_depth);

}  // namespace Draw
