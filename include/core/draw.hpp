#pragma once
#include <opencv2/opencv.hpp>

namespace Draw
{
// 無効画素は赤色へ
cv::Mat visiblizeGrayImage(const cv::Mat& src_image);

// 無効画素は赤色へ
cv::Mat visiblizeDepthImage(const cv::Mat& src_image);

// 無効画素は赤色へ
cv::Mat visiblizeGradientImage(const cv::Mat& x_image, const cv::Mat& y_image);

// TODO: より汎用的に
// window名,画像x5
void showImage(const std::string& window_name, const cv::Mat1f& pre_gray, const cv::Mat1f& pre_depth,
    const cv::Mat1f& warped_gray, const cv::Mat1f& cur_gray, const cv::Mat1f& cur_depth);

}  // namespace Draw
