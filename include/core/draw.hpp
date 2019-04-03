#pragma once
#include <opencv2/opencv.hpp>

namespace Draw
{
// NOTE:無効画素は赤色

// 輝度
cv::Mat visualizeGray(const cv::Mat1f& src_image);

// 深度
cv::Mat visualizeDepth(const cv::Mat1f& src_image);
cv::Mat visualizeDepth(const cv::Mat1f& depth, const cv::Mat1f& sigma);

// 勾配
cv::Mat visualizeGradient(const cv::Mat1f& x_image, const cv::Mat1f& y_image);

// 標準偏差
cv::Mat visualizeSigma(const cv::Mat1f& src_image);

// window名,画像x5
void showImage(const std::string& window_name, const cv::Mat1f& pre_gray, const cv::Mat1f& pre_depth,
    const cv::Mat1f& warped_gray, const cv::Mat1f& cur_gray, const cv::Mat1f& cur_depth, const cv::Mat1f& pre_sigma);

}  // namespace Draw
