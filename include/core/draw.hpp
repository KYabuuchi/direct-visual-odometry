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
cv::Mat visualizeDepthRaw(const cv::Mat1f& src_image);

// 勾配
cv::Mat visualizeGradient(const cv::Mat1f& src_image);

// 標準偏差
cv::Mat visualizeSigma(const cv::Mat1f& src_image);

// 年齢
cv::Mat visualizeAge(const cv::Mat1f& src_image);

void showImage(const std::string& window_name,
    const cv::Mat& ref_gray, const cv::Mat& warped_gray, const cv::Mat& cur_gray,
    const cv::Mat& ref_depth, const cv::Mat& ref_sigma, const cv::Mat& ref_grad);

}  // namespace Draw
