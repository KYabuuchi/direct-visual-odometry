#pragma once
#include <opencv2/opencv.hpp>

namespace Transform
{
// xi,x(3x1) => Rx+t(3x1)
// T(4x4),x(3x1) => Rx+t(3x1)
cv::Mat1f transform(const cv::Mat1f& T, const cv::Mat1f& x);

// x_c => x_i
cv::Mat1f project(const cv::Mat1f& intrinsic, const cv::Mat1f& point);

// x_i => x_c
cv::Mat1f backProject(const cv::Mat1f& intrinsic, const cv::Mat1f& point, float depth);

// 無効な画素にはConvert::INVALIDが入る
cv::Mat mapDepthtoGray(const cv::Mat& depth_image, const cv::Mat& gray_image);

// warp先の座標を返す
cv::Point2f warp(const cv::Mat1f& xi, const cv::Point2f& x_i, const float depth, const cv::Mat1f& intrinsic_matrix);

// warpした画像を返す(gray画像のみを想定)
cv::Mat warpImage(const cv::Mat1f& xi, const cv::Mat& gray_image, const cv::Mat& depth_image, const cv::Mat1f& intrinsic_matrix);
}  // namespace Transform