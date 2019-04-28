#pragma once
#include <opencv2/opencv.hpp>

namespace Transform
{
// xi,x(3x1) => Rx+t(3x1)
// T(4x4),x(3x1) => Rx+t(3x1)
cv::Point3f transform(const cv::Mat1f& T, const cv::Point3f& x);

// x_c => x_i
cv::Point2f project(const cv::Mat1f& K, const cv::Point3f& point);

// x_i => x_c
cv::Point3f backProject(const cv::Mat1f& K, const cv::Point2f& point, float depth);

// <mapped,sigma>
std::pair<cv::Mat1f, cv::Mat1f> mapDepthtoGray(
    const cv::Mat1f& depth_image,
    const cv::Mat1f& gray_image,
    const cv::Mat1f& rgb_K,
    const cv::Mat1f& depth_K,
    const cv::Mat1f& invT);

// warp先の座標を返す
cv::Point2f warp(const cv::Mat1f& xi, const cv::Point2f& x_i, const float depth, const cv::Mat1f& K);

// warpした画像を返す(gray画像のみを想定)
cv::Mat warpImage(const cv::Mat1f& xi, const cv::Mat1f& gray_image, const cv::Mat1f& depth_image, const cv::Mat1f& K);
}  // namespace Transform