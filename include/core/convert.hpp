#pragma once
#include <opencv2/opencv.hpp>

namespace Convert
{
cv::Mat1f toMat1f(int x, int y);
cv::Mat1f toMat1f(float x, float y);
cv::Mat1f toMat1f(float x, float y, float z);

// 正規化
cv::Mat depthNormalize(const cv::Mat& depth_image);
cv::Mat colorNormalize(const cv::Mat& color_image);


// T(4x4) => T(4x4)
cv::Mat1f inversePose(const cv::Mat1f& T);

// 勾配
cv::Mat1f gradiate(const cv::Mat1f& gray_image, bool x);

float getColorSubpix(const cv::Mat1f& img, cv::Point2f pt);

// 間引く
cv::Mat1f cullImage(const cv::Mat1f& src_image, int times = 1);
cv::Mat1f cullIntrinsic(const cv::Mat1f& intrinsic, int times = 1);

}  // namespace Convert