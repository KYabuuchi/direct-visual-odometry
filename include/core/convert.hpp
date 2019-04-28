#pragma once
#include <opencv2/opencv.hpp>

namespace Convert
{
template <typename T>
cv::Mat1f toMat1f(T x, T y)
{
    return cv::Mat1f(2, 1) << x, y;
}
template <typename T>
cv::Mat1f toMat1f(cv::Point_<T> pt)
{
    return cv::Mat1f(2, 1) << pt.x, pt.y;
}
template <typename T>
cv::Mat1f toMat1f(T x, T y, T z)
{
    return cv::Mat1f(3, 1) << x, y, z;
}
template <typename T>
cv::Mat1f toMat1f(cv::Point3_<T> pt)
{
    return cv::Mat1f(3, 1) << pt.x, pt.y, pt.z;
}

// T(4x4) => T(4x4)
cv::Mat1f inversePose(const cv::Mat1f& T);

// 勾配
cv::Mat1f gradiate(const cv::Mat1f& gray_image, bool x);

// subpixelを取得(無効画素を含む場合と含まない場合で挙動が異なる)
float getSubpixel(const cv::Mat1f& img, cv::Point2f pt);
float getSubpixelFromDense(const cv::Mat1f& img, cv::Point2f pt);

// 間引く
cv::Mat1f cullImage(const cv::Mat1f& src_image, int times = 1);
cv::Mat1f cullIntrinsic(const cv::Mat1f& intrinsic, int times = 1);

}  // namespace Convert