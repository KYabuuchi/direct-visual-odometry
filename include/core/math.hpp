#pragma once
#include <cassert>
#include <opencv2/opencv.hpp>

namespace math
{
constexpr float EPSILON = 1e-6f;

inline bool isEpsilon(float num)
{
    return std::abs(num) < EPSILON;
}

inline bool testXi(const cv::Mat1f& xi)
{
    assert(xi.size() == cv::Size(6, 1));
    bool flag = true;

    // TODO:
    for (int i = 0; i < 6; i++)
        flag &= not std::isnan(xi(i));

    return flag;
}

template <typename T = float>
inline bool isRange(float num, T min, T max)
{
    if (num <= static_cast<float>(min))
        return false;
    if (static_cast<float>(max) <= num)
        return false;
    return true;
}


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
