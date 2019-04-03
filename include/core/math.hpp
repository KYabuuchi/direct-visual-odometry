#pragma once
#include <cassert>
#include <opencv2/opencv.hpp>

namespace math
{
constexpr float EPSILON = 1e-6f;

template <typename T>
inline T square(T num)
{
    return num * num;
}

template <typename T>
inline T pow(T base, int exponent)
{
    assert(exponent >= 0);
    if (exponent == 0)
        return 1;
    if (exponent == 1)
        return base;
    return base * pow(base, exponent - 1);
}

inline bool isEpsilon(float num)
{
    return std::abs(num) < EPSILON;
}

inline bool testXi(const cv::Mat1f& xi)
{
    assert(xi.size() == cv::Size(1, 6));

    for (int i = 0; i < 6; i++) {
        if (std::isnan(xi(i)))
            return false;
    }

    return true;
}

template <typename T>
inline bool isRange(T num, T min, T max)
{
    if (num < min)
        return false;
    if (max < num)
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
