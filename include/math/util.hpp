#pragma once
#include <opencv2/opencv.hpp>

namespace math
{
constexpr float EPSILON = 1e-6f;
constexpr float INVALID = -2.0f;

inline bool isValid(float num) { return INVALID < num; }
inline bool isInvalid(float num) { return num <= INVALID; }

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

inline std::function<bool(cv::Point2i)> generateInRange(cv::Size size)
{
    return [=](cv::Point2i pt) -> bool {
        if (pt.x < 0 || size.width <= pt.x)
            return false;
        if (pt.y < 0 || size.height <= pt.y)
            return false;
        return true;
    };
}

inline bool inRange(cv::Point2i pt, cv::Size size)
{
    if (pt.x < 0 || size.width <= pt.x)
        return false;
    if (pt.y < 0 || size.height <= pt.y)
        return false;
    return true;
}

template <typename T>
inline bool inRange(T num, T min, T max)
{
    if (num < min)
        return false;
    if (max < num)
        return false;
    return true;
}

}  // namespace math