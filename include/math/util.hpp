#pragma once
#include "math/constexpr.hpp"

namespace math
{
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


// Gauss分布
struct Gaussian {
    Gaussian(float depth, float sigma) : depth(depth), sigma(sigma) {}
    float depth;
    float sigma;

    void operator()(float d, float s)
    {
        float v1 = math::square(sigma);
        float v2 = math::square(s);
        float v = v1 + v2;

        // 期待値が離れすぎていたら反映しない
        float diff = std::abs(d - depth);
        if (diff > std::max(sigma, s))
            return;

        depth = (v2 * depth + v1 * d) / v;
        sigma = std::sqrt((v1 * v2) / v);
    }
};

}  // namespace math