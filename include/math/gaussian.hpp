#pragma once
#include "math/util.hpp"
#include <cmath>
#include <iostream>

namespace math
{
// Gauss分布
struct Gaussian {
    Gaussian(float depth, float sigma) : depth(depth), sigma(sigma) {}
    float depth;
    float sigma;

    bool update(float d, float s);
    bool operator()(float d, float s);
};
}  // namespace math