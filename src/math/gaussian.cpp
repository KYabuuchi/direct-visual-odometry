#include "math/gaussian.hpp"
#include <random>

namespace math
{
namespace
{
std::mt19937 engine;
// TODO: 平均値は状況による
std::uniform_real_distribution<float> dist(2.5, 0.5);
}  // namespace

bool Gaussian::update(float d, float s)
{
    float v1 = math::square(sigma);
    float v2 = math::square(s);
    float v = v1 + v2;

    // 期待値が離れすぎていたら消去
    float diff = std::abs(d - depth);
    if (diff > std::max(sigma, s)) {
        depth = dist(engine);
        sigma = 0.5f;
        return false;
    }

    depth = (v2 * depth + v1 * d) / v;
    sigma = std::sqrt((v1 * v2) / v);

    return true;
}

bool Gaussian::operator()(float d, float s)
{
    float v1 = math::square(sigma);
    float v2 = math::square(s);
    float v = v1 + v2;

    // 期待値が離れすぎていたら反映しない
    float diff = std::abs(d - depth);
    if (diff > std::max(sigma, s))
        return false;

    depth = (v2 * depth + v1 * d) / v;
    sigma = std::sqrt((v1 * v2) / v);

    return true;
}

}  // namespace math