#include "math/gaussian.hpp"
#include <random>

namespace math
{
namespace
{
std::mt19937 engine;
std::uniform_real_distribution<float> dist(2.0, 0.5);
}  // namespace

bool Gaussian::update(float d, float s)
{
    float v1 = math::square(sigma);
    float v2 = math::square(s);
    float v = v1 + v2;

    // 期待値が離れすぎていたら消去
    float diff = std::abs(d - depth);
    float gain = (std::min(d, diff) < 0.8) ? 0.5f + std::min(d, diff) / 0.8f * 0.5f : 1.0f;
    if (diff > gain * std::max(sigma, s)) {
        depth = std::min(dist(engine), 4.0f);  // NOTE:
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

    // TODO: こっちも近いほうが不遇になるようにする
    // 期待値が離れすぎていたら反映しない
    float diff = std::abs(d - depth);
    if (diff > std::max(sigma, s))
        return false;

    depth = (v2 * depth + v1 * d) / v;
    sigma = std::sqrt((v1 * v2) / v);

    return true;
}

}  // namespace math