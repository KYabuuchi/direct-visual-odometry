#include "math/gaussian.hpp"

namespace math
{
    void Gaussian::update(float d, float s)
    {
        float v1 = math::square(sigma);
        float v2 = math::square(s);
        float v = v1 + v2;

        // 期待値が離れすぎていたら消去
        float diff = std::abs(d - depth);
        if (diff > std::max(sigma, s)) {
            depth = 1.5f;
            sigma = 0.5f;
            std::cout << "reject" << std::endl;
            return;
        }

        depth = (v2 * depth + v1 * d) / v;
        sigma = std::sqrt((v1 * v2) / v);
    }

    void Gaussian::operator()(float d, float s)
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

}  // namespace Math