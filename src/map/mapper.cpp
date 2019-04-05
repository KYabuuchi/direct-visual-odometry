#include "map/mapper.hpp"
#include "core/transform.hpp"

namespace Map
{
struct Gaussian {
    Gaussian(float depth, float sigma) : depth(depth), sigma(sigma) {}
    float depth;
    float sigma;

    void multi(float d, float s)
    {
        float v1 = math::square(sigma);
        float v2 = math::square(s);
        float v = v1 + v2;

        // 期待値が離れすぎていたら反映しない
        float diff = std::abs(d - depth);
        if (diff > std::max(sigma, s))
            return;

        depth = (v2 * depth + v1 * d) / v;
        sigma = (v1 * v2) / v;
    }
};

bool Mapper::needNewFrame(pFrame frame)
{
    cv::Mat1f& xi = frame->m_relative_xi;
    double scalar = cv::norm(xi.rowRange(0, 3));
    return scalar > m_config.minimum_movement;
}

void Mapper::initializeHistory(const cv::Mat1f& depth, cv::Mat1f& sigma)
{
    sigma.forEach(
        [&](float& s, const int p[2]) -> void {
            if (depth(p[0], p[1]) > 0.01f) {
                s = 0.01f;  // 10[mm]
            } else {
                s = m_config.initial_sigma;  // 1[m]
            }
        });
}

void Mapper::propagate(
    const cv::Mat1f& ref_depth,
    const cv::Mat1f& ref_sigma,
    const cv::Mat1f& ref_age,
    cv::Mat1f& depth,
    cv::Mat1f& sigma,
    cv::Mat1f& age,
    const cv::Mat1f& xi,
    const cv::Mat1f& intrinsic)
{
    const float tz = xi(2);
    std::function<bool(cv::Point2i)> inRange = math::generateInRange(age.size());

    ref_depth.forEach(
        [&](float& rd, const int pt[2]) -> void {
            // cv::Point2i x_i(col, row);
            cv::Point2i x_i(pt[1], pt[0]);

            // float rd = ref_depth(x_i);
            if (math::isEpsilon(rd))
                return;

            float s = ref_sigma(x_i);
            float d0 = rd;
            float d1 = d0 - tz;
            if (s > 0.5 or d0 < 0.05)
                s = m_config.initial_sigma;
            else
                s = std::sqrt(math::pow(d1 / d0, 4) * math::square(s)
                              + math::square(m_config.predict_sigma));

            cv::Point2f warped_x_i = Transform::warp(xi, x_i, rd, intrinsic);

            if (inRange(warped_x_i)) {
                depth(warped_x_i) = std::max(d1, 0.0f);
                sigma(warped_x_i) = s;
                age(warped_x_i) = ref_age(x_i) + 1;
            }
        });
}

void Mapper::regularize(cv::Mat1f& depth, const cv::Mat1f& sigma)
{
    cv::Mat1f origin_depth(depth);

    std::vector<std::pair<int, int>> offsets = {{0, -1}, {0, 1}, {1, 0}, {-1, 0}};
    std::function<bool(cv::Point2i)> inRange = math::generateInRange(depth.size());

    depth.forEach(
        [&](float& d, const int p[2]) -> void {
            Gaussian g{d, sigma(p[0], p[1])};

            for (const std::pair<int, int> offset : offsets) {
                cv::Point2i pt(p[1] + offset.second, p[0] + offset.first);
                if (not inRange(pt))
                    continue;

                g.multi(origin_depth.at<float>(pt), sigma.at<float>(pt));
            }
            d = g.depth;
        });
}

}  // namespace Map
