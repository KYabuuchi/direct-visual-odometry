#include "map/mapper.hpp"


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

        // 分散が大きすぎたら反映しない
        float diff = std::abs(d - depth);
        if (diff > std::max(sigma, s))
            return;

        depth = (v2 * depth + v1 * d) / v;
        sigma = (v1 * v2) / v;
    }
};

bool Mapper::insertableHistory(pFrame frame)
{
    cv::Mat1f& xi = frame->m_relative_xi;
    double scalar = cv::norm(xi.rowRange(0, 3));
    return scalar > 0.10;  // 100[mm]
}

void Mapper::initializeHistory(const cv::Mat1f& depth, cv::Mat1f& sigma)
{
    std::cout << "debug" << std::endl;
    sigma.forEach(
        [&](float& s, const int p[2]) -> void {
            if (depth(p[0], p[1]) > 0.01f) {
                s = 0.01f;  // +-10[mm]
            } else {
                s = 1;  // +-1[m]
            }
        });
}


void Mapper::propagate(FrameHistory /*frame_history*/, pFrame /*frame*/)
{
}

void Mapper::update(FrameHistory /*frame_history*/, pFrame /*frame*/)
{
}

void Mapper::regularize(cv::Mat1f& depth, const cv::Mat1f& sigma)
{
    cv::Mat1f origin_depth(depth);

    std::vector<std::pair<int, int>> offsets = {{0, -1}, {0, 1}, {1, 0}, {-1, 0}};
    auto isRange = [=](cv::Point2i pt)
        -> bool { return (0 <= pt.x && pt.x < depth.cols && 0 <= pt.y && pt.y < depth.rows); };

    depth.forEach(
        [&](float& d, const int p[2]) -> void {
            Gaussian g{d, sigma(p[0], p[1])};

            for (const std::pair<int, int> offset : offsets) {
                cv::Point2i pt(p[1] + offset.second, p[0] + offset.first);
                if (not isRange(pt))
                    continue;

                g.multi(origin_depth.at<float>(pt), sigma.at<float>(pt));
            }
            d = g.depth;
        });
}

}  // namespace Map
