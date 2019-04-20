#include "map/mapper.hpp"
#include "core/transform.hpp"
#include "map/implement.hpp"
#include <cmath>

namespace Map
{
namespace
{
constexpr float MINIMUM_MOVEMENT = 0.10f;  // [m]
}  // namespace

void Mapper::estimate(FrameHistory& frame_history, pFrame frame)
{
    if (needNewFrame(frame)) {
        propagate(frame);
        double oldest = 0;
        cv::minMaxIdx(frame->age(), nullptr, &oldest);
        std::cout << "oldest: " << oldest << std::endl;

        frame_history.setRefFrame(frame);
    } else {
        update(frame_history, frame);
    }
    regularize(frame);
}

bool Mapper::needNewFrame(const pFrame frame)
{
    cv::Mat1f& xi = frame->m_relative_xi;
    double scalar = cv::norm(xi.rowRange(0, 3));
    return scalar > MINIMUM_MOVEMENT;
}

void Mapper::propagate(pFrame frame)
{
    std::cout << "Mapper::propagate" << std::endl;

    const pFrame& ref = frame->m_ref_frame;
    auto [depth, sigma, age] = Implement::propagate(
        ref->depth(),
        ref->sigma(),
        ref->age(),
        frame->m_relative_xi,
        frame->K());
    frame->updateDepthSigmaAge(depth, sigma, age);
}

void Mapper::update(const FrameHistory& frame_history, pFrame obj)
{
    std::cout << "Mapper::update" << std::endl;
    const pFrame ref = obj->m_ref_frame;
    const cv::Mat1f xi = obj->m_xi;

    auto inRange = math::generateInRange(ref->depth().size());
    const cv::Mat1f K = obj->K();

    ref->depth().forEach(
        [&](const float d, const int pt[2]) -> void {
            cv::Point2i x_i(pt[1], pt[0]);
            cv::Point2i warped_x_i = Transform::warp(xi, x_i, d, K);
            if (not inRange(warped_x_i))
                return;

            // 生まれ年の
            int age = static_cast<int>(obj->m_age(x_i));
            pFrame born = frame_history[age];

            // 事前分布
            float depth = d - obj->m_relative_xi(2);
            float sigma = ref->sigma()(x_i);

            // 観測
            auto [new_depth, new_sigma] = Implement::update(
                obj->gray(),
                born->gray(),
                born->gradX(),
                born->gradY(),
                math::se3::concatenate(obj->m_xi, -born->m_xi),
                obj->K(),
                warped_x_i,
                depth,
                sigma);

            // 更新
            if (new_depth > 0 and new_depth < 4.0) {
                math::Gaussian g(depth, sigma);
                g(new_depth, new_sigma);
                obj->top()->depth()(x_i) = g.depth;
                obj->top()->sigma()(x_i) = g.sigma;
            }
        });
}

void Mapper::regularize(pFrame frame)
{
    std::cout << "Mapper::regularize" << std::endl;
    cv::Mat1f depth = Implement::regularize(frame->depth(), frame->sigma());
    frame->updateDepth(depth);
}

}  // namespace Map
