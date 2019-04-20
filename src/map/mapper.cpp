#include "map/mapper.hpp"
#include "core/transform.hpp"
#include "map/implement.hpp"
#include <cmath>

namespace Map
{

namespace
{
constexpr float minimum_movement = 0.10f;  // [m]
}  // namespace

void Mapper::estimate(FrameHistory& frame_history, pFrame frame)
{
    if (frame_history.size() == 0) {
        frame_history.setRefFrame(frame);
    } else {
        if (needNewFrame(frame)) {
            pFrame new_frame = propagate(frame_history, frame);
            frame_history.setRefFrame(new_frame);
        } else {
            update(frame_history, frame);
        }
    }
    regularize(frame_history, frame);
}

bool Mapper::needNewFrame(pFrame frame)
{
    cv::Mat1f& xi = frame->m_relative_xi;
    double scalar = cv::norm(xi.rowRange(0, 3));
    return scalar > minimum_movement;
}

pFrame Mapper::propagate(const FrameHistory& /*frame_history*/, const pFrame frame)
{
    const pFrame& ref = frame->m_ref_frame;

    auto [depth, sigma, age] = Implement::propagate(
        ref->depth(),
        ref->sigma(),
        ref->age(),
        frame->m_relative_xi,
        frame->K());

    pFrame new_frame = std::make_shared<Frame>(
        depth,
        sigma,
        age,
        frame->K(),
        frame->level,
        frame->culls);

    return frame;
}

void Mapper::update(const FrameHistory& frame_history, pFrame obj)
{
    const pFrame ref = obj->m_ref_frame;
    const cv::Mat1f xi = obj->m_xi;

    auto inRange = math::generateInRange(ref->depth().size());
    const cv::Mat1f K = obj->K();

    ref->depth().forEach(
        [=](const float d, const int pt[2]) -> void {
            cv::Point2i x_i(pt[1], pt[0]);
            cv::Point2i warped_x_i = Transform::warp(xi, x_i, d, K);
            if (not inRange(warped_x_i))
                return;

            // 生まれた
            int age = static_cast<int>(obj->m_age(x_i));
            pFrame born = frame_history[age];

            // 事前分布
            float depth = d - obj->m_relative_xi(2);
            float sigma = ref->sigma()(x_i);

            // 観測
            auto [new_deptgh, new_sigma] = Implement::update(
                obj->gray(),
                born->gray(),
                born->gradX(),
                born->gradY(),
                math::se3::concatenate(obj->m_xi, -born->m_xi),
                obj->K(),
                warped_x_i,
                depth,
                sigma);

            // TODO: 事後分布
        });
}

void Mapper::regularize(const FrameHistory& /*frame_history*/, pFrame frame)
{
    std::cout << "regularize" << std::endl;

    // TODO: ピラミッドの下のほうが更新されない
    Implement::regularize(frame->top()->depth(), frame->top()->sigma());
    // NOTE: ↓これとかを使う
    // frame->updateDepthSigma();
}

}  // namespace Map
