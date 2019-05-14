#include "map/mapper.hpp"
#include "core/draw.hpp"
#include "core/transform.hpp"
#include "map/implement.hpp"
#include <cmath>

namespace Map
{
namespace
{
constexpr float MINIMUM_MOVEMENT = 0.08f;  // [m]
constexpr int MAXIMUM_FORWARD = 14;        // number of frame
}  // namespace

void Mapper::estimate(FrameHistory& frame_history, pFrame frame)
{
    if (needNewFrame(frame)) {
        propagate(frame);
        frame_history.setRefFrame(frame);
    } else {
        update(frame_history, frame);
    }

    pFrame ref_frame = frame_history.getRefFrame();
    regularize(ref_frame);
    show(ref_frame);
}

void Mapper::show(const pFrame frame)
{
    std::cout << "Mapper::show " << frame->id << std::endl;
    Draw::showImage(
        window_name,
        Draw::visualizeGray(frame->gray()),
        Draw::visualizeDepth(frame->depth(), frame->sigma()),
        Draw::visualizeDepth(frame->depth()),
        Draw::visualizeAge(frame->age()));
}

bool Mapper::needNewFrame(const pFrame frame)
{
    cv::Mat1f& xi = frame->m_relative_xi;
    double scalar = cv::norm(xi.rowRange(0, 3));
    if (scalar > MINIMUM_MOVEMENT) {
        std::cout << "Mapper::needNewFrame: enough movement" << std::endl;
        return true;
    }
    if (frame->id - frame->m_ref_frame->id >= MAXIMUM_FORWARD) {
        std::cout << "Mapper::needNewFrame: much frame accumulated" << std::endl;
        return true;
    }
    // TODO: 回転量におうじたしきい値

    return false;
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
    pFrame ref = obj->m_ref_frame;
    const cv::Mat1f xi = obj->m_relative_xi;
    const cv::Mat1f K = obj->K();

    auto inRange = math::generateInRange(ref->depth().size());
    int valid_update = 0;

    // 320x240
    ref->depth().forEach(
        [&](const float d, const int pt[2]) -> void {
            // 外側は無駄になりがち
            if (pt[1] < 16 or pt[1] > 144 or pt[0] < 12 or pt[0] > 108)
                return;

            cv::Point2i x_i(pt[1], pt[0]);
            cv::Point2i warped_x_i = Transform::warp(xi, x_i, d, K);
            if (not inRange(warped_x_i))
                return;

            // 生まれ年のKeyFrameを参照
            int age = static_cast<int>(ref->m_age(x_i));
            // age = std::min(age, 2);
            pFrame born = frame_history[age];

            // 事前分布
            float depth = d - obj->m_relative_xi(2);
            float sigma = ref->sigma()(x_i);

            cv::Mat1f r_xi = math::se3::concatenate(static_cast<cv::Mat1f>(obj->m_xi), static_cast<cv::Mat1f>(-born->m_xi));
            // std::cout << "Tnow " << born->m_xi.t() << " Tpre " << obj->m_xi.t() << " TrelF" << r_xi.t() << std::endl;

            // 観測
            auto [new_depth, new_sigma] = Implement::update(
                obj->gray(),
                born->gray(),
                born->gradX(),
                born->gradY(),
                r_xi,
                obj->K(),
                warped_x_i,
                depth,
                sigma);

            // 更新
            if (new_depth > 0.2 and new_depth < 6.0 and new_sigma > 0 and new_sigma < 1) {
                math::Gaussian g(depth, sigma);
                if (not g.update(new_depth, new_sigma)) {
                    // NOTE: maybe occulusion
                    ref->m_age(x_i) = 0;
                } else {
                    valid_update++;
                }
                ref->top()->depth()(x_i) = g.depth;
                ref->top()->sigma()(x_i) = g.sigma;
            }
        });

    ref->updateDepthSigma(ref->depth(), ref->sigma());
    std::cout << "\t valid update: " << valid_update << std::endl;
}

void Mapper::regularize(pFrame frame)
{
    std::cout << "Mapper::regularize " << frame->id << std::endl;
    cv::Mat1f depth = Implement::regularize(frame->depth(), frame->sigma());
    frame->updateDepth(depth);
}

}  // namespace Map
