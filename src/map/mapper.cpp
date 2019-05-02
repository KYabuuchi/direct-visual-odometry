#include "map/mapper.hpp"
#include "core/draw.hpp"
#include "core/transform.hpp"
#include "map/implement.hpp"
#include <cmath>

namespace Map
{
namespace
{
constexpr float MINIMUM_MOVEMENT = 0.05f;  // [m]
constexpr int MAXIMUM_FORWARD = 10;        // number of frame
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
        Draw::visualizeDepth(frame->depth()),
        Draw::visualizeSigma(frame->sigma()),
        Draw::visualizeAge(frame->age()));
}

bool Mapper::needNewFrame(const pFrame frame)
{
    cv::Mat1f& xi = frame->m_relative_xi;
    double scalar = cv::norm(xi.rowRange(0, 3));
    return (scalar > MINIMUM_MOVEMENT or frame->id - frame->m_ref_frame->id >= MAXIMUM_FORWARD);
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
    const cv::Mat1f xi = obj->m_xi;

    auto inRange = math::generateInRange(ref->depth().size());
    const cv::Mat1f K = obj->K();

    int valid_update = 0;

    ref->depth().forEach(
        [&](const float d, const int pt[2]) -> void {
            // 外側は無駄になりがち
            if (pt[1] < 16 or pt[1] > 144 or pt[0] < 12 or pt[0] > 108)
                return;

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
                math::se3::concatenate(obj->m_xi, cv::Mat1f(-born->m_xi)),
                obj->K(),
                warped_x_i,
                depth,
                sigma);

            if (new_sigma > 2) {
                std::cout << "tid " << new_sigma << std::endl;
            }


            if (new_depth < 0)
                return;
            // if (new_sigma > 0 and new_sigma < 1)
            //     std::cout << "valid update " << new_depth << " " << new_sigma << " " << x_i << std::endl;


            // 更新
            if (new_depth > 0 and new_depth < 6.0 and new_sigma > 0) {
                math::Gaussian g(depth, sigma);
                g.update(new_depth, new_sigma);
                ref->top()->depth()(x_i) = g.depth;
                ref->top()->sigma()(x_i) = g.sigma;
                valid_update++;
            } else {
                // std::cout << new_depth << std::endl;
            }
        });
    std::cout << "\t valid update: " << valid_update << std::endl;
}

void Mapper::regularize(pFrame frame)
{
    std::cout << "Mapper::regularize" << std::endl;
    cv::Mat1f depth = Implement::regularize(frame->depth(), frame->sigma());
    frame->updateDepth(depth);
}

}  // namespace Map
