#include "map/mapper.hpp"
#include "core/transform.hpp"
#include "map/updater.hpp"
#include <cmath>

namespace Map
{

namespace
{
constexpr float minimum_movement = 0.10f;  // [m]
constexpr float predict_sigma = 0.10f;     // [m]
constexpr float predict_variance = predict_sigma * predict_sigma;

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
    return scalar > m_config.minimum_movement;
}

// ====Propagate====
pFrame Mapper::propagate(const FrameHistory& /*frame_history*/, const pFrame frame)
{
    if (m_config.is_chatty)
        std::cout << "propagate" << std::endl;
    const pFrame& ref = frame->m_ref_frame;

    auto [depth, sigma, age] = propagate(
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

std::tuple<cv::Mat1f, cv::Mat1f, cv::Mat1f> Mapper::propagate(
    const cv::Mat1f& ref_depth,
    const cv::Mat1f& ref_sigma,
    const cv::Mat1f& ref_age,
    const cv::Mat1f& xi,
    const cv::Mat1f& intrinsic)
{
    const float tz = xi(2);
    const cv::Size size = ref_depth.size();
    std::function<bool(cv::Point2i)> inRange = math::generateInRange(size);

    cv::Mat1f depth(cv::Mat1f::ones(size));
    cv::Mat1f sigma(cv::Mat1f::ones(size));
    cv::Mat1f age(cv::Mat1f::zeros(size));

    ref_depth.forEach(
        [&](float& rd, const int pt[2]) -> void {
            cv::Point2i x_i(pt[1], pt[0]);
            if (math::isEpsilon(rd))
                return;

            cv::Point2f warped_x_i = Transform::warp(xi, x_i, rd, intrinsic);
            if (not inRange(warped_x_i))
                return;

            float s = ref_sigma(x_i);
            float d0 = rd;
            float d1 = d0 - tz;
            if (s > 0.5 or d0 < 0.05)
                s = m_config.initial_sigma;
            else
                s = std::sqrt(math::pow(d1 / d0, 4) * math::square(s)
                              + m_config.predict_variance);

            depth(warped_x_i) = std::max(d1, 0.0f);
            sigma(warped_x_i) = s;
            age(warped_x_i) = ref_age(x_i) + 1;
        });

    return {depth, sigma, age};
}

// ====Update====
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
            auto [new_deptgh, new_sigma] = Update::update(
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

// ====Regularize====
void Mapper::regularize(const FrameHistory& /*frame_history*/, pFrame frame)
{
    if (m_config.is_chatty)
        std::cout << "regularize" << std::endl;

    // TODO: ピラミッドの下のほうが更新されない
    regularize(frame->top()->depth(), frame->top()->sigma());
    // NOTE: ↓これとかを使う
    // frame->updateDepthSigma();
}

void Mapper::regularize(cv::Mat1f& depth, const cv::Mat1f& sigma)
{
    cv::Mat1f origin_depth(depth);

    std::vector<std::pair<int, int>> offsets = {{0, -1}, {0, 1}, {1, 0}, {-1, 0}};
    std::function<bool(cv::Point2i)> inRange = math::generateInRange(depth.size());

    depth.forEach(
        [&](float& d, const int p[2]) -> void {
            math::Gaussian gauss{d, sigma(p[0], p[1])};

            for (const std::pair<int, int> offset : offsets) {
                cv::Point2i pt(p[1] + offset.second, p[0] + offset.first);
                if (not inRange(pt))
                    continue;

                gauss(origin_depth.at<float>(pt), sigma.at<float>(pt));
            }
            d = gauss.depth;
        });
}

}  // namespace Map
