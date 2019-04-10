#include "map/mapper.hpp"
#include "core/transform.hpp"
#include <cmath>

namespace Map
{

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
void Mapper::update(const FrameHistory& frame_history, pFrame frame)
{
    const pFrame ref = frame->m_ref_frame;
    const cv::Mat1f xi = frame->m_xi;

    auto inRange = math::generateInRange(ref->depth().size());
    const cv::Mat1f K = frame->top()->K();

    ref->depth().forEach(
        [=](const float d, const int pt[2]) -> void {
            cv::Point2i x_i(pt[1], pt[0]);
            cv::Point2i warped_x_i = Transform::warp(xi, x_i, d, K);
            if (not inRange(warped_x_i))
                return;

            int age = static_cast<int>(frame->m_age(x_i));
            pScene key = frame_history.m_history.at(age)->top();

            float sigma = ref->sigma()(x_i);
            EpipolarSegment es(xi, warped_x_i, K, d + sigma, d - sigma);
            cv::Point2f matched_x_i = doMatching(key->gray(), ref->gray()(warped_x_i), es);

            float new_depth = depthEstimate(
                Convert::toMat1f(matched_x_i),
                Convert::toMat1f(warped_x_i), K, xi);

            // float new_sigma = sigmaEstimate(
            //     frame->m_gradX,
            //     frame->m_gradY,
            //     warped_x_i,
            //     es);

            //  ガウス分布の掛け合わせ
        });
}

float Mapper::depthEstimate(
    const cv::Mat1f& ref_x_i,
    const cv::Mat1f& obj_x_i,
    const cv::Mat1f& K,
    const cv::Mat1f& xi)
{
    if (m_config.is_chatty)
        std::cout << "depthEstimate" << std::endl;
    const cv::Mat1f& x_i = ref_x_i;
    const cv::Mat1f x_q = Transform::backProject(K, obj_x_i, 1);
    const cv::Mat1f t = xi.rowRange(0, 3);
    const cv::Mat1f R = math::se3::exp(xi).colRange(0, 3).rowRange(0, 3);
    const cv::Mat1f r3 = R.row(2);

    const cv::Mat1f a(r3.dot(x_q) * x_i - K * R * x_q);
    const cv::Mat1f b(t(2) * x_i - K * t);

    return static_cast<float>(a.dot(b) / a.dot(a));
}

float Mapper::sigmaEstimate(
    const cv::Mat1f& ref_grad_x,
    const cv::Mat1f& ref_grad_y,
    const cv::Point2f& ref_x_i,
    const EpipolarSegment& es)
{
    if (m_config.is_chatty)
        std::cout << "sigmaEstimate" << std::endl;

    const float alpha = (es.max - es.min) / es.length;

    float gx = ref_grad_x(ref_x_i), gy = ref_grad_y(ref_x_i);
    float lx = (es.start - es.end).x, ly = (es.start - es.end).y;

    // ( \vec{g} \cdot \vec{l} ) ^2
    float gl2 = math::square(gx * lx + gy * ly);
    // ( \vec{g} \cdot \vec{l} ) ^2 /  |\vec{l}|^2
    float g2 = gl2 / math::square(lx * lx + ly * ly);

    float epipolar = m_config.epipolar_variance / gl2;
    float luminance = 2 * m_config.luminance_variance / g2;

    return math::square(alpha) * (epipolar + luminance);
}

cv::Point2f Mapper::doMatching(const cv::Mat1f& ref_gray, const float gray, const EpipolarSegment& es)
{
    cv::Point2f pt = es.start;
    cv::Point2f dir = (es.start - es.end) / es.length;

    cv::Point2f best_pt = pt;
    const int N = 3;
    float min_ssd = N;  // たかだかN

    while (cv::norm(pt - es.start) < es.length) {
        float ssd = 0;
        pt += dir;

        // TODO: 1/Nにできるはず
        for (int i = 0; i < N; i++) {
            float subpixel_gray = Convert::getSubpixel(ref_gray, pt + (i - N / 2) * dir);
            if (math::isInvalid(subpixel_gray)) {
                std::cout << "invalid in doMatching" << std::endl;
            }
            float diff = subpixel_gray - gray;
            ssd += math::square(diff);
        }

        if (ssd < min_ssd) {
            best_pt = pt;
            min_ssd = ssd;
        }
    }
    return best_pt;
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
            Gaussian gauss{d, sigma(p[0], p[1])};

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
