#include "map/mapper.hpp"
#include "core/transform.hpp"
#include <cmath>

namespace Map
{

void Mapper::estimate(FrameHistory frame_history, pFrame frame)
{
    if (frame_history.size() == 0) {
        initializeHistory(frame_history, frame);
    } else {

        if (needNewFrame(frame)) {
            propagate(frame_history, frame);
        } else {
            // update(frame_history, frame);
        }
    }
    regularize(frame_history);
}

bool Mapper::needNewFrame(pFrame frame)
{
    cv::Mat1f& xi = frame->m_relative_xi;
    double scalar = cv::norm(xi.rowRange(0, 3));
    return scalar > m_config.minimum_movement;
}

void Mapper::initializeHistory(const cv::Mat1f& depth, cv::Mat1f& sigma)
{
    if (m_config.is_chatty)
        std::cout << "init History" << std::endl;
    sigma.forEach(
        [&](float& s, const int p[2]) -> void {
            if (depth(p[0], p[1]) > 0.01f) {
                s = 0.1f;  // accuracy of kinect
            } else {
                s = m_config.initial_sigma;
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
            cv::Point2i x_i(pt[1], pt[0]);
            if (math::isEpsilon(rd))
                return;

            float s = ref_sigma(x_i);
            float d0 = rd;
            float d1 = d0 - tz;
            if (s > 0.5 or d0 < 0.05)
                s = m_config.initial_sigma;
            else
                s = std::sqrt(math::pow(d1 / d0, 4) * math::square(s)
                              + m_config.predict_variance);

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

    const cv::Mat1f a = r3.dot(x_q) * x_i - K * R * x_q;
    const cv::Mat1f b = t(2) * x_i - K * t;

    return a.dot(b) / a.dot(a);
}

float Mapper::sigmaEstimate(
    const cv::Mat1f& ref_grad_x,
    const cv::Mat1f& ref_grad_y,
    const cv::Point2f& ref_x_i,
    const EpipolarSegment& es)
{
    if (m_config.is_chatty)
        std::cout << "sigmaEstimate" << std::endl;

    const float lambda = cv::norm(es.start - es.end);
    const float alpha = (es.max - es.min) / lambda;

    float gx = ref_grad_x(ref_x_i), gy = ref_grad_y(ref_x_i);
    float lx = (es.start - es.end).x, ly = (es.start - es.end).y;

    // ( \vec{g} \cdot \vec{l} ) ^2
    float gl2 = math::square(gx * lx + gy * ly);
    // ( \vec{g} \cdot \vec{l} ) ^2 /  |\vec{l}|^2
    float g2 = gl2 / math::square(lx * lx + ly * ly);

    float epipolar = m_config.epipolar_variance / gl2;
    float luminance = 2 * m_config.luminance_variance / g2;

    return  math::square(alpha) * (epipolar + luminance);
}


}  // namespace Map
