#include "map/updater.hpp"
#include "core/convert.hpp"
#include "math/math.hpp"

namespace Map
{
namespace Update
{
namespace
{
constexpr float initial_sigma = 0.50f;  // [m]
constexpr float initial_variance = initial_sigma * initial_sigma;

constexpr float luminance_sigma = 0.01f;  // [pixel]
constexpr float luminance_variance = luminance_sigma * luminance_sigma;

constexpr float epipolar_sigma = 0.5f;  // [pixel]
constexpr float epipolar_variance = epipolar_sigma * epipolar_sigma;

// Eipolar線分
struct EpipolarSegment {
    EpipolarSegment(
        const cv::Mat1f& xi,
        const cv::Point2i& x_i,
        const cv::Mat1f& K,
        const float depth,
        const float sigma)
        : min(depth - sigma), max(depth + sigma),
          start(Transform::warp(xi, x_i, max, K)),
          end(Transform::warp(xi, x_i, min, K)),
          length(static_cast<float>(cv::norm(start - end))) {}

    // copy constractor
    EpipolarSegment(const EpipolarSegment& es)
        : min(es.min), max(es.max), start(es.start), end(es.end), length(es.length) {}

    const float min;
    const float max;
    const cv::Point2f start;
    const cv::Point2f end;
    const float length;
};

float depthEstimate(
    const cv::Point2f& ref_x_i,
    const cv::Point2f& obj_x_i,
    const cv::Mat1f& K,
    const cv::Mat1f& xi)
{
    const cv::Mat1f x_q = Transform::backProject(K, cv::Mat1f(obj_x_i), 1);
    const cv::Mat1f t = xi.rowRange(0, 3);
    const cv::Mat1f R = math::se3::exp(xi).colRange(0, 3).rowRange(0, 3);
    const cv::Mat1f r3 = R.row(2);

    cv::Mat1f x_i = Convert::toMat1f(ref_x_i.x, ref_x_i.y, 1.0f);
    const cv::Mat1f a(r3.dot(x_q.t()) * x_i - K * R * x_q);
    const cv::Mat1f b(t(2) * x_i - K * t);

    return -static_cast<float>(a.dot(b) / a.dot(a));
}

float sigmaEstimate(
    const cv::Mat1f& ref_grad_x,
    const cv::Mat1f& ref_grad_y,
    const cv::Point2f& ref_x_i,
    const EpipolarSegment& es)
{
    std::cout << "sigmaEstimate" << std::endl;

    const float alpha = (es.max - es.min) / es.length;

    float gx = ref_grad_x(ref_x_i), gy = ref_grad_y(ref_x_i);
    float lx = (es.start - es.end).x, ly = (es.start - es.end).y;

    // ( \vec{g} \cdot \vec{l} ) ^2
    float gl2 = math::square(gx * lx + gy * ly);
    // ( \vec{g} \cdot \vec{l} ) ^2 /  |\vec{l}|^2
    float g2 = gl2 / math::square(lx * lx + ly * ly);

    float epipolar = epipolar_variance / gl2;
    float luminance = 2 * luminance_variance / g2;

    return alpha * std::sqrt(epipolar + luminance);
}

cv::Point2f doMatching(const cv::Mat1f& ref_gray, const float gray, const EpipolarSegment& es)
{
    cv::Point2f pt = es.start;
    cv::Point2f dir = (es.end - es.start) / es.length;

    cv::Point2f best_pt = pt;
    const int N = 3;
    // TODO:たかだかN
    float min_ssd = N;

    while (cv::norm(pt - es.start) < es.length) {
        float ssd = 0;
        pt += dir;

        // TODO: 1/Nにできるはず
        for (int i = 0; i < N; i++) {
            float subpixel_gray = Convert::getSubpixel(ref_gray, pt + (i - N / 2) * dir);
            if (math::isInvalid(subpixel_gray)) {
                ssd = N;
                break;
            }
            float diff = subpixel_gray - gray;
            ssd += math::square(diff);
        }

        if (ssd < min_ssd) {
            best_pt = pt;
            min_ssd = ssd;
        }
    }
    if (min_ssd == N) {
        return cv::Point2f(-1, -1);
    }
    std::cout << "best match " << best_pt << " " << min_ssd << std::endl;
    return best_pt;
}

}  // namespace


// 本体
std::tuple<float, float> update(
    const cv::Mat1f& obj_gray,
    const cv::Mat1f& ref_gray,
    const cv::Mat1f& ref_gradx,
    const cv::Mat1f& ref_grady,
    const cv::Mat1f& relative_xi,
    const cv::Mat1f& K,
    const cv::Point2i& x_i,
    float depth,
    float sigma)
{
    EpipolarSegment es(relative_xi, x_i, K, depth, sigma);

    cv::Point2f matched_x_i = doMatching(ref_gray, obj_gray(x_i), es);
    if (matched_x_i.x < 0)
        return {-1, -1};

    float new_depth = depthEstimate(matched_x_i, x_i, K, relative_xi);
    std::cout << new_depth << std::endl;
    float new_sigma = sigmaEstimate(
        ref_gradx,
        ref_grady,
        matched_x_i,
        es);

    return {new_depth, new_sigma};
}

}  // namespace Update
}  // namespace Map