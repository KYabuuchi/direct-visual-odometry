#include "map/implement.hpp"
#include "core/convert.hpp"
#include "math/math.hpp"

namespace Map
{
namespace Implement
{
namespace
{
constexpr float initial_sigma = 0.50f;  // [m]
constexpr float initial_variance = initial_sigma * initial_sigma;

constexpr float luminance_sigma = 0.01f;  // [pixel]
constexpr float luminance_variance = luminance_sigma * luminance_sigma;

constexpr float epipolar_sigma = 0.5f;  // [pixel]
constexpr float epipolar_variance = epipolar_sigma * epipolar_sigma;

constexpr float predict_sigma = 0.10f;  // [m]
constexpr float predict_variance = predict_sigma * predict_sigma;

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
    const cv::Mat1f x_q = cv::Mat1f(Transform::backProject(K, obj_x_i, 1));
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

    // std::cout << es.start << " " << es.end << std::endl;

    cv::Point2f best_pt = pt;
    const int N = 3;
    // TODO:たかだかN
    float min_ssd = N;

    while (cv::norm(pt - es.start) < es.length) {
        float ssd = 0;
        pt += dir;

        // TODO: 1/Nにできるはず
        for (int i = 0; i < N; i++) {
            cv::Point2f target = pt + (i - N / 2) * dir;
            float subpixel_gray = Convert::getSubpixel(ref_gray, target);
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
    return best_pt;
}

}  // namespace

cv::Mat1f regularize(const cv::Mat1f& depth, const cv::Mat1f& sigma)
{
    cv::Mat1f new_depth = depth.clone();

    std::vector<std::pair<int, int>> offsets = {{0, -1}, {0, 1}, {1, 0}, {-1, 0}};
    std::function<bool(cv::Point2i)> inRange = math::generateInRange(depth.size());

    new_depth.forEach(
        [=](float& d, const int p[2]) -> void {
            math::Gaussian g{d, sigma(p[0], p[1])};

            for (const std::pair<int, int> offset : offsets) {
                cv::Point2i pt(p[1] + offset.second, p[0] + offset.first);
                if (not inRange(pt))
                    continue;

                g(depth.at<float>(pt), sigma.at<float>(pt));
            }
            d = g.depth;

            // if (d > 2.0f or d < 0) {
            //     std::cout << d << std::endl;
            //     d = 2.0f;
            // }
        });

    // 遠いのは省く
    new_depth = cv::min(new_depth, 3.0);
    return new_depth;
}

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
    // if (0.1 < new_depth and new_depth < 5)
    //     std::cout << new_depth << " " << matched_x_i << " " << x_i << std::endl;

    float new_sigma = sigmaEstimate(
        ref_gradx,
        ref_grady,
        matched_x_i,
        es);

    return {new_depth, new_sigma};
}

// depth,sigma,age
std::tuple<cv::Mat1f, cv::Mat1f, cv::Mat1f> propagate(
    const cv::Mat1f& ref_depth,
    const cv::Mat1f& ref_sigma,
    const cv::Mat1f& ref_age,
    const cv::Mat1f& xi,
    const cv::Mat1f& K)
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

            cv::Point2f warped_x_i = Transform::warp(xi, x_i, rd, K);
            if (not inRange(warped_x_i))
                return;

            float s = ref_sigma(x_i);
            float d0 = rd;
            float d1 = d0 - tz;
            if (d0 < 0.05)
                s = 0.5;
            else
                s = std::sqrt(math::pow(d1 / d0, 4) * math::square(s)
                              + predict_variance);

            depth(warped_x_i) = std::max(d1, 0.0f);
            sigma(warped_x_i) = s;
            age(warped_x_i) = ref_age(x_i) + 1;
        });

    return {depth, sigma, age};
}

}  // namespace Implement
}  // namespace Map