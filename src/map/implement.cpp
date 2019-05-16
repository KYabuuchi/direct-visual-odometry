#include "map/implement.hpp"
#include "core/convert.hpp"
#include "math/math.hpp"

namespace Map
{
namespace Implement
{
namespace
{
// update
constexpr float luminance_sigma = 0.6f;  // Epipolar直線に沿った輝度勾配による分散
constexpr float luminance_variance = luminance_sigma * luminance_sigma;
constexpr float epipolar_sigma = 0.6f;  // 等高線とEpipolar Lineが成す角度による分散
constexpr float epipolar_variance = epipolar_sigma * epipolar_sigma;
// propagate
constexpr float predict_sigma = 0.08f;  // [m]
constexpr float predict_variance = predict_sigma * predict_sigma;
// doMatching
constexpr double MATCHING_THRESHOLD_RATIO = 0.1;

// Eipolar線分
struct EpipolarSegment {
    EpipolarSegment(
        const cv::Mat1f& xi,
        const cv::Point2i& x_i,
        const cv::Mat1f& K,
        const float depth,
        const float sigma)
        : min(std::max(depth - sigma, 0.10f)), max(depth + sigma),
          start(Transform::warp(static_cast<cv::Mat1f>(xi), x_i, max, K)),
          end(Transform::warp(static_cast<cv::Mat1f>(xi), x_i, min, K)),
          length(static_cast<float>(cv::norm(start - end))),
          x_i(x_i), xi(xi) {}

    // copy constractor
    EpipolarSegment(const EpipolarSegment& es)
        : min(es.min), max(es.max), start(es.start), end(es.end), length(es.length) {}

    const float min;
    const float max;
    const cv::Point2f start;
    const cv::Point2f end;
    const float length;
    const cv::Point2f x_i;
    const cv::Mat1f xi;
};

float depthEstimate(
    const cv::Point2f& ref_x_i,
    const cv::Point2f& obj_x_i,
    const cv::Mat1f& K,
    const cv::Mat1f& xi)
{
    const cv::Mat1f x_q = cv::Mat1f(Transform::backProject(K, obj_x_i, 1));
    const cv::Mat1f t = -xi.rowRange(0, 3);
    // const cv::Mat1f R = cv::Mat1f::eye(3, 3);
    const cv::Mat1f R = math::se3::exp(-xi).colRange(0, 3).rowRange(0, 3);
    const cv::Mat1f r3 = R.row(2);

    cv::Mat1f x_i = Convert::toMat1f(ref_x_i.x, ref_x_i.y, 1.0f);
    const cv::Mat1f a(r3.dot(x_q.t()) * x_i - K * R * x_q);
    const cv::Mat1f b(t(2) * x_i - K * t);

    float depth = -static_cast<float>(a.dot(b) / a.dot(a));

    // if (ref_x_i.x < 50 and ref_x_i.x > 20 and depth < 1)
    // if (ref_x_i.x < 50 and ref_x_i.x > 20 and ref_x_i.y < 50 and ref_x_i.y > 30)
    //     std::cout << ref_x_i << " " << obj_x_i << " " << depth << " " << t.t() << " a " << a.t() << " b " << b.t() << " q " << x_q.t() << std::endl;
    return depth;
}

float sigmaEstimate(
    const cv::Mat1f& ref_grad_x,
    const cv::Mat1f& ref_grad_y,
    const cv::Point2f& ref_x_i,
    const EpipolarSegment& es)
{
    float l = es.length;
    float lx = (es.start - es.end).x / l, ly = (es.start - es.end).y / l;

    const float alpha = (es.max - es.min) / l;

    float gx = ref_grad_x(ref_x_i), gy = ref_grad_y(ref_x_i);
    if (math::isInvalid(gx) or math::isInvalid(gy)) {
        return -1;
    }

    float g_dot_l = std::abs(gx * lx + gy * ly);
    // ( \vec{g} \cdot \vec{l} ) ^2
    float g_dot_l2 = math::square(g_dot_l);
    // ( \vec{g} \cdot \vec{l} )  /  |\vec{l}|
    float gp2 = g_dot_l / l;

    float epipolar = epipolar_variance / std::max(g_dot_l2, math::EPSILON);
    float luminance = 2 * luminance_variance / std::max(gp2, math::EPSILON);

    float sigma = alpha * std::sqrt(epipolar + luminance);

    // if (ref_x_i.x < 90 and ref_x_i.x > 30 and sigma < 1)
    //     std::cout << sigma << " " << ref_x_i << " " << gx << "," << gy << " " << lx << "," << ly << std::endl;

    return sigma;
}

cv::Point2f doMatching(const cv::Mat1f& ref_gray, const float obj_gray, const EpipolarSegment& es)
{
    // 探索幅
    cv::Point2f dir = (es.end - es.start) / es.length;
    cv::Point2f pt = es.start;
    // std::cout << es.start << " " << es.end << " " << es.x_i << " " << es.min << " " << es.max << std::endl;

    cv::Point2f best_pt = pt;
    const int N = 3;
    const int center = (N + 1) / 2;
    // NOTE:たかだかN
    float min_ssd = 2.0f * N;

    while (cv::norm(pt - es.start) < es.length) {
        float ssd = 0;
        pt += dir;

        // TODO: 1/Nにできるはず
        for (int i = 0; i < N; i++) {
            cv::Point2f target = pt + (i - N / 2) * dir;
            float subpixel_gray = Convert::getSubpixel(ref_gray, target);
            if (math::isInvalid(subpixel_gray)) {
                ssd = 2 * N;
                break;
            }
            float diff = subpixel_gray - obj_gray;
            // 中央ほど，強く加味
            ssd += 1.0 * (N - std::abs(i - center)) / N * math::square(diff);
        }

        if (ssd < min_ssd) {
            best_pt = pt;
            min_ssd = ssd;
        }
    }
    if (min_ssd > N * MATCHING_THRESHOLD_RATIO) {
        return cv::Point2f(-1, -1);
    }

    // if (es.x_i.x < 50 and es.x_i.x > 20 and es.x_i.y < 50 and es.x_i.y > 30)
    //     std::cout << es.start << " " << es.end << " " << es.x_i << " " << best_pt << " " << es.xi.t() << std::endl;
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

            // TODO: 近いほうが優遇されないようにする
            for (const std::pair<int, int> offset : offsets) {
                cv::Point2i pt(p[1] + offset.second, p[0] + offset.first);
                if (not inRange(pt))
                    continue;

                g(depth.at<float>(pt), sigma.at<float>(pt));
            }
            d = g.depth;
        });

    // 遠いのは省く
    new_depth = cv::min(new_depth, 6.0);
    return new_depth;
}

std::pair<float, float> update(
    const cv::Mat1f& obj_gray,
    const cv::Mat1f& ref_gray,
    const cv::Mat1f& ref_gradx,
    const cv::Mat1f& ref_grady,
    const cv::Mat1f& relative_xi,
    const cv::Mat1f& K,
    const cv::Point2i& x_i,  // obj座標系
    float depth,
    float sigma)
{
    EpipolarSegment es(-relative_xi, x_i, K, depth, sigma);

    cv::Point2f matched_x_i = doMatching(ref_gray, obj_gray(x_i), es);  // ref座標系
    if (matched_x_i.x < 0
        or matched_x_i.y < 0
        or matched_x_i.x > obj_gray.cols
        or matched_x_i.y > obj_gray.rows)
        return {-1, -1};

    float new_depth = depthEstimate(matched_x_i, x_i, K, relative_xi);
    float new_sigma = sigmaEstimate(
        ref_gradx,
        ref_grady,
        matched_x_i,
        es);

    // if (new_sigma > 0 and new_sigma < 1 and new_depth < 0.8 and x_i.x < 70)
    //     std::cout << "update " << x_i << " " << matched_x_i << " " << new_depth << " " << new_sigma << " " << relative_xi.t()
    //               << std::endl;

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

    // TODO: depthとsigmaを正しく初期化
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
            float d0 = std::max(rd, 0.01f);
            float d1 = d0 + tz;

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