#include "track/optimize.hpp"
#include "core/timer.hpp"
#include "math/math.hpp"

namespace Track
{
namespace
{
constexpr double e = std::numeric_limits<double>::epsilon();
template <typename T = float>
constexpr T sqrt(T s)
{
    T x = s / 2.0f;
    T prev = 0.0f;

    while (std::abs(x - prev) > e) {
        prev = x;
        x = (x + s / x) / 2.0f;
    }
    return x;
}

constexpr float d = 0.5f;
constexpr float sqrt2d = sqrt(2.0f * d);
constexpr float sqrtd2 = sqrt(d / 2.0f);
constexpr float k0 = sqrtd2;
constexpr float k1 = 0.5f / sqrtd2;
constexpr float k2 = -1.0f / (sqrtd2 * d * 2.0f);

// NOTE: huber関数の平方根
inline float huber(float n)
{
    float a = std::abs(n);
    if (a < d)
        return n / sqrt2d;

    // 2次近似
    if (0 < n)
        return k0 + (k1 + k2 * (a - d)) * (a - d);

    return -k0 - (k1 + k2 * (a - d)) * (a - d);
}
}  // namespace

// NOTE: ループ内部でメモリ確保すると遅い
cv::Mat1f jacobi(cv::Mat1f::zeros(1, 6));

Outcome optimize(const Stuff& stuff)
{
    float residual = 0;
    const float fx = stuff.K(0, 0), fy = stuff.K(1, 1);

    int max_size = stuff.cols * stuff.rows;
    int valid_pixels = 0;
    cv::Mat1f A(cv::Mat1f::zeros(max_size, 3));
    cv::Mat1f B(cv::Mat1f::zeros(max_size, 1));

    // TODO:
    // step幅
    float step = 2.0f;
    if (stuff.levels == 1)
        step = 1.5f;
    if (stuff.levels == 2)
        step = 1.0f;

    stuff.ref_depth.forEach(
        [&](float& depth, const int pt[2]) -> void {
            cv::Point2i x_i = cv::Point2i(pt[1], pt[0]);

            // 画素数が多いなら外側を捨てる
            if (stuff.levels == 2) {
                if (x_i.x < 20 or x_i.x > 140 or x_i.y < 20 or x_i.y > 100)
                    return;
            }

            // depth
            if (depth < 0.20) {
                return;
            }

            // luminance
            float I_1 = stuff.obj_gray(x_i);
            float I_2 = stuff.warped_gray(x_i);
            if (math::isInvalid(I_1) or math::isInvalid(I_2)) {
                return;
            }

            // gradient
            cv::Point2f warped_x_i = Transform::warp(static_cast<cv::Mat1f>(-stuff.xi), x_i, depth, stuff.K);
            if (warped_x_i.x < 0
                or warped_x_i.y < 0
                or static_cast<float>(stuff.cols) <= warped_x_i.x
                or static_cast<float>(stuff.rows) <= warped_x_i.y)
                return;

            float gx = Convert::getSubpixelFromDense(stuff.grad_x, warped_x_i);
            float gy = Convert::getSubpixelFromDense(stuff.grad_y, warped_x_i);

            if (math::isInvalid(gx) or math::isInvalid(gy)) {
                return;
            }
            valid_pixels++;

            // calc jacobian
            cv::Point3f x_c = Transform::backProject(stuff.K, x_i, depth);
            float x = x_c.x, y = x_c.y, z = x_c.z;
            float fgx = fx * gx, fgy = fy * gy;
            // float xz = x / z, yz = y / z;
            jacobi = cv::Mat1f::zeros(1, 3);
            jacobi(0) = fgx / z;
            jacobi(1) = fgy / z;
            jacobi(2) = -(fgx * x + fgy * y) / z / z;
            // jacobi(3) = -fgx * xz * yz - fgy * (1.0f + yz * yz);
            // jacobi(4) = fgx * (1.0f + xz * xz) + fgy * xz * yz;
            // jacobi(5) = (-fgx * yz + fgy * xz);

            // float r = huber(I_2 - I_1);  // NOTE: huber関数の恩恵が少ない
            float r = I_2 - I_1;
            residual += r * r;

            // weight of reliability
            float sigma = std::clamp(stuff.ref_sigma(x_i), 0.01f, 0.5f);  // [m]
            float weight = step / sigma;

            // stack
            int id = pt[1] + pt[0] * stuff.cols;
            jacobi.copyTo(A.row(id));
            B(id, 0) = r * weight;
        });

    if (valid_pixels == 0)
        return Outcome{math::se3::xi(), -1, valid_pixels};

    // solve equation (A xi + B = 0)
    cv::Mat1f xi_update, update;
    cv::solve(A, -B, update, cv::DECOMP_SVD);
    xi_update = cv::Mat1f::zeros(6, 1);
    for (int i = 0; i < 3; i++) {
        xi_update(i) = update(i);
    }
    return Outcome{cv::Mat1f(-xi_update), residual / static_cast<float>(valid_pixels), valid_pixels};
}

}  // namespace Track