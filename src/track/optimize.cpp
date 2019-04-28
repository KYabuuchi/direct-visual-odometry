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

Outcome optimize(const Stuff& stuff)
{
    float residual = 0;
    int max_size = stuff.cols * stuff.rows;
    cv::Mat1f A(cv::Mat1f::zeros(max_size, 6));
    cv::Mat1f B(cv::Mat1f::zeros(max_size, 1));
    const float fx = stuff.K(0, 0), fy = stuff.K(1, 1);

    int valid_pixels = 0;

    stuff.ref_depth.forEach(
        [&](float& depth, const int pt[2]) -> void {
            cv::Point2i x_i = cv::Point2i(pt[1], pt[0]);

            // depth
            if (depth < 0.50) {
                return;
                // continue;
            }

            // luminance
            float I_1 = stuff.obj_gray(x_i);
            float I_2 = stuff.warped_gray(x_i);
            if (math::isInvalid(I_1) or math::isInvalid(I_2)) {
                return;
                // continue;
            }

            // gradient
            cv::Point2f warped_x_i = Transform::warp(cv::Mat1f(-stuff.xi), x_i, depth, stuff.K);
            if (warped_x_i.x < 0 or stuff.cols <= warped_x_i.x
                or warped_x_i.y < 0 or stuff.rows <= warped_x_i.y)
                return;
            // continue;

            float gx = Convert::getSubpixel(stuff.grad_x, warped_x_i);
            float gy = Convert::getSubpixel(stuff.grad_y, warped_x_i);

            if (math::isInvalid(gx) or math::isInvalid(gy)) {
                return;
                // continue;
            }

            // calc jacobian
            cv::Point3f x_c = Transform::backProject(stuff.K, x_i, depth);
            float x = x_c.x, y = x_c.y, z = x_c.z;
            float fgx = fx * gx, fgy = fy * gy;
            float xz = x / z, yz = y / z;
            cv::Mat1f jacobi(cv::Mat1f::zeros(1, 6));
            jacobi(0) = fgx / z;
            jacobi(1) = fgy / z;
            jacobi(2) = -(fgx * x + fgy * y) / z / z;
            jacobi(3) = -fgx * xz * yz - fgy * (1.0f + yz * yz);
            jacobi(4) = fgx * (1.0f + xz * xz) + fgy * xz * yz;
            jacobi(5) = (-fgx * yz + fgy * xz);

            // accumulate residual
            float r = huber(I_2 - I_1);
            residual += r * r;

            // weight of reliability
            float sigma = std::max(stuff.ref_sigma(x_i), 0.1f);  // [m]
            float weight = 0.1f / sigma;

            // stack
            int id = pt[1] + pt[0] * stuff.cols;
            jacobi.copyTo(A.row(id));
            B(id, 0) = r * weight;

            valid_pixels++;
        });

    if (valid_pixels == 0)
        return Outcome{math::se3::xi(), -1, B.rows};

    // solve equation (A xi + B = 0)
    cv::Mat1f xi_update;
    cv::solve(A, -B, xi_update, cv::DECOMP_SVD);

    return Outcome{cv::Mat1f(-xi_update), residual / static_cast<float>(B.rows), B.rows};
}

}  // namespace Track