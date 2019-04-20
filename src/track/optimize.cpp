#include "track/optimize.hpp"
#include "math/math.hpp"

namespace Track
{

namespace
{
// NOTE: cv::Mat.atは遅い
inline float at(const cv::Mat& mat, const cv::Point2i& p)
{
    return reinterpret_cast<float*>(mat.data)[p.y * mat.step1() + p.x];
}

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

// NOTE: huber関数の平方根
inline float huber(float n)
{
    constexpr float d = 0.5f;
    constexpr float sqrt2d = sqrt(2.0f * d);
    constexpr float sqrtd2 = sqrt(d / 2.0f);
    constexpr float k0 = sqrtd2;
    constexpr float k1 = 0.5f / sqrtd2;
    constexpr float k2 = -1.0f / (sqrtd2 * d * 2.0f);

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

    cv::Mat1f A(cv::Mat1f::zeros(0, 6));
    cv::Mat1f B(cv::Mat1f::zeros(0, 1));

    for (int col = 0; col < stuff.cols; col++) {
        for (int row = 0; row < stuff.rows; row++) {
            cv::Point2i x_i = cv::Point2i(col, row);


            // depth
            float depth = at(stuff.ref_depth, x_i);
            if (depth < 0.50) {
                continue;
            }

            // luminance
            float I_1 = at(stuff.obj_gray, x_i);
            float I_2 = at(stuff.warped_gray, x_i);
            if (math::isInvalid(I_1) or math::isInvalid(I_2)) {
                continue;
            }

            // gradient
            cv::Point2i warped_x_i = Transform::warp(cv::Mat1f(-stuff.xi), cv::Point2f(x_i), depth, stuff.K);
            if (warped_x_i.x < 0 or stuff.cols <= warped_x_i.x
                or warped_x_i.y < 0 or stuff.rows <= warped_x_i.y)
                continue;

            float gx = at(stuff.grad_x, warped_x_i);
            float gy = at(stuff.grad_y, warped_x_i);

            if (math::isInvalid(gx) or math::isInvalid(gy)) {
                continue;
            }

            // calc jacobian
            cv::Point3f x_c = cv::Point3f(Transform::backProject(stuff.K, Convert::toMat1f(x_i.x, x_i.y), depth));
            float x = x_c.x, y = x_c.y, z = x_c.z;
            float fx = stuff.K(0, 0), fy = stuff.K(1, 1);
            float fgx = fx * gx, fgy = fy * gy;
            cv::Mat1f jacobi = cv::Mat1f(cv::Mat1f::zeros(1, 6));
            jacobi(0) = fgx / z;
            jacobi(1) = fgy / z;
            jacobi(2) = -(fgx * x + fgy * y) / z / z;
            jacobi(3) = -fgx * x * y / z / z - fgy * (1.0f + y / z * y / z);
            jacobi(4) = fgx * (1.0f + x / z * x / z) + fgy * x / z * y / z;
            jacobi(5) = (-fgx * y + fgy * x) / z;


            // stack
            float r = huber(I_2 - I_1);
            residual += r * r;
            cv::vconcat(A, jacobi, A);
            cv::vconcat(B, cv::Mat1f(cv::Mat1f(1, 1) << r), B);
        }
    }
    if (B.rows == 0)
        return Outcome{math::se3::xi(), -1, B.rows};

    // solve equation (A xi + B = 0)
    cv::Mat1f xi_update;
    cv::solve(A, -B, xi_update, cv::DECOMP_SVD);

    return Outcome{cv::Mat1f(-xi_update), residual / static_cast<float>(B.rows), B.rows};
}


}  // namespace Track