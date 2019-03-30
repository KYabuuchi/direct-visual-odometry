#include "track/optimize.hpp"
#include "core/math.hpp"

namespace Track
{
Outcome optimize(const Scene& scene)
{
#define WARPED_GRAD
#ifdef WARPED_GRAD
    cv::Mat gradient_x_image = Convert::gradiate(scene.warped_gray, true);
    cv::Mat gradient_y_image = Convert::gradiate(scene.warped_gray, false);
#else
    cv::Mat gradient_x_image = Convert::gradiate(scene.cur_gray, true);
    cv::Mat gradient_y_image = Convert::gradiate(scene.cur_gray, false);
#endif

    float residual = 0;
    cv::Mat1f A(cv::Mat1f::zeros(0, 6));
    cv::Mat1f B(cv::Mat1f::zeros(0, 1));

    for (int col = 0; col < scene.cols; col++) {
        for (int row = 0; row < scene.rows; row++) {
            cv::Point2i x_i = cv::Point2i(col, row);
            // TODO: cv::Mat.atは遅い

            // depth
            float depth = scene.cur_depth.at<float>(x_i);
            if (depth < 0.50) {
                continue;
            }

            // luminance
            float I_1 = scene.pre_gray.at<float>(x_i);
            float I_2 = scene.warped_gray.at<float>(x_i);
            if (Convert::isInvalid(I_1) or Convert::isInvalid(I_2)) {
                continue;
            }

            // gradient
#ifdef WARPED_GRAD
            float gx = gradient_x_image.at<float>(x_i);
            float gy = gradient_y_image.at<float>(x_i);
#else
            cv::Point2f warped_x_i = Transform::warp(scene.xi, x_i, depth, scene.intrinsic);
            float gx = gradient_x_image.at<float>(warped_x_i);
            float gy = gradient_y_image.at<float>(warped_x_i);
#endif
            if (Convert::isInvalid(gx) or Convert::isInvalid(gy)) {
                continue;
            }

            // calc jacobian
            cv::Point3f x_c = cv::Point3f(Transform::backProject(scene.intrinsic, cv::Mat1f(cv::Point2f(x_i)), depth));
            float x = x_c.x, y = x_c.y, z = x_c.z;
            float fx = scene.intrinsic(0, 0), fy = scene.intrinsic(1, 1);
            float fgx = fx * gx, fgy = fy * gy;
            cv::Mat1f jacobi = cv::Mat1f(cv::Mat1f::zeros(1, 6));
            jacobi(0) = fgx / z;
            jacobi(1) = fgy / z;
            jacobi(2) = -(fgx * x + fgy * y) / z / z;
            jacobi(3) = -fgx * x * y / z / z - fgy * (1.0f + y / z * y / z);
            jacobi(4) = fgx * (1.0f + x / z * x / z) + fgy * x / z * y / z;
            jacobi(5) = (-fgx * y + fgy * x) / z;


            // stack
            float r = I_2 - I_1;
            residual += r * r;
            cv::vconcat(A, jacobi, A);
            cv::vconcat(B, cv::Mat1f(cv::Mat1f(1, 1) << r), B);
        }
    }

    // TODO: A,Bの個数によるreturn
    if (B.rows == 0)
        return Outcome{math::se3::xi(), -1, B.rows};

    // solve equation (A xi + B = 0)
    cv::Mat1f xi_update;
    cv::solve(A, -B, xi_update, cv::DECOMP_SVD);

    return Outcome{cv::Mat1f(-xi_update), residual / static_cast<float>(B.rows), B.rows};
}
}  // namespace Track