#include "track/optimize.hpp"

namespace Track
{
// TODO: jacobi1とjacobi2を分割しない
cv::Mat1f calcJacobi(const cv::Mat1f& intrinsic, cv::Point2f x_i, float depth)
{
    cv::Point3f x_c = cv::Point3f(Transform::backProject(intrinsic, cv::Mat1f(x_i), depth));
    float x = x_c.x, y = x_c.y, z = x_c.z;
    float fx = intrinsic(0, 0);
    float fy = intrinsic(1, 1);

    cv::Mat1f jacobi2 = cv::Mat1f(cv::Mat1f::zeros(2, 6));
    jacobi2(0, 0) = fx / z;
    jacobi2(1, 0) = 0;
    jacobi2(0, 1) = 0;
    jacobi2(1, 1) = fy / z;
    jacobi2(0, 2) = -fx * x / z / z;
    jacobi2(1, 2) = -fy * y / z / z;

    jacobi2(0, 3) = -fx * x * y / z / z;
    jacobi2(1, 3) = -fy * (1.0f + y * y / z / z);
    jacobi2(0, 4) = fx * (1.0f + x * x / z / z);
    jacobi2(1, 4) = fy * x * y / z / z;
    jacobi2(0, 5) = -fx * y / z;
    jacobi2(1, 5) = fy * x / z;
    return jacobi2;
}

Outcome optimize(const Scene& scene)
{
#define WARPED_GRAD
#ifdef WARPED_GRAD
    cv::Mat gradient_x_image = Convert::gradiate(scene.warped_gray, true);
    cv::Mat gradient_y_image = Convert::gradiate(scene.warped_gray, false);
#else
    cv::Mat gradient_x_image = Convert::gradiate(scene.cur_frame.m_gray_image, true);
    cv::Mat gradient_y_image = Convert::gradiate(scene.cur_frame.m_gray_image, false);
#endif

    float residual = 0;
    cv::Mat1f A(cv::Mat1f::zeros(0, 6));
    cv::Mat1f B(cv::Mat1f::zeros(0, 1));

    for (int col = 0; col < scene.cols; col++) {
        for (int row = 0; row < scene.rows; row++) {
            cv::Point2i x_i = cv::Point2i(col, row);
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
            cv::Mat1f jacobi1 = (cv::Mat1f(1, 2) << gx, gy);
            cv::Mat1f jacobi2 = calcJacobi(scene.intrinsic, x_i, depth);

            // stack
            float r = I_2 - I_1;
            residual += r * r;
            cv::vconcat(A, jacobi1 * jacobi2, A);
            cv::vconcat(B, cv::Mat1f(cv::Mat1f(1, 1) << r), B);
        }
    }

    // TODO: A,Bの個数によるreturn

    // solve equation (A xi + B = 0)
    cv::Mat1f xi_update;
    cv::solve(A, -B, xi_update, cv::DECOMP_SVD);

    return Outcome{cv::Mat1f(-xi_update), residual / static_cast<float>(B.rows), B.rows};
}
}  // namespace Track