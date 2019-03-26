#include "track/tracker.hpp"
#include "core/convert.hpp"
#include "core/draw.hpp"
#include "core/math.hpp"
#include "core/transform.hpp"
#include "matplotlibcpp.h"

cv::Mat1f Tracker::track(const cv::Mat& depth_image, const cv::Mat& gray_image)
{
    assert(m_initialized);
    m_vector_of_residuals.clear();
    m_cur_frames = createFramePyramid(depth_image, gray_image, m_config.intrinsic_matrix, m_config.level);
    cv::Mat1f xi = math::se3::xi({0, 0, 0, 0, 0, 0});

    for (int level = 0; level < 5; level++) {
        const Frame& pre_frame = m_pre_frames.at(level);
        const Frame& cur_frame = m_cur_frames.at(level);
        // gradient of gray_image
        cv::Mat gradient_image_x = Convert::gradiate(cur_frame.m_gray_image, true);
        cv::Mat gradient_image_y = Convert::gradiate(cur_frame.m_gray_image, false);

        // vector of residual
        std::vector<float> residuals;

        Scene scene = {
            pre_frame,
            cur_frame,
            pre_frame.m_cols,
            pre_frame.m_rows,
            xi};

        if (m_config.is_chatty)
            std::cout << "\nLEVEL: " << level << " ROW: " << scene.ROW << " COL: " << scene.COL << std::endl;

        optimize(scene);

        xi = scene.xi;
        m_vector_of_residuals.push_back(scene.residuals);
    }

    m_pre_frames = std::move(m_cur_frames);
    return xi;
}

void Tracker::optimize(Scene& scene)
{
    for (int iteration = 0; iteration < 30; iteration++) {
        // A xi + B = 0
        cv::Mat1f A(cv::Mat1f::zeros(0, 6));
        cv::Mat1f B(cv::Mat1f::zeros(0, 1));
        float residual = 0;

        // NOTE: Xi
        cv::Mat warped_gray_image = Transform::warpImage(scene.xi, scene.cur_frame.m_gray_image, scene.cur_frame.m_depth_image, scene.cur_frame.m_intrinsic);
        cv::Mat gradient_x_image = Convert::gradiate(warped_gray_image, true);
        cv::Mat gradient_y_image = Convert::gradiate(warped_gray_image, false);
        // cv::Mat gradient_x_image = Convert::gradiate(scene.cur_frame.m_gray_image, true);
        // cv::Mat gradient_y_image = Convert::gradiate(scene.cur_frame.m_gray_image, false);

        for (int col = 0; col < scene.COL; col++) {
            for (int row = 0; row < scene.ROW; row++) {
                cv::Point2f x_i = cv::Point2f(col, row);

                // depth
                // float depth1 = scene.pre_frame.m_depth_image.at<float>(x_i);
                float depth2 = scene.cur_frame.m_depth_image.at<float>(x_i);
                if (depth2 < 0.50 /*or depth1 < 0.001*/) {
                    // std::cout << "depth: " << x_i << std::endl;
                    continue;
                }

                // luminance
                float I_1 = scene.pre_frame.m_gray_image.at<float>(x_i);
                float I_2 = warped_gray_image.at<float>(x_i);
                if (Convert::isInvalid(I_1) or Convert::isInvalid(I_2)) {
                    // std::cout << "luminance: " << x_i << std::endl;
                    continue;
                }

                // gradient
                cv::Point2f warped_x_i = Transform::warp(scene.xi, x_i, depth2, scene.cur_frame.m_intrinsic);
                float gx = Convert::getColorSubpix(gradient_x_image, warped_x_i);
                float gy = Convert::getColorSubpix(gradient_y_image, warped_x_i);
                // float gx = gradient_x_image.at<float>(warped_x_i);
                // float gy = gradient_y_image.at<float>(warped_x_i);
                if (Convert::isInvalid(gx) or Convert::isInvalid(gy)) {
                    // std::cout << "grad: " << warped_x_i << " " << x_i << " " << gx << " " << gy << std::endl;
                    continue;
                }

                // integral residual
                float r = I_2 - I_1;
                residual += r * r;

                // calc jacobian
                cv::Mat1f jacobi1 = (cv::Mat1f(1, 2) << gx, gy);
                cv::Mat1f jacobi2 = calcJacobi(scene.cur_frame, x_i, depth2);

                // stack coefficient
                cv::vconcat(A, jacobi1 * jacobi2, A);
                cv::vconcat(B, cv::Mat1f(cv::Mat1f(1, 1) << r * 1.5), B);

                std::cout << jacobi1 << " " << x_i << " " << r << std::endl;
            }
        }

        cv::Mat tmp;
        cv::hconcat(A, B, tmp);
        std::cout << tmp << std::endl;

        // 最小二乗法を解く
        cv::Mat1f xi_update;
        cv::solve(A, -B, xi_update, cv::DECOMP_SVD);
        scene.xi = math::se3::concatenate(scene.xi, -xi_update);
        scene.residuals.push_back(residual);
        assert(math::testXi(scene.xi));

        int stack = B.rows;
        if (m_config.is_chatty)
            std::cout << "iteration: " << iteration << " stack: " << stack << " r: " << residual / stack << " update: " << xi_update.t() << /*" xi: " << scene.xi.t() <<*/ std::endl;

        // show image
        showImage(scene, warped_gray_image, Draw::visiblizeGradientImage(gradient_x_image, gradient_y_image));
        if (cv::waitKey(0) == 'q')
            return;
    }
}


cv::Mat1f Tracker::calcJacobi(const Frame& frame, cv::Point2f x_i, float depth)
{
    cv::Point3f x_c = cv::Point3f(Transform::backProject(frame.m_intrinsic, cv::Mat1f(x_i), depth));
    float x = x_c.x, y = x_c.y, z = x_c.z;
    float fx = frame.m_intrinsic(0, 0);
    float fy = frame.m_intrinsic(1, 1);

    cv::Mat1f jacobi2 = cv::Mat1f(cv::Mat1f::zeros(2, 6));
    jacobi2(0, 0) = fx / z;
    jacobi2(1, 0) = 0;
    jacobi2(0, 1) = 0;
    jacobi2(1, 1) = fy / z;
    jacobi2(0, 2) = -fx * x / z / z;
    jacobi2(1, 2) = -fy * y / z / z;

    // jacobi2(0, 3) = -fx * x * y / z / z;
    // jacobi2(1, 3) = -fy * (1 + y * y / z / z);
    // jacobi2(0, 4) = fx * (1 + x * x / z / z);
    // jacobi2(1, 4) = fy * x * y / z / z;
    // jacobi2(0, 5) = -fx * y / z;
    // jacobi2(1, 5) = fy * x / z;
    return jacobi2;
}

// show image
void Tracker::showImage(const Scene& scene, const cv::Mat& warped_image, const cv::Mat& grad_image)
{
    cv::Mat upper_image, under_image;
    cv::Mat show_image;

    cv::hconcat(std::vector<cv::Mat>{
                    Draw::visiblizeGrayImage(scene.pre_frame.m_gray_image),
                    Draw::visiblizeGrayImage(warped_image),
                    Draw::visiblizeGrayImage(scene.cur_frame.m_gray_image)},
        upper_image);
    cv::hconcat(std::vector<cv::Mat>{
                    Draw::visiblizeDepthImage(scene.pre_frame.m_depth_image),
                    grad_image,
                    Draw::visiblizeDepthImage(scene.cur_frame.m_depth_image),
                },
        under_image);
    cv::vconcat(upper_image, under_image, show_image);
    cv::imshow("show", show_image);
}


void Tracker::plot(bool block)
{
    namespace plt = matplotlibcpp;
    for (size_t i = 0; i < m_vector_of_residuals.size(); i++) {
        plt::subplot(1, m_vector_of_residuals.size(), i + 1);
        plt::plot(m_vector_of_residuals.at(i));
    }
    plt::show(block);
}

void Tracker::init(cv::Mat depth_image, cv::Mat gray_image)
{
    m_initialized = true;
    m_pre_frames = createFramePyramid(depth_image, gray_image, m_config.intrinsic_matrix, m_config.level);
}
