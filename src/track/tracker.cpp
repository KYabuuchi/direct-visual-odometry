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
    m_cur_frames = createFramePyramid(depth_image, gray_image, m_config.intrinsic, m_config.level);
    cv::Mat1f xi = math::se3::xi({0, 0, 0, 0, 0, 0});

    for (int level = 0; level < 5 - 1; level++) {
        const Frame& pre_frame = m_pre_frames.at(level);
        const Frame& cur_frame = m_cur_frames.at(level);

        // TODO: 勾配計算は1回でいい
        // // gradient of gray_image
        // cv::Mat gradient_image_x = Convert::gradiate(cur_frame.m_gray_image, true);
        // cv::Mat gradient_image_y = Convert::gradiate(cur_frame.m_gray_image, false);

        const int COLS = pre_frame.m_cols;
        const int ROWS = pre_frame.m_rows;

        if (m_config.is_chatty)
            std::cout << "\nLEVEL: " << level << " ROW: " << ROWS << " COL: " << COLS << std::endl;

        // vector of residual
        std::vector<float> residuals;
        for (int iteration = 0; iteration < 15; iteration++) {
            std::cout << xi.t() << std::endl;

            cv::Mat warped_gray_image = Transform::warpImage(xi, cur_frame.m_gray_image, cur_frame.m_depth_image, cur_frame.m_intrinsic);

            Scene scene = {
                pre_frame,
                cur_frame,
                warped_gray_image,
                COLS,
                ROWS,
                xi,
                0.0f};

            // show image
            showImage(scene);
            cv::waitKey(0);
            cv::Mat1f xi_update = optimize(scene);

            scene.xi = math::se3::concatenate(scene.xi, -xi_update);
            residuals.push_back(scene.residual);
            assert(math::testXi(scene.xi));
            xi = scene.xi;

            if (m_config.is_chatty)
                std::cout << "iteration: " << iteration << " r: " << scene.residual << " update: " << cv::norm(xi_update) << " xi: " << xi.t() << std::endl;


            if (cv::norm(xi_update) < 0.001 or scene.residual < 0.002)
                break;
        }

        m_vector_of_residuals.push_back(residuals);
    }

    m_pre_frames = std::move(m_cur_frames);
    return xi;
}

cv::Mat1f Tracker::optimize(Scene& scene)
{
    // A xi + B = 0
    cv::Mat1f A(cv::Mat1f::zeros(0, 6));
    cv::Mat1f B(cv::Mat1f::zeros(0, 1));
    float residual = 0;

    cv::Mat gradient_x_image = Convert::gradiate(scene.warped_image, true);
    cv::Mat gradient_y_image = Convert::gradiate(scene.warped_image, false);

    for (int col = 0; col < scene.COL; col++) {
        for (int row = 0; row < scene.ROW; row++) {
            cv::Point2f x_i = cv::Point2f(col, row);
            std::cout << x_i << std::endl;

            // depth
            float depth = scene.cur_frame.m_depth_image.at<float>(x_i);
            if (depth < 0.50) {
                std::cout << "d" << std::endl;
                continue;
            }

            // luminance
            float I_1 = scene.pre_frame.m_gray_image.at<float>(x_i);
            float I_2 = scene.warped_image.at<float>(x_i);
            if (Convert::isInvalid(I_1) or Convert::isInvalid(I_2)) {
                std::cout << "l" << std::endl;
                continue;
            }

            // gradient
            // cv::Point2f warped_x_i = Transform::warp(scene.xi, x_i, depth, scene.cur_frame.m_intrinsic);
            float gx = gradient_x_image.at<float>(x_i);
            float gy = gradient_y_image.at<float>(x_i);
            if (Convert::isInvalid(gx) or Convert::isInvalid(gy)) {
                std::cout << "g" << std::endl;
                continue;
            }

            // calc jacobian
            cv::Mat1f jacobi1 = (cv::Mat1f(1, 2) << gx, gy);
            cv::Mat1f jacobi2 = calcJacobi(scene.cur_frame, x_i, depth);

            // stack
            float r = I_2 - I_1;
            residual += r * r;
            cv::vconcat(A, jacobi1 * jacobi2, A);
            cv::vconcat(B, cv::Mat1f(cv::Mat1f(1, 1) << r), B);
        }
    }
    std::cout << "A" << B.rows << std::endl;
    scene.residual = residual;

    // 最小二乗法を解く
    cv::Mat1f xi_update;
    cv::solve(A, -B, xi_update, cv::DECOMP_SVD);
    return xi_update;
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

    jacobi2(0, 3) = -fx * x * y / z / z;
    jacobi2(1, 3) = -fy * (1.0f + y * y / z / z);
    jacobi2(0, 4) = fx * (1.0f + x * x / z / z);
    jacobi2(1, 4) = fy * x * y / z / z;
    jacobi2(0, 5) = -fx * y / z;
    jacobi2(1, 5) = fy * x / z;
    return jacobi2;
}

// show image
void Tracker::showImage(const Scene& scene)
{
    cv::Mat upper_image, under_image;
    cv::Mat show_image;

    cv::hconcat(std::vector<cv::Mat>{
                    Draw::visiblizeGrayImage(scene.pre_frame.m_gray_image),
                    Draw::visiblizeGrayImage(scene.warped_image),
                    Draw::visiblizeGrayImage(scene.cur_frame.m_gray_image)},
        upper_image);
    cv::hconcat(std::vector<cv::Mat>{
                    Draw::visiblizeDepthImage(scene.pre_frame.m_depth_image),
                    Draw::visiblizeGrayImage(scene.warped_image),
                    Draw::visiblizeDepthImage(scene.cur_frame.m_depth_image),
                },
        under_image);
    cv::vconcat(upper_image, under_image, show_image);
    cv::imshow("tracker-show", show_image);
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
    m_pre_frames = createFramePyramid(depth_image, gray_image, m_config.intrinsic, m_config.level);
    cv::namedWindow("tracker-show", cv::WINDOW_NORMAL);
    cv::resizeWindow("tracker-show", 960, 720);
}
