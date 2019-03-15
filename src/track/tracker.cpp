#include "track/tracker.hpp"
#include "core/convert.hpp"
#include "core/math.hpp"
#include "core/transform.hpp"
#include "matplotlibcpp.h"

void Tracker::optimize(Scene& scene)
{
    for (int iteration = 0; iteration < 10; iteration++) {
        // A xi + B = 0
        cv::Mat1f A(cv::Mat1f::zeros(0, 6));
        cv::Mat1f B(cv::Mat1f::zeros(0, 1));
        float residual = 0;

        for (int col = 0; col < scene.COL; col += 1) {
            for (int row = 0; row < scene.ROW; row += 1) {
                cv::Point2f x_i = cv::Point2f(col, row);

                // get depth
                float depth1 = scene.pre_frame.m_depth_image.at<float>(x_i);
                float depth2 = scene.cur_frame.m_depth_image.at<float>(x_i);
                if (depth2 < 0.001 or depth1 < 0.001)  //   1[mm]
                    continue;

                // get warped coordinate
                cv::Point2f wapred_x_i = Transform::warp(scene.xi, x_i, depth2, scene.pre_frame.m_intrinsic);
                if ((!math::isRange(wapred_x_i.x, 0, scene.COL)) or (!math::isRange(wapred_x_i.y, 0, scene.ROW)))
                    continue;

                // get luminance
                float I_1 = scene.pre_frame.m_gray_image.at<float>(x_i);
                float I_2 = Convert::getColorSubpix(scene.cur_frame.m_gray_image, wapred_x_i);
                scene.warped_image.at<float>(x_i) = I_2;
                if (Convert::isInvalid(I_1) or Convert::isInvalid(I_2))
                    continue;

                // get gradient
                float gx = Convert::getColorSubpix(scene.gradient_x_image, wapred_x_i);
                float gy = Convert::getColorSubpix(scene.gradient_y_image, wapred_x_i);
                if (Convert::isInvalid(gx) or Convert::isInvalid(gy))
                    continue;

                // calc residual
                float r = I_2 - I_1;
                residual += r * r;

                // calc jacobian
                cv::Mat1f jacobi1 = (cv::Mat1f(1, 2) << gx, gy);
                cv::Mat1f jacobi2 = calcJacobi(scene.cur_frame, x_i, depth2);

                // stack coefficient
                cv::vconcat(A, jacobi1 * jacobi2, A);
                cv::vconcat(B, cv::Mat1f(cv::Mat1f(1, 1) << r), B);
            }
        }
        showImage(scene);

        // 最小二乗法を解く
        cv::Mat1f xi_update;
        cv::solve(A, -B, xi_update, cv::DECOMP_SVD);
        scene.xi = math::se3::concatenate(scene.xi, xi_update);
        scene.residuals.push_back(residual);
        assert(math::testXi(scene.xi));

        if (m_config.is_chatty)
            std::cout << "iteration: " << iteration << " r : " << residual << " update " << cv::norm(xi_update) << std::endl;
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
    jacobi2(0, 3) = -fx * x * y / z / z;
    jacobi2(1, 3) = -fy * (1 + y * y / z / z);
    jacobi2(0, 4) = fx * (1 + x * x / z / z);
    jacobi2(1, 4) = fy * x * y / z / z;
    jacobi2(0, 5) = -fx * y / z;
    jacobi2(1, 5) = fy * x / z;
    return jacobi2;
}

namespace
{
cv::Mat visiblizeGrayImage(const cv::Mat& src_image)
{
    cv::Mat dst_image;
    src_image.convertTo(dst_image, CV_8UC1, 255);            // change valud-depth
    cv::cvtColor(dst_image, dst_image, cv::COLOR_GRAY2BGR);  // change number of channel
    dst_image.forEach<cv::Vec3b>(
        [](cv::Vec3b& p, const int* position) -> void {
            if (p[0] != 0)
                return;
            p[2] = 255;
        });
    return dst_image;
}

cv::Mat visiblizeDepthImage(const cv::Mat& src_image)
{
    cv::Mat dst_image;
    src_image.convertTo(dst_image, CV_8UC1, 100);            // change valud-depth
    cv::cvtColor(dst_image, dst_image, cv::COLOR_GRAY2BGR);  // change number of channel
    dst_image.forEach<cv::Vec3b>(
        [](cv::Vec3b& p, const int* position) -> void {
            if (p[0] != 0)
                return;
            p[2] = 255;
        });
    return dst_image;
}

cv::Mat visiblizeGradientImage(const cv::Mat& x_image, const cv::Mat& y_image)
{
    cv::Mat dst_image = cv::Mat::zeros(x_image.size(), CV_8UC3);
    cv::Mat normalized_x_image, normalized_y_image;
    x_image.convertTo(normalized_x_image, CV_8UC1, 127, 127);
    y_image.convertTo(normalized_y_image, CV_8UC1, 127, 127);
    dst_image.forEach<cv::Vec3b>(
        [=](cv::Vec3b& p, const int* position) -> void {
            if (Convert::isInvalid(x_image.at<float>(position[0], position[1]))
                or (Convert::isInvalid(x_image.at<float>(position[0], position[1])))) {
                p[2] = 255;
                return;
            }

            p[0] = normalized_x_image.at<unsigned char>(position[0], position[1]);
            p[1] = normalized_y_image.at<unsigned char>(position[0], position[1]);
        });
    return dst_image;
}

}  // namespace

// show image
void Tracker::showImage(const Scene& scene)
{
    cv::Mat upper_image, under_image;
    cv::Mat show_image;

    cv::hconcat(std::vector<cv::Mat>{
                    visiblizeGrayImage(scene.pre_frame.m_gray_image),
                    visiblizeGrayImage(scene.warped_image),
                    visiblizeGrayImage(scene.cur_frame.m_gray_image)},
        upper_image);
    cv::hconcat(std::vector<cv::Mat>{
                    visiblizeDepthImage(scene.pre_frame.m_depth_image),
                    visiblizeGradientImage(scene.gradient_x_image, scene.gradient_y_image),
                    visiblizeDepthImage(scene.cur_frame.m_depth_image),
                },
        under_image);
    cv::vconcat(upper_image, under_image, show_image);
    cv::imshow("show", show_image);
    int key = cv::waitKey(0);
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


cv::Mat1f Tracker::track(const cv::Mat& depth_image, const cv::Mat& gray_image)
{
    assert(m_initialized);
    m_vector_of_residuals.clear();
    m_cur_frames = createFramePyramid(depth_image, gray_image, m_config.intrinsic_matrix, m_config.level);
    cv::Mat1f xi = cv::Mat1f(cv::Mat1f::zeros(6, 1));

    for (int level = 0; level < 4; level++) {
        const Frame& pre_frame = m_pre_frames.at(level);
        const Frame& cur_frame = m_cur_frames.at(level);
        // gradient of gray_image
        cv::Mat gradient_image_x = Convert::gradiate(cur_frame.m_gray_image, true);
        cv::Mat gradient_image_y = Convert::gradiate(cur_frame.m_gray_image, false);

        // vector of residual
        std::vector<float> residuals;

        cv::Mat warped_image(cur_frame.m_depth_image.size(), CV_32FC1, Convert::INVALID);
        Scene scene = {
            pre_frame,
            cur_frame,
            gradient_image_x,
            gradient_image_y,
            pre_frame.m_cols,
            pre_frame.m_rows,
            warped_image,
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
