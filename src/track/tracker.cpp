#include "track/tracker.hpp"
#include "core/convert.hpp"
#include "core/draw.hpp"
#include "core/math.hpp"
#include "core/transform.hpp"
#include "matplotlibcpp.h"

namespace Track
{

cv::Mat1f Tracker::track(const cv::Mat& depth_image, const cv::Mat& gray_image)
{
    assert(m_initialized);
    m_vector_of_residuals.clear();
    m_cur_frames = createFramePyramid(depth_image, gray_image, m_config.intrinsic, m_config.level);
    cv::Mat1f xi = math::se3::xi({0, 0, 0, 0, 0, 0});

    for (int level = 0; level < 5 - 1; level++) {
        const Frame& pre_frame = m_pre_frames.at(level);
        const Frame& cur_frame = m_cur_frames.at(level);
        const int COLS = pre_frame.cols;
        const int ROWS = pre_frame.rows;

        // TODO: 本当は勾配計算はここで1回でいい

        if (m_config.is_chatty)
            std::cout << "\nLEVEL: " << level << " ROW: " << ROWS << " COL: " << COLS << std::endl;

        std::vector<float> residuals;
        for (int iteration = 0; iteration < 15; iteration++) {

            Scene scene = {pre_frame, cur_frame, xi};

            // show image
            showImage(scene);
            cv::waitKey(50);

            // Main Calculation
            Outcome outcome = optimize(scene);

            cv::Mat1f updated_xi = math::se3::concatenate(xi, outcome.xi_update);
            residuals.push_back(outcome.residual);
            if (math::testXi(updated_xi))
                xi = updated_xi;

            if (m_config.is_chatty)
                std::cout << "iteration: " << iteration
                          << " r: " << outcome.residual
                          << " update: " << cv::norm(outcome.xi_update)
                          << " xi: " << xi.t() << std::endl;

            if (cv::norm(outcome.xi_update) < 0.001 or outcome.residual < 0.002)
                break;
        }

        m_vector_of_residuals.push_back(residuals);
    }

    m_pre_frames = std::move(m_cur_frames);
    return xi;
}

// show image
void Tracker::showImage(const Scene& scene)
{
    cv::Mat upper_image, under_image;
    cv::Mat show_image;

    cv::hconcat(std::vector<cv::Mat>{
                    Draw::visiblizeGrayImage(scene.pre_gray),
                    Draw::visiblizeGrayImage(scene.warped_image),
                    Draw::visiblizeGrayImage(scene.cur_gray)},
        upper_image);
    cv::hconcat(std::vector<cv::Mat>{
                    Draw::visiblizeDepthImage(scene.pre_depth),
                    Draw::visiblizeGrayImage(scene.warped_image),
                    Draw::visiblizeDepthImage(scene.cur_depth),
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

}  // namespace Track