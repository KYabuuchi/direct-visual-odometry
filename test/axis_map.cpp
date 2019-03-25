#include "calibration/loader.hpp"
#include "core/draw.hpp"
#include "core/loader.hpp"
#include "core/math.hpp"
#include "core/transform.hpp"
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

int main()
{
    using namespace math;

    // window
    cv::namedWindow("map", cv::WINDOW_NORMAL);
    cv::resizeWindow("map", 960, 720);
    cv::namedWindow("warp", cv::WINDOW_NORMAL);
    cv::resizeWindow("warp", 960, 720);

    // trackbar
    const int MAX = 100;
    std::array<int, 6> params = {MAX / 2, MAX / 2, MAX / 2, MAX / 2, MAX / 2, MAX / 2};
    for (int i = 0; i < 6; i++) {
        cv::createTrackbar("xi" + std::to_string(i), "map", &params.at(i), MAX);
    }

    // viz
    cv::viz::Viz3d viz_window("3D-VIEW");
    viz_window.showWidget("coordinate", cv::viz::WCoordinateSystem(0.5));
    viz_window.setWindowSize(cv::Size(320, 240));

    // load
    cv::Mat rgb_image, depth_image;
    cv::Mat mapped_gray_image, mapped_depth_image;
    Loader image_loader("../data/KINECT_1DEG/info.txt");
    Calibration::Loader config_loader("../camera-calibration/data/kinectv2_00/config.yaml");
    Params::init(config_loader.rgb(), config_loader.depth(), config_loader.extrinsic());
    image_loader.getNormalizedUndistortedImages(0, rgb_image, depth_image);
    mapped_gray_image = Convert::cullImage(Transform::mapDepthtoGray(depth_image, rgb_image));
    mapped_depth_image = Convert::cullImage(depth_image);
    mapped_depth_image = Convert::cullImage(mapped_depth_image);
    mapped_gray_image = Convert::cullImage(mapped_gray_image);

    while (1) {
        std::cout << "1" << std::endl;
        std::array<float, 6> params_f;
        for (int i = 0; i < 6; i++) {
            params_f.at(i) = static_cast<float>(params.at(i) - 50) / 50.0f;
        }

        // TODO:
        cv::Mat1f xi = cv::Mat(6, 1, CV_32FC1, params_f.data());
        std::cout << "A" << xi << std::endl;

        // get warped coordinate
        cv::Mat warped_gray_image = cv::Mat(mapped_gray_image.size(), mapped_gray_image.type(), Convert::INVALID);
        cv::Mat warped_depth_image = cv::Mat(mapped_depth_image.size(), mapped_depth_image.type(), Convert::INVALID);
        const int COL = warped_depth_image.cols;
        const int ROW = warped_depth_image.rows;
        std::cout << COL << " " << ROW << " " << xi << std::endl;
        for (int x = 0; x < COL; x += 1) {
            for (int y = 0; y < ROW; y += 1) {
                cv::Point2i x_i(x, y);
                float gray = mapped_gray_image.at<float>(x_i);
                float depth = mapped_depth_image.at<float>(x_i);
                float warped_depth = 0;
                if (depth < math::EPSILON)
                    continue;
                cv::Point2f warped_x_i = Transform::warp(xi, x_i, depth, config_loader.depth().intrinsic, warped_depth);
                std::cout << x_i << " " << warped_x_i << " " << depth << std::endl;
                if (!math::isRange(warped_x_i.x, 0, COL) or !math::isRange(warped_x_i.y, 0, ROW))
                    continue;
                warped_gray_image.at<float>(warped_x_i) = gray;
                warped_depth_image.at<float>(warped_x_i) = depth;
            }
        }

        // 画像描画
        cv::Mat show_map_image;
        cv::Mat show_warp_image;
        cv::hconcat(Draw::visiblizeGrayImage(mapped_gray_image), Draw::visiblizeDepthImage(mapped_depth_image), show_map_image);
        cv::hconcat(Draw::visiblizeGrayImage(warped_gray_image), Draw::visiblizeDepthImage(warped_depth_image), show_warp_image);
        cv::imshow("map", show_map_image);
        cv::imshow("warp", show_warp_image);

        // 座標変換
        cv::Mat1f T = se3::exp(xi);
        cv::Mat1f arrow(T * (cv::Mat1f(4, 1) << 0, 0, 1, 1));
        cv::viz::WArrow arrow_bottom(cv::Point3f(0, 0, 0), cv::Point3f(0, 0, 1), 0.01, cv::viz::Color::black());
        cv::viz::WArrow arrow_top(cv::Point3f(T.col(3).rowRange(0, 3)), cv::Point3f(arrow.rowRange(0, 3)), 0.01, cv::viz::Color::white());
        viz_window.showWidget("bottom", arrow_bottom);
        viz_window.showWidget("top", arrow_top);
        viz_window.spinOnce(1, true);

        if (cv::waitKey(0) == 'q')
            break;
    }
}