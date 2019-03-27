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
    cv::namedWindow("warp", cv::WINDOW_NORMAL);
    cv::resizeWindow("warp", 960, 720);

    // trackbar
    const int MAX = 50;
    std::array<int, 6> params = {MAX, MAX, MAX, MAX, MAX, MAX};
    for (int i = 0; i < 6; i++) {
        cv::createTrackbar("xi" + std::to_string(i), "warp", &params.at(i), MAX * 2);
    }

    // viz
    cv::viz::Viz3d viz_window("3D-VIEW");
    viz_window.showWidget("coordinate", cv::viz::WCoordinateSystem(0.5));
    viz_window.setWindowSize(cv::Size(320, 240));

    // load
    cv::Mat rgb_image, depth_image;
    cv::Mat mapped_gray_image, mapped_depth_image;
    Loader image_loader("../data/KINECT_1DEG/info.txt");
    Params::init("../camera-calibration/data/kinectv2_00/config.yaml");
    image_loader.getNormalizedUndistortedImages(0, rgb_image, depth_image);
    mapped_gray_image = Transform::mapDepthtoGray(depth_image, rgb_image);
    mapped_depth_image = depth_image;

    mapped_depth_image = Convert::cullImage(mapped_depth_image, 1);
    mapped_gray_image = Convert::cullImage(mapped_gray_image, 1);
    cv::Mat1f intrinsic = Params::DEPTH().intrinsic;
    intrinsic = intrinsic / 2;
    intrinsic(2, 2) = 1;

    while (1) {
        std::array<float, 6> params_f;
        for (int i = 0; i < 6; i++) {
            params_f.at(i) = static_cast<float>(params.at(i) - MAX) * 1.0f / MAX;
        }

        cv::Mat1f xi = se3::xi(params_f);
        std::cout << se3::exp(xi) << std::endl;

        // get warped coordinate
        std::cout << "hoge" << std::endl;
        cv::Mat warped_gray_image = Transform::warpImage(xi, mapped_gray_image, mapped_depth_image, intrinsic);
        std::cout << "hoge" << std::endl;

        // 画像描画
        cv::Mat show_image;
        cv::hconcat(Draw::visiblizeGrayImage(mapped_gray_image), Draw::visiblizeGrayImage(warped_gray_image), show_image);
        cv::imshow("warp", show_image);

        // 座標変換
        cv::Mat1f T = se3::exp(xi);
        cv::Mat1f arrow(T * (cv::Mat1f(4, 1) << 0, 0, 1, 1));
        cv::viz::WArrow arrow_before(cv::Point3f(0, 0, 0), cv::Point3f(0, 0, 1), 0.01, cv::viz::Color::black());
        cv::viz::WArrow arrow_after(cv::Point3f(T.col(3).rowRange(0, 3)), cv::Point3f(arrow.rowRange(0, 3)), 0.01, cv::viz::Color::white());
        viz_window.showWidget("before", arrow_before);
        viz_window.showWidget("after", arrow_after);
        viz_window.spinOnce(1, true);

        if (cv::waitKey(0) == 'q')
            break;
    }
}