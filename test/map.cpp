#include "core/convert.hpp"
#include "core/draw.hpp"
#include "core/loader.hpp"
#include "core/math.hpp"
#include "core/transform.hpp"
#include <opencv2/viz.hpp>

int main()
{
    // window
    cv::viz::Viz3d viz_window("3D-VIEW");
    viz_window.showWidget("coordinate", cv::viz::WCoordinateSystem(0.5));
    viz_window.setWindowSize(cv::Size(320, 240));
    cv::namedWindow("warp", cv::WINDOW_NORMAL);
    cv::resizeWindow("warp", 960, 720);

    // trackbar
    const int HALF = 50;
    std::array<int, 6> params = {HALF, HALF, HALF, HALF, HALF, HALF};
    for (int i = 0; i < 6; i++) {
        cv::createTrackbar("xi" + std::to_string(i), "warp", &params.at(i), HALF * 2);
    }

    // load
    cv::Mat gray_image, depth_image;
    cv::Mat distorted_gray_image, distorted_depth_image;
    Loader loader("../data/KINECT_1DEG/info.txt");
    Params::init("../camera-calibration/data/kinectv2_00/config.yaml");
    loader.getMappedImages(0, gray_image, depth_image);
    loader.getMappedDistortedImages(0, distorted_gray_image, distorted_depth_image);
    cv::Mat1f intrinsic = Params::DEPTH().intrinsic;

    // cull to 1/2^(N+1)
    const int N = 1;
    depth_image = Convert::cullImage(depth_image, N);
    gray_image = Convert::cullImage(gray_image, N);
    distorted_depth_image = Convert::cullImage(distorted_depth_image, N);
    distorted_gray_image = Convert::cullImage(distorted_gray_image, N);
    intrinsic = intrinsic / math::pow(2, N);
    intrinsic(2, 2) = 1;

    while (1) {
        std::array<float, 6> params_f;
        for (int i = 0; i < 6; i++) {
            params_f.at(i) = static_cast<float>(params.at(i) - HALF) * 1.0f / HALF;
        }

        cv::Mat1f xi = math::se3::xi(params_f);
        std::cout << math::se3::exp(xi) << std::endl;
        cv::Mat warped_gray_image = Transform::warpImage(xi, gray_image, depth_image, intrinsic);
        cv::Mat distorted_warped_gray_image = Transform::warpImage(xi, distorted_gray_image, distorted_depth_image, intrinsic);

        // 画像描画
        cv::Mat show_image;
        cv::hconcat(std::vector<cv::Mat>{
                        Draw::visiblizeGray(gray_image),
                        Draw::visiblizeGray(warped_gray_image),
                        Draw::visiblizeGray(distorted_warped_gray_image)},
            show_image);
        cv::imshow("warp", show_image);

        // 座標変換
        cv::Mat1f T = math::se3::exp(xi);
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