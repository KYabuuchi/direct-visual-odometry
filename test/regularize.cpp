#include "core/draw.hpp"
#include "map/implement.hpp"

void show(
    const cv::Mat1f& depth,
    const cv::Mat1f& sigma)
{
    // 描画
    cv::Mat show_image;
    cv::hconcat(
        std::vector<cv::Mat>{
            Draw::visualizeDepth(depth),
            Draw::visualizeDepth(sigma)},
        show_image);

    cv::imshow("show", show_image);
}

int main(/*int argc, char* argv[]*/)
{
    // window
    const std::string window_name = "show";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1280, 720);

    cv::Mat1f depth(50, 50);
    cv::Mat1f sigma(50, 50);
    cv::randn(depth, 1.5, 0.5);
    cv::randn(sigma, 0.3, 0.2);

    depth(25, 25) = 3;
    sigma(25, 25) = 0.1;

    while (true) {
        show(depth, sigma);
        if (cv::waitKey(0) == 'q')
            return 0;
        depth = Map::Implement::regularize(depth, sigma);
    }
}