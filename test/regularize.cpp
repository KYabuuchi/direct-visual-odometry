#include "core/draw.hpp"
#include "core/loader.hpp"
#include "core/params.hpp"
#include "map/mapper.hpp"

int main(/*int argc, char* argv[]*/)
{
    // loading
    Loader loader("../data/KINECT_50MM/info.txt");
    Params::init("../camera-calibration/data/kinectv2_00/config.yaml");

    // window
    const std::string window_name = "show";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1280, 720);

    Map::Mapper mapper;

    while (true) {
        cv::Mat1f depth_image, gray_image, sigma_image;
        bool success = loader.getMappedImages(0, gray_image, depth_image, sigma_image);
        if (not success)
            break;

        // initialize error distribution
        sigma_image = cv::Mat1f::zeros(depth_image.size());
        mapper.initializeHistory(depth_image, sigma_image);

        cv::Mat show_image;
        cv::hconcat(
            Draw::visualizeDepth(depth_image, sigma_image),
            Draw::visualizeSigma(sigma_image),
            show_image);
        cv::imshow("show", show_image);
        if (cv::waitKey(0) == 'q')
            break;

        // regularize
        mapper.regularize(depth_image, sigma_image);

        // 分散は変わらないので，分散考慮の描画はしても意味がない
        cv::hconcat(
            Draw::visualizeDepth(depth_image),
            Draw::visualizeSigma(sigma_image),
            show_image);
        cv::imshow("show", show_image);
        if (cv::waitKey(0) == 'q')
            break;
    }
}