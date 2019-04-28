#include "core/draw.hpp"
#include "core/loader.hpp"
#include "core/params.hpp"
#include "map/implement.hpp"

void show(
    const cv::Mat1f& depth_origin,
    const cv::Mat1f& sigma_origin,
    const cv::Mat1f& age_origin,
    const cv::Mat1f& depth_image,
    const cv::Mat1f& sigma_image,
    const cv::Mat1f& age_image)
{
    // 描画
    cv::Mat show_image1;
    cv::Mat show_image2;
    cv::hconcat(
        std::vector<cv::Mat>{
            Draw::visualizeDepth(depth_origin),
            Draw::visualizeSigma(sigma_origin),
            Draw::visualizeAge(age_origin)},
        show_image1);
    cv::hconcat(
        std::vector<cv::Mat>{
            Draw::visualizeDepth(depth_image),
            Draw::visualizeSigma(sigma_image),
            Draw::visualizeAge(age_image)},
        show_image2);

    cv::vconcat(show_image1, show_image2, show_image1);
    cv::imshow("show", show_image1);
}

int main(/*int argc, char* argv[]*/)
{
    // loading
    Core::KinectLoader loader("../data/KINECT_50MM/info.txt");
    Params::init("../camera-calibration/data/kinectv2_00/config.yaml");

    // window
    const std::string window_name = "show";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1280, 720);

    // initialize
    cv::Mat1f depth_origin, gray_origin, sigma_origin;
    loader.getMappedImages(0, gray_origin, depth_origin, sigma_origin);
    cv::Mat1f age_origin = cv::Mat1f::zeros(depth_origin.size());

    cv::Mat1f depth_last(depth_origin), gray_last(gray_origin);
    cv::Mat1f sigma_last(sigma_origin), age_last(age_origin);

    const cv::Mat1f xi = math::se3::xi({0.01f, -0.01f, 0, 0, 0, 0});
    std::cout << "\nxi:= " << xi.t() << "\n"
              << std::endl;

    while (true) {
        std::cout << "Propagate" << std::endl;
        auto [depth_image, sigma_image, age_image] = Map::Implement::propagate(
            depth_last,
            sigma_last,
            age_last,
            xi,
            Params::DEPTH().intrinsic);

        show(depth_origin, sigma_origin, age_origin, depth_image, sigma_image, age_image);
        if (cv::waitKey(0) == 'q')
            break;

        std::cout << "Regularized" << std::endl;
        Map::Implement::regularize(depth_image, sigma_image);

        show(depth_origin, sigma_origin, age_origin, depth_image, sigma_image, age_image);
        if (cv::waitKey(0) == 'q')
            break;

        depth_last = depth_image;
        sigma_last = sigma_image;
        age_last = age_image;
    }
}