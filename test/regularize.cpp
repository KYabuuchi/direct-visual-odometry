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

    cv::Mat1f depth_image, gray_image, sigma_image, age_image;
    Map::Mapper mapper{Map::Config()};

    // initialize error distribution
    loader.getMappedImages(0, gray_image, depth_image, sigma_image);
    sigma_image = cv::Mat1f::zeros(depth_image.size());
    age_image = cv::Mat1f::zeros(depth_image.size());
    mapper.initializeHistory(depth_image, sigma_image);


    while (true) {

        cv::Mat1f tmp_depth = cv::Mat1f::zeros(depth_image.size());
        cv::Mat1f tmp_sigma = cv::Mat1f::zeros(sigma_image.size());
        cv::Mat1f tmp_age = cv::Mat1f::zeros(age_image.size());

        mapper.propagate(
            cv::Mat1f(depth_image),
            cv::Mat1f(sigma_image),
            cv::Mat1f(age_image),
            tmp_depth,
            tmp_sigma,
            tmp_age,
            math::se3::xi({0, 0, -0.01f, 0, 0, 0}),
            Params::DEPTH().intrinsic);
        std::cout << "propagate" << std::endl;


        // regularize
        mapper.regularize(depth_image, sigma_image);
        std::cout << "regularized" << std::endl;

        // 描画
        cv::Mat show_image1;
        cv::Mat show_image2;
        cv::hconcat(
            std::vector<cv::Mat>{
                Draw::visualizeDepth(depth_image),
                Draw::visualizeSigma(sigma_image),
                Draw::visualizeAge(age_image)},
            show_image1);
        cv::hconcat(
            std::vector<cv::Mat>{
                Draw::visualizeDepth(tmp_depth),
                Draw::visualizeSigma(tmp_sigma),
                Draw::visualizeAge(tmp_age)},
            show_image2);

        cv::vconcat(show_image1, show_image2, show_image1);
        cv::imshow("show", show_image1);
        if (cv::waitKey(0) == 'q')
            break;

        depth_image = tmp_depth;
        sigma_image = tmp_sigma;
        age_image = tmp_age;
    }
}