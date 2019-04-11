#include "core/draw.hpp"
#include "core/loader.hpp"
#include "core/math.hpp"
#include "core/params.hpp"
#include "map/mapper.hpp"

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
    Loader loader("../data/KINECT_50MM/info.txt");
    Params::init("../camera-calibration/data/kinectv2_00/config.yaml");

    // window
    const std::string window_name = "show";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1280, 720);

    Map::Mapper mapper{Map::Config()};

    // initialize
    cv::Mat1f ref_gray, ref_depth, ref_sigma;
    cv::Mat1f obj_gray, obj_depth, obj_sigma;
    loader.getMappedImages(0, ref_gray, ref_depth, ref_sigma);
    loader.getMappedImages(4, obj_gray, obj_depth, obj_sigma);

    cv::Mat1f ref_gradx = Convert::gradiate(ref_gray, true);
    cv::Mat1f ref_grady = Convert::gradiate(ref_gray, false);
    cv::Mat1f K = Params::DEPTH().intrinsic;

    // NOTE: 負かもしれない
    const cv::Mat1f xi = math::se3::xi({0, 0, 0.2f, 0, 0, 0});
    std::cout << "\nxi:= " << xi.t() << "\n"
              << std::endl;

    cv::Size size = obj_gray.size();

    for (int col = 0; col < size.width; col++) {
        for (int row = 0; row < size.height; row++) {

            cv::Point2i x_i(col, row);
            float depth = obj_depth(x_i);
            float sigma = obj_sigma(x_i);
            auto [new_depth, new_sigma] = mapper.update(
                obj_gray,
                ref_gray,
                ref_gradx,
                ref_grady,
                xi,
                K,
                x_i,
                depth,
                sigma);

            std::cout << depth << " " << new_depth << "\t" << sigma << " " << new_sigma << std::endl;
        }
    }
}