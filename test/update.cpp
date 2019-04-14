#include "core/draw.hpp"
#include "core/loader.hpp"
#include "core/params.hpp"
#include "map/updater.hpp"
#include "math/math.hpp"

void show(
    const cv::Mat1f& ref_gray,
    const cv::Mat1f& origin_depth,
    const cv::Mat1f& noise_depth,
    const cv::Mat1f& obj_gray,
    const cv::Mat1f& obj_depth,
    const cv::Mat1f& obj_sigma)
{
    // 描画
    cv::Mat show_image1;
    cv::Mat show_image2;
    cv::hconcat(
        std::vector<cv::Mat>{
            Draw::visualizeGray(ref_gray),
            Draw::visualizeDepth(origin_depth),
            Draw::visualizeDepth(noise_depth)},
        show_image1);
    cv::hconcat(
        std::vector<cv::Mat>{
            Draw::visualizeGray(obj_gray),
            Draw::visualizeDepth(obj_depth, obj_sigma),
            Draw::visualizeSigma(obj_sigma)},
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

    // initialize
    cv::Mat1f K;
    cv::Mat1f ref_gray, ref_depth, ref_sigma;
    cv::Mat1f obj_gray, obj_depth, obj_sigma;
    loader.getCulledMappedImages(0, ref_gray, ref_depth, ref_sigma, K);
    loader.getCulledMappedImages(4, obj_gray, obj_depth, obj_sigma, K);

    cv::Mat1f ref_gradx = Convert::gradiate(ref_gray, true);
    cv::Mat1f ref_grady = Convert::gradiate(ref_gray, false);

    const cv::Mat1f xi = math::se3::xi({0.2f, 0, 0, 0, 0, 0});
    std::cout << "\nxi:= " << xi.t() << "\n"
              << std::endl;

    cv::Mat1f origin_depth = obj_depth.clone();
    cv::Mat1f noise(obj_depth.size());
    cv::randn(noise, 1.5, 0.6);
    obj_depth = noise;
    cv::Mat1f noise_depth = obj_depth.clone();
    obj_sigma = 0.5f * cv::Mat1f::ones(obj_depth.size());

    cv::Size size = obj_gray.size();

    while (true) {
        show(
            ref_gray, origin_depth, noise_depth,
            obj_gray, obj_depth, obj_sigma);
        if (cv::waitKey(0) == 'q')
            return 0;
        for (int col = 0; col < size.width - 0; col++) {
            for (int row = 0; row < size.height - 0; row++) {

                cv::Point2i x_i(col, row);
                float depth = obj_depth(x_i);
                float sigma = obj_sigma(x_i);

                if (depth < 0.5f)
                    continue;


                auto [new_depth, new_sigma]
                    = Map::Update::update(
                        obj_gray,
                        ref_gray,
                        ref_gradx,
                        ref_grady,
                        xi,
                        K,
                        x_i,
                        depth,
                        sigma);
                if (new_depth > 0) {
                    math::Gaussian g(depth, sigma);
                    g(new_depth, new_sigma);
                    obj_depth(x_i) = g.depth;
                    obj_sigma(x_i) = g.sigma;
                } else {
                    // std::cout << "(" << col << "," << row << ") is not updated" << std::endl;
                }
            }
        }
    }
}