#include "core/draw.hpp"
#include "core/loader.hpp"
#include "map/implement.hpp"
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
            Draw::visualizeDepth(noise_depth),
            Draw::visualizeDepth(origin_depth)},
        show_image1);
    cv::hconcat(
        std::vector<cv::Mat>{
            Draw::visualizeGray(obj_gray),
            Draw::visualizeDepthRaw(obj_depth),
            Draw::visualizeDepth(obj_depth, obj_sigma)},
        show_image2);

    cv::vconcat(show_image1, show_image2, show_image1);
    cv::imshow("show", show_image1);
}

int main(/*int argc, char* argv[]*/)
{
    // loading
    Core::KinectLoader loader("../data/KINECT_50MM/info.txt", "../camera-calibration/data/kinectv2_00/config.yaml");

    // window
    const std::string window_name = "show";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1280, 720);

    // initialize
    const cv::Mat1f K = loader.Depth().K();

    cv::Mat1f ref_gray, ref_depth, ref_sigma;
    cv::Mat1f obj_gray, obj_depth, obj_sigma;
    loader.getMappedImages(0, ref_gray, ref_depth, ref_sigma);
    loader.getMappedImages(4, obj_gray, obj_depth, obj_sigma);

    cv::Mat1f ref_gradx = Convert::gradiate(ref_gray, true);
    cv::Mat1f ref_grady = Convert::gradiate(ref_gray, false);

    const cv::Mat1f xi = math::se3::xi({0.2f, 0, 0, 0, 0, 0});
    std::cout << "\nxi:= " << xi.t() << "\n"
              << std::endl;

    cv::Mat1f origin_depth = obj_depth.clone();

    cv::Mat1f noise(obj_depth.size());
    cv::randn(noise, 1.5, 0.5);
    noise = cv::max(noise, 0.1);
    noise = cv::min(noise, 2.0);
    obj_depth = noise;

    cv::Mat1f noise_depth = obj_depth.clone();
    obj_sigma = 0.5f * cv::Mat1f::ones(obj_depth.size());

    while (true) {
        show(ref_gray, origin_depth, noise_depth, obj_gray, obj_depth, obj_sigma);
        if (cv::waitKey(0) == 'q')
            return 0;

        obj_depth.forEach([&](float& depth, const int p[2]) -> void {
            cv::Point2i x_i(p[1], p[0]);
            float sigma = obj_sigma(x_i);

            // 小さすぎると死ぬ
            depth = std::max(depth, 0.1f);

            auto [new_depth, new_sigma] = Map::Implement::update(
                obj_gray,
                ref_gray,
                ref_gradx,
                ref_grady,
                xi,
                K,
                x_i,
                depth,
                sigma);

            if (new_depth > 0 and new_depth < 2.0) {
                math::Gaussian g(depth, sigma);
                g(new_depth, new_sigma);
                obj_depth(x_i) = g.depth;
                obj_sigma(x_i) = g.sigma;
                if (g.depth > 2.0) {
                    std::cout << g.depth << std::endl;
                }
            }
        });

        show(ref_gray, origin_depth, noise_depth, obj_gray, obj_depth, obj_sigma);
        if (cv::waitKey(0) == 'q')
            return 0;

        obj_depth = Map::Implement::regularize(obj_depth, obj_sigma);
        // obj_depth = cv::min(obj_depth, 2.0);
    }
}