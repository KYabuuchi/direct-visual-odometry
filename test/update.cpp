#include "core/draw.hpp"
#include "core/loader.hpp"
#include "map/implement.hpp"
#include "math/math.hpp"

void show(
    const cv::Mat1f& obj_gray,
    const cv::Mat1f& origin_depth,
    const cv::Mat1f& noise_depth,
    const cv::Mat1f& ref_gray,
    const cv::Mat1f& ref_depth,
    const cv::Mat1f& ref_sigma)
{
    // 描画
    cv::Mat show_image1;
    cv::Mat show_image2;
    cv::hconcat(
        std::vector<cv::Mat>{
            Draw::visualizeGray(obj_gray),
            Draw::visualizeDepth(noise_depth),
            Draw::visualizeDepth(origin_depth)},
        show_image1);
    cv::hconcat(
        std::vector<cv::Mat>{
            Draw::visualizeGray(ref_gray),
            Draw::visualizeSigma(ref_sigma),
            Draw::visualizeDepth(ref_depth, ref_sigma)},
        show_image2);

    cv::vconcat(show_image1, show_image2, show_image1);
    cv::imshow("show", show_image1);
}

int main(/*int argc, char* argv[]*/)
{
    // loading
    Core::KinectLoader loader("../data/KINECT_50MM/info.txt", "../external/camera-calibration/data/kinectv2_00/config.yaml");

    // window
    const std::string window_name = "show";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1280, 720);

    // initialize
    const cv::Mat1f K = loader.Depth().K();

    cv::Mat1f ref_gray, ref_depth, ref_sigma;
    cv::Mat1f obj_gray;
    {
        cv::Mat1f obj_depth, obj_sigma;
        loader.getMappedImages(0, ref_gray, ref_depth, ref_sigma);
        loader.getMappedImages(4, obj_gray, obj_depth, obj_sigma);
    }

    cv::Mat1f ref_gradx = Convert::gradiate(ref_gray, true);
    cv::Mat1f ref_grady = Convert::gradiate(ref_gray, false);

    const cv::Mat1f xi = math::se3::xi({-0.2f, 0, 0, 0, 0, 0});
    std::cout << "\nxi:= " << xi.t() << "\n"
              << std::endl;

    cv::Mat1f origin_depth = ref_depth.clone();

    cv::Mat1f noise(ref_depth.size());
    cv::randn(noise, 1.7, 0.5);
    noise = cv::max(noise, 1.0);
    noise = cv::min(noise, 4.0);
    ref_depth = noise;

    cv::Mat1f noise_depth = ref_depth.clone();
    ref_sigma = 0.5f * cv::Mat1f::ones(ref_depth.size());

    while (true) {
        show(obj_gray, origin_depth, noise_depth, ref_gray, ref_depth, ref_sigma);
        if (cv::waitKey(0) == 'q')
            return 0;

        std::cout << "update" << std::endl;
        ref_depth.forEach([&](float& depth, const int p[2]) -> void {
            cv::Point2i x_i(p[1], p[0]);
            float sigma = ref_sigma(x_i);

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

            // if (x_i.y < 40 and x_i.x < 40) {
            //     std::cout << new_depth << " " << new_sigma << std::endl;
            //     // return;
            // }

            if (new_depth > 0 and new_depth < 6.0) {
                math::Gaussian g(depth, sigma);
                g(new_depth, new_sigma);
                ref_depth(x_i) = g.depth;
                ref_sigma(x_i) = g.sigma;
            }
        });

        show(obj_gray, origin_depth, noise_depth, ref_gray, ref_depth, ref_sigma);
        if (cv::waitKey(0) == 'q')
            return 0;

        std::cout << "regularize" << std::endl;
        ref_depth = Map::Implement::regularize(ref_depth, ref_sigma);
        // obj_depth = cv::min(obj_depth, 2.0);
    }
}