// Track::optimize()のテスト
#include "core/loader.hpp"
#include "core/math.hpp"
#include "core/params.hpp"
#include "matplotlibcpp.h"
#include "track/optimize.hpp"

constexpr int LEVEL = 5;
int main(int argc, char* argv[])
{
    // argumentation
    int num1 = 8, num2 = 10;
    if (argc == 2)
        num1 = std::atoi(argv[1]) + 10;
    std::cout << num1 << " " << num2 << std::endl;
    assert(0 <= num1 and num1 <= 20);

    // loading
    Loader loader("../data/KINECT_50MM/info.txt");
    Params::init("../camera-calibration/data/kinectv2_00/config.yaml");
    cv::Mat depth_image1, depth_image2;
    cv::Mat gray_image1, gray_image2;
    loader.getMappedImages(num1, gray_image1, depth_image1);
    loader.getMappedImages(num2, gray_image2, depth_image2);
    std::vector<std::shared_ptr<Track::Scene>> pre_frames = Track::Scene::createScenePyramid(depth_image1, gray_image1, Params::DEPTH().intrinsic, LEVEL);
    std::vector<std::shared_ptr<Track::Scene>> cur_frames = Track::Scene::createScenePyramid(depth_image2, gray_image2, Params::DEPTH().intrinsic, LEVEL);

    // window
    const std::string window_name = "show";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1280, 720);

    cv::Mat1f xi = math::se3::xi({0, 0, 0, 0, 0, 0});
    std::vector<std::vector<float>> vector_of_residuals;

    // iteration for pyramid
    for (int level = 0; level < LEVEL - 1; level++) {
        std::shared_ptr<Track::Scene> pre_scene = pre_frames.at(level);
        std::shared_ptr<Track::Scene> cur_scene = cur_frames.at(level);
        const int COLS = pre_scene->cols;
        const int ROWS = pre_scene->rows;
        std::vector<float> residuals;

        std::cout << "\nLEVEL: " << level << " ROW: " << ROWS << " COL: " << COLS << std::endl;

        Track::Stuff stuff = {pre_scene, cur_scene, xi};

        // iteration
        for (int iteration = 0; iteration < 15; iteration++) {
            Track::Outcome outcome = Track::optimize(stuff);

            // update
            cv::Mat1f updated_xi = math::se3::concatenate(xi, outcome.xi_update);
            residuals.push_back(outcome.residual);
            if (math::testXi(updated_xi))
                xi = updated_xi;
            stuff.update(xi);

            // show
            std::cout << "itr: " << iteration
                      << " r: " << outcome.residual
                      << " upd: " << cv::norm(outcome.xi_update)
                      << " rows : " << outcome.valid_pixels
                      << " xi: " << xi.t() << std::endl;

            stuff.show(window_name);
            cv::waitKey(10);

            if (cv::norm(outcome.xi_update) < 0.005 or outcome.residual < 0.005f)
                break;
        }
        cv::waitKey(100);

        vector_of_residuals.push_back(residuals);
    }

    // plot residuals
    namespace plt = matplotlibcpp;
    for (size_t i = 0; i < vector_of_residuals.size(); i++) {
        plt::subplot(1, vector_of_residuals.size(), i + 1);
        plt::plot(vector_of_residuals.at(i));
    }
    plt::show(true);

    std::cout << "\n"
              << math::se3::exp(xi) << "\n"
              << std::endl;

    // wait
    std::cout << "press 'q' to finish" << std::endl;
    int key = -1;
    while (key != 'q')
        key = cv::waitKey(0);
}