// 相対姿勢の計算(重みなし)
#include "core/convert.hpp"
#include "core/loader.hpp"
#include "core/params.hpp"
#include "core/transform.hpp"
#include "track/tracker.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

int main(int argc, char* argv[])
{
    // argumentation
    int num1 = 8, num2 = 10;
    if (argc == 2)
        num1 = std::atoi(argv[1]) + 10;
    std::cout << num1 << " " << num2 << std::endl;
    assert(0 <= num1 and num1 <= 20);

    // loader
    Loader image_loader("../data/KINECT_50MM/info.txt");
    Params::init("../camera-calibration/data/kinectv2_00/config.yaml");

    // loading
    cv::Mat depth_image1, depth_image2;
    cv::Mat color_image1, color_image2;
    cv::Mat gray_image1, gray_image2;
    image_loader.getNormalizedUndistortedImages(num1, color_image1, depth_image1);
    image_loader.getNormalizedUndistortedImages(num2, color_image2, depth_image2);
    gray_image1 = Transform::mapDepthtoGray(depth_image1, color_image1);
    gray_image2 = Transform::mapDepthtoGray(depth_image2, color_image2);


    // initialize
    Track::Config config = {Params::DEPTH().intrinsic, 6, true};
    Track::Tracker tracker(config);
    tracker.init(depth_image1, gray_image1);

    // tracking
    cv::Mat1f T = tracker.track(depth_image2, gray_image2);
    std::cout << "\n"
              << T << std::endl;
    // tracker.plot(false);

    // wait
    std::cout << "press 'q' to finish" << std::endl;
    int key = -1;
    while (key != 'q')
        key = cv::waitKey(0);
}