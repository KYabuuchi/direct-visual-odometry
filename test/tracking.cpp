// 相対姿勢の計算(重みなし)
#include "core/convert.hpp"
#include "core/io.hpp"
#include "core/params.hpp"
#include "core/transform.hpp"
#include "track/tracker.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

//#define DEBUG

int main(int argc, char* argv[])
{
    // argumentation
    int num1 = 5, num2 = 10;
    if (argc == 2)
        num1 = std::atoi(argv[1]) + 10;
    std::cout << num1 << " " << num2 << std::endl;
    assert(0 <= num1 and num1 <= 20);

    // loading
    io::Loader loader("../data/KINECT_1DEG/info.txt");
    cv::Mat depth_image1, depth_image2;
    cv::Mat color_image1, color_image2;
    cv::Mat gray_image1, gray_image2;
#ifndef DEBUG
    loader.readImages(num1, color_image1, depth_image1);
    loader.readImages(num2, color_image2, depth_image2);
    depth_image1 = Convert::depthNormalize(depth_image1);
    depth_image2 = Convert::depthNormalize(depth_image2);
    gray_image1 = Transform::mapDepthtoGray(depth_image1, Convert::colorNormalize(color_image1));
    gray_image2 = Transform::mapDepthtoGray(depth_image2, Convert::colorNormalize(color_image2));
#else
    depth_image1 = cv::imread("depth01.png", cv::IMREAD_UNCHANGED);
    depth_image2 = cv::imread("depth02.png", cv::IMREAD_UNCHANGED);
    color_image1 = cv::imread("rgb01.png", cv::IMREAD_UNCHANGED);
    color_image2 = cv::imread("rgb02.png", cv::IMREAD_UNCHANGED);
    cv::cvtColor(color_image1, gray_image1, cv::COLOR_BGR2GRAY);
    cv::cvtColor(color_image2, gray_image2, cv::COLOR_BGR2GRAY);
    gray_image1.convertTo(gray_image1, CV_32FC1, 1 / 255.0f);
    gray_image2.convertTo(gray_image2, CV_32FC1, 1 / 255.0f);
    depth_image1.convertTo(depth_image1, CV_32FC1, 1 / 5000.0f);
    depth_image2.convertTo(depth_image2, CV_32FC1, 1 / 5000.0f);
#endif
    assert(gray_image1.type() == CV_32FC1);
    assert(gray_image2.type() == CV_32FC1);

    // initialize
    Tracker::Config config = {Params::KINECTV2_INTRINSIC_DEPTH, 5, true};
    Tracker tracker(config);
    tracker.init(depth_image1, gray_image1);

    // tracking
    tracker.track(depth_image2, gray_image2);
    tracker.plot(false);

    // wait
    int key = -1;
    while (key != 'q')
        key = cv::waitKey(0);
}