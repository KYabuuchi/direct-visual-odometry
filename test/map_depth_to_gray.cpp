// converter::mapDepthToColorのテスト
#include "converter.hpp"
#include "params.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    cv::Mat color_image = cv::imread("../data/KINECT_5MM/rgb01.png", cv::IMREAD_COLOR);
    cv::Mat depth_image = cv::imread("../data/KINECT_5MM/depth01.png", cv::IMREAD_UNCHANGED);
    cv::Mat gray_image;

    std::cout << Params::KINECTV2_EXTRINSIC_INVERSE << std::endl;

    cv::cvtColor(color_image, gray_image, cv::COLOR_BGR2GRAY);
    depth_image.convertTo(depth_image, CV_32FC1, 1.0 / 5000.0);  // [mm]
    gray_image.convertTo(gray_image, CV_32FC1, 1.0 / 255.0);     // 0~1

    // map Depth->Color
    cv::Mat mapped_image = Converter::mapDepthtoGray(depth_image, gray_image);

    cv::Mat show_image;
    cv::hconcat(depth_image, mapped_image, show_image);
    show_image.convertTo(show_image, CV_8UC1, 255);

    // show
    cv::namedWindow("window", cv::WINDOW_NORMAL);
    cv::imshow("window", show_image);
    cv::waitKey(0);
}
