// 相対姿勢の計算(without using image-pyramid)
#include "converter.hpp"
#include "io.hpp"
#include "matplotlibcpp.h"
#include "params.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

namespace plt = matplotlibcpp;

int main()
{
    // prepare depth-image & gray-image
    io::Loader loader("../data/KINECT_1DEG/info.txt");
    cv::Mat depth_image1, color_image1;
    cv::Mat depth_image2, color_image2;
    loader.readImages(0, color_image1, depth_image1);
    loader.readImages(1, color_image2, depth_image2);
    cv::Mat mapped_image1 = Converter::mapDepthtoGray(Converter::depthNormalize(depth_image1), Converter::colorNormalize(color_image1));
    cv::Mat mapped_image2 = Converter::mapDepthtoGray(Converter::depthNormalize(depth_image2), Converter::colorNormalize(color_image2));


    // show graph
    //plt::plot({1, 3, 2, 4});
    //plt::show(false);

    // show image
    cv::Mat show_image;
    cv::hconcat(mapped_image1, mapped_image2, show_image);
    show_image.convertTo(show_image, CV_8UC1, 255);
    cv::namedWindow("window", cv::WINDOW_NORMAL);
    cv::imshow("window", show_image);
    cv::waitKey(0);
}