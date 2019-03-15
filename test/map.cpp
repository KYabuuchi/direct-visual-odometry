// converter::mapDepthToColorのテスト
#include "core/convert.hpp"
#include "core/params.hpp"
#include "core/transform.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    cv::Mat color_image = cv::imread("../data/KINECT_5MM/rgb01.png", cv::IMREAD_COLOR);
    cv::Mat depth_image = cv::imread("../data/KINECT_5MM/depth01.png", cv::IMREAD_UNCHANGED);
    cv::Mat gray_image;

    cv::cvtColor(color_image, gray_image, cv::COLOR_BGR2GRAY);
    depth_image.convertTo(depth_image, CV_32FC1, 1.0 / 5000.0);  // [m]
    gray_image.convertTo(gray_image, CV_32FC1, 1.0 / 255.0);     // 0~1

    // map Depth->Color
    cv::Mat mapped_image = Transform::mapDepthtoGray(depth_image, gray_image);

    // cull to half
    cv::Mat1f half_depth_image = Convert::cullImage(depth_image);
    cv::Mat1f half_mapped_image = Convert::cullImage(mapped_image);

    // show
    {
        cv::Mat show_image;
        cv::hconcat(depth_image, mapped_image, show_image);
        show_image.convertTo(show_image, CV_8UC1, 255);
        cv::namedWindow("origin", cv::WINDOW_AUTOSIZE);
        cv::imshow("origin", show_image);
    }

    // show
    {
        cv::Mat half_show_image;
        cv::hconcat(half_depth_image, half_mapped_image, half_show_image);
        half_show_image.convertTo(half_show_image, CV_8UC1, 255);
        cv::namedWindow("half", cv::WINDOW_AUTOSIZE);
        cv::imshow("half", half_show_image);
    }

    cv::waitKey(0);
}
