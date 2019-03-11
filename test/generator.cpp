#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    const cv::Size2i size = cv::Size2i(512, 424);

    cv::Mat depth = cv::Mat::ones(size, CV_16UC1) * 5000;
    cv::Mat gray1 = cv::Mat::ones(size, CV_8UC1) * 0;
    cv::Mat gray2 = cv::Mat::ones(size, CV_8UC1) * 0;

    cv::Mat texture = cv::imread("lena.png");
    cv::cvtColor(texture, texture, cv::COLOR_BGR2GRAY);
    cv::resize(texture, texture, cv::Size(500, 400));
    texture.copyTo(gray1.colRange(0, 500).rowRange(0, 400));
    texture.copyTo(gray1.colRange(0, 500).rowRange(0, 400));
    texture.copyTo(gray2.colRange(12, 512).rowRange(24, 424));


    cv::imshow("gray1", gray1);
    cv::imshow("gray2", gray2);
    cv::imshow("depth", depth);
    cv::waitKey(0);

    cv::imwrite("gray1.png", gray1);
    cv::imwrite("gray2.png", gray2);
    cv::imwrite("depth.png", depth);
}