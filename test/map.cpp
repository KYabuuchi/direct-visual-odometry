// Convert::mapDepthToColorのテスト
#include "calibration/loader.hpp"
#include "core/draw.hpp"
#include "core/loader.hpp"
#include "core/transform.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    cv::namedWindow("mapped", CV_WINDOW_NORMAL);
    cv::resizeWindow("mapped", 960, 720);

    Loader image_loader("../data/KINECT_1DEG/info.txt");
    Calibration::Loader config_loader("../camera-calibration/data/kinectv2_00/config.yaml");
    Params::init(config_loader.rgb(), config_loader.depth(), config_loader.extrinsic());

    std::cout << "'q': quit, 'other': load next image\n"
              << std::endl;

    int num = 0;
    while (true) {
        cv::Mat rgb_image, depth_image;
        cv::Mat undistorted_rgb_image, undistorted_depth_image;
        bool flag1 = image_loader.getNormalizedImages(num, rgb_image, depth_image);
        bool flag2 = image_loader.getNormalizedUndistortedImages(num, undistorted_rgb_image, undistorted_depth_image);
        if (flag1 == false or flag2 == false)
            return 0;

        cv::Mat undistorted_mapped_image = Transform::mapDepthtoGray(undistorted_depth_image, undistorted_rgb_image);
        cv::Mat mapped_image = Transform::mapDepthtoGray(depth_image, rgb_image);
        cv::Mat show_image;
        cv::hconcat(Draw::visiblizeGrayImage(mapped_image), Draw::visiblizeGrayImage(undistorted_mapped_image), show_image);
        cv::imshow("mapped", show_image);

        num++;
        if (cv::waitKey(0) == 'q')
            break;
    }
}
