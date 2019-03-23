// Convert::mapDepthToColorのテスト
#include "calibration/loader.hpp"
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

        // map Depth->Color
        // cv::Mat undistorted_mapped_image = Transform::mapDepthtoGray(undistorted_depth_image, undistorted_rgb_image);
        cv::Mat tmp_mapped_image = Transform::mapDepthtoGray(undistorted_depth_image, undistorted_rgb_image);
        //  cv::Mat tmp_mapped_image = Transform::mapDepthtoGray(depth_image, rgb_image);

        cv::Mat mapped_image = cv::Mat(tmp_mapped_image.size(), CV_8UC3);
        mapped_image.forEach<cv::Vec3b>(
            [=](cv::Vec3b& p, const int position[2]) -> void {
                float v = tmp_mapped_image.at<float>(position[0], position[1]);
                if (v < -1)
                    p = cv::Vec3b(0, 0, 255);
                else
                    p = cv::Vec3b(v * 255, 0, 0);
            });
        cv::imshow("mapped", mapped_image);
        // cv::Mat upper_image, lower_image, show_image;
        // cv::hconcat(undistorted_mapped_image, mapped_image, upper_image);
        // cv::hconcat(undistorted_depth_image, depth_image, lower_image);
        // upper_image.convertTo(upper_image, CV_8UC1, 255);
        // lower_image.convertTo(lower_image, CV_8UC1, 100);
        // cv::vconcat(upper_image, lower_image, show_image);
        // cv::imshow("mapped", show_image);

        num++;
        if (cv::waitKey(0) == 'q')
            break;
    }
}
