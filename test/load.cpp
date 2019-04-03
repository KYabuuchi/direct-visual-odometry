// Loader,cv::undistortのテスト
#include "calibration/loader.hpp"
#include "core/loader.hpp"
#include "core/params.hpp"

int main()
{
    cv::namedWindow("rgb", CV_WINDOW_NORMAL);
    cv::namedWindow("depth", CV_WINDOW_NORMAL);
    cv::resizeWindow("rgb", 960, 720);
    cv::resizeWindow("depth", 960, 720);

    Loader image_loader("../data/KINECT_1DEG/info.txt");
    Params::init("../camera-calibration/data/kinectv2_00/config.yaml");

    std::cout << "'q': quit, 'other': load next image\n"
              << std::endl;

    int num = 0;
    while (true) {
        cv::Mat rgb_image, depth_image;
        cv::Mat undistorted_rgb_image, undistorted_depth_image;
        bool flag1 = image_loader.getUndistortedImages(num, undistorted_rgb_image, undistorted_depth_image);
        bool flag2 = image_loader.getRawImages(num, rgb_image, depth_image);
        if (flag1 == false or flag2 == false) {
            return 0;
        }
        std::cout << "RETURN" << std::endl;

        cv::vconcat(rgb_image, undistorted_rgb_image, undistorted_rgb_image);
        cv::vconcat(depth_image, undistorted_depth_image, undistorted_depth_image);
        cv::imshow("rgb", undistorted_rgb_image);
        cv::imshow("depth", undistorted_depth_image);

        if (cv::waitKey(0) == 'q')
            break;

        num++;
    }
}
