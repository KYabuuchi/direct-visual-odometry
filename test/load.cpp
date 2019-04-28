// Loader,cv::undistortのテスト
#include "calibration/loader.hpp"
#include "core/loader.hpp"

int main()
{
    cv::namedWindow("rgb", CV_WINDOW_NORMAL);
    cv::namedWindow("depth", CV_WINDOW_NORMAL);
    cv::resizeWindow("rgb", 960, 720);
    cv::resizeWindow("depth", 960, 720);

    std::cout << "'q': quit, 'other': load next image\n"
              << std::endl;

    {
        int num = 0;
        Core::KinectLoader image_loader(
            "../data/KINECT_1DEG/info.txt",
            "../camera-calibration/data/kinectv2_00/config.yaml");

        while (true) {
            cv::Mat rgb_image, depth_image;
            cv::Mat undistorted_rgb_image, undistorted_depth_image;
            bool flag1 = image_loader.getUndistortedImages(num, undistorted_rgb_image, undistorted_depth_image);
            bool flag2 = image_loader.getRawImages(num, rgb_image, depth_image);
            if (flag1 == false or flag2 == false) {
                return 0;
            }

            cv::vconcat(rgb_image, undistorted_rgb_image, undistorted_rgb_image);
            cv::vconcat(depth_image, undistorted_depth_image, undistorted_depth_image);
            cv::imshow("rgb", undistorted_rgb_image);
            cv::imshow("depth", undistorted_depth_image);
            if (cv::waitKey(0) == 'q')
                break;

            num++;
        }
    }
    {
        int num = 0;
        Core::Loader loader(
            "../data/logicool0/info.txt",
            "../camera-calibration/data/logicool_00/config.yaml");

        while (true) {
            cv::Mat rgb_image;
            cv::Mat undistorted_rgb_image;
            bool flag1 = loader.getUndistortedImages(num, undistorted_rgb_image);
            bool flag2 = loader.getRawImages(num, rgb_image);
            if (flag1 == false or flag2 == false) {
                return 0;
            }
            std::cout << rgb_image.size() << " " << undistorted_rgb_image.size() << std::endl;

            cv::vconcat(rgb_image, undistorted_rgb_image, undistorted_rgb_image);
            cv::imshow("rgb", undistorted_rgb_image);
            if (cv::waitKey(0) == 'q')
                break;

            num++;
        }
    }
}
