#include "core/loader.hpp"
#include "system/system.hpp"

int main(/*int argc, char* argv[]*/)
{
    // loading
    Core::KinectLoader loader("../data/KINECT_50MM/info.txt", "../camera-calibration/data/kinectv2_00/config.yaml");

    // main system
    System::VisualOdometry vo(loader.Depth().K());

    int num = 0;
    while (true) {
        cv::Mat1f depth_image, gray_image, sigma_image;
        bool success = loader.getMappedImages(num++, gray_image, depth_image, sigma_image);
        if (not success)
            break;

        // odometrize
        cv::Mat1f T = vo.odometrizeUsingDepth(gray_image, depth_image, sigma_image);
        std::cout << "\n"
                  << T << "\n"
                  << std::endl;

        // wait
        if (cv::waitKey(0) == 'q')
            break;
    }
}