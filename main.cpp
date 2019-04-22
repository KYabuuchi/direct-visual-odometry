#include "core/loader.hpp"
#include "core/params.hpp"
#include "system/system.hpp"

int main(/*int argc, char* argv[]*/)
{
    // loading
    Loader loader("../data/kinectv2_01/info.txt");
    Params::init("../camera-calibration/data/kinectv2_00/config.yaml");

    // data
    int num = 0;
    cv::Mat1f gray_image, depth_image, sigma_image;
    bool success = loader.getMappedImages(num++, gray_image, depth_image, sigma_image);
    if (not success)
        return -1;

    // main system
    System::VisualOdometry vo(
        gray_image,
        depth_image,
        sigma_image,
        Params::DEPTH().K());

    while (true) {
        success = loader.getMappedImages(num++, gray_image, depth_image, sigma_image);
        if (not success)
            break;

        // odometrize
        cv::Mat1f T = vo.odometrize(gray_image);
        std::cout << "\n"
                  << T << "\n"
                  << std::endl;

        // wait
        if (cv::waitKey(0) == 'q')
            break;
    }
}