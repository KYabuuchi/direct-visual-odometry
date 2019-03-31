// track::trackerのテスト
#include "core/loader.hpp"
#include "core/params.hpp"
#include "track/tracker.hpp"

int main(/*int argc, char* argv[]*/)
{
    // loading
    Loader loader("../data/KINECT_50MM/info.txt");
    Params::init("../camera-calibration/data/kinectv2_00/config.yaml");

    // window
    const std::string window_name = "show";
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);
    cv::resizeWindow(window_name, 1280, 720);

    Track::Tracker tracker({Params::DEPTH().intrinsic, 5, true, 0.005f, 0.005f});

    int num = 0;
    while (true) {
        cv::Mat depth_image, gray_image;
        bool success = loader.getMappedImages(num++, gray_image, depth_image);
        if (not success)
            break;

        cv::Mat1f T = tracker.track(depth_image, gray_image);
        std::cout << "\n"
                  << T << "\n"
                  << std::endl;

        // wait
        if (cv::waitKey(0) == 'q')
            break;
    }
}