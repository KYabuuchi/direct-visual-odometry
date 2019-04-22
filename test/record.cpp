#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <sstream>

int main(int argc, char* argv[])
{

    cv::VideoCapture video;
    {
        std::string device = "/dev/video0";
        if (argc == 2) {
            device = argv[1];
        }
        if (not video.open(device)) {
            std::cout << "cannot open " << device << std::endl;
            return -1;
        }
        std::cout << "open " << device << std::endl;
    }

    const std::string dir_name = "recorded";
    {
        std::uintmax_t num = std::filesystem::remove_all(dir_name);
        if (num > 0)
            std::cout << num << " files or directories are Deleted" << std::endl;
        std::filesystem::create_directory(dir_name);
    }

    const std::string window = "show";
    cv::namedWindow(window, cv::WINDOW_NORMAL);
    bool capture = false;
    int num = 0;

    while (true) {
        cv::Mat img;
        video >> img;
        cv::imshow(window, img);
        int key = cv::waitKey(30);

        if (key == 'c') {
            capture = !capture;
        }
        if (key == 'q')
            break;

        if (capture) {
            std::stringstream file_name;
            file_name << dir_name << "/" << std::setfill('0')
                      << std::right << std::setw(4)
                      << std::to_string(num++) << ".png";
            cv::imwrite(file_name.str(), img);
            std::cout << "saved " << file_name.str() << std::endl;
        }
    }
}