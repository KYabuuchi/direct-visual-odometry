#include "core/timer.hpp"
#include <omp.h>
#include <opencv2/opencv.hpp>
#include <thread>

int main(void)
{
    // set number of threads
    int nProc = omp_get_num_procs();
    std::cout << "omp_get_num_procs() " << nProc << std::endl;
    int nTh = nProc * 4;
    std::cout << "omp_set_num_threads() " << nTh << std::endl;
    omp_set_num_threads(nTh);

    cv::Mat src = cv::Mat::zeros(2000, 2000, CV_8UC3);
    cv::namedWindow("src", CV_WINDOW_NORMAL);

    cv::Vec3b color(255, 255, 255);

    {
        Timer t("OpenMP");
#pragma omp parallel for
        for (int y = 0; y < src.cols; y++) {
            for (int x = 0; x < src.rows; x++) {
                src.at<cv::Vec3b>(y, x) = color;
            }
        }
    }
    {
        Timer t("forEach");
        src.forEach<cv::Vec3b>([=](cv::Vec3b& c, const int pt[2]) -> void {
            c = color;
        });
    }

    cv::imshow("src", src);
    cv::waitKey(0);
}