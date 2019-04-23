#include <omp.h>
#include <opencv2/opencv.hpp>

int main(void)
{
    cv::Mat src = cv::Mat::zeros(100, 100, CV_8UC3);
    cv::namedWindow("src", CV_WINDOW_NORMAL);

    int nProc = omp_get_num_procs();
    std::cout << nProc << std::endl;
    int nTh = nProc * 4;
    omp_set_num_threads(nTh);
    cv::Scalar colors;
    colors = CV_RGB(255, 255, 255);


    // dynamic で y のparallel
    // スレッドでxを共有するので private(x)が必要
    int x;
#pragma omp parallel for schedule(dynamic) private(x)
    for (int y = 0; y < src.cols; y++) {
        for (x = 0; x < src.rows; x++) {
            src.at<cv::Vec3b>(y, x) = colors[omp_get_thread_num()];
        }
    }

    cv::imshow("src", src);
    cv::waitKey(0);
}