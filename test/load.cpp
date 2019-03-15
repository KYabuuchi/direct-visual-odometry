// io::Loaderのテスト
#include "core/io.hpp"

int main()
{
    io::Loader loader("../data/KINECT_1DEG/info.txt");
    cv::Mat image1, image2;
    int num = 0;

    while (loader.readImages(num, image1, image2)) {
        cv::imshow("window1", image1);
        cv::imshow("window2", image2);
        if (cv::waitKey(0) == 'q')
            break;
        num++;
    }
}
