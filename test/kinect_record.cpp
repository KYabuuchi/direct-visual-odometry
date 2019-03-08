#include <iostream>
#include <libfreenect2/frame_listener_impl.h>
#include <libfreenect2/libfreenect2.hpp>
#include <libfreenect2/logger.h>
#include <libfreenect2/packet_pipeline.h>
#include <opencv2/opencv.hpp>

int main()
{
    //libfreenect2::setGlobalLogger(libfreenect2::createConsoleLogger(libfreenect2::Logger::None));
    libfreenect2::Freenect2 freenect2;

    if (freenect2.enumerateDevices() == 0) {
        std::cerr << "no device connected!" << std::endl;
        return 1;
    }

    std::string serial = freenect2.getDefaultDeviceSerialNumber();
    libfreenect2::PacketPipeline* pipeline = new libfreenect2::CpuPacketPipeline();
    libfreenect2::Freenect2Device* dev = freenect2.openDevice(serial, pipeline);

    if (not dev) {
        std::cerr << "failure opening device!" << std::endl;
        return 1;
    }

    unsigned types = libfreenect2::Frame::Color | libfreenect2::Frame::Ir | libfreenect2::Frame::Depth;
    libfreenect2::SyncMultiFrameListener listener{types};
    libfreenect2::FrameMap frames;

    dev->setColorFrameListener(&listener);
    dev->setIrAndDepthFrameListener(&listener);

    if (not dev->start()) {
        std::cerr << "start failed!" << std::endl;
        return 1;
    }

    std::cout << "device serial: " << dev->getSerialNumber() << std::endl;
    std::cout << "device firmware: " << dev->getFirmwareVersion() << std::endl;

    cv::namedWindow("Color", 0);
    cv::namedWindow("Ir", 0);
    cv::namedWindow("Depth", 0);

    cv::Mat color_mat, depth_mat, ir_mat;
    int framecount = 0;
    while (true) {
        if (not listener.waitForNewFrame(frames, 10 * 1000)) {  // 10 sconds
            std::cerr << "timeout!" << std::endl;
            return 1;
        }

        bool save = false;
        int key = cv::waitKey(10);
        if (key == 'q')
            break;
        if (key == 's') {
            save = true;
            framecount++;
        }

        {  // Color
            libfreenect2::Frame* color = frames[libfreenect2::Frame::Color];
            cv::Mat{
                static_cast<int>(color->height),
                static_cast<int>(color->width),
                CV_8UC4,
                (char*)(void*)(color->data)}
                .copyTo(color_mat);
            cv::imshow("Color", color_mat);
            if (save) {
                cv::imwrite("rgb" + std::to_string(framecount) + ".png", color_mat);
                std::cout << "rgb save" << std::endl;
            }
        }

        {  // IR
            libfreenect2::Frame* ir = frames[libfreenect2::Frame::Ir];
            cv::Mat{
                static_cast<int>(ir->height),
                static_cast<int>(ir->width),
                CV_32FC1,
                (char*)(void*)(ir->data)}
                .copyTo(ir_mat);
            ir_mat.convertTo(ir_mat, CV_16UC1);
            cv::imshow("Ir", ir_mat);
            if (save) {
                cv::imwrite("ir" + std::to_string(framecount) + ".png", ir_mat);
                std::cout << "ir save" << std::endl;
            }
        }

        {  // Depth
            libfreenect2::Frame* depth = frames[libfreenect2::Frame::Depth];
            cv::Mat{
                static_cast<int>(depth->height),
                static_cast<int>(depth->width),
                CV_32FC1,
                (char*)(void*)(depth->data)}
                .copyTo(depth_mat);
            depth_mat.convertTo(depth_mat, CV_16UC1, 5);
            cv::imshow("Depth", depth_mat);
            if (save) {
                cv::imwrite("depth" + std::to_string(framecount) + ".png", depth_mat);
                std::cout << "depth save" << std::endl;
            }
        }

        listener.release(frames);
    }

    dev->stop();
    dev->close();

    return 0;
}