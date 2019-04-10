#pragma once
#include "core/convert.hpp"
#include "core/math.hpp"
#include <memory>

namespace System
{

class Scene
{
public:
    Scene(const cv::Mat& gray_image, const cv::Mat1f& K)
        : cols(gray_image.cols), rows(gray_image.rows),
          m_gray(gray_image),
          m_K(K) {}

    Scene(const cv::Mat& gray_image, const cv::Mat& depth_image, const cv::Mat& sigma_image, const cv::Mat1f& K)
        : cols(gray_image.cols), rows(gray_image.rows),
          m_gray(gray_image), m_depth(depth_image), m_sigma(sigma_image),
          m_K(K) {}

    // Copy
    Scene(const Scene& scene)
        : cols(scene.cols), rows(scene.rows),
          m_gray(scene.m_gray), m_depth(scene.m_depth), m_sigma(scene.m_sigma),
          m_K(scene.m_K)
    {
        if (not scene.m_grad_x.empty())
            m_grad_x = scene.m_grad_x;
        if (not scene.m_grad_y.empty())
            m_grad_y = scene.m_grad_y;
    }

    const int cols;
    const int rows;

    cv::Mat1f gray() const { return m_gray; }
    cv::Mat1f K() const { return m_K; }
    cv::Mat1f depth() const { return m_depth; }
    cv::Mat1f& depth() { return m_depth; }
    cv::Mat1f sigma() const { return m_sigma; }
    cv::Mat1f& sigma() { return m_sigma; }

    cv::Mat1f gradX()
    {
        if (m_grad_x.empty())
            m_grad_x = Convert::gradiate(m_gray, true);
        return m_grad_x;
    }
    cv::Mat1f gradY()
    {
        if (m_grad_y.empty())
            m_grad_y = Convert::gradiate(m_gray, false);
        return m_grad_y;
    }

private:
    // TODO: 深度->逆深度
    cv::Mat1f m_gray, m_depth, m_sigma;
    cv::Mat1f m_grad_x, m_grad_y;
    cv::Mat1f m_K;
};

class Frame
{
public:
    static int latest_id;

    const int id;
    const int cols;
    const int rows;
    const int level;  // 保有する画像ピラミッドのサイズ
    const int culls;  // 最大でも 1/2^ のサイズまでしか扱わない

    cv::Mat1f m_age;
    std::vector<std::shared_ptr<Scene>> m_scenes;

    // 姿勢
    cv::Mat1f m_xi;
    cv::Mat1f m_relative_xi;
    std::shared_ptr<Frame> m_ref_frame;

    Frame(const cv::Mat1f& gray_image, const cv::Mat1f& depth_image, const cv::Mat1f& sigma_image, const cv::Mat1f& K, int level, int culls)
        : id(++latest_id), cols(gray_image.cols), rows(gray_image.rows), level(level), culls(culls),
          m_age(cv::Mat::zeros(gray_image.size(), CV_32FC1)),
          m_xi(math::se3::xi()), m_relative_xi(math::se3::xi()), m_ref_frame(nullptr)
    {
        Scene base = {
            Convert::cullImage(gray_image, culls),
            Convert::cullImage(depth_image, culls),
            Convert::cullImage(sigma_image, culls),
            Convert::cullIntrinsic(K, culls)};
        m_scenes = createScenePyramid(base);
    }

    Frame(const cv::Mat1f& gray_image, const cv::Mat1f& K, int level, int cull)
        : id(++latest_id), cols(gray_image.cols), rows(gray_image.rows), level(level), culls(culls),
          m_age(cv::Mat::zeros(gray_image.size(), CV_32FC1)),
          m_xi(math::se3::xi()), m_relative_xi(math::se3::xi()), m_ref_frame(nullptr)
    {
        Scene base = {
            Convert::cullImage(gray_image, culls),
            Convert::cullIntrinsic(K, culls)};
        m_scenes = createScenePyramid(base);
    }

    // copy constructor
    Frame(const Frame& frame)
        : id(frame.id), cols(frame.cols), rows(frame.rows), level(level), culls(culls),
          m_age(frame.m_age), m_scenes(frame.m_scenes),
          m_xi(frame.m_xi), m_relative_xi(frame.m_relative_xi), m_ref_frame(frame.m_ref_frame) {}

    std::shared_ptr<Scene> at(int level) const { return m_scenes.at(level); }

    std::shared_ptr<Scene> top() { return m_scenes.at(level - 1); }

    // VisualOdometry::estimateで呼ばれる
    void updateXi(const cv::Mat1f& relative_xi, std::shared_ptr<Frame> ref_frame);

    // Mapper::updateで呼ばれる
    void updateDepthSigma(const cv::Mat1f& depth_image, const cv::Mat1f& sigma_image);

private:
    std::shared_ptr<Scene> downscaleScene(const Scene& scene, int times = 1);
    std::vector<std::shared_ptr<Scene>> createScenePyramid(const Scene& scene);
};

class FrameHistory
{
public:
    FrameHistory() {}

    void reduceHistory(size_t remain)
    {
        if (remain < m_history.size())
            m_history.erase(m_history.begin() + remain, m_history.end());
        else
            std::cout << " invalid remains" << std::endl;
    }
    void setRefFrame(std::shared_ptr<Frame> frame) { m_history.push_back(frame); }

    std::shared_ptr<Frame> getRefFrame()
    {
        if (m_history.empty())
            return nullptr;
        return m_history.at(0);
    }

    const std::shared_ptr<Frame> getRefFrame() const
    {
        if (m_history.empty())
            return nullptr;
        return m_history.at(0);
    }


    int size() { return static_cast<int>(m_history.size()); }

    std::shared_ptr<Frame> operator[](size_t i) { return m_history.at(i); }

    std::vector<std::shared_ptr<Frame>> m_history;
};

}  // namespace System