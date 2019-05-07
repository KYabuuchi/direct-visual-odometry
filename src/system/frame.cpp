#include "system/frame.hpp"

namespace System
{
int Frame::latest_id = -1;

void Frame::updateXi(const cv::Mat1f& relative_xi, std::shared_ptr<Frame> ref_frame)
{
    if (ref_frame == nullptr)
        return;
    m_relative_xi = relative_xi;
    m_ref_frame = ref_frame;
    m_xi = math::se3::concatenate(ref_frame->m_xi, relative_xi);
}

std::shared_ptr<Scene> Frame::downscaleScene(const Scene& scene, int times)
{
    if (times == 0)
        return std::make_shared<Scene>(scene);
    cv::Mat1f gray_image = Convert::cullImage(scene.gray(), times);
    cv::Mat1f K = Convert::cullIntrinsic(scene.K(), times);
    if (scene.depth().empty())
        return std::make_shared<Scene>(gray_image, K);

    cv::Mat1f depth_image = Convert::cullImage(scene.depth(), times);
    cv::Mat1f sigma_image = Convert::cullImage(scene.sigma(), times);
    return std::make_shared<Scene>(gray_image, depth_image, sigma_image, K);
}

std::vector<std::shared_ptr<Scene>> Frame::createScenePyramid(const Scene& scene)
{
    std::vector<std::shared_ptr<Scene>> scenes;
    for (int i = 0; i < levels; i++) {
        scenes.push_back(downscaleScene(scene, levels - 1 - i));  // level-1 , ... , 1 , 0
    }
    return scenes;
}

void Frame::updateDepthSigma(const cv::Mat1f& depth_image, const cv::Mat1f& sigma_image)
{
    for (int i = 0; i < levels; i++) {
        m_scenes.at(i)->depth() = Convert::cullImage(depth_image, levels - 1 - i);
        m_scenes.at(i)->sigma() = Convert::cullImage(sigma_image, levels - 1 - i);
    }
}

void Frame::updateDepthSigmaAge(const cv::Mat1f& depth_image, const cv::Mat1f& sigma_image, const cv::Mat1f& age_image)
{
    m_age = age_image;
    for (int i = 0; i < levels; i++) {
        m_scenes.at(i)->depth() = Convert::cullImage(depth_image, levels - 1 - i);
        m_scenes.at(i)->sigma() = Convert::cullImage(sigma_image, levels - 1 - i);
    }
}

void Frame::updateDepth(const cv::Mat1f& depth_image)
{
    for (int i = 0; i < levels; i++) {
        m_scenes.at(i)->depth() = Convert::cullImage(depth_image, levels - 1 - i);
    }
}

}  // namespace System