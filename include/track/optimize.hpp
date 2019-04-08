#pragma once
#include "core/convert.hpp"
#include "core/draw.hpp"
#include "core/transform.hpp"
#include "system/frame.hpp"
#include <memory>

namespace Track
{
class Scene
{
public:
    Scene(const std::shared_ptr<System::Frame> frame)
        : cols(frame->cols), rows(frame->rows), id(frame->id),
          m_depth(frame->m_depth), m_gray(frame->m_gray), m_sigma(frame->m_sigma),
          m_intrinsic(frame->m_intrinsic) {}

    Scene(const cv::Mat& gray_image, const cv::Mat& depth_image, const cv::Mat& sigma_image, const cv::Mat1f& intrinsic, int id)
        : cols(depth_image.cols), rows(depth_image.rows), id(id),
          m_depth(depth_image), m_gray(gray_image), m_sigma(sigma_image),
          m_intrinsic(intrinsic) {}

    // Copy
    Scene(const Scene& scene)
        : cols(scene.cols), rows(scene.rows), id(scene.id),
          m_depth(scene.m_depth), m_gray(scene.m_gray),
          m_intrinsic(scene.m_intrinsic)
    {
        if (not scene.m_grad_x.empty())
            m_grad_x = scene.m_grad_x;
        if (not scene.m_grad_y.empty())
            m_grad_y = scene.m_grad_y;
    }

    const int cols;
    const int rows;
    const int id;

    static std::vector<std::shared_ptr<Scene>> createScenePyramid(
        const std::shared_ptr<System::Frame> frame,
        const int level)
    {
        std::vector<std::shared_ptr<Scene>> scenes;
        Scene origin = Scene(frame);
        for (int i = 0; i < level; i++) {
            scenes.push_back(downscaleScene(origin, level - i - 1));  // level-1 , ... , 1 , 0
        }
        return scenes;
    }

    // Only use in test/track.cpp
    static std::vector<std::shared_ptr<Scene>> createScenePyramid(
        const cv::Mat1f& gray_image,
        const cv::Mat1f& depth_image,
        const cv::Mat1f& sigma_image,
        const cv::Mat1f& intrinsic,
        const int level)
    {
        std::vector<std::shared_ptr<Scene>> scenes;
        Scene origin = Scene(gray_image, depth_image, sigma_image, intrinsic, -1);
        for (int i = 0; i < level; i++) {
            scenes.push_back(downscaleScene(origin, level - i - 1));  // level-1 , ... , 1 , 0
        }
        return scenes;
    }

    cv::Mat1f depth() const { return m_depth; }
    cv::Mat1f gray() const { return m_gray; }
    cv::Mat1f sigma() const { return m_sigma; }
    cv::Mat1f intrinsic() const { return m_intrinsic; }
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
    cv::Mat1f m_depth;
    cv::Mat1f m_gray;
    cv::Mat1f m_sigma;
    cv::Mat1f m_intrinsic;
    cv::Mat1f m_grad_x;
    cv::Mat1f m_grad_y;

    static std::shared_ptr<Scene> downscaleScene(const Scene& scene, int times = 1)
    {
        if (times == 0)
            return std::make_shared<Scene>(scene);
        cv::Mat depth_image = Convert::cullImage(scene.depth(), times);
        cv::Mat gray_image = Convert::cullImage(scene.gray(), times);
        cv::Mat sigma_image = Convert::cullImage(scene.sigma(), times);
        cv::Mat1f intrinsic = Convert::cullIntrinsic(scene.intrinsic(), times);
        return std::make_shared<Scene>(gray_image, depth_image, sigma_image, intrinsic, scene.id);
    }
};

struct Stuff {
    Stuff(
        std::shared_ptr<Scene> now,
        std::shared_ptr<Scene> ref,
        const cv::Mat1f& xi)
        : now_gray(now->gray()), now_depth(now->depth()), ref_sigma(ref->sigma()),
          ref_gray(ref->gray()), ref_depth(ref->depth()),
          grad_x(ref->gradX()), grad_y(ref->gradY()),
          warped_gray(Transform::warpImage(xi, ref->gray(), ref->depth(), ref->intrinsic())),
          xi(xi), intrinsic(ref->intrinsic()),
          cols(ref->cols), rows(ref->rows) {}

    void update(const cv::Mat1f _xi)
    {
        xi = _xi;
        warped_gray = Transform::warpImage(xi, ref_gray, ref_depth, intrinsic);
    }

    void show(const std::string& window_name) const
    {
        Draw::showImage(window_name,
            Draw::visualizeGray(now_gray), Draw::visualizeGray(warped_gray), Draw::visualizeGray(ref_gray),
            Draw::visualizeDepth(ref_depth, ref_sigma), Draw::visualizeGradient(grad_x), Draw::visualizeGradient(grad_y));
    }

    const cv::Mat1f now_gray;
    const cv::Mat1f now_depth;
    const cv::Mat1f ref_sigma;
    const cv::Mat1f ref_gray;
    const cv::Mat1f ref_depth;
    const cv::Mat1f grad_x;
    const cv::Mat1f grad_y;
    cv::Mat1f warped_gray;
    cv::Mat1f xi;
    const cv::Mat1f intrinsic;
    const int cols;
    const int rows;
};

struct Outcome {
    Outcome(const cv::Mat1f& xi_update, float residual, int valid_pixels)
        : xi_update(xi_update), residual(residual), valid_pixels(valid_pixels) {}
    const cv::Mat1f xi_update;
    const float residual;
    const int valid_pixels;
};

Outcome optimize(const Stuff& stuff);

}  // namespace Track