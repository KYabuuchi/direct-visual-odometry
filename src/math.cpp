#include "math.hpp"

namespace math
{
// 3x1 => 3x3
cv::Mat1f hat(const cv::Mat1f& vec)
{
    cv::Mat1f mat_hat = (cv::Mat1f(3, 3) << 0, -vec(2), vec(1),
        vec(2), 0, -vec(0),
        -vec(1), vec(0), 0);

    return mat_hat;
}

namespace so3
{

// 3x1 => 3x3
cv::Mat1f exp(const cv::Mat1f& twist)
{
    assert(twist.size() == cv::Size(1, 3));
    cv::Mat1f R;
    cv::Rodrigues(twist, R);
    R.convertTo(R, CV_32FC1);
    return R;
}

// 3x3 => 3x1
cv::Mat1f log(const cv::Mat1f& R)
{
    assert(R.size() == cv::Size(3, 3));
    cv::Mat1f twist(cv::Mat1f::zeros(3, 1));

    float w_length = static_cast<float>(std::acos((cv::trace(R)[0] - 1.0f) * 0.5f));  // NOTE: OpenCVのtraceはチャンネル毎に計算される
    if (w_length > 1e-6f) {
        cv::Mat1f tmp
            = (cv::Mat1f(3, 1) << R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
        twist = 1.0f / (2.0f * static_cast<float>(std::sin(w_length))) * tmp * w_length;
    }
    return twist;
}

}  // namespace so3

namespace se3
{
// 6x1 => 4x4
cv::Mat1f exp(const cv::Mat1f& twist)
{
    assert(twist.size() == cv::Size(1, 6));

    cv::Mat1f v = twist.rowRange(0, 3);
    cv::Mat1f w = twist.rowRange(3, 6);
    cv::Mat1f w_hat = hat(w);
    float w_length = static_cast<float>(cv::norm(w));

    // rotation
    cv::Mat1f R = so3::exp(w);

    // translation
    cv::Mat1f t(cv::Mat1f::zeros(3, 1));
    if (w_length > 1e-6f) {
        cv::Mat1f V(cv::Mat1f::eye(3, 3)
                    + w_hat * (1.0f - std::cos(w_length)) / (w_length * w_length)
                    + (w_hat * w_hat) * (w_length - static_cast<float>(std::sin(w_length))) / (w_length * w_length * w_length));
        t = V * v;
    }

    // 4x4
    cv::Mat1f mat(cv::Mat1f::eye(4, 4));
    R.copyTo(mat.colRange(0, 3).rowRange(0, 3));
    t.copyTo(mat.col(3).rowRange(0, 3));
    return mat;
}

// 4x4 => 6x1
cv::Mat1f log(const cv::Mat1f& T)
{
    assert(T.size() == cv::Size(4, 4));

    cv::Mat1f R = T.colRange(0, 3).rowRange(0, 3);
    cv::Mat1f t = T.col(3).rowRange(0, 3);

    cv::Mat1f w = so3::log(R);

    cv::Mat1f w_hat = hat(w);
    float w_length = static_cast<float>(cv::norm(w));
    cv::Mat1f V_inv(cv::Mat1f::eye(3, 3));
    if (w_length > 1e-6f) {
        V_inv = cv::Mat1f::eye(3, 3) - 0.5f * w_hat
                + (1.0f - (w_length * std::cos(w_length * 0.5f)) / (2.0f * std::sin(w_length * 0.5f)))
                      * (w_hat * w_hat) / (w_length * w_length);
    }
    cv::Mat1f v(V_inv * t);

    cv::Mat1f twist(cv::Mat1f::zeros(6, 1));
    v.copyTo(twist.rowRange(0, 3));
    w.copyTo(twist.rowRange(3, 6));
    return twist;
}

// {6x1,6x1} => 6x1
cv::Mat1f concatenate(const cv::Mat1f& xi0, const cv::Mat1f& xi1)
{
    assert(xi0.size() == cv::Size(6, 1) and xi1.size() == cv::Size(6, 1));
    return se3::log(cv::Mat1f(se3::exp(xi0) * se3::exp(xi1)));
}

}  // namespace se3
}  // namespace math
