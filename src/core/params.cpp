#include "core/params.hpp"
#include "calibration/loader.hpp"

namespace Params
{
namespace
{
bool initialized = false;
Calibration::IntrinsicParams rgb_intrinsic;
Calibration::IntrinsicParams depth_intrinsic;
Calibration::ExtrinsicParams extrinsic;
}  // namespace

void init(const std::string& config_path)
{
    Calibration::Loader config(config_path);
    rgb_intrinsic = config.rgb();
    depth_intrinsic = config.depth();
    extrinsic = config.extrinsic();
    initialized = true;
}

const Calibration::IntrinsicParams RGB()
{
    if (not initialized)
        assert(false);
    return rgb_intrinsic;
}
const Calibration::IntrinsicParams DEPTH()
{
    if (not initialized)
        assert(false);
    return depth_intrinsic;
}
const Calibration::ExtrinsicParams EXT()
{
    if (not initialized)
        assert(false);
    return extrinsic;
}


}  // namespace Params