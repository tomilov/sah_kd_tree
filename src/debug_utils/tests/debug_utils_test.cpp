#include <debug_utils/renderdoc.hpp>

#include <gtest/gtest.h>

TEST(Renderdoc, Basic)
{
    [[maybe_unused]] const auto & renderdoc = debug_utils::Renderdoc::renderdoc();
    [[maybe_unused]] auto capture = debug_utils::Renderdoc::renderdoc().makeFrameCapture();
}
