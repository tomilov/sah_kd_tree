#include <debug/renderdoc.hpp>

#include <gtest/gtest.h>

TEST(Renderdoc, Basic)
{
    [[maybe_unused]] const auto & renderdoc = debug::Renderdoc::renderdoc();
    [[maybe_unused]] auto capture = debug::Renderdoc::renderdoc().makeFrameCapture();
}
