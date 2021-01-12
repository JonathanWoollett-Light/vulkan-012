#include <gtest/gtest.h>
#include "../Example.hpp"

const uint32_t WORKGROUP_SIZE = 1024;

TEST(Boilerplate, ten) {
    uint32_t const size = 10;
    char const shader[] = "../../glsl/012.spv";
    ComputeApp app = ComputeApp(
        shader,
        size, // Buffer sizes
        new float[size], // Buffer data
        new uint32_t[3]{ size,1,1 }, // Invocations
        new uint32_t[3]{ WORKGROUP_SIZE,1,1 } // Workgroup sizes
    );
    uint32_t* out = static_cast<uint32_t*>(Utility::map(app.device,app.bufferMemory));
    for(uint32_t i = 0; i < size; ++i) {
        ASSERT_EQ(i,out[i]);
    }
}