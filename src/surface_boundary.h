#pragma once                                                                                   

#include <stdint.h>
#include <optix.h>

namespace immc {
    struct SurfaceBoundary {
    public:
        float4 norm_and_mediumID;
        OptixTraversableHandle manifold;
    };
}
