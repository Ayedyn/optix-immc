#ifndef _MMC_OPTIX_UTILS_H
#define _MMC_OPTIX_UTILS_H

#include "mmc_utils.h"
#include "mmc_mesh.h"
#include "CUDABuffer.h"
#include "mmc_optix_launchparam.h"
#include "optix7.h"
#include <vector>
#include "implicit_capsule.h"
#include "implicit_sphere.h"

/*! SBT record for a raygen program */
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void *data;
};

/*! SBT record for a miss program */
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    void *data;
};

/*! SBT record for a hitgroup program */
struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
{
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    TriangleMeshSBTData data;
};

// structure for optix pipeline setup
struct OptixParams {
    CUcontext          cudaContext;
    CUstream           stream;
    cudaDeviceProp     deviceProps;
    /*! @} */

    //! the optix context that our pipeline will run in.
    OptixDeviceContext optixContext;

    /*! @{ the pipeline we're building */
    OptixPipeline               pipeline;
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions    pipelineLinkOptions = {};
    /*! @} */

    /*! @{ the module that contains out device programs */
    OptixModule                 module;
    OptixModuleCompileOptions   moduleCompileOptions = {};
    /* @} */

    /*! vector of all our program(group)s, and the SBT built around
        them */
    std::vector<OptixProgramGroup> raygenPGs;
    osc::CUDABuffer raygenRecordsBuffer;
    std::vector<OptixProgramGroup> missPGs;
    osc::CUDABuffer missRecordsBuffer;
    std::vector<OptixProgramGroup> hitgroupPGs;
    osc::CUDABuffer hitgroupRecordsBuffer;
    OptixShaderBindingTable sbt = {};

    /*! @{ our launch parameters, on the host, and the buffer to store
        them on the device */
    unsigned int launchWidth;
    MMCParam launchParams;
    osc::CUDABuffer launchParamsBuffer;
    /*! @} */

    /*! the model we are going to trace rays against */
    osc::CUDABuffer vertexBuffer;
    osc::CUDABuffer indexBuffer;
    osc::CUDABuffer fnormBuffer;
    osc::CUDABuffer nbgashandleBuffer;
    //! buffer that keeps the (final, compacted) accel structure
    osc::CUDABuffer asBuffer;

    /*! buffer for RNG seed of each thread */
    osc::CUDABuffer seedBuffer;

    /*! buffer for output storage */
    float *outputHostBuffer;
    unsigned int outputBufferSize;
    osc::CUDABuffer outputBuffer;

    /*! vector of gashandles */
    std::vector<OptixTraversableHandle> gashandles;
    std::vector<OptixTraversableHandle> inside_primitive_handles;
    std::vector<OptixTraversableHandle> outside_primitive_handles;

    /*! vector of capsules for immc */
    std::vector<immc::ImplicitCapsule> capsules;

    /*! vector of spheres for immc */
    std::vector<immc::ImplicitSphere> spheres;
};

// struct for surface mesh of each medium
typedef struct surfaceMesh {
    std::vector<uint3> face;
    std::vector<float3> norm; // normal vector
    std::vector<unsigned int> nbtype; // neighboring medium type
} surfmesh;

#ifdef __cplusplus
extern "C" {
#endif

void optix_run_simulation(mcconfig* cfg, tetmesh* mesh, raytracer* tracer,
    GPUInfo* gpu, void (*progressfun)(float, void*), void* handle);

void initOptix();
void createContext(mcconfig* cfg, OptixParams* optixcfg);
void createModule(mcconfig* cfg, OptixParams* optixcfg, std::string ptxcode);
void createRaygenPrograms(OptixParams* optixcfg);
void createMissPrograms(OptixParams* optixcfg);
void createHitgroupPrograms(OptixParams* optixcfg);
void prepareSurfMesh(tetmesh *tmesh, surfmesh *smesh);
OptixTraversableHandle buildSurfMeshAccel(tetmesh *tmesh, surfmesh* smesh, OptixParams* optixcfg,
    const unsigned int primitiveoffset);

static OptixTraversableHandle buildSurfacesWithPrimitives(tetmesh* mesh, 
		surfmesh* smesh, OptixParams* optixcfg, 
		unsigned int& primitiveoffset, const float capsuleWidthAdjustment, 
		const float sphereRadiusAdjustment);

static void buildImplicitASHierarchy(
    tetmesh* tmesh, surfmesh* smesh, 
    OptixParams* optixcfg, unsigned int& primitiveoffset, 
    const float WIDTH_ADJ); 

static OptixTraversableHandle createCapsuleAccelStructure(
     OptixParams* optixcfg,
     std::vector<float3>& vertexVector,
     std::vector<float>& widthVector,
     const unsigned int primitiveOffset);
 

static OptixTraversableHandle createSphereAccelStructure(                                     
     OptixParams* optixcfg,
     const std::vector <float3>& sphere_centers, 
     const std::vector<float>& sphere_radii,
     const unsigned int primitiveOffset
     );

static OptixTraversableHandle createInstanceAccelerationStructure(
                 OptixParams* optixcfg, 
                 std::vector<OptixTraversableHandle> handles_to_combine);

void createPipeline(OptixParams* optixcfg);
void buildSBT(tetmesh* tmesh, surfmesh* smesh, OptixParams* optixcfg);
void prepLaunchParams(mcconfig* cfg, tetmesh* mesh, GPUInfo* gpu,
    OptixParams *optixcfg);
void clearOptixParams(OptixParams* optixcfg);

#ifdef __cplusplus
}
#endif

#endif
