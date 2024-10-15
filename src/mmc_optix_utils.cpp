#include <limits.h>
#include <optix_function_table_definition.h>
#include <sutil/vec_math.h>
#include <time.h>

#include <CUDABuffer.h>

#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <unordered_map>
#include <unordered_set>
#ifdef _OPENMP
    #include <omp.h>
#endif

#include "incbin.h"
#include "mmc_cuda_query_gpu.h"
#include "mmc_optix_utils.h"
#include "mmc_tictoc.h"
#include "surface_boundary.h"

INCTXT(mmcShaderPtx, mmcShaderPtxSize, "built/mmc_optix_core.ptx");
const int out[4][3] = {{0, 3, 1}, {3, 2, 1}, {0, 2, 3}, {0, 1, 2}};
const int ifaceorder[] = {3, 0, 2, 1};
const unsigned int IMPLICIT_MATERIAL = 1;

bool usingImplicitPrimitives = true;  // TODO remove temporary test variable

void optix_run_simulation(mcconfig* cfg, tetmesh* mesh, raytracer* tracer,
                          GPUInfo* gpu, void (*progressfun)(float, void*),
                          void* handle) {
    uint tic0 = StartTimer();
    // ==================================================================
    // prepare optix pipeline
    // ==================================================================
    OptixParams optixcfg {};

    initOptix();

    // TODO: make this inputted by user
    if (usingImplicitPrimitives) {
        immc::ImplicitCapsule test_capsule_1{make_float3(20.0, 20.0, 20.0), make_float3(50.0, 50.0, 20.0), 5.0}; //v1, v2, width
        immc::ImplicitCapsule test_capsule_2{make_float3(0.0, 0.0, 0.0), make_float3(50.0, 0.0, 0.0), 1.0}; //v1, v2, width

        optixcfg.capsules.resize(2);
        optixcfg.capsules.push_back(test_capsule_1);
        optixcfg.capsules.push_back(test_capsule_2);

        optixcfg.spheres.resize(1);
        immc::ImplicitSphere test_sphere_1{make_float3(1, 1, 1), 0.3};
        optixcfg.spheres.push_back(test_sphere_1);
    }

    // temp variable:
    float WIDTH_ADJ_TEMP = 1 / 1024.0;

    createContext(cfg, &optixcfg);
    MMC_FPRINTF(cfg->flog, "optix init complete:  \t%d ms\n",
                GetTimeMillis() - tic0);
    fflush(cfg->flog);

    std::string ptxcodestr = std::string(mmcShaderPtx);
    createModule(cfg, &optixcfg, ptxcodestr, usingImplicitPrimitives);
    MMC_FPRINTF(cfg->flog, "optix module complete:  \t%d ms\n",
                GetTimeMillis() - tic0);
    fflush(cfg->flog);

    createRaygenPrograms(&optixcfg);
    createMissPrograms(&optixcfg);
    createHitgroupPrograms(&optixcfg);
    MMC_FPRINTF(cfg->flog, "optix device programs complete:  \t%d ms\n",
                GetTimeMillis() - tic0);
    fflush(cfg->flog);

    // prepare surface meshes and initialize count of primitives
    // This raw pointer practice is a lower level C practice that should be
    // overhauled to use std::vector instead
    surfmesh* smesh_lowlevel_array =
        (surfmesh*)calloc((mesh->prop + 1), sizeof(surfmesh));
    prepareSurfMesh(mesh, smesh_lowlevel_array);
    unsigned int primitiveoffset = 0;

    // build acceleration structures for all sphere primitives and capsule
    // primitives
    if (usingImplicitPrimitives) {
        // Need to populate gas handles
        buildImplicitASHierarchy(mesh, smesh_lowlevel_array, &optixcfg,
                                 primitiveoffset, WIDTH_ADJ_TEMP);
        buildSurfaceData(mesh, smesh_lowlevel_array, &optixcfg);
    }

    // build acceleration structure for only triangle meshes (one per
    // medium) shijie's original approach
    else {


        for (int i = 0; i <= mesh->prop; ++i) {
            optixcfg.gashandles.push_back(
                buildSurfMeshAccel(mesh, smesh_lowlevel_array + i,
                                   &optixcfg, primitiveoffset));

            primitiveoffset += smesh_lowlevel_array[i].norm.size();
        }
    }

    MMC_FPRINTF(cfg->flog,
                "optix acceleration structure complete:  \t%d ms\n",
                GetTimeMillis() - tic0);
    fflush(cfg->flog);

    createPipeline(&optixcfg);
    MMC_FPRINTF(cfg->flog, "optix pipeline complete:  \t%d ms\n",
                GetTimeMillis() - tic0);
    fflush(cfg->flog);

    buildSBT(mesh, smesh_lowlevel_array, &optixcfg);
    free(smesh_lowlevel_array);
    MMC_FPRINTF(cfg->flog,
                "optix shader binding table complete:  \t%d ms\n",
                GetTimeMillis() - tic0);
    fflush(cfg->flog);

    // ==================================================================
    // prepare launch parameters
    // ==================================================================
    prepLaunchParams(cfg, mesh, gpu, &optixcfg);
    CUDA_ASSERT(cudaDeviceSynchronize());
    MMC_FPRINTF(cfg->flog, "optix launch parameters complete:  \t%d ms\n",
                GetTimeMillis() - tic0);
    fflush(cfg->flog);

    // ==================================================================
    // Launch simulation
    // ==================================================================
    MMC_FPRINTF(cfg->flog,
                "lauching OptiX for time window [%.1fns %.1fns] ...\n",
                cfg->tstart * 1e9, cfg->tend * 1e9);
    fflush(cfg->flog);
    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                    optixcfg.pipeline, optixcfg.stream,
                    /*! parameters and SBT */
                    optixcfg.launchParamsBuffer.d_pointer(),
                    optixcfg.launchParamsBuffer.sizeInBytes,
                    &optixcfg.sbt,
                    /*! dimensions of the launch: */
                    optixcfg.launchWidth, 1, 1));
    CUDA_ASSERT(cudaDeviceSynchronize());
    MMC_FPRINTF(cfg->flog,
                "kernel complete:  \t%d ms\nretrieving flux ... \t",
                GetTimeMillis() - tic0);
    fflush(cfg->flog);

    // ==================================================================
    // Save output
    // ==================================================================
    optixcfg.outputBuffer.download(optixcfg.outputHostBuffer,
                                   optixcfg.outputBufferSize);
    MMC_FPRINTF(cfg->flog, "transfer complete:        %d ms\n",
                GetTimeMillis() - tic0);
    fflush(cfg->flog);

    for (size_t i = 0; i < optixcfg.launchParams.crop0.w; i++) {
        // combine two outputs into one
        #pragma omp atomic
        mesh->weight[i] +=
            optixcfg.outputHostBuffer[i] +
            optixcfg
            .outputHostBuffer[i + optixcfg.launchParams.crop0.w];
    }

    // ==================================================================
    // normalize output
    // ==================================================================
    if (cfg->isnormalized) {
        MMC_FPRINTF(cfg->flog, "normalizing raw data ...\t");
        fflush(cfg->flog);

        // not used if cfg->method == rtBLBadouelGrid
        cfg->energyabs = 0.0f;

        // for now assume initial weight of each photon is 1.0
        cfg->energytot = cfg->nphoton;
        mesh_normalize(mesh, cfg, cfg->energyabs, cfg->energytot, 0);
        MMC_FPRINTF(cfg->flog, "normalization complete:    %d ms\n",
                    GetTimeMillis() - tic0);
        fflush(cfg->flog);
    }

    #pragma omp master
    {
        if (cfg->issave2pt && cfg->parentid == mpStandalone) {
            MMC_FPRINTF(cfg->flog, "saving data to file ...\t");
            mesh_saveweight(mesh, cfg, 0);
            MMC_FPRINTF(cfg->flog,
                        "saving data complete : %d ms\n\n",
                        GetTimeMillis() - tic0);
            fflush(cfg->flog);
        }
    }

    // ==================================================================
    // Free memory
    // ==================================================================
    clearOptixParams(&optixcfg);
}

/**
 * @brief extract surface mesh for each medium
 */
void prepareSurfMesh(tetmesh* tmesh, surfmesh* smesh) {
    int* fnb = (int*)calloc(tmesh->ne * tmesh->elemlen, sizeof(int));
    memcpy(fnb, tmesh->facenb, (tmesh->ne * tmesh->elemlen) * sizeof(int));

    float3 v0, v1, v2, vec01, vec02, vnorm;

    for (int i = 0; i < tmesh->ne; ++i) {
        // iterate over each tetrahedra
        unsigned int currmedid = tmesh->type[i];

        for (int j = 0; j < tmesh->elemlen; ++j) {
            // iterate over each triangle
            int nexteid = fnb[(i * tmesh->elemlen) + j];

            if (nexteid == INT_MIN) {
                continue;
            }

            unsigned int nextmedid =
                ((nexteid < 0) ? 0 : tmesh->type[nexteid - 1]);

            if (currmedid != nextmedid) {
                // face nodes
                unsigned int n0 =
                    tmesh->elem[(i * tmesh->elemlen) +
                                                     out[ifaceorder[j]][0]] -
                    1;
                unsigned int n1 =
                    tmesh->elem[(i * tmesh->elemlen) +
                                                     out[ifaceorder[j]][1]] -
                    1;
                unsigned int n2 =
                    tmesh->elem[(i * tmesh->elemlen) +
                                                     out[ifaceorder[j]][2]] -
                    1;

                // face vertex indices
                smesh[currmedid].face.push_back(
                    make_uint3(n0, n1, n2));
                smesh[nextmedid].face.push_back(
                    make_uint3(n0, n2, n1));

                // outward-pointing face norm
                v0 = *(float3*)&tmesh->fnode[n0];
                v1 = *(float3*)&tmesh->fnode[n1];
                v2 = *(float3*)&tmesh->fnode[n2];
                vec_diff((MMCfloat3*)&v0, (MMCfloat3*)&v1,
                         (MMCfloat3*)&vec01);
                vec_diff((MMCfloat3*)&v0, (MMCfloat3*)&v2,
                         (MMCfloat3*)&vec02);
                vec_cross((MMCfloat3*)&vec01,
                          (MMCfloat3*)&vec02,
                          (MMCfloat3*)&vnorm);
                float mag =
                    1.0f / sqrtf(vec_dot((MMCfloat3*)&vnorm,
                                         (MMCfloat3*)&vnorm));
                vec_mult((MMCfloat3*)&vnorm, mag,
                         (MMCfloat3*)&vnorm);
                smesh[currmedid].norm.push_back(vnorm);
                smesh[nextmedid].norm.push_back(-vnorm);

                // neighbour medium types
                smesh[currmedid].nbtype.push_back(nextmedid);
                smesh[nextmedid].nbtype.push_back(currmedid);

                fnb[(i * tmesh->elemlen) + j] = INT_MIN;

                if (nexteid > 0) {
                    for (int k = 0; k < tmesh->elemlen;
                            ++k) {
                        if (fnb[((nexteid - 1) *
                                                tmesh->elemlen) +
                                               k] == i + 1) {
                            fnb[((nexteid - 1) *
                                                tmesh->elemlen) +
                                               k] = INT_MIN;
                            break;
                        }
                    }
                }
            }
        }
    }
}

/**
 * @brief prepare launch parameters
 */
void prepLaunchParams(mcconfig* cfg, tetmesh* mesh, GPUInfo* gpu,
                      OptixParams* optixcfg) {
    if (cfg->method != rtBLBadouelGrid) {
        mcx_error(-1, "Optix MMC only supports dual grid mode",
                  __FILE__, __LINE__);
    }

    int timeSteps = (int)((cfg->tend - cfg->tstart) / cfg->tstep + 0.5);

    if (timeSteps < 1) {
        mcx_error(-1, "There must be at least one time step.", __FILE__,
                  __LINE__);
    }

    // set up optical properties
    if (mesh->prop + 1 > MAX_PROP_OPTIX) {
        mcx_error(-1, "Medium type count exceeds limit.", __FILE__,
                  __LINE__);
    }

    for (int i = 0; i <= mesh->prop; ++i) {
        optixcfg->launchParams.medium[i].mua = mesh->med[i].mua;
        optixcfg->launchParams.medium[i].mus = mesh->med[i].mus;
        optixcfg->launchParams.medium[i].g = mesh->med[i].g;
        optixcfg->launchParams.medium[i].n = mesh->med[i].n;
    }

    // source setup
    optixcfg->launchParams.srcpos =
        make_float3(cfg->srcpos.x, cfg->srcpos.y, cfg->srcpos.z);
    optixcfg->launchParams.srcdir =
        make_float3(cfg->srcdir.x, cfg->srcdir.y, cfg->srcdir.z);

    // sets the capsule geometry information in order of capsule creation
    if (usingImplicitPrimitives) {
        osc::CUDABuffer capsules_buf {};
        capsules_buf.alloc_and_upload(optixcfg->capsules);

        optixcfg->launchParams.capsuleData = capsules_buf.d_pointer();
    }

    // set the surface data information
    if (usingImplicitPrimitives) {
        osc::CUDABuffer surfaceData_buf {};
        surfaceData_buf.alloc_and_upload(optixcfg->surfaceData);
        optixcfg->launchParams.surfaceData = surfaceData_buf.d_pointer();
    }

    // parameters of dual grid
    optixcfg->launchParams.nmin =
        make_float3(mesh->nmin.x, mesh->nmin.y, mesh->nmin.z);
    optixcfg->launchParams.nmax = make_float3(mesh->nmax.x - mesh->nmin.x,
                                  mesh->nmax.y - mesh->nmin.y,
                                  mesh->nmax.z - mesh->nmin.z);
    optixcfg->launchParams.crop0 = make_uint4(
                                       cfg->crop0.x, cfg->crop0.y, cfg->crop0.z, cfg->crop0.z * timeSteps);
    optixcfg->launchParams.dstep = 1.0f / cfg->unitinmm;

    // time-gate settings
    optixcfg->launchParams.tstart = cfg->tstart;
    optixcfg->launchParams.tend = cfg->tend;
    optixcfg->launchParams.Rtstep = 1.0f / cfg->tstep;
    optixcfg->launchParams.maxgate = cfg->maxgate;

    if (usingImplicitPrimitives){
        // starting handle needs to be chosen based on whether photon is starting in or out
        // of an implicitly defined shape

        // initialize starting handle and medium if the ray is in an implicit shape
        if(startInSphere(optixcfg) || startInCapsule(optixcfg)){
           optixcfg->launchParams.mediumid0 = IMPLICIT_MATERIAL;   
           optixcfg->launchParams.gashandle0 = 
               optixcfg->inside_primitive_handles[optixcfg->launchParams.mediumid0];
        }
        // initialize starting handle and medium if the ray is outside an implicit shape 
        else{
            // init medium ID using element based
            optixcfg->launchParams.mediumid0 = mesh->type[cfg->e0 - 1];

            // init gashandle using initial medium ID
            optixcfg->launchParams.gashandle0 = 
            optixcfg->outside_primitive_handles[optixcfg->launchParams.mediumid0];
        }
    }
    else{
    // init medium ID using element based
    optixcfg->launchParams.mediumid0 = mesh->type[cfg->e0 - 1];

    // init gashandle using initial medium ID
    optixcfg->launchParams.gashandle0 =
        optixcfg->gashandles[optixcfg->launchParams.mediumid0];
    }

    // simulation flags
    optixcfg->launchParams.isreflect = cfg->isreflect;

    // output type
    optixcfg->launchParams.outputtype = static_cast<int>(cfg->outputtype);

    // number of photons for each thread
    int totalthread = cfg->nthread;

    int gpuid, threadid = 0;
#ifdef _OPENMP
    threadid = omp_get_thread_num();
#endif
    gpuid = cfg->deviceid[threadid] - 1;

    if (cfg->autopilot) {
        totalthread = gpu[gpuid].autothread;
    }

    optixcfg->launchWidth = totalthread;
    optixcfg->launchParams.threadphoton =
        cfg->nphoton / optixcfg->launchWidth;
    optixcfg->launchParams.oddphoton =
        cfg->nphoton - optixcfg->launchParams.threadphoton * totalthread;

    // output buffer (single precision)
    optixcfg->outputBufferSize = (optixcfg->launchParams.crop0.w << 1);
    optixcfg->outputHostBuffer =
        (float*)calloc(optixcfg->outputBufferSize, sizeof(float));
    optixcfg->outputBuffer.alloc_and_upload(optixcfg->outputHostBuffer,
                                            optixcfg->outputBufferSize);
    optixcfg->launchParams.outputbuffer =
        optixcfg->outputBuffer.d_pointer();

    // photon seed buffer
    if (cfg->seed > 0) {
        srand(cfg->seed);
    } else {
        srand(time(0));
    }

    uint4* hseed = (uint4*)calloc(totalthread, sizeof(uint4));

    for (int i = 0; i < totalthread; ++i) {
        hseed[i] = make_uint4(rand(), rand(), rand(), rand());
    }

    optixcfg->seedBuffer.alloc_and_upload(hseed, totalthread);
    optixcfg->launchParams.seedbuffer = optixcfg->seedBuffer.d_pointer();

    if (hseed) {
        free(hseed);
    }

    // upload launch parameters to device
    optixcfg->launchParamsBuffer.alloc_and_upload(&optixcfg->launchParams,
            1);
}

/**************************************************************************
 * helper functions for Optix pipeline creation
 ******************************************************************************/

/**
 * @brief initialize optix
 */
void initOptix() {
    cudaFree(0);
    OPTIX_CHECK(optixInit());
}

/**
 * @brief creates and configures a optix device context
 */
void createContext(mcconfig* cfg, OptixParams* optixcfg) {
    int gpuid, threadid = 0;

#ifdef _OPENMP
    threadid = omp_get_thread_num();
#endif

    gpuid = cfg->deviceid[threadid] - 1;

    if (gpuid < 0) {
        mcx_error(-1, "GPU ID must be non-zero", __FILE__, __LINE__);
    }

    CUDA_ASSERT(cudaSetDevice(gpuid));
    CUDA_ASSERT(cudaStreamCreate(&optixcfg->stream));

    cudaGetDeviceProperties(&optixcfg->deviceProps, gpuid);
    std::cout << "Running on device: " << optixcfg->deviceProps.name
              << std::endl;

    CUresult cuRes = cuCtxGetCurrent(&optixcfg->cudaContext);

    if (cuRes != CUDA_SUCCESS)
        fprintf(stderr,
                "Error querying current context: error code %d\n",
                cuRes);

    OptixDeviceContextOptions options = {};
    options.logCallbackFunction = [](unsigned int level, const char* tag,
    const char* message, void*) {
        std::cerr << "[" << std::setw(2) << level << "]["
                  << std::setw(12) << tag << "]: " << message << "\n";
    };
#ifndef NDEBUG
    printf("DEBUG MODE ACTIVATED");
    options.logCallbackLevel = 4;
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
#else
    options.logCallbackLevel = 0;
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_OFF;
#endif

    OPTIX_CHECK(optixDeviceContextCreate(optixcfg->cudaContext, &options,
                                         &optixcfg->optixContext));
}

/**
 * @brief creates the module that contains all programs
 */
void createModule(mcconfig* cfg, OptixParams* optixcfg, std::string ptxcode,
                  bool usingImplicitPrimitives) {
    // moduleCompileOptions
    optixcfg->moduleCompileOptions.maxRegisterCount =
        OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
#ifndef NDEBUG
    optixcfg->moduleCompileOptions.debugLevel =
        OPTIX_COMPILE_DEBUG_LEVEL_FULL;
    optixcfg->moduleCompileOptions.optLevel =
        OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
#else
    optixcfg->moduleCompileOptions.debugLevel =
        OPTIX_COMPILE_DEBUG_LEVEL_NONE;
    optixcfg->moduleCompileOptions.optLevel =
        OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#endif

    // pipelineCompileOptions
    optixcfg->pipelineCompileOptions = {};
    optixcfg->pipelineCompileOptions.traversableGraphFlags =
        OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    optixcfg->pipelineCompileOptions.usesMotionBlur = false;

    // usingImplicitPrimitives is a placeholder for now
    if (usingImplicitPrimitives) {
        optixcfg->pipelineCompileOptions.numPayloadValues = 18;
        optixcfg->pipelineCompileOptions.numAttributeValues = 4;
        optixcfg->pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_TRIANGLE | OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE | OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
    } else {
        optixcfg->pipelineCompileOptions.numPayloadValues = 16;
        optixcfg->pipelineCompileOptions.numAttributeValues =
            2;  // for triangle
    }

#ifndef NDEBUG
    optixcfg->pipelineCompileOptions.exceptionFlags =
        OPTIX_EXCEPTION_FLAG_DEBUG | OPTIX_EXCEPTION_FLAG_TRACE_DEPTH |
        OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW;
#else
    optixcfg->pipelineCompileOptions.exceptionFlags =
        OPTIX_EXCEPTION_FLAG_NONE;
#endif
    optixcfg->pipelineCompileOptions.pipelineLaunchParamsVariableName =
        "gcfg";

    // pipelineLinkOptions
    optixcfg->pipelineLinkOptions.maxTraceDepth = 1;

    char log[2048];
    size_t logsize = sizeof(log);
    OPTIX_CHECK(optixModuleCreateFromPTX(
                    optixcfg->optixContext, &optixcfg->moduleCompileOptions,
                    &optixcfg->pipelineCompileOptions, ptxcode.c_str(), ptxcode.size(),
                    log, &logsize, &optixcfg->module));

    if (logsize > 1) {
        std::cout << log << std::endl;
    }
}

/**
 * @brief set up ray generation programs
 */
void createRaygenPrograms(OptixParams* optixcfg) {
    optixcfg->raygenPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    pgDesc.raygen.module = optixcfg->module;
    pgDesc.raygen.entryFunctionName = "__raygen__rg";

    char log[2048];
    size_t logsize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixcfg->optixContext, &pgDesc, 1,
                                        &pgOptions, log, &logsize,
                                        &optixcfg->raygenPGs[0]));

    if (logsize > 1) {
        std::cout << log << std::endl;
    }
}

/**
 * @brief set up miss programs
 */
void createMissPrograms(OptixParams* optixcfg) {
    optixcfg->missPGs.resize(1);

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};
    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    pgDesc.raygen.module = optixcfg->module;
    pgDesc.raygen.entryFunctionName = "__miss__ms";

    char log[2048];
    size_t logsize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixcfg->optixContext, &pgDesc, 1,
                                        &pgOptions, log, &logsize,
                                        &optixcfg->missPGs[0]));

    if (logsize > 1) {
        std::cout << log << std::endl;
    }
}

// helper function for getting built-in intersection shader for sphere
static OptixModule getSphereModule(OptixDeviceContext ctx, OptixParams* optixcfg) {
    OptixModuleCompileOptions module_compile_options = {};
#if !defined( NDEBUG )
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_FULL;
#endif

    OptixBuiltinISOptions builtin_is_options = {};

    builtin_is_options.usesMotionBlur = false;
    builtin_is_options.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
    OptixModule sphere_module;
    OPTIX_CHECK(optixBuiltinISModuleGet(ctx, &module_compile_options, &optixcfg->pipelineCompileOptions,
                                        &builtin_is_options, &sphere_module));
    return sphere_module;
}


/**
 * @brief set up hitgroup programs
 */
void createHitgroupPrograms(OptixParams* optixcfg) {
    if (usingImplicitPrimitives) {
        optixcfg->hitgroupPGs.resize(3);  // TODO fix this global mutation

        OptixDeviceContext context = optixcfg->optixContext;
        const unsigned int numProgramGroups = 3;

        OptixProgramGroupDesc capsules_grp_desc {};
        capsules_grp_desc.kind = OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        capsules_grp_desc.hitgroup.moduleCH = optixcfg->module;
        capsules_grp_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        capsules_grp_desc.hitgroup.moduleIS = optixcfg->module;
        capsules_grp_desc.hitgroup.entryFunctionNameIS = "__intersection__customlinearcapsule";

        OptixProgramGroupDesc spheres_grp_desc {};
        spheres_grp_desc.kind = OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        spheres_grp_desc.hitgroup.moduleCH = optixcfg->module;
        spheres_grp_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        spheres_grp_desc.hitgroup.moduleIS = getSphereModule(optixcfg->optixContext, optixcfg);
        spheres_grp_desc.hitgroup.entryFunctionNameIS = nullptr;

        OptixProgramGroupDesc triangles_grp_desc {};
        triangles_grp_desc.kind = OptixProgramGroupKind::OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        triangles_grp_desc.hitgroup.moduleCH = optixcfg->module;
        triangles_grp_desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        //triangles_grp_desc.hitgroup.entryFunctionNameIS = nullptr;  // TODO figure out what this defaults to

        std::array<OptixProgramGroupDesc, 3> grp_descs {capsules_grp_desc, spheres_grp_desc, triangles_grp_desc};
        std::array<OptixProgramGroupOptions, 3> grp_options{};  // uses default options

        char log[2048];
        size_t logsize = sizeof(log);

        OPTIX_CHECK(optixProgramGroupCreate(optixcfg->optixContext,
                                            grp_descs.data(),
                                            numProgramGroups,
                                            grp_options.data(),
                                            log,
                                            &logsize,
                                            optixcfg->hitgroupPGs.data()));

        return;
    }

    OptixProgramGroupOptions pgOptions = {};
    OptixProgramGroupDesc pgDesc = {};

    optixcfg->hitgroupPGs.resize(1);

    pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    pgDesc.hitgroup.moduleCH = optixcfg->module;
    pgDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";



    char log[2048];
    size_t logsize = sizeof(log);
    OPTIX_CHECK(optixProgramGroupCreate(optixcfg->optixContext, &pgDesc, 1,
                                        &pgOptions, log, &logsize,
                                        &optixcfg->hitgroupPGs[0]));

    if (logsize > 1) {
        std::cout << log << std::endl;
    }
}

// build acceleration structure hierarchy for Implicit Primitives superimposed
// on surface mesh
static void buildImplicitASHierarchy(tetmesh* mesh, surfmesh* smesh,
                                     OptixParams* optixcfg,
                                     unsigned int& primitiveoffset,
                                     const float WIDTH_ADJ) {
    std::vector<OptixTraversableHandle> outside_prim_handles;
    std::vector<OptixTraversableHandle> inside_prim_handles;

    std::cout << "Building acceleration structures." << mesh->prop
              << std::endl;

    unsigned int primitiveCount = 0;

    // two separate vectors of traversable handles are created, one for
    // outside primitives with one handle for each medium
    for (int i = 0; i <= mesh->prop; ++i) {
        outside_prim_handles.push_back(buildSurfacesWithPrimitives(
                                           mesh, smesh + i, optixcfg, primitiveoffset, 0.0, 0.0));
    }

    printf("\nThe number of inside primitives is: %d", primitiveCount);

    // one for inside primitives with one handle for each medium
    for (int i = 0; i <= mesh->prop; ++i) {
        inside_prim_handles.push_back(buildSurfacesWithPrimitives(
                                          mesh, smesh + i, optixcfg, primitiveoffset, WIDTH_ADJ,
                                          WIDTH_ADJ));
    }

    optixcfg->inside_primitive_handles = inside_prim_handles;
    optixcfg->outside_primitive_handles = outside_prim_handles;
}

// vector to look up a given material id, surface normal, and handle for
// an instance acceleration structure
// used in only implicit MMC
static void buildSurfaceData(tetmesh* mesh, surfmesh* smesh, OptixParams* optixcfg) {

    // add handles and material ID for all entering-implicit acceleration structures
    for (int i = 0; i <= mesh->prop; ++i) {
        // add handles and material ID for capsules superimposed on a surface mesh
        for (const immc::ImplicitCapsule& capsule : optixcfg->capsules) {
            immc::SurfaceBoundary c_data{make_float4(0, 0, 0, static_cast<float>(IMPLICIT_MATERIAL)), 
                optixcfg->inside_primitive_handles[i]};
            optixcfg->surfaceData.push_back(c_data);
        }

        // add handles and material ID for spheres superimposed on a surface mesh
        for (const immc::ImplicitSphere& sphere : optixcfg->spheres) {
            immc::SurfaceBoundary s_data{make_float4(sphere.position.x,
                                            sphere.position.y,
                                            sphere.position.z,
                                            static_cast<float>(IMPLICIT_MATERIAL)), 
                                             optixcfg->inside_primitive_handles[i]};
            optixcfg->surfaceData.push_back(s_data);
        }

        // add handles and material ID for triangles composing the surface mesh
        // TODO: figure out why this conditional is here:
        for (int j = 0; j < smesh[i].norm.size(); ++j) {
            immc::SurfaceBoundary t_data{make_float4(
                                                smesh[i].norm[j].x, smesh[i].norm[j].y,
                                                smesh[i].norm[j].z, 
                                                static_cast<float>(smesh[i].nbtype[j])),
                                                optixcfg->outside_primitive_handles[i]};
            optixcfg->surfaceData.push_back(t_data);
        }
    }

    // TODO: check if the material property of exiting implicits needs to be set to
    // surface mesh material instead of 0.
    // add handles and material ID for all exiting-implicit acceleration structures
    for (int i = 0; i <= mesh->prop; ++i) {
        // add handles and material ID for capsules superimposed on a surface mesh
        for (const immc::ImplicitCapsule& capsule : optixcfg->capsules) {
            // give 0,0,0 as normal vector for capsule since we will be
            // calculating normals on intersection           
            immc::SurfaceBoundary c_data{make_float4(
                                        0, 0, 0, 0),
                                        optixcfg->inside_primitive_handles[i]};
            optixcfg->surfaceData.push_back(c_data);
        }

        // add handles and material ID for spheres superimposed on a surface mesh
        for (const immc::ImplicitSphere& sphere : optixcfg->spheres) {
            immc::SurfaceBoundary s_data{make_float4(sphere.position.x,
                                            sphere.position.y,
                                            sphere.position.z,
                                            0), optixcfg->inside_primitive_handles[i]};
            // give sphere center location for calculating normal on intersection
            optixcfg->surfaceData.push_back(s_data);
        }

        // add handles and material ID for triangles composing the surface mesh
        // TODO: figure out why this conditional is here:
        for (int j = 0; j < smesh[i].norm.size(); ++j) {
            immc::SurfaceBoundary t_data{make_float4(
                                        smesh[i].norm[j].x, smesh[i].norm[j].y,
                                        smesh[i].norm[j].z, 
                                        static_cast<float>(IMPLICIT_MATERIAL)),
                                            optixcfg->outside_primitive_handles[i]};
            optixcfg->surfaceData.push_back(t_data);
        }
    }
}

// Takes a single triangle surface mesh for a given material property
// builds a geometry acceleration structure (GAS) for spheres,
// then a GAS for capsules, then a GAS for triangles of the surface mesh
// then combines them into a compressed instance acceleration structure (IAS)
// with one handle for a single material.
static OptixTraversableHandle buildSurfacesWithPrimitives(
    tetmesh* mesh, surfmesh* smesh, OptixParams* optixcfg,
    unsigned int& primitiveoffset, const float capsuleWidthAdjustment,
    const float sphereRadiusAdjustment) {
    // vector contains all handles to make an instance acceleration
    // structure (IAS)
    std::vector<OptixTraversableHandle> handles_to_combine;

    // prep and modify vectors of capsule geometry:
    std::vector<float3> capsuleVertices;
    std::vector<float> capsuleWidths;

    // in the future this might check if capsule coordinates are
    // inside of tetrahedral manifold
    unsigned int capsulecount = 0;
    unsigned int vertices_per_capsule = 2;

    for (immc::ImplicitCapsule capsule : optixcfg->capsules) {
        capsuleVertices.push_back(capsule.vertex1);
        capsuleVertices.push_back(capsule.vertex2);
        capsuleWidths.push_back(capsule.width + capsuleWidthAdjustment);
        capsulecount = capsulecount + 1;
    }

    printf("\n finished pushing back geometries from config");

    // create the acceleration structures for capsules
    if (capsuleWidths.size() > 0) {
        OptixTraversableHandle capsulesHandle =
            createCapsuleAccelStructure(optixcfg, capsuleVertices,
                                        capsuleWidths, primitiveoffset);
        primitiveoffset += capsuleWidths.size();

        handles_to_combine.push_back(capsulesHandle);
    }

    // prep the sphere acceleration structures geometry values
    std::vector<float3> sphereCenters;
    std::vector<float> sphereRadii;

    // Ideally this checks if
    // sphere coordinates and radii is inside of tetrahedral manifold
    for (immc::ImplicitSphere sphere : optixcfg->spheres) {
        sphereCenters.push_back(sphere.position);
        sphereRadii.push_back(sphere.radius + sphereRadiusAdjustment);
    }

    // create the acceleration structures for spheres
    if (sphereCenters.size() > 0) {
        // this creates the acceleration structures for spheres
        OptixTraversableHandle spheresHandle =
            createSphereAccelStructure(optixcfg, sphereCenters,
                                       sphereRadii, primitiveoffset);
        primitiveoffset += sphereCenters.size();
        handles_to_combine.push_back(spheresHandle);
    }

    // build surface mesh acceleration structure for one particular surface
    OptixTraversableHandle surfaceHandle =
        buildSurfMeshAccel(mesh, smesh, optixcfg, primitiveoffset);
    primitiveoffset += smesh->norm.size();
    handles_to_combine.push_back(surfaceHandle);

    // Combine the handles into one instance acceleration structure and
    // return it
    return createInstanceAccelerationStructure(optixcfg,
            handles_to_combine);
}

// Builds a an acceleration structure of linear (cylindrical with spherical
// end-caps) custom capsules. Had to make it custom because OptiX does not
// support in-to-out ray tracing vertexBuffer represents a list of the endpoints
// of each capsule segment widthBuffer represents a list of the swept radius
// between each vertex Takes returns the AS as a devicebytebuffer and adds the
// AS to the traversable handle passed by reference
static OptixTraversableHandle createCapsuleAccelStructure(
    OptixParams* optixcfg, std::vector<float3>& vertexVector,
    std::vector<float>& widthVector, const unsigned int primitiveOffset) {
    // traversable handle we will ultimately build the accel structure to:
    OptixTraversableHandle capsulesHandle;

#ifndef NDBUG
    printf("\nCreating capsule acceleration structures");
#endif
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;

    // tells the number of capsule segments (pills) to create
    // there are two vertices per capsule segment
    if (vertexVector.size() % 2 == 0) {
        buildInput.customPrimitiveArray.numPrimitives =
            vertexVector.size() / 2;
    } else {
        throw std::runtime_error(
            "Number of capsule vertices is not an even number");
    }

    std::vector<OptixAabb> aabb;

    // prepare axis aligned bounding boxes
    for (int i = 0; i < vertexVector.size(); i += 2) {
        // create one per capsule primitive,
        //  figure out the maximum/min vertex in each direction
        //  and then calculate the maximum height/width/depth
        //  via adding or subtracting radii
        OptixAabb temp_aabb;
        float width = widthVector[i / 2];
        float3 vertex_one = vertexVector[i];
        float3 vertex_two = vertexVector[i + 1];

        temp_aabb.minX = std::min(vertex_one.x, vertex_two.x) - width;
        temp_aabb.minY = std::min(vertex_one.y, vertex_two.y) - width;
        temp_aabb.minZ = std::min(vertex_one.z, vertex_two.z) - width;

        temp_aabb.maxX = std::max(vertex_one.x, vertex_two.x) + width;
        temp_aabb.maxY = std::max(vertex_one.y, vertex_two.y) + width;
        temp_aabb.maxZ = std::max(vertex_one.z, vertex_two.z) + width;
        aabb.push_back(temp_aabb);
    }

    osc::CUDABuffer aabbBuffer;
    aabbBuffer.alloc_and_upload(aabb);

    const CUdeviceptr aabb_cudapointer = aabbBuffer.d_pointer();
    buildInput.customPrimitiveArray.aabbBuffers = &aabb_cudapointer;

    uint32_t aabbInputFlags = OPTIX_GEOMETRY_FLAG_NONE;
    buildInput.customPrimitiveArray.flags = &aabbInputFlags;
    buildInput.customPrimitiveArray.numSbtRecords = 1;
    buildInput.customPrimitiveArray.primitiveIndexOffset = primitiveOffset;

    // compact and build the acceleration structure
    OptixAccelBuildOptions accelerationOptions = {};
    accelerationOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelerationOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    // blas stands for "bottom level acceleration structure", geometric data
    // that must be sent to GPU.
    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
                    optixcfg->optixContext, &accelerationOptions, &buildInput,
                    1,  // num of build inputs
                    &blasBufferSizes));

    // temporary buffers needed to build/compress acceleration structure

    osc::CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    osc::CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;

    osc::CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    emitDesc.result = compactedSizeBuffer.d_pointer();
    OPTIX_CHECK(
        optixAccelBuild(optixcfg->optixContext, optixcfg->stream,
                        &accelerationOptions, &buildInput,
                        1,  // num of build inputs
                        tempBuffer.d_pointer(), tempBuffer.sizeInBytes,
                        outputBuffer.d_pointer(), outputBuffer.sizeInBytes,
                        &capsulesHandle, &emitDesc, 1));

    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);
    printf("\n reached this section test");
    osc::CUDABuffer compactedOutputBuffer;
    compactedOutputBuffer.alloc(compactedSize);
    printf("\n reached after the alloc");
    OPTIX_CHECK(
        optixAccelCompact(optixcfg->optixContext, optixcfg->stream,
                          capsulesHandle, compactedOutputBuffer.d_pointer(),
                          compactedOutputBuffer.sizeInBytes, &capsulesHandle));
    CUDA_SYNC_CHECK();
    // ==================================================================
    // clean up
    // ==================================================================
    outputBuffer.free();  // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();
    compactedOutputBuffer.free();

#ifndef NDEBUG
    printf("\nBuilt an optix acceleration structure of type: custom capsule");
#endif

    return capsulesHandle;
}

// Creates a sphere acceleration structure as a DeviceByteBuffer, adds it to the
// OptixTraversableHandle which is passed by reference this AS contains a series
// of spheres with the same material and varying radii & center points
static OptixTraversableHandle createSphereAccelStructure(
    OptixParams* optixcfg, const std::vector<float3>& sphere_centers,
    const std::vector<float>& sphere_radii,
    const unsigned int primitiveOffset) {

#ifndef NDEBUG
    printf("\nCreating sphere acceleration structures");
#endif

    // creates a handle that will ultimately point to the built
    // accelerations structure
    OptixTraversableHandle spheresHandle;

    // allocate sphere geometry vectors to GPU and get a cudapointer
    osc::CUDABuffer centerBuffer;
    centerBuffer.alloc_and_upload(sphere_centers);
    osc::CUDABuffer radiusBuffer;
    radiusBuffer.alloc_and_upload(sphere_radii);

    uint32_t sphere_input_flags[1] = {OPTIX_GEOMETRY_FLAG_NONE};
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_SPHERES;
    buildInput.sphereArray.vertexStrideInBytes = sizeof(float3);
    CUdeviceptr sphereCenterDevicePtr = centerBuffer.d_pointer();
    buildInput.sphereArray.vertexBuffers = &sphereCenterDevicePtr;
    buildInput.sphereArray.numVertices = sphere_radii.size();
    CUdeviceptr sphereRadiusDevicePtr = radiusBuffer.d_pointer();
    buildInput.sphereArray.radiusBuffers = &sphereRadiusDevicePtr;
    buildInput.sphereArray.radiusStrideInBytes = 0;
    buildInput.sphereArray.flags = sphere_input_flags;
    buildInput.sphereArray.numSbtRecords = 1;
    buildInput.sphereArray.primitiveIndexOffset = primitiveOffset;
    buildInput.sphereArray.singleRadius = false;

    OptixAccelBuildOptions accelerationOptions = {};
    accelerationOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelerationOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    // blas stands for "bottom level acceleration structure", geometric data
    // that must be sent to GPU.
    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(
                    optixcfg->optixContext, &accelerationOptions, &buildInput,
                    1,  // num of build inputs
                    &blasBufferSizes));

    // temporary buffers needed to build/compress acceleration structure

    osc::CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    osc::CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    osc::CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();

    OPTIX_CHECK(
        optixAccelBuild(optixcfg->optixContext, optixcfg->stream,
                        &accelerationOptions, &buildInput,
                        1,  // num of build inputs
                        tempBuffer.d_pointer(), tempBuffer.sizeInBytes,
                        outputBuffer.d_pointer(), outputBuffer.sizeInBytes,
                        &spheresHandle, &emitDesc, 1));

    // ==================================================================
    // perform compaction
    // ==================================================================
    printf("b1");
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);
    printf("b2");
    osc::CUDABuffer compactedOutputBuffer;
    compactedOutputBuffer.alloc(compactedSize);
    OPTIX_CHECK(
        optixAccelCompact(optixcfg->optixContext, optixcfg->stream,
                          spheresHandle, compactedOutputBuffer.d_pointer(),
                          compactedOutputBuffer.sizeInBytes, &spheresHandle));
    CUDA_SYNC_CHECK();
    printf("b3");
    // ==================================================================
    // clean up
    // ==================================================================
    outputBuffer.free();  // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();
    compactedOutputBuffer.free();

#ifndef NDEBUG
    printf("\nBuilt an optix acceleration structure of type: sphere");
#endif

    return spheresHandle;
}

bool startInSphere(const OptixParams* optixcfg){
    if(optixcfg->spheres.size() == 0)
        return false;

    float3 startPos = optixcfg->launchParams.srcpos;

    for (immc::ImplicitSphere sphere : optixcfg->spheres) {
        float3 displacement = startPos - sphere.position;
        if (dot(displacement, displacement) <= sphere.radius*sphere.radius) {
            return true;
        }
    }
    return false;
}

bool startInCapsule(const OptixParams* optixcfg){
    if(optixcfg->capsules.size() == 0)
        return false;
   
    // starting position of ray
    float3 position = optixcfg->launchParams.srcpos; 
    for (immc::ImplicitCapsule capsule : optixcfg->capsules) {
        // vector along the line segment of the capsule
        float3 vector_alongcapsule =
            make_float3(capsule.vertex2.x - capsule.vertex1.x,
                capsule.vertex2.y - capsule.vertex1.y,
                capsule.vertex2.z - capsule.vertex1.z);
        // vector from the first vertex to the position
        float3 vector_toposition = make_float3(
            position.x - capsule.vertex1.x, position.y - capsule.vertex1.y,
            position.z - capsule.vertex1.z);
        // computes the scalar projection of vector to position onto the
        // line segment
        float scalarproj = dot(vector_alongcapsule, vector_toposition) /
                   dot(vector_alongcapsule, vector_alongcapsule);
        // computes the projection of the vector to position onto the
        // line segment
        float3 projection =
            make_float3(vector_alongcapsule.x * scalarproj,
                vector_alongcapsule.y * scalarproj,
                vector_alongcapsule.z * scalarproj);
        // gets the vector for the shortest distance from line segment
        float3 distance_vector =
            make_float3(vector_toposition.x - projection.x,
                vector_toposition.y - projection.y,
                vector_toposition.z - projection.z);
        // get the magnitude of that vector
        float distance = sqrt(distance_vector.x * distance_vector.x +
                      distance_vector.y * distance_vector.y +
                      distance_vector.z * distance_vector.z);
        if (distance < capsule.width) {
            return true;
        }                                                                                     
    }

}

/**
 * @brief set up geometry acceleration structure for one surface mesh
 */
OptixTraversableHandle buildSurfMeshAccel(tetmesh* tmesh, surfmesh* smesh,
        OptixParams* optixcfg,
        const unsigned int primitiveoffset) {
    OptixTraversableHandle asHandle{0};

    if (smesh->face.empty()) {
        return asHandle;
    }

    // ==================================================================
    // upload the model to the device
    // note: mesh->fnode needs to be float3
    // mesh->face needs to be uint3 (zero-indexed)
    // ==================================================================

    osc::CUDABuffer vert_buff;
    vert_buff.alloc_and_upload(tmesh->fnode, tmesh->nn);

    osc::CUDABuffer idx_buff;
    idx_buff.alloc_and_upload(smesh->face);

    // ==================================================================
    // triangle inputs
    // ==================================================================
    OptixBuildInput triangleInput = {};
    triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    // create local variables, because we need a *pointer* to the
    // device pointers
    CUdeviceptr d_vertices = vert_buff.d_pointer();
    CUdeviceptr d_indices = idx_buff.d_pointer();

    triangleInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    triangleInput.triangleArray.vertexStrideInBytes = sizeof(float3);
    triangleInput.triangleArray.numVertices = tmesh->nn;
    triangleInput.triangleArray.vertexBuffers = &d_vertices;

    triangleInput.triangleArray.indexFormat =
        OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    triangleInput.triangleArray.indexStrideInBytes = sizeof(uint3);
    triangleInput.triangleArray.numIndexTriplets = smesh->face.size();
    triangleInput.triangleArray.indexBuffer = d_indices;

    uint32_t triangleInputFlags[1] = {OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT};

    // in this example we have one SBT entry, and no per-primitive
    // materials:
    triangleInput.triangleArray.flags = triangleInputFlags;
    triangleInput.triangleArray.numSbtRecords = 1;
    triangleInput.triangleArray.sbtIndexOffsetBuffer = 0;
    triangleInput.triangleArray.sbtIndexOffsetSizeInBytes = 0;
    triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0;
    triangleInput.triangleArray.primitiveIndexOffset = primitiveoffset;

    // ==================================================================
    // BLAS setup
    // ==================================================================
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags =
        OPTIX_BUILD_FLAG_NONE | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelOptions.motionOptions.numKeys = 1;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optixcfg->optixContext,
                &accelOptions, &triangleInput,
                1,  // num_build_inputs
                &blasBufferSizes));

    // ==================================================================
    // prepare compaction
    // ==================================================================
    osc::CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();

    // ==================================================================
    // execute build (main stage)
    // ==================================================================
    osc::CUDABuffer tempBuffer;
    tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

    osc::CUDABuffer outputBuffer;
    outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

    OPTIX_CHECK(optixAccelBuild(
                    optixcfg->optixContext, optixcfg->stream, &accelOptions,
                    &triangleInput, 1, tempBuffer.d_pointer(), tempBuffer.sizeInBytes,
                    outputBuffer.d_pointer(), outputBuffer.sizeInBytes, &asHandle,
                    &emitDesc, 1));
    CUDA_SYNC_CHECK();

    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    osc::CUDABuffer compactedOutputBuffer;
    compactedOutputBuffer.alloc(compactedSize);

    printf("\nBuilding a surface mesh acceleration structure");
    OPTIX_CHECK(optixAccelCompact(optixcfg->optixContext,
                                  optixcfg->stream,
                                  asHandle, compactedOutputBuffer.d_pointer(),
                                  compactedOutputBuffer.sizeInBytes,
                                  &asHandle));
    CUDA_SYNC_CHECK();
    printf("\nSuccessfully built a surface mesh acceleration structure");
    // ==================================================================
    // clean up
    // ==================================================================
    outputBuffer.free();  // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();
    compactedOutputBuffer.free();

    return asHandle;
}

static OptixTraversableHandle createInstanceAccelerationStructure(
    OptixParams* optixcfg,
    std::vector<OptixTraversableHandle> handles_to_combine) {

#ifndef NDBUG
    printf("\nCreating instance acceleration structures");
#endif

    OptixTraversableHandle instanceHandle;

    std::vector<OptixInstance> instances = std::vector<OptixInstance>();
    int i = 0;

    for (OptixTraversableHandle handle : handles_to_combine) {
        OptixInstance instance = {};
        float transform[12] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0};
        memcpy(instance.transform, transform, sizeof(float) * 12);
        instance.flags = OPTIX_INSTANCE_FLAG_NONE;
        instance.instanceId = 0;
        instance.visibilityMask = 255;
        instance.sbtOffset = i;
        instance.traversableHandle = handle;
        instances.push_back(instance);
        i++;
    }

    printf("\n before alloc and upload of instances");
    osc::CUDABuffer instanceBuffer;
    instanceBuffer.alloc_and_upload(instances);

    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    buildInput.instanceArray.instances = instanceBuffer.d_pointer();
    buildInput.instanceArray.instanceStride = 0;
    buildInput.instanceArray.numInstances = instances.size();

    OptixAccelBuildOptions accelerationOptions = {};
    accelerationOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    accelerationOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes bufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optixcfg->optixContext,
                &accelerationOptions,
                &buildInput, 1, &bufferSizes));
    osc::CUDABuffer compactedSizeBuffer;
    compactedSizeBuffer.alloc(sizeof(uint64_t));

    OptixAccelEmitDesc emitDesc;
    emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emitDesc.result = compactedSizeBuffer.d_pointer();

    osc::CUDABuffer tempBuffer;
    tempBuffer.alloc(bufferSizes.tempSizeInBytes);

    osc::CUDABuffer outputBuffer;
    outputBuffer.alloc(bufferSizes.outputSizeInBytes);
    printf("\n before building instance accel structure");
    OPTIX_CHECK(
        optixAccelBuild(optixcfg->optixContext, optixcfg->stream,
                        &accelerationOptions, &buildInput,
                        1,  // num of build inputs
                        tempBuffer.d_pointer(), tempBuffer.sizeInBytes,
                        outputBuffer.d_pointer(), outputBuffer.sizeInBytes,
                        &instanceHandle, &emitDesc, 1));
    printf("\n after building instance accel structure");
    // ==================================================================
    // perform compaction
    // ==================================================================
    uint64_t compactedSize;
    compactedSizeBuffer.download(&compactedSize, 1);

    // temporary buffer to define location and size to store compacted accel structure on the GPU
    osc::CUDABuffer compactedOutputBuffer;
    compactedOutputBuffer.alloc(compactedSize);
    OPTIX_CHECK(
        optixAccelCompact(optixcfg->optixContext, optixcfg->stream,
                          instanceHandle, compactedOutputBuffer.d_pointer(),
                          compactedOutputBuffer.sizeInBytes, &instanceHandle));
    CUDA_SYNC_CHECK();
    printf("\nafter compacting instance accel structure");
    // ==================================================================
    // clean up
    // ==================================================================
    outputBuffer.free();  // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    compactedSizeBuffer.free();
    compactedOutputBuffer.free();
    return instanceHandle;
}

/**
 * @brief assemble the pipeline of all programs
 */
void createPipeline(OptixParams* optixcfg) {
    std::vector<OptixProgramGroup> programGroups;

    for (auto pg : optixcfg->raygenPGs) {
        programGroups.push_back(pg);
    }

    for (auto pg : optixcfg->missPGs) {
        programGroups.push_back(pg);
    }

    for (auto pg : optixcfg->hitgroupPGs) {
        programGroups.push_back(pg);
    }

    char log[2048];
    size_t logsize = sizeof(log);
    OPTIX_CHECK(optixPipelineCreate(
                    optixcfg->optixContext, &optixcfg->pipelineCompileOptions,
                    &optixcfg->pipelineLinkOptions, programGroups.data(),
                    (int)programGroups.size(), log, &logsize, &optixcfg->pipeline));

    if (logsize > 1) {
        std::cout << log << std::endl;
    }
}

/**
 * @ set up the shader binding table
 */
void buildSBT(tetmesh* mesh, surfmesh* smesh, OptixParams* optixcfg) {
    // ==================================================================
    // build raygen records
    // ==================================================================
    std::vector<RaygenRecord> raygenRecords;

    for (size_t i = 0; i < optixcfg->raygenPGs.size(); i++) {
        RaygenRecord rec;
        OPTIX_CHECK(
            optixSbtRecordPackHeader(optixcfg->raygenPGs[i], &rec));
        rec.data = nullptr;
        raygenRecords.push_back(rec);
    }

    optixcfg->raygenRecordsBuffer.alloc_and_upload(raygenRecords);
    optixcfg->sbt.raygenRecord = optixcfg->raygenRecordsBuffer.d_pointer();

    // ==================================================================
    // build miss records
    // ==================================================================
    std::vector<MissRecord> missRecords;

    for (size_t i = 0; i < optixcfg->missPGs.size(); i++) {
        MissRecord rec;
        OPTIX_CHECK(
            optixSbtRecordPackHeader(optixcfg->missPGs[i], &rec));
        rec.data = nullptr; /* for now ... */
        missRecords.push_back(rec);
    }

    optixcfg->missRecordsBuffer.alloc_and_upload(missRecords);
    optixcfg->sbt.missRecordBase = optixcfg->missRecordsBuffer.d_pointer();
    optixcfg->sbt.missRecordStrideInBytes = sizeof(MissRecord);
    optixcfg->sbt.missRecordCount = (int)missRecords.size();

    // ==================================================================
    // build hitgroup records
    // ==================================================================

    if (usingImplicitPrimitives) {
        HitgroupRecord capsules_rec;
        HitgroupRecord spheres_rec;
        HitgroupRecord triangles_rec;

        OPTIX_CHECK(optixSbtRecordPackHeader(optixcfg->hitgroupPGs[0], &capsules_rec));
        OPTIX_CHECK(optixSbtRecordPackHeader(optixcfg->hitgroupPGs[1], &spheres_rec));
        OPTIX_CHECK(optixSbtRecordPackHeader(optixcfg->hitgroupPGs[2], &triangles_rec));

        std::vector<HitgroupRecord> hitgroupRecords;
        hitgroupRecords.push_back(capsules_rec);
        hitgroupRecords.push_back(spheres_rec);
        hitgroupRecords.push_back(triangles_rec);

        // allocate and upload nothing for the implicit primitives. All data
        // is going on the launchParams
        // TODO: performance test if putting SurfaceData on hitgroup record data is better
        optixcfg->hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
        optixcfg->sbt.hitgroupRecordBase = optixcfg->hitgroupRecordsBuffer.d_pointer();
        optixcfg->sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
        optixcfg->sbt.hitgroupRecordCount = (int)hitgroupRecords.size();

        return;
    }

    std::vector<HitgroupRecord> hitgroupRecords;
    HitgroupRecord rec;

    // all meshes use the same code, so all same hit group
    // pack the header for the SBT record
    OPTIX_CHECK(optixSbtRecordPackHeader(optixcfg->hitgroupPGs[0], &rec));

    // combine face normal + front + back into a float4 array
    std::vector<float4> fnorm;
    std::vector<OptixTraversableHandle> nbgashandle;

    for (int i = 0; i <= mesh->prop; ++i) {
        for (size_t j = 0; j < smesh[i].norm.size(); ++j) {
            fnorm.push_back(make_float4(
                                smesh[i].norm[j].x, smesh[i].norm[j].y,
                                smesh[i].norm[j].z, *(float*)&smesh[i].nbtype[j]));
            nbgashandle.push_back(
                optixcfg->gashandles[smesh[i].nbtype[j]]);
        }
    }

    // prepare buffers on the gpu for holding face-normal data and
    // triangle mesh handles
    optixcfg->fnormBuffer.alloc_and_upload(fnorm);
    optixcfg->nbgashandleBuffer.alloc_and_upload(nbgashandle);

    // save the locations of the face-normals and triangle mesh handles on
    // SBT record data
    rec.data.fnorm = (float4*)optixcfg->fnormBuffer.d_pointer();
    rec.data.nbgashandle =
        (OptixTraversableHandle*)optixcfg->nbgashandleBuffer.d_pointer();

    hitgroupRecords.push_back(rec);

    optixcfg->hitgroupRecordsBuffer.alloc_and_upload(hitgroupRecords);
    optixcfg->sbt.hitgroupRecordBase =
        optixcfg->hitgroupRecordsBuffer.d_pointer();
    optixcfg->sbt.hitgroupRecordStrideInBytes = sizeof(HitgroupRecord);
    optixcfg->sbt.hitgroupRecordCount = (int)hitgroupRecords.size();

}

/**
 * @ Free allocated memory
 */
void clearOptixParams(OptixParams* optixcfg) {
    optixcfg->raygenRecordsBuffer.free();
    optixcfg->missRecordsBuffer.free();
    optixcfg->hitgroupRecordsBuffer.free();
    optixcfg->launchParamsBuffer.free();
    optixcfg->vertexBuffer.free();
    optixcfg->indexBuffer.free();
    optixcfg->fnormBuffer.free();
    optixcfg->nbgashandleBuffer.free();
    optixcfg->asBuffer.free();
    optixcfg->seedBuffer.free();
    optixcfg->outputBuffer.free();
    free(optixcfg->outputHostBuffer);
}
