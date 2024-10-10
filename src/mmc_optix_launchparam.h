#ifndef _MMC_OPTIX_LAUNCHPARAM_H
#define _MMC_OPTIX_LAUNCHPARAM_H

#define MAX_PROP_OPTIX 4000              /*maximum property number*/

/**
 * @brief struct for medium optical properties
 */
typedef struct __attribute__((aligned(16))) MCX_medium {
    float mua;                     /**<absorption coeff in 1/mm unit*/
    float mus;                     /**<scattering coeff in 1/mm unit*/
    float g;                       /**<anisotropy*/
    float n;                       /**<refractive index*/
} Medium;

/**
 * @brief struct for simulation configuration paramaters
 */
typedef struct __attribute__((aligned(16))) MMC_Parameter {
    OptixTraversableHandle gashandle0;

    CUdeviceptr seedbuffer;             /**< rng seed for each thread */
    CUdeviceptr outputbuffer;
    CUdeviceptr capsuleData;           // this points to a device buffer with immc capsules
    CUdeviceptr surfaceData;

    float3 srcpos;
    float3 srcdir;
    float3 nmin;
    float3 nmax;
    uint4 crop0;
    float dstep;
    float tstart, tend;
    float Rtstep;

    // variables for implicit tracing:
    float WIDTH_ADJ;
    unsigned int num_inside_prims;

    int maxgate;
    unsigned int mediumid0;             /**< initial medium type */

    uint isreflect;
    int outputtype;

    int threadphoton;
    int oddphoton;

    Medium medium[MAX_PROP_OPTIX];
} MMCParam;

struct __attribute__((aligned(16))) TriangleMeshSBTData {
    float4* fnorm; /**< x,y,z: face normal; w: neighboring medium type */
    OptixTraversableHandle* nbgashandle;
};

#endif
