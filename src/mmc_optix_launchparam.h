#ifndef _MMC_OPTIX_LAUNCHPARAM_H
#define _MMC_OPTIX_LAUNCHPARAM_H

#define MAX_PROP_OPTIX 2000              /*maximum property number*/

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
    OptixTraversableHandle gashandle[MAX_PROP_OPTIX];
    unsigned int gasoffset[MAX_PROP_OPTIX + 1];

    CUdeviceptr seedbuffer;             /**< rng seed for each thread */
    CUdeviceptr outputbuffer;

    float3 srcpos;
    float3 srcdir;
    float3 nmin;
    uint4 crop0;
    float dstep;
    float tstart, tend;
    float Rtstep;
    int maxgate;
    unsigned int mediumid0;             /**< initial medium type */

    uint isreflect;
    int outputtype;

    int threadphoton;
    int oddphoton;

    Medium medium[MAX_PROP_OPTIX];
} MMCParam;

struct TriangleMeshSBTData {
    /**< x,y,z: face normal; w: outside medium type */
    float4 *fnorm;
};

#endif
