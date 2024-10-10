#include <iostream>
#include <limits>
#include <math.h>
#include <stdint.h>

#include <optix.h>
#include <cuda_runtime.h>

#include <optix_device.h>
#include <optix_types.h>
#include <sutil/vec_math.h>

#include "implicit_capsule.h"
#include "mmc_optix_ray.h"
#include "mmc_optix_launchparam.h"

// as of 09/17/2022, only otFlux, otFluence, otEnergy are supported
enum TOutputType {otFlux, otFluence, otEnergy, otJacobian, otWL, otWP};

constexpr float R_C0 = 3.335640951981520e-12f; // 1/C0 in s/mm
constexpr float MAX_ACCUM = 1000.0f;
constexpr float SAFETY_DISTANCE = 0.0001f; // ensure ray cut through triangle
constexpr float DOUBLE_SAFETY_DISTANCE = 0.0002f;

// simulation configuration and medium optical properties
extern "C" {
    __constant__ MMCParam gcfg;
}

/**
 * @brief Init RNG seed for each thread
 */
__device__ __forceinline__ void initRNGSeed(mcx::Random& rng, const int& idx) {
    rng = mcx::Random(((uint4*)gcfg.seedbuffer)[idx]);
}

/**
 * @brief Launch a new photon
 */
__device__ __forceinline__ void launchPhoton(optixray& r, mcx::Random& rng) {
    r.p0 = gcfg.srcpos;
    r.dir = gcfg.srcdir;
    r.slen = rng.rand_next_scatlen();
    r.weight = 1.0f;
    r.photontimer = 0.0f;
    r.mediumid = gcfg.mediumid0;
    r.gashandle = gcfg.gashandle0;
}

/**
 * @brief Move a photon one step forward
 */
__device__ __forceinline__ void movePhoton(optixray& r, mcx::Random& rng) {
    optixTrace(r.gashandle, r.p0, r.dir, 0.0f, r.slen / gcfg.medium[r.mediumid].mus,
               0.0f, OptixVisibilityMask(255), OptixRayFlags::OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES, 0, 1, 0,
               *(uint32_t*) & (r.p0.x), *(uint32_t*) & (r.p0.y), *(uint32_t*) & (r.p0.z),
               *(uint32_t*) & (r.dir.x), *(uint32_t*) & (r.dir.y), *(uint32_t*) & (r.dir.z),
               *(uint32_t*) & (r.slen), *(uint32_t*) & (r.weight), *(uint32_t*) & (r.photontimer),
               r.mediumid,
               rng.intSeed.x, rng.intSeed.y, rng.intSeed.z, rng.intSeed.w,
               *((uint32_t*) & (r.gashandle) + 1), *(uint32_t*) & (r.gashandle));
}

/**
 * @brief Rotate a vector with given azimuth and zenith angles
 */
__device__ __forceinline__ float3 rotateVector(const float3& vec, const float2& zen,
        const float2& azi) {
    if (vec.z > -1.0f + std::numeric_limits<float>::epsilon() &&
            vec.z < 1.0f - std::numeric_limits<float>::epsilon()) {
        float tmp0 = 1.0f - vec.z * vec.z;
        float tmp1 = zen.x * rsqrtf(tmp0);
        return tmp1 * (azi.y * make_float3(vec.x, vec.y, -tmp0) *
                       make_float3(vec.z, vec.z, 1.f) + azi.x * make_float3(-vec.y, vec.x, 0.0f))
               + zen.y * vec;
    } else {
        return make_float3(zen.x * azi.y, zen.x * azi.x, (vec.z > 0.0f) ? zen.y : -zen.y);
    }
}

/**
 * @brief Returns the sine and cosine from the Henyey-Greenstein distribution
 */
__device__ __forceinline__ float2 henyeyGreenstein(const float& g, mcx::Random& rand) {
    float ctheta;

    if (fabs(g) > std::numeric_limits<float>::epsilon()) {
        ctheta = (1.0f - g * g) / (1.0f - g + 2.0f * g * rand.uniform(0.0f, 1.0f));
        ctheta *= ctheta;
        ctheta = (1.0f + g * g - ctheta) / (2.0f * g);
        ctheta = fmax(-1.0f, fmin(1.0f, ctheta));
    } else {
        ctheta = 2.0f * rand.uniform(0.0f, 1.0f) - 1.0f;
    }

    return make_float2(sinf(acosf(ctheta)), ctheta);
}

/**
 * @brief Update ray direction after a scattering event
 */
__device__ __forceinline__ float3 selectScatteringDirection(const float3& dir,
        const float& g, mcx::Random& rand) {
    float2 aziScat;
    sincosf(rand.uniform(0.0f, 2.0f * M_PIf), &aziScat.x, &aziScat.y);

    float2 zenScat = henyeyGreenstein(g, rand);

    return rotateVector(dir, zenScat, aziScat);
}

/**
 * @brief Convert 3D indices to 1D
 */
__device__ __forceinline__ uint subToInd(const uint3& idx3d) {
    return idx3d.z * gcfg.crop0.y + idx3d.y * gcfg.crop0.x + idx3d.x;
}

/**
 * @brief Get the index of the voxel that encloses p
 */
__device__ __forceinline__ uint getVoxelIdx(const float3& p) {
    return subToInd(make_uint3(p.x > 0.0f ? __float2int_rd(min(p.x, gcfg.nmax.x) * gcfg.dstep) : 0,
                               p.y > 0.0f ? __float2int_rd(min(p.y, gcfg.nmax.y) * gcfg.dstep) : 0,
                               p.z > 0.0f ? __float2int_rd(min(p.z, gcfg.nmax.z) * gcfg.dstep) : 0));
}

/**
 * @brief Get the offset of the current time frame
 */
__device__ __forceinline__ uint getTimeFrame(const float& tof) {
    return min(((int)((tof - gcfg.tstart) * gcfg.Rtstep)),
               gcfg.maxgate - 1) * gcfg.crop0.z;
}

/**
 * @brief Get the coordinates and width of a capsule
 **/
__device__ __forceinline__ immc::ImplicitCapsule& getCapsuleFromID(int id) {
    return ((immc::ImplicitCapsule*)gcfg.capsuleData)[id];
}

// tests intersection of ray with a sphere and returns intersections as floats of distance from     start of ray
// geometric solution taken from: scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-trac    er-rendering-simple-shapes/ray-sphere-intersection.html
__device__ __forceinline__ float2 get_sphere_intersections(const float3 center,
        const float width, const float3 ray_origin, const float3 ray_dir) {

    float3 L = center - ray_origin;
    float tca = dot(L, ray_dir);

    float d2 = dot(L, L) - tca * tca;

    // result if no intersections, set both to negatives
    if (d2 > width * width) {
        float2 hitTs = make_float2(-1, -1);
        return hitTs;
    }

    // calculates both intersections
    float thc = sqrt(width * width - d2);
    float2 hitTs;
    hitTs.x = tca - thc;
    hitTs.y = tca + thc;
    return hitTs;
}

// tests intersection of ray with an infinite cylinder (derived from parametric substitution)
// from davidjcobb.github.io/articles/ray-cylinder-intersection
// solve a quadratic equation of at^2+bt+c = 0 for t
__device__ __forceinline__ float2 get_inf_cyl_intersections(const float3 vertex1, const float3     vertex2, const float width, const float3 ray_origin, const float3 ray_dir) {
    float3 R1 = ray_origin - vertex2;
    float3 Cs = vertex1 - vertex2;
    float  Ch = length(Cs);
    float3 Ca = Cs / Ch;

    float Ca_dot_Rd = dot(Ca, ray_dir);
    float Ca_dot_R1 = dot(Ca, R1);
    float R1_dot_R1 = dot(R1, R1);

    float a = 1 - (Ca_dot_Rd * Ca_dot_Rd);
    float b = 2 * (dot(ray_dir, R1) - Ca_dot_Rd * Ca_dot_R1);
    float c = R1_dot_R1 - Ca_dot_R1 * Ca_dot_R1 - (width * width);

    float2 cyl_hits;
    float discriminant = b * b - 4.0f * a * c;

    if (discriminant < 0) {
        cyl_hits.x = -1;
        cyl_hits.y = -1;
        return cyl_hits;
    }

    cyl_hits.x = (-b - sqrt(discriminant)) / (2 * a);
    cyl_hits.y = (-b + sqrt(discriminant)) / (2 * a);
    return cyl_hits;
}

/**
 * @brief Save output to a buffer
 */
__device__ __forceinline__ void saveToBuffer(const uint& eid, const float& w) {
    // to minimize numerical error, use the same trick as MCX
    // becomes much slower when using atomicAdd(*double, double)
    float accum = atomicAdd(&((float*)gcfg.outputbuffer)[eid], w);

    if (accum > MAX_ACCUM) {
        if (atomicAdd(&((float*)gcfg.outputbuffer)[eid], -accum) < 0.0f) {
            atomicAdd(&((float*)gcfg.outputbuffer)[eid], accum);
        } else {
            atomicAdd(&((float*)gcfg.outputbuffer)[eid + gcfg.crop0.w], accum);
        }
    }
}

/**
 * @brief Accumulate output quantities to a 3D grid
 */
__device__ __forceinline__ void accumulateOutput(const optixray& r, const Medium& prop,
        const float& lmove) {
    // divide path into segments of equal length
    int segcount = ((int)(lmove * gcfg.dstep) + 1) << 1;
    float seglen = lmove / segcount;
    float segdecay = expf(-prop.mua * seglen);
    float segloss = (gcfg.outputtype == otEnergy) ? r.weight * (1.0f - segdecay) :
                    (prop.mua ? r.weight * (1.0f - segdecay) / prop.mua : 0.0f);

    // deposit weight loss of each segment to the corresponding grid
    float3 step = seglen * r.dir;
    float3 segmid = r.p0 - gcfg.nmin + 0.5f * step; // segment midpoint
    float currtof = r.photontimer + seglen * R_C0 * prop.n; // current time of flight

    // find the information of the first segment
    uint oldeid = getTimeFrame(currtof) + getVoxelIdx(segmid);
    float oldweight = segloss;

    // iterater over the rest of the segments
    for (int i = 1; i < segcount; ++i) {
        // update information for the curr segment
        segloss *= segdecay;
        segmid += step;
        currtof += seglen * R_C0 * prop.n;
        uint neweid = getTimeFrame(currtof) + getVoxelIdx(segmid);

        // save when entering a new element or during the last segment
        if (neweid != oldeid) {
            saveToBuffer(oldeid, oldweight);
            // reset oldeid and weight bucket
            oldeid = neweid;
            oldweight = 0.0f;
        }

        oldweight += segloss;
    }

    // save the weight loss of the last segment
    saveToBuffer(oldeid, oldweight);
}

/**
 * @brief reflect or refract a ray at mismatched boundary
 */
__device__ __forceinline__ bool reflectray(const float3& norm, const float& n1,
        const float& n2, mcx::Random& rng, optixray& r) {
    float Icos, Re, Im, Rtotal, tmp0, tmp1, tmp2;

    Icos = fabs(dot(r.dir, norm));
    tmp0 = n1 * n1;
    tmp1 = n2 * n2;
    tmp2 = 1.f - tmp0 / tmp1 * (1.f - Icos * Icos);

    if (tmp2 > 0.0f) {
        // if no total internal reflection
        Re = tmp0 * Icos * Icos + tmp1 * tmp2; /*transmission angle*/
        tmp2 = sqrtf(tmp2); /*to save one sqrt*/
        Im = 2.f * n1 * n2 * Icos * tmp2;
        Rtotal = (Re - Im) / (Re + Im); /*Rp*/
        Re = tmp1 * Icos * Icos + tmp0 * tmp2 * tmp2;
        Rtotal = (Rtotal + (Re - Im) / (Re + Im)) * 0.5f; /*(Rp+Rs)/2*/

        if (rng.uniform(0.0f, 1.0f) <= Rtotal) {
            // do reflection, correct photon position
            r.p0 -= r.dir * DOUBLE_SAFETY_DISTANCE;
            r.dir += -2.0f * Icos * norm;
        } else {
            // do transmission
            r.dir += -Icos * norm;
            r.dir = tmp2 * norm + n1 / n2 * r.dir;
            return false;
        }
    } else {
        // total internel reflection, correct photon position
        r.p0 -= r.dir * DOUBLE_SAFETY_DISTANCE;
        r.dir += -2.0f * Icos * norm;
    }

    // normalize new direction
    tmp0 = rsqrtf(dot(r.dir, r.dir));
    r.dir *= tmp0;
    return true;
}

/**
 * @brief Launch photon and trace ray iteratively
 */
extern "C" __global__ void __raygen__rg() {
    uint3 launchindex = optixGetLaunchIndex();

    // init RNG seed for each thread
    mcx::Random rng;
    initRNGSeed(rng, launchindex.x);

    // init a ray
    optixray r;
    launchPhoton(r, rng);

    int ndone = 0;  // number of simulated photons

    while (ndone < (gcfg.threadphoton + (launchindex.x < gcfg.oddphoton))) {
        movePhoton(r, rng);

        // when a photon escapes or tof reaches the upper limit
        if (!(r.mediumid && r.photontimer < gcfg.tend)) {
            launchPhoton(r, rng);
            ++ndone;
        }
    }
}

/**
 * @brief when a photon hits a triangle
 */
extern "C" __global__ void __closesthit__ch() {
    // get photon and ray information from payload
    optixray r = getRay();

    // get rng
    mcx::Random rng = getRNG();

    // distance to the intersection
    const float lmove = optixGetRayTmax();

    // get medium properties
    const Medium currprop = gcfg.medium[r.mediumid];

    // save output
    accumulateOutput(r, currprop, lmove);

    // update photon position
    r.p0 += r.dir * (lmove + SAFETY_DISTANCE);

    // update photon weight
    r.weight *= expf(-currprop.mua * lmove);

    // update photon timer
    r.photontimer += lmove * R_C0 * currprop.n;

    // after hitting a boundary, update remaining scattering length
    r.slen -= lmove * currprop.mus;

    // intersected triangle id
    const int primid = optixGetPrimitiveIndex();

    // get info of triangle, including face normal, neighbouring medium
    const TriangleMeshSBTData& sbtData =
        *(const TriangleMeshSBTData*)(optixGetSbtDataPointer());
    float4 fnorm = sbtData.fnorm[primid];

    // assume transmission
    uint origmed = r.mediumid;
    OptixTraversableHandle origgashandle = r.gashandle;

    const unsigned int hitkind = optixGetHitKind();
    const OptixPrimitiveType hit_type = optixGetPrimitiveType(hitkind);

    if (hit_type == OptixPrimitiveType::OPTIX_PRIMITIVE_TYPE_CUSTOM) {
        // TODO do something with the medium since it's a capsule
        r.mediumid = 2.0;

        printf("HIT\n");
    }

    r.mediumid = __float_as_uint(fnorm.w);
    r.gashandle = sbtData.nbgashandle[primid];

    // update ray direction at mismatched boundary
    if (gcfg.isreflect && currprop.n != gcfg.medium[r.mediumid].n) {
        if (reflectray(*(float3*)&fnorm, currprop.n, gcfg.medium[r.mediumid].n,
                       rng, r)) {
            // if the ray is reflected
            r.mediumid = origmed;
            r.gashandle = origgashandle;
        }
    }

    // update rng
    setRNG(rng);

    // update ray
    setRay(r);
}

/**
 * @brief when a photon has a scattering event
 */
extern "C" __global__ void __miss__ms() {
    // get photon and ray information from payload
    optixray r = getRay();

    // get rng
    mcx::Random rng = getRNG();

    // get medium properties
    const Medium currprop = gcfg.medium[r.mediumid];

    // distance to the scattering event site
    float lmove = r.slen / currprop.mus;

    // save output
    accumulateOutput(r, currprop, lmove);

    // update photon position
    r.p0 += r.dir * lmove;

    // update photon weight
    r.weight *= expf(-currprop.mua * lmove);

    // update photon timer
    r.photontimer += lmove * R_C0 * currprop.n;

    // scattering event
    r.dir = selectScatteringDirection(r.dir, currprop.g, rng);
    r.slen = rng.rand_next_scatlen();

    // update rng
    setRNG(rng);

    // update ray
    setRay(r);
}


// intersection test for capsule primitives
extern "C" __global__ void __intersection__customlinearcapsule() {
    // 1. initialize variables for geometry
    int primIdx = optixGetPrimitiveIndex();
    unsigned int capsuleIdx;

    float width_offset = 0;

    if (primIdx >= gcfg.num_inside_prims) {
        capsuleIdx = primIdx - gcfg.num_inside_prims;
        width_offset = gcfg.WIDTH_ADJ;
    } else {
        capsuleIdx = primIdx;
    }

    const immc::ImplicitCapsule& capsule = getCapsuleFromID(capsuleIdx);

    // vector going from pt2 to pt1:
    float3 lineseg_AB = capsule.vertex1 - capsule.vertex2;
    float width = capsule.width + width_offset;
    float t_min = optixGetRayTmin();
    float t_max = optixGetRayTmax();
    // get normalized ray direction
    float3 ray_dir = normalize(optixGetWorldRayDirection());
    float3 ray_origin = optixGetWorldRayOrigin();

    // test for intersections with sphere 1:
    float2 sphere_one_hits = get_sphere_intersections(capsule.vertex1, width,
                             ray_origin, ray_dir);

    // get coordinates for intersections
    float3 sphere_one_hit_one = ray_origin + ray_dir * sphere_one_hits.x;
    float3 sphere_one_hit_two = ray_origin + ray_dir * sphere_one_hits.y;

    // discard intersections on interior side of sphere 1:
    // (discard if negative when taking dot product with vector AB)
    if (dot(sphere_one_hit_one - capsule.vertex1, lineseg_AB) < 0) {
        sphere_one_hits.x = -1;
    }

    if (dot(sphere_one_hit_two - capsule.vertex1, lineseg_AB) < 0) {
        sphere_one_hits.y = -1;
    }

    optixReportIntersection(sphere_one_hits.x, 1);
    optixReportIntersection(sphere_one_hits.y, 1);

    // test for intersections with sphere 2:
    float2 sphere_two_hits = get_sphere_intersections(capsule.vertex2, width,
                             ray_origin, ray_dir);
    // get coordinates for intersections
    float3 sphere_two_hit_one = ray_origin + (ray_dir * sphere_two_hits.x);
    float3 sphere_two_hit_two = ray_origin + (ray_dir * sphere_two_hits.y);

    // discard intersections on interior side of sphere 2:
    if (dot(capsule.vertex2 - sphere_two_hit_one, lineseg_AB) < 0) {
        sphere_two_hits.x = -1;
    }

    if (dot(capsule.vertex2 - sphere_two_hit_two, lineseg_AB) < 0) {
        sphere_two_hits.y = -1;
    }

    optixReportIntersection(sphere_two_hits.x, 2);
    optixReportIntersection(sphere_two_hits.y, 2);

    // test for intersections with infinite cylinder:
    float2 cyl_hits = get_inf_cyl_intersections(capsule.vertex1, capsule.vertex2, width,
                      ray_origin, ray_dir);

    // discard intersections on exterior of cylinder:
    float3 cyl_hit_one = ray_origin + ray_dir * cyl_hits.x;
    float3 cyl_hit_two = ray_origin + ray_dir * cyl_hits.y;

    if (dot(cyl_hit_one - capsule.vertex1, lineseg_AB) > 0 ||
            dot(cyl_hit_one - capsule.vertex2, lineseg_AB) < 0) {
        cyl_hits.x = -1;
    }

    if (dot(cyl_hit_two - capsule.vertex1, lineseg_AB) > 0 ||
            dot(cyl_hit_two - capsule.vertex2, lineseg_AB) < 0) {
        cyl_hits.y = -1;
    }

    optixReportIntersection(cyl_hits.x, 3);
    optixReportIntersection(cyl_hits.y, 3);

    return;

}



