/*******************************************************************************
**  Mesh-based Monte Carlo (MMC)
**
**  Author: Qianqian Fang <fangq at nmr.mgh.harvard.edu>
**
**  Reference:
**  (Fang2010) Qianqian Fang, "Mesh-based Monte Carlo Method Using Fast Ray-Tracing 
**          in Pl�cker Coordinates," Biomed. Opt. Express, 1(1) 165-175 (2010)
**
**  (Fang2009) Qianqian Fang and David A. Boas, "Monte Carlo Simulation of Photon 
**          Migration in 3D Turbid Media Accelerated by Graphics Processing 
**          Units," Optics Express, 17(22) 20178-20190 (2009)
**
**  simpmesh.c: basic vector math and mesh operations
**
**  License: GPL v3, see LICENSE.txt for details
**
*******************************************************************************/

#ifndef _MMC_MESH_UNIT_H
#define _MMC_MESH_UNIT_H

#include <stdio.h>
#include <math.h>
#include "mcx_utils.h"

#ifdef MMC_USE_SSE
#include <smmintrin.h>
#endif

#ifdef MMC_LOGISTIC
  #include "logistic_rand.h"
#else
  #include "posix_randr.h"
#endif

#define MMC_UNDEFINED (3.40282347e+38F)
#define R_RAND_MAX (1.f/RAND_MAX)
#define TWO_PI     (M_PI*2.0)
#define EPS        1e-9f
#define LOG_MT_MAX 22.1807097779182f
#define R_MIN_MUS  1e9f
#define R_C0       3.335640951981520e-12f  //1/C0 in s/mm

typedef struct MMCMedium{
	float mua,mus,g,n;
} medium;

typedef struct femmesh{
	int nn; // number of nodes
	int ne; // number of elems
	int prop;
	float3 *node;
	int4 *elem;
	int  *type;
	int4 *facenb;
	medium *med;
	float *atte;
	float *weight;
	float *evol; /*volume of an element*/
	float *nvol; /*veronio volume of a node*/
} tetmesh;

typedef struct tplucker{
	tetmesh *mesh;
	float3 *d;
	float3 *m;
} tetplucker;
/*
static inline void vec_add(float3 *a,float3 *b,float3 *res);
static inline void vec_diff(float3 *a,float3 *b,float3 *res);
static inline void vec_cross(float3 *a,float3 *b,float3 *res);
static inline void vec_mult_add(float3 *a,float3 *b,float sa,float sb,float3 *res);
static inline float vec_dot(float3 *a,float3 *b);
static inline float pinner(float3 *Pd,float3 *Pm,float3 *Ad,float3 *Am);
static inline float dist2(float3 *p0,float3 *p1);
static inline float dist(float3 *p0,float3 *p1);
*/
void mesh_init(tetmesh *mesh);
void mesh_loadnode(tetmesh *mesh,Config *cfg);
void mesh_loadelem(tetmesh *mesh,Config *cfg);
void mesh_loadfaceneighbor(tetmesh *mesh,Config *cfg);
void mesh_loadmedia(tetmesh *mesh,Config *cfg);
void mesh_loadelemvol(tetmesh *mesh,Config *cfg);

void mesh_clear(tetmesh *mesh);
float mesh_normalize(tetmesh *mesh,Config *cfg, float Eabsorb, float Etotal);
void mesh_build(tetmesh *mesh);
void mesh_error(char *msg);
void mesh_filenames(char *format,char *foutput,Config *cfg);
void mesh_saveweight(tetmesh *mesh,Config *cfg);

void plucker_init(tetplucker *plucker,tetmesh *mesh);
void plucker_build(tetplucker *plucker);
void plucker_clear(tetplucker *plucker);
float mc_next_scatter(float g, float3 *dir,RandType *ran,RandType *ran0,Config *cfg);


static inline void vec_add(float3 *a,float3 *b,float3 *res){
	res->x=a->x+b->x;
	res->y=a->y+b->y;
	res->z=a->z+b->z;
}
static inline void vec_diff(float3 *a,float3 *b,float3 *res){
        res->x=b->x-a->x;
        res->y=b->y-a->y;
        res->z=b->z-a->z;
}
static inline void vec_mult_add(float3 *a,float3 *b,float sa,float sb,float3 *res){
	res->x=sb*b->x+sa*a->x;
	res->y=sb*b->y+sa*a->y;
	res->z=sb*b->z+sa*a->z;
}
static inline void vec_cross(float3 *a,float3 *b,float3 *res){
	res->x=a->y*b->z-a->z*b->y;
	res->y=a->z*b->x-a->x*b->z;
	res->z=a->x*b->y-a->y*b->x;
}

static inline void mmc_sincosf(float x, float * sine, float * cosine){
#if defined(__GNUC__) && defined(__linux__)
    __builtin_sincosf(x, sine, cosine);
#else
    *sine = sinf(x);
    *cosine = cosf(x);
#endif
} 

#ifndef MMC_USE_SSE
static inline float vec_dot(float3 *a,float3 *b){
        return a->x*b->x+a->y*b->y+a->z*b->z;
}
#else

#ifndef __SSE4_1__
static inline float vec_dot(float3 *a,float3 *b){
        float dot;
        __m128 na,nb,res;
        na=_mm_load_ps(&a->x);
        nb=_mm_load_ps(&b->x);
        res=_mm_mul_ps(na,nb);
        res=_mm_hadd_ps(res,res);
        res=_mm_hadd_ps(res,res);
        _mm_store_ss(&dot,res);
        return dot;   
}
#else
static inline float vec_dot(float3 *a,float3 *b){
        float dot;
        __m128 na,nb,res;
        na=_mm_load_ps(&a->x);
        nb=_mm_load_ps(&b->x);
        res=_mm_dp_ps(na,nb,0x7f);
        _mm_store_ss(&dot,res);
        return dot;
}
#endif
        
#endif
 
static inline float pinner(float3 *Pd,float3 *Pm,float3 *Ad,float3 *Am){
        return vec_dot(Pd,Am)+vec_dot(Pm,Ad);
}


static inline float dist2(float3 *p0,float3 *p1){
    return (p1->x-p0->x)*(p1->x-p0->x)+(p1->y-p0->y)*(p1->y-p0->y)+(p1->z-p0->z)*(p1->z-p0->z);
}

static inline float dist(float3 *p0,float3 *p1){
    return sqrt(dist2(p0,p1));
}
#endif
