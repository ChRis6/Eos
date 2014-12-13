/*
 * Copyright (c) 2014 Christos Papaioannou
 *
 * This software is provided 'as-is', without any express or implied
 * warranty. In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 * claim that you wrote the original software. If you use this software
 * in a product, an acknowledgment in the product documentation would be
 * appreciated but is not required.
 *
 * 2. Altered source versions must be plainly marked as such, and must not be
 * misrepresented as being the original software.
 *
 * 3. This notice may not be removed or altered from any source distribution.
 */

#ifndef _KERNEL_INVOCATIONS_H
#define _KERNEL_INVOCATIONS_H

#include "kernelWrapper.h"
#include "cudaQualifiers.h"
#include "cudaStructures.h"
#include "BVH.h"		// for SURFACES_PER_LEAF

typedef struct __align__(16) intr{
    int triIndex;
    glm::vec3 baryCoords;
}intersection_t;


// new cuda Scene kernels
__global__ void __rayTrace_MegaKernel( cudaScene_t* deviceScene, Camera* camera, int width, int height, uchar4* imageBuffer);
__global__ void __rayTrace_WarpShuffle_MegaKernel( cudaScene_t* deviceScene, Camera* camera, int width, int height, uchar4* imageBuffer);


// new cudaScene device functions

DEVICE FORCE_INLINE int rayIntersectsCudaAABB(const Ray& ray, const glm::aligned_vec4& minBoxBounds, const glm::aligned_vec4& maxBoxBounds, float dist){

	const glm::aligned_vec4& lb = minBoxBounds;
	const glm::aligned_vec4& rt = maxBoxBounds;

	const glm::aligned_vec4& rayOrigin = glm::aligned_vec4(ray.getOrigin(), 1.0f);
	const glm::aligned_vec4& rayInvDirection = glm::aligned_vec4(ray.getInvDirection(), 0.0f);

	float tmin = fmaxf(fmaxf(fminf(((lb.x - rayOrigin.x) * rayInvDirection.x), ((rt.x - rayOrigin.x) * rayInvDirection.x)), fminf(((lb.y - rayOrigin.y) * rayInvDirection.y), ((rt.y - rayOrigin.y) * rayInvDirection.y))), fminf(((lb.z - rayOrigin.z) * rayInvDirection.z), ((rt.z - rayOrigin.z) * rayInvDirection.z)));
	float tmax = fminf(fminf(fmaxf(((lb.x - rayOrigin.x) * rayInvDirection.x), ((rt.x - rayOrigin.x) * rayInvDirection.x)), fmaxf(((lb.y - rayOrigin.y) * rayInvDirection.y), ((rt.y - rayOrigin.y) * rayInvDirection.y))), fmaxf(((lb.z - rayOrigin.z) * rayInvDirection.z), ((rt.z - rayOrigin.z) * rayInvDirection.z)));

	return ((tmin < tmax && tmax > 0) && dist >= tmin);

}

DEVICE FORCE_INLINE int rayIntersectsCudaAABB(const cudaRay& ray, const glm::aligned_vec4& minBoxBounds, const glm::aligned_vec4& maxBoxBounds, float dist){

  const glm::aligned_vec4& lb = minBoxBounds;
  const glm::aligned_vec4& rt = maxBoxBounds;

  const glm::aligned_vec4& rayOrigin = glm::aligned_vec4(ray.getOrigin(), 1.0f);
  const glm::aligned_vec4& rayInvDirection = glm::aligned_vec4(ray.getInvDirection(), 0.0f);

  float tmin = fmaxf(fmaxf(fminf(((lb.x - rayOrigin.x) * rayInvDirection.x), ((rt.x - rayOrigin.x) * rayInvDirection.x)), fminf(((lb.y - rayOrigin.y) * rayInvDirection.y), ((rt.y - rayOrigin.y) * rayInvDirection.y))), fminf(((lb.z - rayOrigin.z) * rayInvDirection.z), ((rt.z - rayOrigin.z) * rayInvDirection.z)));
  float tmax = fminf(fminf(fmaxf(((lb.x - rayOrigin.x) * rayInvDirection.x), ((rt.x - rayOrigin.x) * rayInvDirection.x)), fmaxf(((lb.y - rayOrigin.y) * rayInvDirection.y), ((rt.y - rayOrigin.y) * rayInvDirection.y))), fmaxf(((lb.z - rayOrigin.z) * rayInvDirection.z), ((rt.z - rayOrigin.z) * rayInvDirection.z)));

  return ((tmin < tmax && tmax > 0) && dist >= tmin);

}


DEVICE void intersectRayWithCudaLeaf( const Ray& ray, cudaScene_t* __restrict__  deviceScene, int bvhLeafIndex, float* __restrict__  minDistace, intersection_t*  __restrict__  intersection, int threadID);

DEVICE FORCE_INLINE void  intersectRayWithCudaLeafRestricted( const Ray& ray,// ray
                                    int bvhLeafIndex, int* __restrict__ numSurfacesEncapulated, int* __restrict__ surfacesIndices, glm::mat4* __restrict__ inverseTransformation, // bvh
                                    glm::aligned_vec3* __restrict__ v1, glm::aligned_vec3* __restrict__ v2, glm::aligned_vec3* __restrict__ v3, // triangle vertices
                                    int* __restrict__ triTransIndex,    // triangle transformations
                                    float* __restrict__ minDistace, intersection_t* __restrict__ threadIntersection, int threadID );



DEVICE FORCE_INLINE bool rayIntersectsCudaTriangle( const Ray& ray, const glm::aligned_vec3& v1, const glm::aligned_vec3& v2, const glm::aligned_vec3& v3, glm::vec3& baryCoords){
    const glm::aligned_vec3& P = glm::aligned_vec3(glm::cross(ray.getDirection(), v3 - v1));
    float det = glm::dot(v2 - v1, P);
    if (det > -0.00001f && det < 0.00001f)
        return false;

    det = 1.0f / det;
    const glm::aligned_vec3& T = glm::aligned_vec3(ray.getOrigin() - v1);
    const glm::aligned_vec3& Q = glm::aligned_vec3(glm::cross(T, v2 - v1));
    
    baryCoords.x = glm::dot(v3 - v1, Q) * det;
    baryCoords.y = glm::dot(T, P) * det;
    baryCoords.z = glm::dot(ray.getDirection(), Q) * det;

    //if ((baryCoords.x < 0.0f) || (baryCoords.y < 0.0f || baryCoords.y > 1.0f) || ( baryCoords.z < 0.0f || baryCoords.y + baryCoords.z > 1.0f) )
    //    return false;
    
    //return true;

    return !((baryCoords.x < 0.0f) || (baryCoords.y < 0.0f || baryCoords.y > 1.0f) || ( baryCoords.z < 0.0f || baryCoords.y + baryCoords.z > 1.0f)) ;
}

DEVICE FORCE_INLINE bool rayIntersectsCudaTriangle( const cudaRay& ray, const glm::aligned_vec3& v1, const glm::aligned_vec3& v2, const glm::aligned_vec3& v3, glm::vec3& baryCoords){
    const glm::aligned_vec3& P = glm::cross(ray.getDirection(), v3 - v1);
    float det = glm::dot(v2 - v1, P);
    if (det > -0.00001f && det < 0.00001f)
        return false;

    det = 1.0f / det;
    const glm::aligned_vec3& T = ray.getOrigin() - v1;
    const glm::aligned_vec3& Q = glm::cross(T, v2 - v1);
    
    baryCoords.x = glm::dot(v3 - v1, Q) * det;
    baryCoords.y = glm::dot(T, P) * det;
    baryCoords.z = glm::dot(ray.getDirection(), Q) * det;

    //if ((baryCoords.x < 0.0f) || (baryCoords.y < 0.0f || baryCoords.y > 1.0f) || ( baryCoords.z < 0.0f || baryCoords.y + baryCoords.z > 1.0f) )
    //    return false;
    
    //return true;

    return !((baryCoords.x < 0.0f) || (baryCoords.y < 0.0f || baryCoords.y > 1.0f) || ( baryCoords.z < 0.0f || baryCoords.y + baryCoords.z > 1.0f)) ;
}



DEVICE FORCE_INLINE void  intersectCudaRayWithCudaLeafRestricted( const cudaRay& ray,// ray
                                    int bvhLeafIndex, int* __restrict__ numSurfacesEncapulated, /*int* __restrict__ surfacesIndices,*/ glm::mat4* __restrict__ inverseTransformation, // bvh
                                    glm::aligned_vec3* __restrict__ v1, glm::aligned_vec3* __restrict__ v2, glm::aligned_vec3* __restrict__ v3, // triangle vertices
                                    int* __restrict__ triTransIndex,    // triangle transformations
                                    float* __restrict__ minDistace, intersection_t* __restrict__ threadIntersection, int firstTri, int threadID ){

    cudaRay localRay;
    glm::vec3 baryCoords(0.0f);
    int minTriangleIndex;
    int i;
    minTriangleIndex = -1;

    #pragma unroll 4
    for( i = 0; i < numSurfacesEncapulated[bvhLeafIndex]; i++){
        

        /*
         * transform Ray to local Triangle Coordinates
         */

        // first get triangle index in the triangle buffer
        // every leaf has SURFACES_PER_LEAF(look at BVH.h) triangles
        // get triangle i
        //const int triangleIndex = surfacesIndices[ bvhLeafIndex * SURFACES_PER_LEAF + i];
        const int triangleIndex = i + firstTri;

        // get Transformation index
        //int triangleTransformationIndex = deviceScene->triangles->transformationIndex[ triangleIndex];

        // transform ray origin and direction
        localRay.setOrigin( glm::aligned_vec3( inverseTransformation[ triTransIndex[ triangleIndex]] * glm::vec4(ray.getOrigin(), 1.0f)));
        localRay.setDirection( glm::aligned_vec3( inverseTransformation[ triTransIndex[ triangleIndex]] * glm::vec4(ray.getDirection(), 0.0f)));

        // Now intersect Triangle with local ray
        if( rayIntersectsCudaTriangle( localRay, v1[triangleIndex], v2[triangleIndex], v3[triangleIndex], baryCoords) && baryCoords.x < *minDistace){
            // intersection is found
            minTriangleIndex = triangleIndex;
            *minDistace = baryCoords.x;

            threadIntersection->triIndex = minTriangleIndex;
            threadIntersection->baryCoords = baryCoords;
        }// endif

    }// end for
}

#endif