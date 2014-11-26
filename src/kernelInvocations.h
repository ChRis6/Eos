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

typedef struct intr{
    int triIndex;
    glm::vec3 baryCoords;
}intersection_t;


__global__ void __oneThreadPerPixel_kernel();
__global__ void __renderToBuffer_kernel(char* buffer, unsigned int buffer_len, Camera* camera, DScene* scene, DRayTracer* rayTracer, int width, int height);
__global__ void __calculateIntersections_kernel(Camera* camera, cudaIntersection_t* intersectionBuffer, int intersectionBufferSize, 
	                                            DTriangle* trianglesBuffer, int trianglesBufferSize, BvhNode* bvh, int width, int height);
__global__ void __shadeIntersectionsToBuffer_kernel(char* imageBuffer, unsigned int imageSize, DRayTracer* rayTracer, Camera* camera,
													DLightSource* lights, int numLights,
													cudaIntersection_t* intersectionBuffer, int intersectionBufferSize,
													DMaterial* materialsBuffer, int materialsBufferSize, 
										 			int width, int height);

// new cuda Scene kernels
__global__ void __calculateCudaSceneIntersections_kernel( cudaScene_t* deviceScene, Camera* camera, cudaIntersection_t* intersectionBuffer, int width, int height);
__global__ void __shadeCudaSceneIntersections_kernel( cudaScene_t* deviceScene, Camera* camera, cudaIntersection_t* intersectionBuffer, int width, int height, uchar4* imageBuffer);


DEVICE void traverseTreeAndStore(const Ray& ray, cudaIntersection_t* intersectionBuffer, int intersectionBufferSize, DTriangle* trianglesBuffer, int trianglesBufferSize, BvhNode* bvh, int threadID );
DEVICE bool intersectRayWithLeafNode(const Ray& ray, BvhNode* node, cudaIntersection_t* intersection, float& distance, DTriangle* triangles, int threadID);


// new cudaScene device functions
DEVICE void traverseCudaTreeAndStore( cudaScene_t* deviceScene, const Ray& ray, cudaIntersection_t* intersectionBuffer, int threadID);
DEVICE void traverseCudaTreeAndStoreSharedStack( int* sharedStack, int* sharedCurrNodeIndex, int* sharedVotes,
												cudaScene_t* deviceScene, const Ray& ray, cudaIntersection_t* intersectionBuffer, int threadID, int threadBlockID);

DEVICE void traverseCudaTreeAndStoreNew( cudaScene_t* deviceScene, const Ray& ray, cudaIntersection_t* intersectionBuffer, int threadID);

DEVICE FORCE_INLINE int rayIntersectsCudaAABB(const Ray& ray, const glm::vec4& minBoxBounds, const glm::vec4& maxBoxBounds, float& dist){
   /*
   glm::vec4 tmin = (minBoxBounds - glm::vec4(ray.getOrigin(), 1.0f)) * glm::vec4( ray.getInvDirection(), 0.0f);
   glm::vec4 tmax = (maxBoxBounds - glm::vec4(ray.getOrigin(), 1.0f)) * glm::vec4( ray.getInvDirection(), 0.0f);
   
   glm::vec4 real_min = glm::min(tmin, tmax);
   glm::vec4 real_max = glm::max(tmin, tmax);
   
   float minmax = fminf( fminf(real_max.x, real_max.y), real_max.z);
   float maxmin = fmaxf( fmaxf(real_min.x, real_min.y), real_min.z);
    
   return (minmax >= maxmin ); //&& limit - 1e-3 <= minmax);
*/

	// lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
	const glm::vec4& lb = minBoxBounds;
	const glm::vec4& rt = maxBoxBounds;

	const glm::vec4& rayOrigin = glm::vec4(ray.getOrigin(), 1.0f);
	const glm::vec4& rayInvDirection = glm::vec4(ray.getInvDirection(), 0.0f);

	float tmin = fmaxf(fmaxf(fminf(((lb.x - rayOrigin.x) * rayInvDirection.x), ((rt.x - rayOrigin.x) * rayInvDirection.x)), fminf(((lb.y - rayOrigin.y) * rayInvDirection.y), ((rt.y - rayOrigin.y) * rayInvDirection.y))), fminf(((lb.z - rayOrigin.z) * rayInvDirection.z), ((rt.z - rayOrigin.z) * rayInvDirection.z)));
	float tmax = fminf(fminf(fmaxf(((lb.x - rayOrigin.x) * rayInvDirection.x), ((rt.x - rayOrigin.x) * rayInvDirection.x)), fmaxf(((lb.y - rayOrigin.y) * rayInvDirection.y), ((rt.y - rayOrigin.y) * rayInvDirection.y))), fmaxf(((lb.z - rayOrigin.z) * rayInvDirection.z), ((rt.z - rayOrigin.z) * rayInvDirection.z)));

	return (tmin < tmax && tmax > 0);
	
	//if( (tmax < 0 || tmin > tmax) )//|| tmin > limit)
    //	return 0;
	//return 1;

}
DEVICE void intersectRayWithCudaLeaf( const Ray& ray, cudaScene_t* __restrict__  deviceScene, int bvhLeafIndex, float* __restrict__  minDistace, intersection_t*  __restrict__  intersection, int threadID);

DEVICE void intersectRayWithCudaLeafRestricted( const Ray& ray,// ray
                                    int bvhLeafIndex, int* __restrict__ numSurfacesEncapulated, int* __restrict__ surfacesIndices, glm::mat4* __restrict__ inverseTransformation, // bvh
                                    glm::vec3* __restrict__ v1, glm::vec3* __restrict__ v2, glm::vec3* __restrict__ v3, // triangle vertices
                                    int* __restrict__ triTransIndex,    // triangle transformations
                                    float* __restrict__ minDistace, intersection_t* __restrict__ threadIntersection, int threadID );

DEVICE FORCE_INLINE bool rayIntersectsCudaTriangle( const Ray& ray, const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3, glm::vec3& baryCoords){
    const glm::vec3& P = glm::cross(ray.getDirection(), v3 - v1);
    float det = glm::dot(v2 - v1, P);
    if (det > -0.00001f && det < 0.00001f)
        return false;

    det = 1.0f / det;
    const glm::vec3& T = ray.getOrigin() - v1;
    const glm::vec3& Q = glm::cross(T, v2 - v1);
    
    baryCoords.x = glm::dot(v3 - v1, Q) * det;
    baryCoords.y = glm::dot(T, P) * det;
    baryCoords.z = glm::dot(ray.getDirection(), Q) * det;

    if ((baryCoords.x < 0.0f) || (baryCoords.y < 0.0f || baryCoords.y > 1.0f) || ( baryCoords.z < 0.0f || baryCoords.y + baryCoords.z > 1.0f) )
        return false;
    
    return true;
}
#endif