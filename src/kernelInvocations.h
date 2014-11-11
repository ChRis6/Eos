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

__global__ void __oneThreadPerPixel_kernel();
__global__ void __renderToBuffer_kernel(char* buffer, unsigned int buffer_len, Camera* camera, DScene* scene, DRayTracer* rayTracer, int width, int height);
__global__ void __calculateIntersections_kernel(Camera* camera, DRayIntersection* intersectionBuffer, int intersectionBufferSize, 
	                                            DTriangle* trianglesBuffer, int trianglesBufferSize, BvhNode* bvh, int width, int height);
__global__ void __shadeIntersectionsToBuffer_kernel(char* imageBuffer, unsigned int imageSize, DRayTracer* rayTracer, Camera* camera,
													DLightSource* lights, int numLights,
													DRayIntersection* intersectionBuffer, int intersectionBufferSize,
													DMaterial* materialsBuffer, int materialsBufferSize, 
										 			int width, int height);


DEVICE void traverseTreeAndStore(const Ray& ray, DRayIntersection* intersectionBuffer, int intersectionBufferSize, DTriangle* trianglesBuffer, int trianglesBufferSize, BvhNode* bvh, int threadID );
DEVICE bool intersectRayWithLeafNode(const Ray& ray, BvhNode* node, DRayIntersection* intersection, float& distance, DTriangle* triangles);
#endif