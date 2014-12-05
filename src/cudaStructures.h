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

#ifndef _CUDA_STRUCTURES_H
#define _CUDA_STRUCTURES_H

#include <glm/glm.hpp>


#include "cudaQualifiers.h"
#include "Scene.h"
#include "Triangle.h"
#include "BVH.h"

// Bounding volume hierachy
typedef struct cudaBvh{
	int* type;
	glm::vec4* minBoxBounds;
	glm::vec4* maxBoxBounds;
	int* numSurfacesEncapulated;
	int* rightChildIndex;
	int* leftChildIndex;

	int* surfacesIndices;
}cudaBvhNode_t;

// lights
typedef struct cudaLightSource{
	glm::vec4* positions;
	glm::vec4* colors;
	int        numLights;
}cudaLightSource_t;

// transformations
typedef struct cudaTransformation{
	glm::mat4* transformation;
	glm::mat4* inverseTransformation;
	glm::mat4* inverseTransposeTransformation;
	int numTransformations;
}cudaTransformations_t;

// triangles
typedef struct cudaTriangle{
	// vertices
	glm::vec3* v1;
	glm::vec3* v2;
	glm::vec3* v3;
	// normals
	glm::vec3* n1;
	glm::vec3* n2;
	glm::vec3* n3;
	//materials
	int* materialIndex;
	// transformations
	int* transformationIndex;
}cudaTriangle_t;

typedef struct cudaMaterial{
	glm::vec4* diffuse;
	glm::vec4* specular;
	float*     ambientIntensity;
	float*     reflectivity;
	int*       shininess;
}cudaMaterial_t;

typedef struct cudaScene{
	cudaBvhNode_t*         bvh;
	cudaLightSource_t*     lights;
	cudaTransformations_t* transformations;
	cudaTriangle_t*        triangles;
	cudaMaterial_t*        materials;
}cudaScene_t;



// create a cuda scene object.The returned pointer points to gpu memory.
// DO NOT DEREFERENCE on host
HOST cudaScene_t* createCudaScene(Scene* h_scene);

#endif