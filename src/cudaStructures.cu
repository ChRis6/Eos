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

#include "cudaStructures.h"

#include <stdio.h>
#include <string.h>



/*
 * returns pointer to GPU memory
 *
 * DO NOT DEREFERNCE ON HOST
 */
HOST cudaBvhNode_t* copyBVH(Scene* h_scene){

	cudaBvhNode_t*	d_bvh;

	int* h_Bvh_type;
	int* d_Bvh_type;
	glm::vec3* h_minBoxBounds, *d_minBoxBounds;
	glm::vec3* h_maxBoxBounds, *d_maxBoxBounds;
	int* h_numSurfacesEncapulated, *d_numSurfacesEncapulated;
	int* h_rightChildIndex, *d_rightChildIndex;
	int* h_leftChildIndex, *d_leftChildIndex;
	int* h_surfacesIndices, *d_surfacesIndices;


	const BVH& sceneBVH = h_scene->getBVH();
	int numBvhNodes   = sceneBVH.getNodesBufferSize();

	cudaErrorCheck( cudaDeviceSynchronize());

	printf("Copying BVH to GPU.BVH tree has %d nodes\n", numBvhNodes);

	//TODO: make it pinned memory for async copies in case scene is too big
	
	cudaErrorCheck( cudaMalloc((void**) &d_Bvh_type, sizeof(int) * numBvhNodes));	
	// aabb
	cudaErrorCheck( cudaMalloc((void**)&d_minBoxBounds, sizeof(glm::vec3) * numBvhNodes));
	cudaErrorCheck( cudaMalloc((void**)&d_maxBoxBounds, sizeof(glm::vec3) * numBvhNodes));
	// numSurfaces per node
	cudaErrorCheck( cudaMalloc((void**)&d_numSurfacesEncapulated, sizeof(int) * numBvhNodes));
	// left child
	cudaErrorCheck( cudaMalloc((void**)&d_leftChildIndex, sizeof(int) * numBvhNodes));
	// right child
	cudaErrorCheck( cudaMalloc((void**)&d_rightChildIndex, sizeof(int) * numBvhNodes));
	// leaf surface indices. Too many maybe? Profile memory later
	cudaErrorCheck( cudaMalloc((void**)&d_surfacesIndices, sizeof(int) * numBvhNodes * SURFACES_PER_LEAF)); // <------------------------------------------


	// set to zero
	cudaErrorCheck( cudaMemset( d_Bvh_type, 0, sizeof(int) * numBvhNodes));
	cudaErrorCheck( cudaMemset( d_minBoxBounds, 0, sizeof(glm::vec3) * numBvhNodes));
	cudaErrorCheck( cudaMemset( d_maxBoxBounds, 0, sizeof(glm::vec3) * numBvhNodes));
	cudaErrorCheck( cudaMemset( d_numSurfacesEncapulated, 0, sizeof(int) * numBvhNodes));
	cudaErrorCheck( cudaMemset( d_leftChildIndex, 0, sizeof(int) * numBvhNodes));
	cudaErrorCheck( cudaMemset( d_rightChildIndex, 0, sizeof(int) * numBvhNodes));
	cudaErrorCheck( cudaMemset( d_surfacesIndices, 0, sizeof(int) * numBvhNodes * SURFACES_PER_LEAF));


	// create temp buffers on host
 	h_Bvh_type = new int[numBvhNodes];
 	h_minBoxBounds = new glm::vec3[numBvhNodes];
 	h_maxBoxBounds = new glm::vec3[numBvhNodes];
 	h_numSurfacesEncapulated = new int[numBvhNodes];
 	h_leftChildIndex = new int[numBvhNodes];
 	h_rightChildIndex = new int[numBvhNodes];
 	h_surfacesIndices = new int[ numBvhNodes * SURFACES_PER_LEAF];

 	memset( h_Bvh_type, 0, sizeof(int) * numBvhNodes);
 	memset( h_minBoxBounds, 0, sizeof(glm::vec3) * numBvhNodes);
 	memset( h_maxBoxBounds, 0, sizeof(glm::vec3) * numBvhNodes);
 	memset( h_numSurfacesEncapulated, 0, sizeof(int) * numBvhNodes);
 	memset( h_leftChildIndex, 0, sizeof(int) * numBvhNodes);
 	memset( h_rightChildIndex, 0, sizeof(int) * numBvhNodes);
 	memset( h_surfacesIndices, 0, sizeof(int) * numBvhNodes * SURFACES_PER_LEAF);

 	BvhNode* h_bvhTree = sceneBVH.getNodesBuffer();
 	for( int i = 0; i < numBvhNodes; i++){

 		h_Bvh_type[i]     = h_bvhTree[i].type;
 		h_minBoxBounds[i] = h_bvhTree[i].aabb.getMinVertex();
 		h_maxBoxBounds[i] = h_bvhTree[i].aabb.getMaxVertex();
 		h_numSurfacesEncapulated[i] = h_bvhTree[i].numSurfacesEncapulated;
 		h_leftChildIndex[i] = h_bvhTree[i].leftChildIndex;
 		h_rightChildIndex[i] = h_bvhTree[i].rightChildIndex;
 		for( int j = 0; j < SURFACES_PER_LEAF; j++)
 			h_surfacesIndices[i * SURFACES_PER_LEAF + j] = h_bvhTree[i].surfacesIndices[j];

 	}


 	// types
 	cudaErrorCheck( cudaMemcpy(d_Bvh_type, h_Bvh_type, sizeof(int) * numBvhNodes, cudaMemcpyHostToDevice));
 	// aabb
 	cudaErrorCheck( cudaMemcpy(d_minBoxBounds, h_minBoxBounds, sizeof(glm::vec3) * numBvhNodes, cudaMemcpyHostToDevice));
 	cudaErrorCheck( cudaMemcpy(d_maxBoxBounds, h_maxBoxBounds, sizeof(glm::vec3) * numBvhNodes, cudaMemcpyHostToDevice));
 	// surfaces encapulated
 	cudaErrorCheck( cudaMemcpy(d_numSurfacesEncapulated, h_numSurfacesEncapulated, sizeof(int) * numBvhNodes, cudaMemcpyHostToDevice));
 	// left child
 	cudaErrorCheck( cudaMemcpy(d_leftChildIndex, h_leftChildIndex, sizeof(int) * numBvhNodes, cudaMemcpyHostToDevice));
 	// right child
 	cudaErrorCheck( cudaMemcpy(d_rightChildIndex, h_rightChildIndex, sizeof(int) * numBvhNodes, cudaMemcpyHostToDevice));
 	// surface indices
 	cudaErrorCheck( cudaMemcpy(d_surfacesIndices, h_surfacesIndices, sizeof(int) * numBvhNodes * SURFACES_PER_LEAF, cudaMemcpyHostToDevice));

 	// deallocate temp
 	delete h_Bvh_type;
 	delete h_minBoxBounds;
 	delete h_maxBoxBounds;
 	delete h_numSurfacesEncapulated;
 	delete h_leftChildIndex;
 	delete h_rightChildIndex;
 	delete h_surfacesIndices;

 	cudaBvhNode_t* h_cudaBvhNode = new cudaBvhNode_t;
 	memset( h_cudaBvhNode, 0, sizeof(cudaBvhNode_t));

 	cudaErrorCheck( cudaMalloc((void**) &d_bvh, sizeof(cudaBvhNode_t)));
 	cudaErrorCheck( cudaMemset(d_bvh, 0, sizeof(cudaBvhNode_t)));

 	h_cudaBvhNode->type = d_Bvh_type;
 	h_cudaBvhNode->minBoxBounds = d_minBoxBounds;
 	h_cudaBvhNode->maxBoxBounds = d_maxBoxBounds;
 	h_cudaBvhNode->numSurfacesEncapulated = d_numSurfacesEncapulated;
 	h_cudaBvhNode->rightChildIndex = d_rightChildIndex;
 	h_cudaBvhNode->leftChildIndex  = d_leftChildIndex;
 	h_cudaBvhNode->surfacesIndices = d_surfacesIndices;
 	cudaErrorCheck( cudaMemcpy(d_bvh, h_cudaBvhNode, sizeof(cudaBvhNode_t), cudaMemcpyHostToDevice));

 	delete h_cudaBvhNode;

 	return d_bvh;
}


HOST cudaLightSource_t* copyLights(Scene* h_scene){

	glm::vec4* h_positions, *d_positions;
	glm::vec4* h_colors, *d_colors;
	cudaLightSource_t* h_light, *d_light;

	int numSceneLights = h_scene->getNumLightSources();

	// allocate device mem
	cudaErrorCheck( cudaMalloc((void**) &d_positions, sizeof(glm::vec4) * numSceneLights));
	cudaErrorCheck( cudaMalloc((void**) &d_colors, sizeof(glm::vec4) * numSceneLights));
	// set zero
	cudaErrorCheck( cudaMemset( d_positions, 0, sizeof(glm::vec4) * numSceneLights));
	cudaErrorCheck( cudaMemset( d_colors, 0, sizeof(glm::vec4) * numSceneLights));

	h_positions = new glm::vec4[numSceneLights];
	h_colors = new glm::vec4[numSceneLights];
	memset(h_positions, 0 , sizeof(glm::vec4) * numSceneLights);
	memset(h_colors, 0, sizeof(glm::vec4) * numSceneLights);

	for( int i = 0; i < numSceneLights; i++){
		h_positions[i] = h_scene->getLightSource(i)->getPosition();
		h_colors[i]    = h_scene->getLightSource(i)->getLightColor();
	}

	cudaErrorCheck( cudaMemcpy( d_positions, h_positions, sizeof(glm::vec4) * numSceneLights, cudaMemcpyHostToDevice));
	cudaErrorCheck( cudaMemcpy( d_colors, h_colors, sizeof(glm::vec4) * numSceneLights, cudaMemcpyHostToDevice));

	h_light = new cudaLightSource_t;
	h_light->positions = d_positions;
	h_light->colors = d_colors;
	h_light->numLights = numSceneLights;

	cudaErrorCheck( cudaMalloc((void**) &d_light, sizeof(cudaLightSource_t)));
	cudaErrorCheck( cudaMemcpy( d_light, h_light, sizeof(cudaLightSource_t), cudaMemcpyHostToDevice));

	delete h_positions;
	delete h_colors;
	delete h_light;

	return d_light;
}

HOST cudaTransformations_t* copyTransformations(Scene* h_scene){

	glm::mat4* h_transformation, *d_transformation;
	glm::mat4* h_inverseTransformation, *d_inverseTransformation;
	glm::mat4* h_inverseTransposeTransformation, *d_inverseTransposeTransformation;
	cudaTransformations_t* h_cudaTransformation, *d_cudaTransformation;

	int numTransformations = h_scene->getNumTransformations();
	int numInverseTrans    = numTransformations / 3;
	int numInverseTransposeTras = numTransformations / 3;
	numTransformations     = numTransformations / 3;

	// allocate memory on device
	cudaErrorCheck( cudaMalloc((void**) &d_transformation, sizeof(glm::mat4) * numTransformations));
	cudaErrorCheck( cudaMalloc((void**) & d_inverseTransformation, sizeof(glm::mat4) * numInverseTrans));
	cudaErrorCheck( cudaMalloc((void**) &d_inverseTransposeTransformation, sizeof(glm::mat4) * numInverseTransposeTras));

	cudaErrorCheck( cudaMemset( d_transformation, 0, sizeof(glm::mat4) * numTransformations));
	cudaErrorCheck( cudaMemset( d_inverseTransformation, 0, sizeof(glm::mat4) * numInverseTrans));
	cudaErrorCheck( cudaMemset( d_inverseTransposeTransformation, 0, sizeof(glm::mat4) * numInverseTransposeTras));

	h_transformation = new glm::mat4[numTransformations];
	h_inverseTransformation = new glm::mat4[numInverseTrans];
	h_inverseTransposeTransformation = new glm::mat4[numInverseTransposeTras];

	int k = 0;
	for( int i = 0; k < numTransformations; i += 3, k++){
		h_transformation[k]                  = h_scene->getTransformationAtIndex(i);
		h_inverseTransformation[k]           = h_scene->getTransformationAtIndex(i+1);
		h_inverseTransposeTransformation[k]  = h_scene->getTransformationAtIndex(i+2); 
	}

	cudaErrorCheck( cudaMemcpy( d_transformation, h_transformation, sizeof(glm::mat4) * numTransformations, cudaMemcpyHostToDevice));
	cudaErrorCheck( cudaMemcpy( d_inverseTransformation, h_inverseTransformation, sizeof(glm::mat4) * numInverseTrans, cudaMemcpyHostToDevice));
	cudaErrorCheck( cudaMemcpy( d_inverseTransposeTransformation, h_inverseTransposeTransformation, sizeof(glm::mat4) * numInverseTransposeTras, cudaMemcpyHostToDevice));

	h_cudaTransformation = new cudaTransformations_t;
	h_cudaTransformation->transformation                 = d_transformation;
	h_cudaTransformation->inverseTransformation          = d_inverseTransformation;
	h_cudaTransformation->inverseTransposeTransformation = d_inverseTransposeTransformation;
	h_cudaTransformation->numTransformations             = numTransformations;

	cudaErrorCheck( cudaMalloc((void**) &d_cudaTransformation, sizeof(cudaTransformations_t)));
	cudaErrorCheck( cudaMemset(d_cudaTransformation, 0, sizeof(cudaTransformations_t)));
	cudaErrorCheck( cudaMemcpy(d_cudaTransformation, h_cudaTransformation, sizeof(cudaTransformations_t), cudaMemcpyHostToDevice));

	delete h_transformation;
	delete h_inverseTransformation;
	delete h_inverseTransposeTransformation;
	delete h_cudaTransformation;

	return d_cudaTransformation;
}

HOST cudaTriangle_t* copyTriangles(Scene* h_scene){

	
	glm::vec3* h_v1, *d_v1;
	glm::vec3* h_v2, *d_v2;
	glm::vec3* h_v3, *d_v3;

	glm::vec3* h_n1, *d_n1;
	glm::vec3* h_n2, *d_n2;
	glm::vec3* h_n3, *d_n3;
	int* h_materialIndex, *d_materialIndex;
	int* h_transformationIndex, *d_transformationIndex;


	int numTriangles = h_scene->getNumSurfaces();

	// allocate gpu memory
	cudaErrorCheck( cudaMalloc((void**) &d_v1, sizeof(glm::vec3) * numTriangles));
	cudaErrorCheck( cudaMalloc((void**) &d_v2, sizeof(glm::vec3) * numTriangles));
	cudaErrorCheck( cudaMalloc((void**) &d_v3, sizeof(glm::vec3) * numTriangles));

	cudaErrorCheck( cudaMalloc((void**) &d_n1, sizeof(glm::vec3) * numTriangles));
	cudaErrorCheck( cudaMalloc((void**) &d_n2, sizeof(glm::vec3) * numTriangles));
	cudaErrorCheck( cudaMalloc((void**) &d_n3, sizeof(glm::vec3) * numTriangles));

	cudaErrorCheck( cudaMalloc((void**) &d_materialIndex, sizeof(int) * numTriangles));
	cudaErrorCheck( cudaMalloc((void**) &d_transformationIndex, sizeof(int) * numTriangles));

	// set to zero
	cudaErrorCheck( cudaMemset(d_v1, 0, sizeof(glm::vec3) * numTriangles));
	cudaErrorCheck( cudaMemset(d_v2, 0, sizeof(glm::vec3) * numTriangles));
	cudaErrorCheck( cudaMemset(d_v3, 0, sizeof(glm::vec3) * numTriangles));

	cudaErrorCheck( cudaMemset(d_n1, 0, sizeof(glm::vec3) * numTriangles));
	cudaErrorCheck( cudaMemset(d_n2, 0, sizeof(glm::vec3) * numTriangles));
	cudaErrorCheck( cudaMemset(d_n3, 0, sizeof(glm::vec3) * numTriangles));

	cudaErrorCheck( cudaMemset(d_materialIndex, 0, sizeof(int) * numTriangles));
	cudaErrorCheck( cudaMemset(d_transformationIndex, 0, sizeof(int) * numTriangles));

	// allocate temp host buffers
	h_v1 = new glm::vec3[numTriangles];
	h_v2 = new glm::vec3[numTriangles];
	h_v3 = new glm::vec3[numTriangles];

	h_n1 = new glm::vec3[numTriangles];
	h_n2 = new glm::vec3[numTriangles];
	h_n3 = new glm::vec3[numTriangles];

	h_materialIndex = new int[numTriangles];
	h_transformationIndex = new int[numTriangles];

	memset( h_v1, 0, sizeof(glm::vec3) * numTriangles);
	memset( h_v2, 0, sizeof(glm::vec3) * numTriangles);
	memset( h_v3, 0, sizeof(glm::vec3) * numTriangles);

	memset( h_n1, 0, sizeof(glm::vec3) * numTriangles);
	memset( h_n2, 0, sizeof(glm::vec3) * numTriangles);
	memset( h_n3, 0, sizeof(glm::vec3) * numTriangles);

	memset( h_materialIndex, 0, sizeof(int) * numTriangles);
	memset( h_transformationIndex, 0, sizeof(int) * numTriangles);

	for( int i = 0; i < numTriangles; i++){

		Surface* h_Surface = h_scene->getSurface(i);
		Triangle* h_Triangle = dynamic_cast<Triangle*>(h_Surface);

		if( !h_Triangle){
			// this surface is not a Triangle.
			// Only scenes containing Triangles will be copied to gpu.
			cudaDeviceReset();
			fprintf(stderr, "GPU scenes can contain only triangles.\n");
			exit(1);
		}

		h_v1[i] = h_Triangle->getV1();
		h_v2[i] = h_Triangle->getV2();
		h_v3[i] = h_Triangle->getV3();

		h_n1[i] = h_Triangle->getN1();
		h_n2[i] = h_Triangle->getN2();
		h_n3[i] = h_Triangle->getN3();

		h_materialIndex[i] = h_Triangle->getMaterialIndex();
		h_transformationIndex[i] = h_Triangle->getTransformationIndex();
	}


	cudaErrorCheck( cudaMemcpy( d_v1, h_v1, sizeof(glm::vec3) * numTriangles, cudaMemcpyHostToDevice));
	cudaErrorCheck( cudaMemcpy( d_v2, h_v2, sizeof(glm::vec3) * numTriangles, cudaMemcpyHostToDevice));
	cudaErrorCheck( cudaMemcpy( d_v3, h_v3, sizeof(glm::vec3) * numTriangles, cudaMemcpyHostToDevice));

	cudaErrorCheck( cudaMemcpy( d_n1, h_n1, sizeof(glm::vec3) * numTriangles, cudaMemcpyHostToDevice));
	cudaErrorCheck( cudaMemcpy( d_n2, h_n2, sizeof(glm::vec3) * numTriangles, cudaMemcpyHostToDevice));
	cudaErrorCheck( cudaMemcpy( d_n3, h_n3, sizeof(glm::vec3) * numTriangles, cudaMemcpyHostToDevice));

	cudaErrorCheck( cudaMemcpy( d_materialIndex, h_materialIndex, sizeof(int) * numTriangles, cudaMemcpyHostToDevice));
	cudaErrorCheck( cudaMemcpy( d_transformationIndex, h_transformationIndex, sizeof(int) * numTriangles, cudaMemcpyHostToDevice));

	cudaTriangle_t* h_cudaTriangles, *d_cudaTriangles;

	h_cudaTriangles = new cudaTriangle_t;
	h_cudaTriangles->v1 = d_v1;
	h_cudaTriangles->v2 = d_v2;
	h_cudaTriangles->v3 = d_v3;

	h_cudaTriangles->n1 = d_n1;
	h_cudaTriangles->n2 = d_n2;
	h_cudaTriangles->n3 = d_n3;

	h_cudaTriangles->materialIndex       = d_materialIndex;
	h_cudaTriangles->transformationIndex = d_transformationIndex;

	cudaErrorCheck( cudaMalloc((void**) &d_cudaTriangles, sizeof(cudaTriangle_t)));
	cudaErrorCheck( cudaMemcpy(d_cudaTriangles, h_cudaTriangles, sizeof(cudaTriangle_t), cudaMemcpyHostToDevice));


	delete h_v1;
	delete h_v2;
	delete h_v3;

	delete h_n1;
	delete h_n2;
	delete h_n3;

	delete h_materialIndex;
	delete h_transformationIndex;

	return d_cudaTriangles; 
}

HOST cudaMaterial_t* copyMaterials(Scene* h_scene){

	glm::vec4* h_diffuse, *d_diffuse;
	glm::vec4* h_specular,  *d_specular;
	float*     h_ambientIntensity, *d_ambientIntensity;
	float*     h_reflectivity, *d_reflectivity;
	int*       h_shininess, *d_shininess;

	int numMaterials = h_scene->getNumMaterials();

	cudaErrorCheck( cudaMalloc((void**) &d_diffuse, sizeof(glm::vec4) * numMaterials));
	cudaErrorCheck( cudaMalloc((void**) &d_specular, sizeof(glm::vec4) * numMaterials));
	cudaErrorCheck( cudaMalloc((void**) &d_ambientIntensity, sizeof(float) * numMaterials));
	cudaErrorCheck( cudaMalloc((void**) &d_reflectivity, sizeof(float) * numMaterials));
	cudaErrorCheck( cudaMalloc((void**) &d_shininess, sizeof(float) * numMaterials));

	cudaErrorCheck( cudaMemset( d_diffuse, 0, sizeof(glm::vec4) * numMaterials));
	cudaErrorCheck( cudaMemset(d_specular, 0, sizeof(glm::vec4) * numMaterials));
	cudaErrorCheck( cudaMemset(d_ambientIntensity, 0, sizeof(float) * numMaterials));
	cudaErrorCheck( cudaMemset(d_reflectivity, 0, sizeof(float) * numMaterials));
	cudaErrorCheck( cudaMemset(d_shininess, 0, sizeof(int) * numMaterials));


	h_diffuse  = new glm::vec4[numMaterials];
	h_specular = new glm::vec4[numMaterials];
	h_ambientIntensity = new float[numMaterials];
	h_reflectivity = new float[numMaterials];
	h_shininess = new int[numMaterials];

	memset( h_diffuse, 0, sizeof(glm::vec4) * numMaterials);
	memset( h_specular, 0, sizeof(glm::vec4) * numMaterials);
	memset( h_ambientIntensity, 0, sizeof(float) * numMaterials);
	memset( h_reflectivity, 0, sizeof(float) * numMaterials);
	memset( h_shininess, 0, sizeof(int) * numMaterials);

	for( int i = 0 ; i < numMaterials; i++){

		const Material& sceneMaterial = h_scene->getMaterialAtIndex(i);

		h_diffuse[i]          = sceneMaterial.getDiffuseColor();
		h_specular[i]         = sceneMaterial.getSpecularColor();
		h_ambientIntensity[i] = sceneMaterial.getAmbientIntensity();
		h_reflectivity[i]     = sceneMaterial.getReflectiveIntensity();
		h_shininess[i]        = sceneMaterial.getShininess();  
	} 

	cudaErrorCheck( cudaMemcpy( d_diffuse, h_diffuse, sizeof(glm::vec4) * numMaterials, cudaMemcpyHostToDevice));
	cudaErrorCheck( cudaMemcpy( d_specular, h_specular, sizeof(glm::vec4) * numMaterials, cudaMemcpyHostToDevice));
	cudaErrorCheck( cudaMemcpy( d_ambientIntensity, h_ambientIntensity, sizeof(float) * numMaterials, cudaMemcpyHostToDevice));
	cudaErrorCheck( cudaMemcpy( d_reflectivity, h_reflectivity, sizeof(float) * numMaterials, cudaMemcpyHostToDevice));
	cudaErrorCheck( cudaMemcpy( d_shininess, h_shininess, sizeof(int) * numMaterials, cudaMemcpyHostToDevice));


	cudaMaterial_t* h_materal, *d_material;

	h_materal = new cudaMaterial_t;

	h_materal->diffuse          = d_diffuse;
	h_materal->specular         = d_specular;
	h_materal->ambientIntensity = d_ambientIntensity;
	h_materal->reflectivity     = d_reflectivity;
	h_materal->shininess        = d_shininess;

	cudaErrorCheck( cudaMalloc((void**) &d_material, sizeof(cudaMaterial_t)));
	cudaErrorCheck( cudaMemcpy(d_material, h_materal, sizeof(cudaMaterial_t), cudaMemcpyHostToDevice));

	delete h_diffuse;
	delete h_specular;
	delete h_ambientIntensity;
	delete h_reflectivity;
	delete h_shininess;
	delete h_materal;

	return d_material;
}

HOST cudaScene_t* createCudaScene(Scene* h_scene){

 	cudaBvhNode_t*         d_bvh;
	cudaLightSource_t*     d_lights;
	cudaTransformations_t* d_transformations;
	cudaTriangle_t*        d_triangles;
	cudaMaterial_t*        d_materials;

	cudaScene_t* h_cudaScene, *d_cudaScene;

	h_cudaScene = new cudaScene_t;
	memset( h_cudaScene, 0, sizeof(cudaScene_t));

	/**
	 * Copy BVH
	 *
	 */
	if( h_scene->isUsingBvh() == false){

		// only scenes containing a bvh tree are supported
		delete h_cudaScene;
		return NULL;
	}
	
	d_bvh = copyBVH(h_scene);

 	/*
 	 * 
 	 * Copy light sources to gpu
 	 *
 	 */
 	d_lights = copyLights(h_scene);

 	/*
 	 * Copy transformations
	 */
	d_transformations = copyTransformations(h_scene);

	/*
	 * triangles
	 */
	d_triangles = copyTriangles(h_scene);

	/*
	 * materials
	 */
	d_materials = copyMaterials(h_scene);

	h_cudaScene->bvh = d_bvh;
	h_cudaScene->lights = d_lights;
	h_cudaScene->transformations = d_transformations;
	h_cudaScene->triangles = d_triangles;
	h_cudaScene->materials = d_materials;

	cudaErrorCheck( cudaMalloc((void**) &d_cudaScene, sizeof(cudaScene_t)));
	cudaErrorCheck( cudaMemcpy( d_cudaScene, h_cudaScene, sizeof(cudaScene_t), cudaMemcpyHostToDevice));

	delete h_cudaScene;

	return d_cudaScene;
 }
