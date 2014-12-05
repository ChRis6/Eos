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

#include <glm/glm.hpp>
#include "cudaWrapper.h"
#include "cudaPrint.h"
#include <cuda_gl_interop.h>

__global__ void printHelloFromGPUKernel(){
	printf("Hello from GPU.Printing a new vector\n");
	glm::vec3 newVector(0.0f, 0.0f, 0.0f);
	glm::vec3 otherVector(10.0f, 102.0f, 14.6f);
	printf("(%f, %f, %f)\n", newVector.x, newVector.y, newVector.z);

	newVector = newVector + otherVector;
	printf("Addition:\n");
	printf("(%f, %f, %f)\n", newVector.x, newVector.y, newVector.z);


	printf("Creating a Ray on GPU...\n");
	Ray ray(newVector, otherVector);

	glm::vec3 newVector1   = ray.getOrigin();
	glm::vec3 otherVector2 = ray.getDirection();
	printf("Ray Origin:(%f, %f, %f)\n", newVector1.x, newVector1.y, newVector1.z);
	printf("Ray Direction:(%f, %f, %f)\n", otherVector2.x, otherVector2.y, otherVector2.z);

	return;

}

__global__ void printDeviceSceneGPUKernel(DScene* d_scene){
	int numLights;
	int numTriangles;
	int i;

	numLights = d_scene->getNumLights();
	numTriangles = d_scene->getNumTriangles();
	printf("Printing Message From GPU...\n");
	printf("There are %d lights and %d triangles on the CUDA-enabled scene !!!\n\n\n", numLights, numTriangles);

	for( i = 0 ; i < numLights; i++){
		DLightSource* light = d_scene->getLightSource(i);
		glm::vec4 pos = light->getPosition();
		glm::vec4 color = light->getColor();

		printf("LightSource: %d\n", i+1);
		printf("pos: %f %f %f\n", pos.x, pos.y, pos.z);
		printf("color: %f %f %f\n\n",color.x, color.y, color.z);
	}

	for( i = 0 ; i < numTriangles; i++){

		DTriangle* tri = d_scene->getTriangle(i);
		glm::vec3 v1 = tri->m_V1;
		glm::vec3 v2 = tri->m_V2;
		glm::vec3 v3 = tri->m_V3;

		glm::vec3 n1 = tri->m_N1;
		glm::vec3 n2 = tri->m_N2;
		glm::vec3 n3 = tri->m_N3;

		printf("Triangle: %d\n", i+1);
		printf("V1: %f %f %f\n", v1.x, v1.y, v1.z);
		printf("V2: %f %f %f\n", v2.x, v2.y, v2.z);
		printf("V3: %f %f %f\n\n", v3.x, v3.y, v3.z);

		printf("N1: %f %f %f\n", n1.x, n1.y, n1.z);
		printf("N2: %f %f %f\n", n2.x, n2.y, n2.z);
		printf("N3: %f %f %f\n\n", n3.x, n3.y, n3.z);

	}



}

__global__ void __debug_printCudaScene_kernel(cudaScene_t* deviceScene){

    cudaBvhNode_t*         bvh;
    cudaLightSource_t*     lights;
    cudaTransformations_t* transformations;
    cudaTriangle_t*        triangles;
    cudaMaterial_t*        materials;

    bvh = deviceScene->bvh;
    lights = deviceScene->lights;
    transformations = deviceScene->transformations;
    triangles = deviceScene->triangles;

    int numPrintableNodes = 15;
    printf("GPU:Printing first %d BVH nodes\n\n", numPrintableNodes);

    for( int i = 0; i < numPrintableNodes; i++){
        printf("Node: %d\n",i);

        if( bvh->type[i] == BVH_NODE){
        	printf("Type: %d, a.k.a BVH NODE\n", bvh->type[i]);
        	printf("min box vertex: ( %f, %f, %f)\n", bvh->minBoxBounds[i].x, bvh->minBoxBounds[i].y, bvh->minBoxBounds[i].z);
        	printf("max box vertex: ( %f, %f, %f)\n", bvh->maxBoxBounds[i].x, bvh->maxBoxBounds[i].y, bvh->maxBoxBounds[i].z);
        	printf("Num surfaces encapulated: %d\n", bvh->numSurfacesEncapulated[i]);
        	printf("Left child index: %d\n", bvh->leftChildIndex[i]);
        	printf("Right Child index: %d\n\n", bvh->rightChildIndex[i]);
        }
        else if( bvh->type[i] == BVH_LEAF) {
        	// this is a leaf
        	printf("Type: %d, a.k.a BVH Leaf\n", bvh->type[i]);
        	int numTriangles = bvh->numSurfacesEncapulated[i];
        	printf("This leaf has %d triangles.Printing triangles\n\n", numTriangles);

        	for( int j = 0; j < numTriangles; j++){
        		int triangleIndex = bvh->surfacesIndices[ i * SURFACES_PER_LEAF + j];
        		
        		printf("Triangle Index:%d\n", triangleIndex);
        		printf("Triangle Transformation Index: %d\n", triangles->transformationIndex[ triangleIndex]);
        		printf("V1: (%f, %f, %f)\n", triangles->v1[triangleIndex].x, triangles->v1[triangleIndex].y, triangles->v1[triangleIndex].z );
        		printf("V2: (%f, %f, %f)\n", triangles->v2[triangleIndex].x, triangles->v2[triangleIndex].y, triangles->v2[triangleIndex].z );
        		printf("V3: (%f, %f, %f)\n\n", triangles->v3[triangleIndex].x, triangles->v3[triangleIndex].y, triangles->v3[triangleIndex].z );

        		printf("N1: (%f, %f, %f)\n", triangles->n1[triangleIndex].x, triangles->n1[triangleIndex].y, triangles->n1[triangleIndex].z );
        		printf("N2: (%f, %f, %f)\n", triangles->n2[triangleIndex].x, triangles->n2[triangleIndex].y, triangles->n2[triangleIndex].z );
        		printf("N3: (%f, %f, %f)\n\n", triangles->n3[triangleIndex].x, triangles->n3[triangleIndex].y, triangles->n3[triangleIndex].z );


        	}

        }
    }

    printf("End of GPU bvh\n\n");

}

__global__ void __shuffleTest_kernel(){

	int i = threadIdx.x % 2;

	printf("Thread id %d: i = %d ( before)\n", threadIdx.x, i);

	#pragma unroll
	for( int mask = 1 ; mask < 32 ; mask *= 2)
		i += __shfl_xor(i, mask);


	//i = __shfl(i, 0);

	printf("Thread id %d: i = %d (after)\n", threadIdx.x, i);

}



void printHelloGPU(){
	printHelloFromGPUKernel<<<1,1>>>();
	cudaDeviceSynchronize();
}

void printDeviceScene(DScene* d_scene){
	printDeviceSceneGPUKernel<<<1,1>>>(d_scene);
	cudaDeviceSynchronize();
}

void resetDevice(){
	cudaDeviceReset();
}
void setGLDevice(int dev_id){
	cudaGLSetGLDevice(dev_id);
}

void debug_printCudaScene(cudaScene_t* deviceScene){

    __debug_printCudaScene_kernel<<<1,1>>>(deviceScene);
}

void shuffleTest(){

	__shuffleTest_kernel<<<1, 32>>>();
}

void gpuSynch(){
	cudaDeviceSynchronize();
}

void cudaPreferL1Cache(){
	cudaDeviceSetCacheConfig( cudaFuncCachePreferL1);
}