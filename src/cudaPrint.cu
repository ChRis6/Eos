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

	for( i = 0 ; i < 5; i++){
		DTriangle* d_triangle = d_scene->getTriangle(i);
		glm::vec3 v1 = d_triangle->getV1();
		printf("Triangle: %d\n",i );
		printf("V1: %f, %f, %f\n\n", v1.x, v1.y, v1.z);
	}

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