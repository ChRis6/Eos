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

#include "DeviceRenderer.h"
#include "kernelWrapper.h"
#include "getTime.h"
#include <cuda_gl_interop.h>
#include <iostream>

//#define PRINT_GPU_RENDER_TIME

HOST void DeviceRenderer::renderToGLPixelBuffer(GLuint pbo)const {

	/* 
	 * 1. bind opengl resources (just the pbo for now)
	 * 2. invoke cuda kernel and rendering 
	 * 3. unbing gl resources
	 * 4. return
	 */

	void* d_pbo = NULL;
	size_t d_pboSize;
	cudaGraphicsResource_t cudaResourcePBO;
	
	// bind
	cudaErrorCheck( cudaGraphicsGLRegisterBuffer(&cudaResourcePBO, pbo, cudaGraphicsRegisterFlagsWriteDiscard));	// only write
	cudaErrorCheck( cudaGraphicsMapResources ( 1, &cudaResourcePBO, 0));
	// get pointer
	cudaErrorCheck( cudaGraphicsResourceGetMappedPointer(&d_pbo, &d_pboSize, cudaResourcePBO)); 

	// render
	this->renderToCudaBuffer(d_pbo, d_pboSize);
	// wait for kernel
	cudaErrorCheck( cudaDeviceSynchronize());
	// unbind
	cudaErrorCheck( cudaGraphicsUnmapResources( 1, &cudaResourcePBO, 0));
	return;
}

HOST void DeviceRenderer::renderSceneToGLPixelBuffer(DScene* h_Dscene, DRayIntersection* intersectionBuffer, int bufferSize, GLuint pbo) const{
	/*
	Camera* d_camera;
	DRayTracer* d_tracer;
	void* d_pbo = NULL;
	size_t d_pboSize;
	cudaGraphicsResource_t cudaResourcePBO;
	
	int blockdim[2];
	int threadPerBlock[2];


	// bind
	cudaErrorCheck( cudaGraphicsGLRegisterBuffer(&cudaResourcePBO, pbo, cudaGraphicsRegisterFlagsWriteDiscard));	// only write
	cudaErrorCheck( cudaGraphicsMapResources ( 1, &cudaResourcePBO, 0));
	// get pointer
	cudaErrorCheck( cudaGraphicsResourceGetMappedPointer(&d_pbo, &d_pboSize, cudaResourcePBO));

	int width = this->getWidth();
	int height = this->getHeight();

	// (16,16) threads per block
	threadPerBlock[0] = 16;
	threadPerBlock[1] = 16;

	blockdim[0] = width / threadPerBlock[0];
	blockdim[1] = height / threadPerBlock[1];

	d_camera = this->getDeviceCamera();
	d_tracer = this->getDeviceRayTracer();

	cudaErrorCheck( cudaMemset(intersectionBuffer, 0, bufferSize * sizeof(DRayIntersection)));
	// invoke intersectio kernel.Traverse the BVH first
	calculateIntersections(d_camera, intersectionBuffer, bufferSize, h_Dscene->m_Triangles, h_Dscene->m_NumTriangles,
						   h_Dscene->m_BvhBuffer, width, height, blockdim, threadPerBlock);
	// wait for kernel
	cudaErrorCheck( cudaDeviceSynchronize());

	// shadeIntersections
	shadeIntersectionsToBuffer((uchar4*)d_pbo, d_pboSize, d_tracer, d_camera, h_Dscene->m_Lights, h_Dscene->m_NumLights, intersectionBuffer, bufferSize, 
										h_Dscene->m_Materials, h_Dscene->m_NumMaterials, width, height, blockdim, threadPerBlock);

	cudaErrorCheck( cudaDeviceSynchronize());
	cudaErrorCheck( cudaGraphicsUnmapResources( 1, &cudaResourcePBO, 0));
	*/
}

HOST void DeviceRenderer::renderToCudaBuffer(void* d_buffer, unsigned int buffer_len)const{
	Camera* d_camera;
	DScene* d_scene;
	DRayTracer* d_tracer;

	int blockdim[2];
	int threadPerBlock[2];

	int width = this->getWidth();
	int height = this->getHeight();

	// (16,16) threads per block
	threadPerBlock[0] = 16;
	threadPerBlock[1] = 16;

	blockdim[0] = width / threadPerBlock[0];
	blockdim[1] = height / threadPerBlock[1];

	d_camera = this->getDeviceCamera();
	d_scene  = this->getDeviceScene();
	d_tracer = this->getDeviceRayTracer();

	renderToBuffer((char*)d_buffer, buffer_len, d_camera, d_scene, d_tracer, blockdim, threadPerBlock, width, height);
}


HOST void  DeviceRenderer::renderToHostBuffer(void* h_buffer, unsigned int buffer_len)const{

	void* d_buffer;
	cudaErrorCheck( cudaMalloc(&d_buffer, buffer_len));
	cudaErrorCheck( cudaMemset(d_buffer, 0, buffer_len));

	// call kernel
	this->renderToCudaBuffer(d_buffer, buffer_len);

	// wait for computation
	cudaErrorCheck( cudaDeviceSynchronize());

	// copy result to host buffer
	cudaErrorCheck( cudaMemcpy( h_buffer, d_buffer, buffer_len, cudaMemcpyDeviceToHost));
	cudaErrorCheck( cudaFree(d_buffer));
}


HOST void DeviceRenderer::renderSceneToHostBuffer(DScene* h_Dscene, cudaIntersection_t* intersectionBuffer, int bufferSize, void* imageBuffer, int imageBufferSize){
	Camera* d_camera;
	DRayTracer* d_tracer;
	void* d_image;
	cudaErrorCheck( cudaMalloc(&d_image, imageBufferSize));
	cudaErrorCheck( cudaMemset(d_image, 0, imageBufferSize));

	int blockdim[2];
	int threadPerBlock[2];

	int width = this->getWidth();
	int height = this->getHeight();

	threadPerBlock[0] = 16;
	threadPerBlock[1] = 16;

	blockdim[0] = width / threadPerBlock[0];
	blockdim[1] = height / threadPerBlock[1];

	d_camera = this->getDeviceCamera();

	// reset intersections

	//cudaErrorCheck( cudaMemset(intersectionBuffer, 0, bufferSize * sizeof(DRayIntersection)));
	cudaErrorCheck( cudaMemset(m_CudaHostIntersection->points, 0, sizeof(glm::vec4) * width * height));
	cudaErrorCheck( cudaMemset(m_CudaHostIntersection->normals, 0, sizeof(glm::vec4) * width * height));
	cudaErrorCheck( cudaMemset(m_CudaHostIntersection->materialsIndices, 0, sizeof(int) * width * height));

	calculateIntersections(d_camera, m_CudaDeviceIntersection, bufferSize, h_Dscene->m_Triangles, h_Dscene->m_NumTriangles,
						   h_Dscene->m_BvhBuffer, width, height, blockdim, threadPerBlock);

	d_tracer = this->getDeviceRayTracer();

	cudaErrorCheck( cudaDeviceSynchronize());

		// shadeIntersections
	shadeIntersectionsToBuffer((uchar4*)d_image, imageBufferSize, d_tracer, d_camera, h_Dscene->m_Lights, h_Dscene->m_NumLights, m_CudaDeviceIntersection, bufferSize, 
										h_Dscene->m_Materials, h_Dscene->m_NumMaterials, width, height, blockdim, threadPerBlock);

	cudaErrorCheck( cudaDeviceSynchronize());

	cudaErrorCheck( cudaMemcpy( imageBuffer, d_image, imageBufferSize, cudaMemcpyDeviceToHost));
	cudaErrorCheck( cudaFree(d_image));

}
HOST void DeviceRenderer::setCamera(Camera* d_camera){
	m_Camera = d_camera;
}

HOST void DeviceRenderer::allocateCudaIntersectionBuffer(){
	// allocate cudaIntersection		
	glm::vec4* d_pointsVec4;
	glm::vec4* d_normalsVec4;
	int* d_materialIndices;
	int width;
	int height;

	width = this->getWidth();
	height = this->getHeight();

	cudaErrorCheck( cudaMalloc((void**) &d_pointsVec4, sizeof(glm::vec4) * width * height));
	cudaErrorCheck( cudaMalloc((void**) &d_normalsVec4, sizeof(glm::vec4) * width * height));
	cudaErrorCheck( cudaMalloc((void**) &d_materialIndices, sizeof(int) * width * height));

	//cudaErrorCheck( cudaMemset(d_pointsVec4, 0, sizeof(glm::vec4) * width * height));
	//cudaErrorCheck( cudaMemset(d_normalsVec4, 0, sizeof(glm::vec4) * width * height));
	//cudaErrorCheck( cudaMemset(d_materialIndices, 0, sizeof(int) * width * height));

	
	cudaIntersection_t* deviceIntersection;

	m_CudaHostIntersection = new cudaIntersection_t;
	m_CudaHostIntersection->points  = d_pointsVec4 ;
	m_CudaHostIntersection->normals = d_normalsVec4;
	m_CudaHostIntersection->materialsIndices = d_materialIndices;

	cudaErrorCheck( cudaMalloc((void**) &deviceIntersection, sizeof(cudaIntersection_t)));
	cudaErrorCheck( cudaMemcpy( deviceIntersection, m_CudaHostIntersection, sizeof(cudaIntersection_t), cudaMemcpyHostToDevice));

	m_CudaDeviceIntersection = deviceIntersection;
}

HOST void DeviceRenderer::renderCudaSceneToHostBuffer(cudaScene_t* deviceScene, void* imageBuffer){

	void* d_image;
	Camera* d_camera;
	int width;
	int height;
	int blockdim[2];
	int threadPerBlock[2];

	width  = this->getWidth();
	height = this->getHeight();

	// 
	cudaErrorCheck( cudaMalloc((void**) &d_image, sizeof(uchar4) * width * height ));
	cudaErrorCheck( cudaMemset( d_image, 0, sizeof(uchar4) * width * height));

	d_camera = this->getDeviceCamera();

	// make sure its zero
	cudaErrorCheck( cudaMemset(m_CudaHostIntersection->points, 0, sizeof(glm::vec4) * width * height));
	cudaErrorCheck( cudaMemset(m_CudaHostIntersection->normals, 0, sizeof(glm::vec4) * width * height));
	cudaErrorCheck( cudaMemset(m_CudaHostIntersection->materialsIndices, 0, sizeof(int) * width * height));


	threadPerBlock[0] = 16;
	threadPerBlock[1] = 16;

	blockdim[0] = width / threadPerBlock[0];
	blockdim[1] = height / threadPerBlock[1];

#ifdef PRINT_GPU_RENDER_TIME
	std::cout << "Rendering Once to Image file (GPU)..." << std::endl;
    double start,end,diff;
    double msecs;
    int hours,minutes,seconds;

    start = getRealTime();
#endif

	// calculate intersections.Traverse Tree on gpu
	calculateCudaSceneIntersections( deviceScene, d_camera, m_CudaDeviceIntersection, width, height, blockdim, threadPerBlock);
	// wait for kernel
	cudaErrorCheck( cudaDeviceSynchronize());

	// shade intersections
	shadeCudaSceneIntersections( deviceScene, d_camera, m_CudaDeviceIntersection, width, height, (uchar4*) d_image, blockdim, threadPerBlock);
	// wait.(no need cause there is memcpy next)
	cudaErrorCheck( cudaDeviceSynchronize());

#ifdef PRINT_GPU_RENDER_TIME
	end = getRealTime();

    diff = end - start;
    minutes = diff / 60;
    seconds = ((int)diff) % 60;
    msecs = diff * 1000;

    std::cout << "Rendering Once: Completed in " << minutes << "minutes, " << seconds << "sec " << std::endl;
    std::cout << "That's About " << msecs << "ms" << std::endl;
#endif

	cudaErrorCheck( cudaMemcpy( imageBuffer, d_image, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));

	// free temp gpu buffer
	cudaErrorCheck( cudaFree(d_image));
}

HOST void DeviceRenderer::renderCudaSceneToGLPixelBuffer( cudaScene_t* deviceScene, GLuint pbo){

	void* d_pbo = NULL;
	size_t d_pboSize;
	cudaGraphicsResource_t cudaResourcePBO;

	Camera* d_camera;
	int width;
	int height;
	int blockdim[2];
	int threadPerBlock[2];

	width  = this->getWidth();
	height = this->getHeight();
	d_camera = this->getDeviceCamera();

	// bind
	cudaErrorCheck( cudaGraphicsGLRegisterBuffer(&cudaResourcePBO, pbo, cudaGraphicsRegisterFlagsWriteDiscard));	// only write
	cudaErrorCheck( cudaGraphicsMapResources ( 1, &cudaResourcePBO, 0));
	// get pointer
	cudaErrorCheck( cudaGraphicsResourceGetMappedPointer(&d_pbo, &d_pboSize, cudaResourcePBO));


	// make sure its zero
	cudaErrorCheck( cudaMemset(m_CudaHostIntersection->points, 0, sizeof(glm::vec4) * width * height));
	cudaErrorCheck( cudaMemset(m_CudaHostIntersection->normals, 0, sizeof(glm::vec4) * width * height));
	cudaErrorCheck( cudaMemset(m_CudaHostIntersection->materialsIndices, 0, sizeof(int) * width * height));

	threadPerBlock[0] = 16;
	threadPerBlock[1] = 16;

	blockdim[0] = width / threadPerBlock[0];
	blockdim[1] = height / threadPerBlock[1];

	// calculate intersections.Traverse Tree on gpu
	calculateCudaSceneIntersections( deviceScene, d_camera, m_CudaDeviceIntersection, width, height, blockdim, threadPerBlock);
	cudaErrorCheck( cudaDeviceSynchronize());

	// shade intersections
	shadeCudaSceneIntersections( deviceScene, d_camera, m_CudaDeviceIntersection, width, height, (uchar4*) d_pbo, blockdim, threadPerBlock);
	cudaErrorCheck( cudaDeviceSynchronize());

	// unbind pbo for opengl
	cudaErrorCheck( cudaGraphicsUnmapResources( 1, &cudaResourcePBO, 0));
	cudaErrorCheck( cudaGraphicsUnregisterResource(cudaResourcePBO));
}


HOST void DeviceRenderer::renderCudaSceneToHostBufferMegaKernel( cudaScene_t* deviceScene, void* imageBuffer){


	cudaEvent_t start, stop;


	void* d_image;
	Camera* d_camera;
	int width;
	int height;
	int blockdim[2];
	int threadPerBlock[2];

	width  = this->getWidth();
	height = this->getHeight();

	// 
	cudaErrorCheck( cudaMalloc((void**) &d_image, sizeof(uchar4) * width * height ));
	cudaErrorCheck( cudaMemset( d_image, 0, sizeof(uchar4) * width * height));

	d_camera = this->getDeviceCamera();


	threadPerBlock[0] = 16;
	threadPerBlock[1] = 16;

	blockdim[0] = width / threadPerBlock[0];
	blockdim[1] = height / threadPerBlock[1];

	cudaErrorCheck( cudaEventCreate(&start));
	cudaErrorCheck( cudaEventCreate(&stop));

	cudaEventRecord(start);
	rayTrace_MegaKernel( deviceScene, d_camera, width, height, (uchar4*) d_image, blockdim, threadPerBlock);
	cudaDeviceSynchronize();
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	

	cudaErrorCheck( cudaMemcpy( imageBuffer, d_image, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));
	

	float milliseconds = 0.0f;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Cuda Mega Kernel finished in %f milliseconds\n", milliseconds );

	// free temp gpu buffer
	cudaErrorCheck( cudaFree(d_image));
}

HOST void DeviceRenderer::renderCudaSceneToHostBufferWarpShuffleMegaKernel( cudaScene_t* deviceScene, void* imageBuffer){

	cudaEvent_t start, stop;


	void* d_image;
	Camera* d_camera;
	int width;
	int height;
	int blockdim[2];
	int threadPerBlock;

	width  = this->getWidth();
	height = this->getHeight();

	// 
	cudaErrorCheck( cudaMalloc((void**) &d_image, sizeof(uchar4) * width * height ));
	cudaErrorCheck( cudaMemset( d_image, 0, sizeof(uchar4) * width * height));

	d_camera = this->getDeviceCamera();


	threadPerBlock = 256;

	blockdim[0] = width / threadPerBlock;
	blockdim[1] = height;

	cudaErrorCheck( cudaEventCreate(&start));
	cudaErrorCheck( cudaEventCreate(&stop));

	cudaErrorCheck( cudaEventRecord(start));
	rayTrace_WarpShuffle_MegaKernel( deviceScene, d_camera, width, height, (uchar4*) d_image, blockdim, threadPerBlock);
	cudaErrorCheck( cudaDeviceSynchronize());
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	

	cudaErrorCheck( cudaMemcpy( imageBuffer, d_image, sizeof(uchar4) * width * height, cudaMemcpyDeviceToHost));
	

	float milliseconds = 0.0f;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("Cuda Mega Kernel finished in %f milliseconds\n", milliseconds );

	// free temp gpu buffer
	cudaErrorCheck( cudaFree(d_image));


}