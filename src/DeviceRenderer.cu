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

/*
HOST void DeviceRenderer::renderToGLPixelBuffer(GLuint pbo)const {


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
*/


HOST void DeviceRenderer::setCamera(Camera* d_camera){
	m_Camera = d_camera;
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