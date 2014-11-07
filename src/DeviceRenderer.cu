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
#include <cuda_gl_interop.h>

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

	renderToBuffer(d_buffer, buffer_len, d_camera, d_scene, d_tracer, blockdim, threadPerBlock, width, height);
}


HOST void  DeviceRenderer::renderToHostBuffer(void* h_buffer, unsigned int buffer_len)const{

	void* d_buffer;
	cudaErrorCheck( cudaMalloc(&d_buffer, buffer_len));
	cudaErrorCheck( cudaMemset(d_buffer, 0, buffer_len));

	// call kernel
	this->renderToCudaBuffer(d_buffer, buffer_len);

	// wait for computation
	cudaDeviceSynchronize();

	// copy result to host buffer
	cudaErrorCheck( cudaMemcpy( h_buffer, d_buffer, buffer_len, cudaMemcpyDeviceToHost));
}
HOST void DeviceRenderer::setCamera(Camera* d_camera){
	m_Camera = d_camera;
}

