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

HOST void DeviceRenderer::renderSceneToGLPixelBuffer(DScene* h_Dscene, DRayIntersection* intersectionBuffer, int bufferSize, GLuint pbo) const{
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
	shadeIntersectionsToBuffer((char*)d_pbo, d_pboSize, d_tracer, d_camera, h_Dscene->m_Lights, h_Dscene->m_NumLights, intersectionBuffer, bufferSize, 
										width, height, blockdim, threadPerBlock);

	cudaErrorCheck( cudaDeviceSynchronize());
	cudaErrorCheck( cudaGraphicsUnmapResources( 1, &cudaResourcePBO, 0));
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


HOST void DeviceRenderer::renderSceneToHostBuffer(DScene* h_Dscene, DRayIntersection* intersectionBuffer, int bufferSize, void* imageBuffer, int imageBufferSize){
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

	cudaErrorCheck( cudaMemset(intersectionBuffer, 0, bufferSize * sizeof(DRayIntersection)));
	calculateIntersections(d_camera, intersectionBuffer, bufferSize, h_Dscene->m_Triangles, h_Dscene->m_NumTriangles,
						   h_Dscene->m_BvhBuffer, width, height, blockdim, threadPerBlock);

	d_tracer = this->getDeviceRayTracer();

	cudaErrorCheck( cudaDeviceSynchronize());

		// shadeIntersections
	shadeIntersectionsToBuffer((char*)d_image, imageBufferSize, d_tracer, d_camera, h_Dscene->m_Lights, h_Dscene->m_NumLights, intersectionBuffer, bufferSize, 
										width, height, blockdim, threadPerBlock);

	cudaErrorCheck( cudaDeviceSynchronize());

	cudaErrorCheck( cudaMemcpy( imageBuffer, d_image, imageBufferSize, cudaMemcpyDeviceToHost));
	cudaErrorCheck( cudaFree(d_image));

}
HOST void DeviceRenderer::setCamera(Camera* d_camera){
	m_Camera = d_camera;
}

