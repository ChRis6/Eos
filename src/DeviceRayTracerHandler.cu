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

#include "DeviceRayTracerHandler.h"

HOST DRayTracer* DeviceRayTracerHandler::createDeviceTracer(RayTracer* h_tracer){
	
	DRayTracer* d_rayTracer = NULL;
	DRayTracer* h_DRayTracer;
	
	h_DRayTracer = new DRayTracer;
	h_DRayTracer->setAASamples( h_tracer->getAASamples());
	h_DRayTracer->setTracedDepth( h_tracer->getTracedDepth());

	// allocate device memory and copy
	cudaErrorCheck( cudaMalloc((void**)&d_rayTracer, sizeof(DRayTracer)));
	cudaErrorCheck( cudaMemcpy(d_rayTracer, h_DRayTracer, sizeof(DRayTracer), cudaMemcpyHostToDevice));

	delete h_DRayTracer;
	return d_rayTracer;
}

HOST void  DeviceRayTracerHandler::freeDeviceTracer(){
	cudaErrorCheck( cudaFree(m_DeviceRayTracer));
}

HOST void DeviceRayTracerHandler::setHostTracer(RayTracer* h_tracer){
	
	m_HostRayTracer = h_tracer;
	if( m_DeviceRayTracer != NULL){

		// free and make a new one
		this->freeDeviceTracer();
		m_DeviceRayTracer = this->createDeviceTracer(h_tracer);
	}
	else{
		// make a new
		m_DeviceRayTracer = this->createDeviceTracer(h_tracer);
	}
}