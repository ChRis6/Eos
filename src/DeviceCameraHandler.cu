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

#include "DeviceCameraHandler.h"

HOST DeviceCameraHandler::DeviceCameraHandler(Camera* h_Camera){
	Camera* d_camera = NULL;

	cudaErrorCheck( cudaMalloc((void**) &d_camera, sizeof(Camera)));
	cudaErrorCheck( cudaMemcpy(d_camera, h_Camera, sizeof(Camera), cudaMemcpyHostToDevice));
	m_DeviceCamera = d_camera;
}

HOST void DeviceCameraHandler::setDeviceCamera(Camera* h_Camera){
	if( this->getDeviceCamera() == NULL ){
		Camera* d_camera = NULL;
		cudaErrorCheck( cudaMalloc((void**) &d_camera, sizeof(Camera)));
		m_DeviceCamera = d_camera;
	}

	cudaErrorCheck( cudaMemcpy(m_DeviceCamera, h_Camera, sizeof(Camera), cudaMemcpyHostToDevice));
}

HOST Camera* DeviceCameraHandler::getDeviceCamera(){
	return m_DeviceCamera;
}

HOST void DeviceCameraHandler::updateDeviceCamera(Camera* h_Camera){
	cudaErrorCheck( cudaMemcpy(m_DeviceCamera, h_Camera, sizeof(Camera), cudaMemcpyHostToDevice));
}

HOST void DeviceCameraHandler::freeDeviceCamera(){
	cudaErrorCheck( cudaFree(m_DeviceCamera));
}