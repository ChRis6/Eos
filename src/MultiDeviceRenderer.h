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

#ifndef _MULTI_DEVICE_RENDERER_H
#define _MULTI_DEVICE_RENDERER_H


#include "cudaQualifiers.h"
#include "Camera.h"
#include "cudaStructures.h"
#include "DeviceCameraHandler.h"

#include <GL/glew.h>

class MultiDeviceRenderer{
public:
	HOST MultiDeviceRenderer( Scene* hostScene, Camera* hostCamera, int width, int height);

	HOST void renderSceneToHostBuffer( void* imageBuffer, Camera* hostCamera);
private:

	int m_NumDevices;
	cudaScene_t** m_MultiGpuScene;
	int m_Width;
	int m_Height;

	char** m_DeviceImageBuffer;
	DeviceCameraHandler* m_CameraHandler; 
};



#endif