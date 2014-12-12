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

#ifndef _DEVICE_RENDERER_H
#define _DEVICE_RENDERER_H

#include "cudaQualifiers.h"
#include "Camera.h"
#include "cudaStructures.h"


#include <GL/glew.h>

class DeviceRenderer{

public:
	
	
	HOST DeviceRenderer( Camera* camera, int width, int height):m_Camera(camera),m_Width(width),m_Height(height){}

	
	HOST void renderCudaSceneToHostBufferMegaKernel( cudaScene_t* deviceScene, void* imageBuffer);
	HOST void renderCudaSceneToHostBufferWarpShuffleMegaKernel( cudaScene_t* deviceScene, void* imageBuffer);

	HOST void setCamera(Camera* d_camera);	// d_camera must point to GPU memory
	HOST int getWidth()const	{ return m_Width;}
	HOST int getHeight()const   {return m_Height;}
	HOST Camera* getDeviceCamera()const 	{return m_Camera;}

private:
	HOST DeviceRenderer():m_Camera(NULL),m_Width(0),m_Height(0){}
private:
	Camera*     m_Camera;
	int m_Width;
	int m_Height;

};

#endif