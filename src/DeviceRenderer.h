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
#include "DScene.h"
#include "Camera.h"
#include "DRayTracer.h"


#include <GL/glew.h>

class DeviceRenderer{

public:
	HOST DeviceRenderer(DScene* scene, DRayTracer* tracer, Camera* camera):m_Dscene(scene),m_DrayTracer(tracer),m_Camera(camera){}

	HOST void renderToGLPixelBuffer(GLuint pbo)const;
	HOST void renderToHostBuffer(void* h_buffer, unsigned int buffer_len)const;	// call to render only once.

	HOST void setCamera(Camera* d_camera);	// d_camera must point to GPU memory

private:
	HOST DeviceRenderer():m_Dscene(NULL),m_DrayTracer(NULL),m_Camera(NULL){}
	HOST void renderToCudaBuffer(void* d_buffer, int buffer_len)const;
private:
	/* 
	 * all the pointers MUST point to GPU memory.Each one has to be allocated
	 * by the appropriate device handler.
	 * DO NOT DEREFERENCE ON HOST
	 */
	DScene*     m_Dscene;		
	DRayTracer* m_DrayTracer;
	Camera*     m_Camera;
};

#endif