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

HOST void DeviceRenderer::renderToGLPixelBuffer(GLuint pbo)const {

	/* 
	 * 1. bind opengl resources (just the pbo for now)
	 * 2. invoke cuda kernel and render scene
	 * 3. unbing gl resources
	 * 4. return
	 */


	return;
}

HOST void DeviceRenderer::renderToCudaBuffer(void* d_buffer, int buffer_len)const{


}


HOST void  DeviceRenderer::renderToHostBuffer(void* h_buffer, unsigned int buffer_len)const{


}
HOST void DeviceRenderer::setCamera(Camera* d_camera){
	m_Camera = d_camera;
}