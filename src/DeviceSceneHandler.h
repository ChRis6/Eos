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

#ifndef _DEVICE_SCENE_HANDLER_H
#define _DEVICE_SCENE_HANDLER_H

#include "cudaQualifiers.h"
#include "Scene.h"
#include "DScene.h"
#include "Camera.h"

class DeviceSceneHandler{

public:
	HOST DeviceSceneHandler():m_HostScene(0){}
	HOST DeviceSceneHandler(Scene* h_scene){ m_HostScene = h_scene;}

	HOST DScene* createDeviceScene();
private:
	Scene* getScene(){ return m_HostScene;}
private:
	Scene* m_HostScene;
};
#endif 