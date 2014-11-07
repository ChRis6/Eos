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

#ifndef _DRAYTRACER_H
#define _DRAYTRACER_H

#include "cudaQualifiers.h"
#include "Camera.h"
#include "DScene.h"
#include "Ray.h"
#include "DRayIntersection.h"

class DRayTracer{

public:
	HOST DEVICE DRayTracer():m_AASamples(1),m_TraceDepth(4){}

	HOST DEVICE int getTracedDepth()const { return m_TraceDepth;}
	HOST DEVICE int getAASamples()const   { return m_AASamples;}

	HOST DEVICE void setTracedDepth(int depth)	{ m_TraceDepth = depth; }
	HOST DEVICE void setAASamples(int samples)  { m_AASamples  = samples;}

	DEVICE glm::vec4 rayTrace(DScene* scene, Camera* camera, const Ray& ray,  int depth);
private:
	
	DEVICE glm::vec4 shadeIntersection(DScene* scene, const Ray& ray, Camera* camera, DRayIntersection& intersection, int depth);
	DEVICE glm::vec4 calcPhong(Camera* camera, DLightSource* lightSource, DRayIntersection& intersection);
	DEVICE glm::vec4 findDiffuseColor(DLightSource* lightSource, const glm::vec4& intersectionToLight, DRayIntersection& intersection);

private:
	int m_AASamples;
	int m_TraceDepth;
};

#endif