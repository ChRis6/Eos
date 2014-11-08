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


#ifndef _DSCENE_H
#define _DSCENE_H

#include "cudaQualifiers.h"
#include "DTriangle.h"
#include "DMaterial.h"
#include "DLightSource.h"
#include "DRayIntersection.h"
#include "BVH.h"

class DeviceSceneHandler;

class DScene{
	friend class DeviceSceneHandler;

private:
	HOST DEVICE DScene():m_Triangles(0),m_NumTriangles(0),m_Lights(0),m_NumLights(0){}

public:
	
	DEVICE void setNumTriangles(int num) { m_NumTriangles = num;}
	DEVICE void setTrianglesArray(DTriangle* array){m_Triangles = array;}

	DEVICE DTriangle* getTriangle(int index)const {return &(m_Triangles[index]);}
	DEVICE int getNumTriangles()const			{return m_NumTriangles;}

	DEVICE DLightSource* getLightSource(int index)	{ return &(m_Lights[index]);}
	DEVICE int getNumLights()	{return m_NumLights;}
	DEVICE bool isUsingBVH()	{ return m_UsingBVH;}
	DEVICE bool findMinDistanceIntersectionLinear(const Ray& ray, DRayIntersection& intersection) const;
	DEVICE bool findMinDistanceIntersectionBVH(const Ray& ray, DRayIntersection& intersection) const;
	DEVICE bool visibilityTest(const Ray& ray) const;

private:
	DEVICE bool intersectRayWithLeaf(const Ray& ray, BvhNode* node, DRayIntersection& intersection, float& distance) const;
private:
	DTriangle* m_Triangles;
	int m_NumTriangles;

	DLightSource* m_Lights;
	int m_NumLights;

	BvhNode* m_BvhBuffer;
	int m_BvhBufferSize;

	bool m_UsingBVH;

};
#endif