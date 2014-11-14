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

#ifndef _BOX_H
#define _BOX_H

#include <glm/glm.hpp>
#include "cudaQualifiers.h"
#include "Ray.h"

/*
 * Axis aligned Bounding Box
 */
class Box{

public:
	Box(): m_MinVertex(0.0f), m_MaxVertex(0.0f){ m_Bounds[0] = m_MinVertex; m_Bounds[1] = m_MaxVertex;}
	Box(const glm::vec3& minVertex, const glm::vec3& maxVertex): m_MinVertex(minVertex), m_MaxVertex(maxVertex){ m_Bounds[0] = m_MinVertex; m_Bounds[1] = m_MaxVertex; }
	Box(const glm::vec3& vertex): m_MinVertex(vertex), m_MaxVertex(vertex){ m_Bounds[0] = m_MinVertex; m_Bounds[1] = m_MaxVertex;}
	Box(const Box& other) {m_MinVertex = other.getMinVertex(); m_MaxVertex = other.getMaxVertex();}

	void setMinVertex(const glm::vec3& min);
	void setMaxVertex(const glm::vec3& max);

	HOST DEVICE const glm::vec3& getMinVertex() const;
	HOST DEVICE const glm::vec3& getMaxVertex() const;
	int getBiggestDimension() const;
	glm::vec3 getBoxCentroid();

	float computeVolume();
	float computeSurfaceArea();
	void expandToIncludeBox(const Box& newBox);
	void expandToIncludeVertex(const glm::vec3& vertex);
	HOST DEVICE bool intersectWithRay(const Ray& ray, float& distance) const;
	HOST DEVICE bool intersectWithRayNew(const Ray& ray);
	HOST DEVICE bool intersectWithRayOptimized(const Ray& ray, float t0, float t1) const;
	void transformBoundingBox(const glm::mat4& transformation);
	bool isPointInBox(glm::vec3& point);

private:
	glm::vec3 m_Bounds[2];

	glm::vec3 m_MinVertex;
	glm::vec3 m_MaxVertex;
};

#endif