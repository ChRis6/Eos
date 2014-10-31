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

#ifndef _DRAYINTERSECTION_H
#define _DRAYINTERSECTION_H

#include "cudaQualifiers.h"
#include "DMaterial.h"
#include <glm/glm.hpp>

class DRayIntersection{
public:
	DEVICE DRayIntersection():m_Point(0.0f),m_Normal(0.0f),m_Material(){}
	DEVICE DRayIntersection(const glm::vec3& point, const glm::vec3& normal, const DMaterial& material):
							m_Point(point),m_Normal(normal),m_Material(material){}

	DEVICE const glm::vec3& getIntersectionPoint()    { return m_Point;}
	DEVICE const glm::vec3& getIntersectionNormal()   { return m_Normal;}
	DEVICE const DMaterial& getIntersectionMaterial() { return m_Material;}

	DEVICE void setIntersectionPoint(const glm::vec3& point)	{ m_Point    = point; }
	DEVICE void setIntersectionNormal(const glm::vec3& normal)  { m_Normal   = normal;}
	DEVICE void setIntersectionMaterial(const DMaterial& mat)   { m_Material = mat;   }
private:
	glm::vec3 m_Point;
	glm::vec3 m_Normal;
	DMaterial m_Material;
};

#endif