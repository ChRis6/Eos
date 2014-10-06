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

#ifndef _RAYINTERSECTION_H
#define _RAYINTERSECTION_H

#include <glm/glm.hpp>
#include "Material.h"

class RayIntersection{

public:
	RayIntersection(): m_Point(0.0), m_Normal(0.0){}
	RayIntersection(glm::vec3 point, glm::vec3 normal): m_Point(point), m_Normal(normal)
	{}
	RayIntersection(const glm::vec3& point, const glm::vec3& normal, const Material& material ): m_Point(point), m_Normal(normal), m_Material(material)
	{}

	void setPoint(glm::vec3 point);
	void setNormal(glm::vec3 normal);
	void setMaterial(Material& material);

	glm::vec3 getPoint() const;
	glm::vec3 getNormal() const;
	Material& getMaterial();

private:
	glm::vec3 m_Point;
	glm::vec3 m_Normal;

	Material m_Material;
};


#endif