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
#ifndef _PLANE_H
#define _PLANE_H

#include <glm/glm.hpp>
#include "surface.h"
#include "Material.h"

class Plane: public Surface{

public:
	Plane():m_PlanePoint(0.0f), m_PlaneNormal(0.0f, 1.0f, 0.0f){}
	Plane(const glm::vec3& planePoint, const glm::vec3& planeNormal):m_PlanePoint(planePoint), m_PlaneNormal(planeNormal){}
	Plane(const glm::vec3& planePoint, const glm::vec3& planeNormal, const Material& material):
		 m_PlanePoint(planePoint), m_PlaneNormal(planeNormal), m_Material(material){}

	void setPlanePoint(const glm::vec3& point)  {m_PlanePoint  = point;}
	void setPlaneNormal(const glm::vec3& normal){m_PlaneNormal = glm::normalize(normal);}

	const glm::vec3& getPlanePoint()  const {return m_PlanePoint;}
	const glm::vec3& getPlaneNormal() const {return m_PlaneNormal;}

	virtual bool hit(const Ray& ray, RayIntersection& intersection, float& distance);
	virtual Box getLocalBoundingBox();
	virtual const glm::mat4& transformation()               {return m_Transformation;}
	virtual void setTransformation(glm::mat4 transformation){m_Transformation = transformation;}
	virtual const Material& getMaterial()             {return m_Material;}
	virtual void setMaterial(const Material& material){m_Material = material;}   

private:
	glm::vec3 m_PlanePoint;
	glm::vec3 m_PlaneNormal;

	glm::mat4 m_Transformation;
	Material m_Material;
};

#endif