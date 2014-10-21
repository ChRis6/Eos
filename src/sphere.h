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

#ifndef _SPHERE_H
#define _SPHERE_H

#include <glm/glm.hpp>
#include "surface.h"
#include "Material.h"
#include "Texture.h"

class Sphere: public Surface{

public:
	Sphere(): m_Origin(0.0), m_RadiusSquared(1), m_LocalToWorldTransformation(1.0f){}
	Sphere(glm::vec3 origin, float radiusSquared): m_Origin(origin), m_RadiusSquared(radiusSquared){ m_LocalToWorldTransformation = glm::mat4(1.0f);}
	Sphere(glm::vec3 origin, float radiusSquared, Material material): m_Origin(origin), m_RadiusSquared(radiusSquared), m_Material(material)
	      { m_LocalToWorldTransformation = glm::mat4(1.0f);}
	

	virtual Box getLocalBoundingBox();
	virtual glm::vec3 getCentroid();
	virtual const glm::mat4& transformation();
	virtual bool hit(const Ray& ray, RayIntersection& intersection, float& distance);
	virtual void setTransformation(glm::mat4& transformation); 
	virtual const glm::mat4& getInverseTransformation();
	virtual const glm::mat4& getInverseTransposeTransformation();
	virtual Material& getMaterial(){return m_Material;}
	virtual void setMaterial(Material& material){m_Material = material;}
	virtual bool isPointInside(glm::vec3& point);

private:
	bool quadSolve(float a, float b, float c, float& t);


private:
	glm::vec3 m_Origin;
	float     m_RadiusSquared;

	glm::mat4 m_LocalToWorldTransformation;
	glm::mat4 m_Inverse;
	glm::mat4 m_InverseTranspose;
	Material  m_Material;
	
};


#endif
