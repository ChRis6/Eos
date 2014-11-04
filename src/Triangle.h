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

#include <glm/glm.hpp>
#include "surface.h"

class DeviceSceneHandler;

class Triangle: public Surface{
	friend class DeviceSceneHandler;
public:
	Triangle(const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3, const glm::vec3& n1, const glm::vec3& n2, const glm::vec3& n3):
		m_V1(v1),m_V2(v2),m_V3(v3),m_N1(n1),m_N2(n2),m_N3(n3){
			m_Centroid = ( v1 + v2 + v3) / 3.0f;
			m_BoundingBox.expandToIncludeVertex(v1);
			m_BoundingBox.expandToIncludeVertex(v2);
			m_BoundingBox.expandToIncludeVertex(v3);
		}

public:
	virtual bool hit(const Ray& ray, RayIntersection& intersection, float& distance);
	virtual const Box& getLocalBoundingBox();
	virtual glm::vec3 getCentroid();
	virtual const glm::mat4& transformation();
	virtual void setTransformation(glm::mat4& transformation);
	virtual const glm::mat4& getInverseTransformation();
	virtual const glm::mat4& getInverseTransposeTransformation();
	virtual Material& getMaterial();
	virtual void setMaterial(Material& material);
	virtual bool isPointInside(glm::vec3& point); 

	void setInverseTransformation(glm::mat4& inverse);
	void setInverseTransposeTransformation(glm::mat4& inverseTranspose);

private:
	bool RayTriangleIntersection(const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3, const Ray& ray, glm::vec3& barycetricCoords);

private:
	// triangle vertices
	glm::vec3 m_V1;
	glm::vec3 m_V2;
	glm::vec3 m_V3;

	// triangle normal/vertex
	glm::vec3 m_N1;
	glm::vec3 m_N2;
	glm::vec3 m_N3;

	Box m_BoundingBox;

	// centroid
	glm::vec3 m_Centroid;

	Material* m_Material;
	glm::mat4* m_LocalToWorldTransformation;
	glm::mat4* m_Inverse;
  	glm::mat4* m_InverseTranspose;
};