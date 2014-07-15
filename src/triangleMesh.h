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

#ifndef _TRIANGLEMESH_H
#define _TRIANGLEMESH_H

#include <glm/glm.hpp>
#include "surface.h"
#include "Material.h"

class TriangleMesh: public Surface{

public:
	TriangleMesh();
	~TriangleMesh();

	// implement interface methods
	virtual bool hit(const Ray& ray, RayIntersection& intersection, float& distance);
	virtual Box getLocalBoundingBox();
	virtual glm::vec3 getCentroid();
	virtual const glm::mat4& transformation();
	virtual void setTransformation(glm::mat4 transformation);
	virtual const Material& getMaterial();
	virtual void setMaterial(const Material& material); 
	   
	bool loadFromFile(const char* filename);

	unsigned int getNumTriangles();
	unsigned int getNumVertices();

private:
	bool RayTriangleIntersection(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, const Ray& ray, glm::vec3& barycetricCoords);
	bool RayBoxIntersection(const Ray& ray, const Box& box);
private:
	glm::vec3* m_Vertices;
	glm::vec3* m_Normals;
	unsigned int* m_Indices;
	
	unsigned int m_NumVertices;
	unsigned int m_NumIndices;

	glm::vec3 m_BoxMin;
	glm::vec3 m_BoxMax;

	glm::mat4 m_LocalToWorldTransformation;

	Material m_Material;
	bool m_Valid;

	// need tangent vectors,textures,materials etc...
};

#endif