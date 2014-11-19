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
#include "Triangle.h"
#include "Material.h"

class TriangleMesh{

public:
	TriangleMesh();
	~TriangleMesh();
	   
	bool loadFromFile(const char* filename);

	const glm::mat4& transformation();
	void  setTransformation(glm::mat4& transformation);
	Material& getMaterial();
	void setMaterial(const Material& material);

	unsigned int getNumTriangles(){ return m_NumTriangles;}
	Triangle* getTriangle(int index);
	void setSceneStartIndex(int start){m_SceneIndexStart = start;}
	void setSceneEndIndex(int end){m_SceneIndexEnd = end;}

	void setMaterialIndex(int index) { m_MaterialIndex = index;}
	int getMaterialIndex(int index)const { return m_MaterialIndex;}

	void setTransformationIndex(int index) { m_TransformationIndex = index;}
	int getTransformationIndex()const 	   { return m_TransformationIndex; }

private:
	bool RayTriangleIntersection(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, const Ray& ray, glm::vec3& barycetricCoords);
	bool RayBoxIntersection(const Ray& ray, const Box& box);
private:

	Triangle** m_Triangles;
	unsigned int m_NumTriangles;
	
	glm::mat4 m_LocalToWorldTransformation;
	glm::mat4 m_Inverse;
	glm::mat4 m_InverseTranspose;
	int m_TransformationIndex;

	Material m_Material;
	int m_MaterialIndex;
	
	unsigned int m_SceneIndexStart;
	unsigned int m_SceneIndexEnd;
	bool m_Valid;

	// need tangent vectors,textures,materials etc...
};

#endif