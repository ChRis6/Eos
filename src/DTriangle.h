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

#ifndef _DTRIANGLE_H
#define _DTRIANGLE_H

#include <glm/glm.hpp>

#include "cudaQualifiers.h"
#include "Ray.h"
#include "DRayIntersection.h"
#include "DMaterial.h"

class DeviceSceneHandler;

class DTriangle{
	friend class DeviceSceneHandler;
public: // constructor
	DEVICE DTriangle(const glm::vec3& v1, const glm::vec3& v2, const glm::vec3& v3,
		             const glm::vec3& n1, const glm::vec3& n2, const glm::vec3& n3):
						m_V1(v1), m_V2(v2), m_V3(v3),
						m_N1(n1), m_N2(n2), m_N3(n3){}

public: // device methods
	DEVICE bool hit(const Ray& ray, DRayIntersection& intersection, float& distance);
	DEVICE bool hit(const Ray& ray, DRayIntersection* intersection, float& distance);

	DEVICE void setTransformation(const glm::mat4& mat);
	DEVICE const glm::mat4& getTransformation();
	DEVICE const glm::mat4& getInverseTrasformation();
	DEVICE const glm::mat4& getInverseTransposeTransformation();
	DEVICE const DMaterial& getMaterial();
	DEVICE int getMaterialIndex() const{ return m_MaterialIndex;}

	DEVICE glm::vec3 getV1()	{ return m_V1;}
private:
	DEVICE bool rayTriangleIntersectionTest(const Ray& ray, glm::vec3& baryCoords);

private: // host methods.Used only by DeviceImporter class
	HOST DEVICE DTriangle(): m_V1(0.0f),m_V2(0.0f),m_V3(0.0f),m_N1(0.0f),m_N2(0.0f),m_N3(0.0f){}

public:
	// triangle vertices
	glm::vec3 m_V1;
	glm::vec3 m_V2;
	glm::vec3 m_V3;

	// triangle normals
	glm::vec3 m_N1;
	glm::vec3 m_N2;
	glm::vec3 m_N3;

	glm::mat4 m_Transformation;
	glm::mat4 m_Inverse;
	glm::mat4 m_InverseTranspose;

	DMaterial m_Material;
	int m_MaterialIndex;
};
#endif