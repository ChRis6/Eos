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

#ifndef _DTRIANGLEMESH_H
#define _DTRIANGLEMESH_H

#include <glm/glm.hpp>

#include "cudaQualifiers.h"
#include "DTriangle.h"
#include "DMaterial.h"

class DTriangleMesh{

public:
	DEVICE DTriangleMesh():m_Triangles(0),m_NumTriangles(0){}

public:
	DEVICE int getNumTriangles()         { return m_NumTriangles;}
	DEVICE void setNumTriangles(int num) { m_NumTriangles = num;}

	DEVICE void setTransformation(const glm::mat4& mat);

private:
	DTriangle** m_Triangles;
	int m_NumTriangles;
	
	DMaterial m_Material;
	glm::mat4 m_WorldTransformation;
	glm::mat4 m_Inverse;
	glm::mat4 m_InverseTranspose;
};
#endif