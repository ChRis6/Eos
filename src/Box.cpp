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


#include "Box.h"


void Box::setMinVertex(glm::vec3 min){
	m_MinVertex = min;
}

void Box::setMaxVertex(glm::vec3 max){
	m_MaxVertex = max;
}

glm::vec3 Box::getMinVertex() const{
	return m_MinVertex;
}

glm::vec3 Box::getMaxVertex() const{
	return m_MaxVertex;
}

void Box::expandToIncludeBox(const Box& newBox){
	glm::vec3 newBoxMinVertex = newBox.getMinVertex();
	glm::vec3 newBoxMaxVertex = newBox.getMaxVertex();

	m_MinVertex.x = glm::min(m_MinVertex.x, newBoxMinVertex.x);
	m_MinVertex.y = glm::min(m_MinVertex.y, newBoxMinVertex.y);
	m_MinVertex.z = glm::min(m_MinVertex.z, newBoxMinVertex.z);

	m_MaxVertex.x = glm::max(m_MaxVertex.x, newBoxMaxVertex.x);
	m_MaxVertex.y = glm::max(m_MaxVertex.y, newBoxMaxVertex.y);
	m_MaxVertex.z = glm::max(m_MaxVertex.z, newBoxMaxVertex.z);
}

bool Box::intersectWithRay(const Ray& ray, float& distance){


	// lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
	glm::vec3 lb = this->getMinVertex();
	glm::vec3 rt = this->getMaxVertex();

	glm::vec3 rayOrigin = ray.getOrigin();
	glm::vec3 rayInvDirection = ray.getInvDirection();

	float t1 = (lb.x - rayOrigin.x) * rayInvDirection.x;
	float t2 = (rt.x - rayOrigin.x) * rayInvDirection.x;
	float t3 = (lb.y - rayOrigin.y) * rayInvDirection.y;
	float t4 = (rt.y - rayOrigin.y) * rayInvDirection.y;
	float t5 = (lb.z - rayOrigin.z) * rayInvDirection.z;
	float t6 = (rt.z - rayOrigin.z) * rayInvDirection.z;

	float tmin = glm::max(glm::max(glm::min(t1, t2), glm::min(t3, t4)), glm::min(t5, t6));
	float tmax = glm::min(glm::min(glm::max(t1, t2), glm::max(t3, t4)), glm::max(t5, t6));

	// if tmax < 0, ray (line) is intersecting AABB, but whole AABB is behing
	if (tmax < 0)
	{
    	//t = tmax;
    	return false;
	}

	// if tmin > tmax, ray doesn't intersect AABB
	if (tmin > tmax)
	{
    	//t = tmax;
    	return false;
	}

	//t = tmin;
	distance = tmin;
	return true;
}


void Box::transformBoundingBox(const glm::mat4& transformation){

	/*
	 * Transform the axis aligned bounding box 
	 * and create a new one AABB
	 */
	glm::vec4 transformedMin = transformation * glm::vec4(m_MinVertex, 1.0f);
	glm::vec4 transformedMax = transformation * glm::vec4(m_MaxVertex, 1.0f);
	float newMinX,newMinY,newMinZ;
	float newMaxX,newMaxY,newMaxZ;

	newMinX = newMinY = newMinZ = 99999999.0f;
	newMaxX = newMaxY = newMaxZ = -999999999.0f;

	// find new min
	newMinX = glm::min(newMinX, transformedMin.x);
	newMinY = glm::min(newMinY, transformedMin.y);
	newMinZ = glm::min(newMinZ, transformedMin.z);

	newMinX = glm::min(newMinX, transformedMax.x);
	newMinY = glm::min(newMinY, transformedMax.y);
	newMinZ = glm::min(newMinZ, transformedMax.z);

	// find new max
	newMaxX = glm::max(newMaxX, transformedMax.x);
	newMaxY = glm::max(newMaxY, transformedMax.y);
	newMaxZ = glm::max(newMaxZ, transformedMax.z);

	newMaxX = glm::max(newMaxX, transformedMin.x);
	newMaxY = glm::max(newMaxY, transformedMin.y);
	newMaxZ = glm::max(newMaxZ, transformedMin.z);

	m_MinVertex.x = newMinX;
	m_MinVertex.y = newMinY;
	m_MinVertex.z = newMinZ;

	m_MaxVertex.x = newMaxX;
	m_MaxVertex.y = newMaxY;
	m_MaxVertex.z = newMaxZ;
}

float Box::computeVolume(){

	float width  = m_MaxVertex.x - m_MinVertex.x;
	float height = m_MaxVertex.y - m_MinVertex.y;
	float length = m_MaxVertex.z - m_MinVertex.z;

	return width * height * length; 
}

void Box::expandToIncludeVertex(const glm::vec3& vertex){

	m_MinVertex.x = glm::min(m_MinVertex.x, vertex.x);
	m_MinVertex.y = glm::min(m_MinVertex.y, vertex.y);
	m_MinVertex.z = glm::min(m_MinVertex.z, vertex.z);

	m_MaxVertex.x = glm::max(m_MaxVertex.x, vertex.x);
	m_MaxVertex.y = glm::max(m_MaxVertex.y, vertex.y);
	m_MaxVertex.z = glm::max(m_MaxVertex.z, vertex.z);
}

int Box::getBiggestDimension() const{
	int biggest;
	float diffX,diffY,diffZ;

	diffX = m_MaxVertex.x - m_MinVertex.x;
	diffY = m_MaxVertex.y - m_MinVertex.y;
	diffZ = m_MaxVertex.z - m_MinVertex.z;

	// assume X
	biggest = 0;
	if(diffY > diffX) 
		biggest = 1; // Y
	if(diffZ > diffY)
		biggest = 2; // Z

	return biggest;
}