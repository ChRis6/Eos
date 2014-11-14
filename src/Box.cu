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


void Box::setMinVertex(const glm::vec3& min){
	m_MinVertex = min;
	m_Bounds[0] = min;
	
}

void Box::setMaxVertex(const glm::vec3& max){
	m_MaxVertex = max;
	m_Bounds[1] = max;
	
}

const glm::vec3& Box::getMinVertex() const{
	return m_MinVertex;
}

const glm::vec3& Box::getMaxVertex() const{
	return m_MaxVertex;
}

void Box::expandToIncludeBox(const Box& newBox){
	const glm::vec3& newBoxMinVertex = newBox.getMinVertex();
	const glm::vec3& newBoxMaxVertex = newBox.getMaxVertex();

	m_MinVertex.x = glm::min(m_MinVertex.x, newBoxMinVertex.x);
	m_MinVertex.y = glm::min(m_MinVertex.y, newBoxMinVertex.y);
	m_MinVertex.z = glm::min(m_MinVertex.z, newBoxMinVertex.z);

	m_MaxVertex.x = glm::max(m_MaxVertex.x, newBoxMaxVertex.x);
	m_MaxVertex.y = glm::max(m_MaxVertex.y, newBoxMaxVertex.y);
	m_MaxVertex.z = glm::max(m_MaxVertex.z, newBoxMaxVertex.z);

	m_Bounds[0] = m_MinVertex;
	m_Bounds[1] = m_MaxVertex;

}

HOST DEVICE bool Box::intersectWithRay(const Ray& ray, float& distance) const{


	// lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
	const glm::vec3& lb = this->getMinVertex();
	const glm::vec3& rt = this->getMaxVertex();

	const glm::vec3& rayOrigin = ray.getOrigin();
	const glm::vec3& rayInvDirection = ray.getInvDirection();

	float tmin = fmaxf(fmaxf(fminf(((lb.x - rayOrigin.x) * rayInvDirection.x), ((rt.x - rayOrigin.x) * rayInvDirection.x)), fminf(((lb.y - rayOrigin.y) * rayInvDirection.y), ((rt.y - rayOrigin.y) * rayInvDirection.y))), fminf(((lb.z - rayOrigin.z) * rayInvDirection.z), ((rt.z - rayOrigin.z) * rayInvDirection.z)));
	float tmax = fminf(fminf(fmaxf(((lb.x - rayOrigin.x) * rayInvDirection.x), ((rt.x - rayOrigin.x) * rayInvDirection.x)), fmaxf(((lb.y - rayOrigin.y) * rayInvDirection.y), ((rt.y - rayOrigin.y) * rayInvDirection.y))), fmaxf(((lb.z - rayOrigin.z) * rayInvDirection.z), ((rt.z - rayOrigin.z) * rayInvDirection.z)));

	distance = tmin;
	if (tmax < 0 || tmin > tmax)
    	return false;
	return true;
}


void Box::transformBoundingBox(const glm::mat4& transformation){

	/*
	 * Transform the axis aligned bounding box 
	 * and create a new one AABB
	 */
	glm::vec4 transformedMin = transformation * glm::vec4(m_MinVertex, 1.0f);
	glm::vec4 transformedMax = transformation * glm::vec4(m_MaxVertex, 1.0f);
	//float newMinX,newMinY,newMinZ;
	//float newMaxX,newMaxY,newMaxZ;

	//newMinX = newMinY = newMinZ = 99999999.0f;
	//newMaxX = newMaxY = newMaxZ = -999999999.0f;

	glm::vec4 newMin;
	glm::vec4 newMax;
	// find new min
	/*
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

	*/
	newMin = glm::min(transformedMin, transformedMax);
	newMax = glm::max(transformedMax, transformedMin);

	m_MaxVertex = glm::vec3(newMax);
	m_MinVertex = glm::vec3(newMin);

	m_Bounds[0] = m_MinVertex;
	m_Bounds[1] = m_MaxVertex; 

}

float Box::computeVolume(){

	float width  = m_MaxVertex.x - m_MinVertex.x;
	float height = m_MaxVertex.y - m_MinVertex.y;
	float length = m_MaxVertex.z - m_MinVertex.z;

	return width * height * length; 
}
float Box::computeSurfaceArea(){

	float width  = m_MaxVertex.x - m_MinVertex.x;
	float height = m_MaxVertex.y - m_MinVertex.y;
	float length = m_MaxVertex.z - m_MinVertex.z;

	return 2.0f * (width * length + height * length + height * width); 
}

void Box::expandToIncludeVertex(const glm::vec3& vertex){

	m_MinVertex.x = glm::min(m_MinVertex.x, vertex.x);
	m_MinVertex.y = glm::min(m_MinVertex.y, vertex.y);
	m_MinVertex.z = glm::min(m_MinVertex.z, vertex.z);

	m_MaxVertex.x = glm::max(m_MaxVertex.x, vertex.x);
	m_MaxVertex.y = glm::max(m_MaxVertex.y, vertex.y);
	m_MaxVertex.z = glm::max(m_MaxVertex.z, vertex.z);
	
	m_Bounds[0] = m_MinVertex;
	m_Bounds[1] = m_MaxVertex;


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
	if(diffZ > diffY && diffZ > diffX)
		biggest = 2; // Z

	return biggest;
}

glm::vec3 Box::getBoxCentroid(){

	float midX;
	float midY;
	float midZ;

	midX = (m_MaxVertex.x - m_MinVertex.x) / 2.0f;
	midY = (m_MaxVertex.y - m_MinVertex.y) / 2.0f;
	midZ = (m_MaxVertex.z - m_MinVertex.z) / 2.0f;

	return glm::vec3(midX, midY, midZ);
}
bool Box::isPointInBox(glm::vec3& point){

	bool x = (point.x < m_MaxVertex.x) && (point.x > m_MinVertex.x);
	bool y = (point.y < m_MaxVertex.y) && (point.y > m_MinVertex.y);
	bool z = (point.z < m_MaxVertex.z) && (point.z > m_MinVertex.z);
	if(x && y && z)
		return true;
	return false;
}



HOST DEVICE bool Box::intersectWithRayNew(const Ray& ray){

   glm::vec3 tmin = (m_MinVertex - ray.getOrigin()) * ray.getInvDirection();
   glm::vec3 tmax = (m_MaxVertex - ray.getOrigin()) * ray.getInvDirection();
   
   glm::vec3 real_min = glm::min(tmin, tmax);
   glm::vec3 real_max = glm::max(tmin, tmax);
   
   float minmax = fminf( fminf(real_max.x, real_max.y), real_max.z);
   float maxmin = fmaxf( fmaxf(real_min.x, real_min.y), real_min.z);
   	
   return ( minmax >= maxmin);
}

HOST DEVICE bool Box::intersectWithRayOptimized(const Ray& ray, float t0, float t1) const{
	float tmin,tmax,tymin,tymax,tzmin,tzmax;
	
	const glm::vec3& rayOrigin = ray.getOrigin();
	const glm::vec3& rayInvDirection = ray.getInvDirection();
	
	tmin = ( m_Bounds[ray.m_sign[0]].x - rayOrigin.x ) * rayInvDirection.x;
	tmax = ( m_Bounds[1 - ray.m_sign[0]].x - rayOrigin.x) * rayInvDirection.x;
	tymin = ( m_Bounds[ray.m_sign[1]].y - rayOrigin.y ) * rayInvDirection.y;
	tymax = ( m_Bounds[1 - ray.m_sign[1]].y - rayOrigin.y) * rayInvDirection.y;

	if( (tmin > tymax) || (tymin > tmax) )
		return false;
	if( tymin > tmin )
		tmin = tymin;
		if( tymax < tmax)

	tmax = tymax;
	tzmin = ( m_Bounds[ray.m_sign[2]].z - rayOrigin.z) * rayInvDirection.z ;
	tzmax = ( m_Bounds[1- ray.m_sign[2]].z - rayOrigin.z) * rayInvDirection.z;
	if( (tmin > tzmax) || (tzmin > tmax))
		return false;
	if( tzmin > tmin)
		tmin = tzmin;
	if( tzmax < tmax)
		tmax = tzmax;
	
	return ( (tmin < t1) && (tmax > t0));
}