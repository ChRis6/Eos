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

#include "Triangle.h"
#include <iostream>

bool Triangle::hit(const Ray& ray, RayIntersection& intersection, float& distance){

	bool collision;
	glm::vec3 barCoords(0.0f);
	glm::vec3 pos(0.0f);
	glm::vec3 norm(0.0f);
	float u,v;

	collision = this->RayTriangleIntersection(m_V1, m_V2, m_V3, ray, barCoords);
	if(collision){

		distance = barCoords.x;
		pos = ray.getOrigin() + distance * ray.getDirection();

		// interpolate normals
		u = barCoords.y;
		v = barCoords.z;
		norm = glm::normalize(m_N1 * ( 1.0f - u - v) + (m_N2 * u) + (m_N3*v));
		
		intersection.setPoint(pos);
		intersection.setNormal(norm);
		intersection.setMaterial(this->getMaterial());

		return true;
	}
	return false;
}

bool Triangle::RayTriangleIntersection(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, const Ray& ray, glm::vec3& barycetricCoords){

   glm::vec3 e1, e2;  //Edge1, Edge2
   glm::vec3 P, Q, T;
   float det, inv_det, u, v;
   float t;
 
   //Find vectors for two edges sharing V1
   e1  = v2 - v1;
   e2 =  v3 -  v1;
  
   P = glm::cross(ray.getDirection(), e2);
  
   det = glm::dot(e1, P);

   // culling
   if( det < 1e-6) return false;

   T = ray.getOrigin() - v1;
   u = glm::dot( T, P);
   if( u < 0.0 || u > det) return false;

   Q = glm::cross(T, e1);
   v = glm::dot( ray.getDirection(), Q);
   if( v < 0.0 || u + v > det) return false;

   t = glm::dot(e2, Q);
   inv_det = 1.0 / (float)det;

   t *= inv_det;
   u *= inv_det;
   v *= inv_det;

   barycetricCoords.x = t;
   barycetricCoords.y = u;
   barycetricCoords.z = v;

   return true;
}


const glm::mat4& Triangle::transformation(){
	return *m_LocalToWorldTransformation;
}

void Triangle::setTransformation(glm::mat4& transformation){
	m_LocalToWorldTransformation = &transformation;
}

void Triangle::setInverseTransformation(glm::mat4& inverse){
	m_Inverse = &inverse;
}

void Triangle::setInverseTransposeTransformation(glm::mat4& inverseTranspose){
	m_InverseTranspose = &inverseTranspose;
}

const glm::mat4& Triangle::getInverseTransformation(){
	return *m_Inverse;
}

const glm::mat4& Triangle::getInverseTransposeTransformation(){
	return *m_InverseTranspose;
}

Material& Triangle::getMaterial(){
	return *m_Material;
}

void Triangle::setMaterial(Material& material){
	m_Material = &material;
}

Box Triangle::getLocalBoundingBox(){
	Box triangleBox;

	triangleBox.expandToIncludeVertex(m_V1);
	triangleBox.expandToIncludeVertex(m_V2);
	triangleBox.expandToIncludeVertex(m_V3);
	return triangleBox;
}
glm::vec3 Triangle::getCentroid(){
	return m_Centroid;
}

bool Triangle::isPointInside(glm::vec3& point){
	return false;
}