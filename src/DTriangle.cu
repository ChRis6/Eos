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

#include "DTriangle.h"


DEVICE void DTriangle::setTransformation(const glm::mat4& mat){
	m_Transformation = mat;
	m_Inverse = glm::inverse(mat);
	m_InverseTranspose = glm::transpose(m_Inverse);
}

DEVICE const glm::mat4& DTriangle::getTransformation(){
	return m_Transformation;
}

DEVICE const glm::mat4& DTriangle::getInverseTrasformation(){
	return m_Inverse;
}

DEVICE const glm::mat4& DTriangle::getInverseTransposeTransformation(){
	return m_InverseTranspose;
}

DEVICE const DMaterial& DTriangle::getMaterial(){
	return m_Material;
}

DEVICE bool DTriangle::hit(const Ray& ray, DRayIntersection& intersection, float& distance){
	
	bool collision;
	glm::vec3 barCoords(0.0f);
	glm::vec3 pos(0.0f);
	glm::vec3 norm(0.0f);
	
	collision = this->rayTriangleIntersectionTest( ray, barCoords);
	if(collision){
		float u,v;

		distance = barCoords.x;
		pos = ray.getOrigin() + distance * ray.getDirection();

		// interpolate normals
		u = barCoords.y;
		v = barCoords.z;
		norm = glm::normalize(m_N1 * ( 1.0f - u - v) + (m_N2 * u) + (m_N3*v));
		
		intersection.setIntersectionPoint(pos);
		intersection.setIntersectionNormal(norm);
		intersection.setIntersectionMaterial(this->getMaterial());

		return true;
	}
	return false;
}

DEVICE bool DTriangle::rayTriangleIntersectionTest(const Ray& ray, glm::vec3& baryCoords){
	const glm::vec3& v1 = m_V1;
	const glm::vec3& v2 = m_V2;
	const glm::vec3& v3 = m_V3; 

	float t,u,v;
	glm::vec3 e1 = v2 - v1;
	glm::vec3 e2 = v3 - v1;
	glm::vec3 P = glm::cross(ray.getDirection(), e2);
	
	float det = glm::dot(e1, P);
	if (det > -0.00001f && det < 0.00001f)
    	return false;

    float inv_det = 1.0f / det;
 
  	glm::vec3 T = ray.getOrigin() - v1;
  	glm::vec3 Q = glm::cross(T, e1);
  	
  	t = glm::dot(e2, Q) * inv_det;
	

  	if (t < 0.0f)
    	return false;
  	
  	u = glm::dot(T, P) * inv_det;
  	if (u < 0.0f || u > 1.0f)
    	return false;

  	v = glm::dot(ray.getDirection(), Q) * inv_det;
  	if (v < 0.0f || u + v > 1.0f)
    	return false;

   	baryCoords.x = t;
   	baryCoords.y = u;
	baryCoords.z = v;
  	return true;
}