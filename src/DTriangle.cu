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
	
	glm::vec3 barCoords(0.0f);
	if( this->rayTriangleIntersectionTest( ray, barCoords)){
		if( barCoords.x < distance){
			distance = barCoords.x;	

			intersection.setIntersectionPoint(ray.getOrigin() + barCoords.x * ray.getDirection());
			intersection.setIntersectionNormal(glm::normalize(m_N1 * ( 1.0f - barCoords.y - barCoords.z) + (m_N2 * barCoords.y) + (m_N3*barCoords.z)));
			intersection.setIntersectionMaterial(this->getMaterial());
			return true;
		}
	}
	return false;
}

DEVICE bool DTriangle::rayTriangleIntersectionTest(const Ray& ray, glm::vec3& baryCoords){

	const glm::vec3& P = glm::cross(ray.getDirection(), m_V3 - m_V1);
	float det = glm::dot(m_V2 - m_V1, P);
	if (det > -0.00001f && det < 0.00001f)
    	return false;

    det = 1.0f / det;
  	const glm::vec3& T = ray.getOrigin() - m_V1;
  	const glm::vec3& Q = glm::cross(T, m_V2 - m_V1);
  	
  	baryCoords.x = glm::dot(m_V3 - m_V1, Q) * det;
	baryCoords.y = glm::dot(T, P) * det;
	baryCoords.z = glm::dot(ray.getDirection(), Q) * det;

  	if ((baryCoords.x < 0.0f) || (baryCoords.y < 0.0f || baryCoords.y > 1.0f) || ( baryCoords.z < 0.0f || baryCoords.y + baryCoords.z > 1.0f) )
    	return false;
  	
  	return true;
}