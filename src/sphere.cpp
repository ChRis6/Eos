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

#include <math.h>
#include "sphere.h"

#ifndef EPSILON
#define EPSILON 1e-4
#endif


bool Sphere::hit(const Ray& ray, RayIntersection& intersection, float& distance){

	float t;
	float a,b,c;
	bool collision;
	glm::vec3 displacement;

	displacement = ray.getOrigin() - m_Origin;

	a = glm::dot(ray.getDirection(), ray.getDirection());
  b = 2.0f * glm::dot(ray.getDirection(), displacement);
  c = glm::dot(displacement, displacement) - m_RadiusSquared;

    t = -1;
    collision = this->quadSolve(a, b, c, t);

    if(collision){
    	// set intersection point and normal
    	intersection.setPoint(ray.getOrigin() + t * ray.getDirection());
    	intersection.setNormal(glm::normalize(intersection.getPoint() - m_Origin));
      intersection.setMaterial(m_Material);
      // also return the distance
      distance = t;
    	return true;
    }

	return false;
}

bool Sphere::quadSolve(float a, float b, float c, float& t){

   float dis = (b * b) - (4.0 * a * c);
   if( dis < 0.0 )
      return false;

   float t1,t2;

   t1 = ( -b + sqrt(dis)) / (float) ( 2.0 * a);
   t2 = ( -b - sqrt(dis)) / (float) ( 2.0 * a);


   if( t1 > EPSILON){
      if( t2 > EPSILON){
        if( t1 > t2 ){
        	t = t2;
        }
        else{
        	t = t1;
        }
        return true;
     	}
      return true;      
   }

   if( t2 > EPSILON){
      t = t2;
      return true;
   }

   return false;

}

Box Sphere::getLocalBoundingBox(){

  float minX,minY,minZ;
  float maxX,maxY,maxZ;


  minX = m_Origin.x - m_RadiusSquared; 
  minY = m_Origin.y - m_RadiusSquared; 
  minZ = m_Origin.z - m_RadiusSquared; 

  maxX = m_Origin.x + m_RadiusSquared;
  maxY = m_Origin.y + m_RadiusSquared;
  maxZ = m_Origin.z + m_RadiusSquared;

  return Box(glm::vec3(minX, minY, minZ), glm::vec3(maxX, maxY, maxZ));
}

glm::vec3 Sphere::getCentroid(){
  return m_Origin;
}

const glm::mat4& Sphere::transformation(){
  return m_LocalToWorldTransformation;
}

void Sphere::setTransformation(glm::mat4 transformation){
  m_LocalToWorldTransformation = transformation;
}