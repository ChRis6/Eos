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

#include "Disk.h"
#include <glm/gtx/norm.hpp>

bool Disk::hit(const Ray& ray, RayIntersection& intersection, float& distance){

	RayIntersection dummyIntersection;
	bool collisionFound;
	float t;

	collisionFound = inherited::hit(ray, dummyIntersection, t);
	if(collisionFound){
		// is the intersection point inside the disk ?
		if( glm::length2(intersection.getPoint() - this->getPlanePoint()) <  m_RadiusSquared){
			// yes
			distance = t;
			intersection = dummyIntersection;
			return true;
		}
	} 
	return false;
}