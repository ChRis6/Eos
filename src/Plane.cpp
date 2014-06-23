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

#include "Plane.h"

bool Plane::hit(const Ray& ray, RayIntersection& intersection, float& distance){

	float denom;
	const float eps = 1e-5;

	denom = glm::dot( ray.getDirection(), this->getPlaneNormal());
	if( denom > eps ){
		distance = glm::dot( this->getPlanePoint() - ray.getOrigin(), this->getPlaneNormal()) / (float) denom;
		glm::vec3 point = ray.getOrigin() + ray.getDirection() * distance;

		intersection.setPoint(point);
		intersection.setNormal(this->getPlaneNormal());
		intersection.setMaterial(this->getMaterial());
		return distance >= 0.0f;
	}

	return false;
}

Box Plane::boundingBox(){
	return Box();
}
