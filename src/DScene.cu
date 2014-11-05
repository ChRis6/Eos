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
#include "DScene.h"

DEVICE bool DScene::findMinDistanceIntersectionLinear(const Ray& ray, DRayIntersection& intersection){
	int numTriangles;
	int i;
	bool intersectionFound = false;
	float minDist;
	Ray localRay;
	DRayIntersection dummyIntersection;
	glm::vec4 min_point;
	glm::vec4 min_normal;

	numTriangles = this->getNumTriangles();
	minDist = 9999999.0f;

	for( i = 0; i < numTriangles; i++){

		float triDistance;
		DTriangle* tri = this->getTriangle(i);

		// transform ray to local coordinates
		glm::vec3& localRayOrigin    = glm::vec3( tri->getInverseTrasformation() * glm::vec4(ray.getOrigin(), 1.0f));
		glm::vec3& localRayDirection = glm::vec3( tri->getInverseTrasformation() * glm::vec4(ray.getDirection(), 0.0f));

		localRay.setOrigin(localRayOrigin);
		localRay.setDirection(localRayDirection);
		
		if( tri->hit(localRay, dummyIntersection, triDistance) ){
			// hit found.Transform intersection to world coordinates
			intersectionFound = true;
			if( triDistance < minDist && triDistance > 0.0f){

				minDist = triDistance;
				min_point  = tri->getTransformation() * glm::vec4(dummyIntersection.getPoint(), 1.0f);
				min_normal = tri->getInverseTransposeTransformation() * glm::vec4(dummyIntersection.getNormal(), 0.0f);
				intersection.setIntersectionMaterial(tri->getMaterial()); 
			}
		}
	}

	intersection.setIntersectionPoint(glm::vec3(min_normal));
	intersection.setIntersectionNormal(glm::vec3(min_normal));
	return intersectionFound;
}