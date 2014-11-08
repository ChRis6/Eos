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

DEVICE bool DScene::findMinDistanceIntersectionLinear(const Ray& ray, DRayIntersection& intersection) const{
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
		const glm::vec3& localRayOrigin    = glm::vec3( tri->getInverseTrasformation() * glm::vec4(ray.getOrigin(), 1.0f));
		const glm::vec3& localRayDirection = glm::vec3( tri->getInverseTrasformation() * glm::vec4(ray.getDirection(), 0.0f));

		localRay.setOrigin(localRayOrigin);
		localRay.setDirection(localRayDirection);
		
		if( tri->hit(localRay, dummyIntersection, triDistance) ){
			// hit found.Transform intersection to world coordinates
			intersectionFound = true;
			if( triDistance < minDist && triDistance > 0.0f){

				minDist = triDistance;
				min_point  = tri->getTransformation() * glm::vec4(dummyIntersection.getIntersectionPoint(), 1.0f);
				min_normal = tri->getInverseTransposeTransformation() * glm::vec4(dummyIntersection.getIntersectionNormal(), 0.0f);
				intersection.setIntersectionMaterial(tri->getMaterial()); 
			}
		}
	}

	intersection.setIntersectionPoint(glm::vec3(min_normal));
	intersection.setIntersectionNormal(glm::vec3(min_normal));
	return intersectionFound;
}

DEVICE bool DScene::intersectRayWithLeaf(const Ray& ray, BvhNode* node, DRayIntersection& intersection, float& distance) const{
		 
	Ray localRay;
	DTriangle* minTri = NULL;

	#pragma unroll 2
	for( int i = 0; i < node->numSurfacesEncapulated; i++){
		int triIndex = node->surfacesIndices[i];

		DTriangle* tri = this->getTriangle(triIndex);

		// transform ray to local coordinates
		localRay.setOrigin(glm::vec3( tri->getInverseTrasformation() * glm::vec4(ray.getOrigin(), 1.0f)));
		localRay.setDirection(glm::vec3( tri->getInverseTrasformation() * glm::vec4(ray.getDirection(), 0.0f)));

		if( tri->hit(localRay, intersection, distance) ){
			// intersection found
			minTri = tri;
		}
	}

	if( minTri != NULL ){
		intersection.setIntersectionPoint(glm::vec3(minTri->getTransformation() * glm::vec4(intersection.getIntersectionPoint(), 1.0f)));
		intersection.setIntersectionNormal(glm::vec3(minTri->getInverseTransposeTransformation() * glm::vec4(intersection.getIntersectionNormal(), 0.0f)));
		return true;
	}
	return false;
}


DEVICE bool DScene::findMinDistanceIntersectionBVH(const Ray& ray, DRayIntersection& intersection) const{
	
	BvhNode* stack[64];
	BvhNode** stack_ptr = stack;
	
	float minDistace = 99999.0f;
	bool surfaceIntersectionFound = false;
	BvhNode* currNode = &m_BvhBuffer[0];

	if( currNode->aabb.intersectWithRay(ray, minDistace) == false )
		return false;
	minDistace = 99999.0f;
	// push null
	*stack_ptr++ = NULL;

	while(currNode != NULL){

		if( currNode->type == BVH_NODE ){

			float leftDistance;
			float rightDistance;


			BvhNode* leftChild  = &m_BvhBuffer[currNode->leftChildIndex];	
			BvhNode* rightChild = &m_BvhBuffer[currNode->rightChildIndex];

			bool leftChildIntersected = leftChild->aabb.intersectWithRay(ray, leftDistance);
			bool rightChildIntersected = rightChild->aabb.intersectWithRay(ray, rightDistance);

			if(leftChildIntersected){
				currNode = leftChild;
				if( rightChildIntersected){

					// push right child to stack
					*stack_ptr++ = rightChild;
				}
			}
			else if(rightChildIntersected){
				currNode = rightChild;
			} 
			else{ // none of  the children hit the ray. POP stack
				currNode = *--stack_ptr;
			}
		}
		else{

			// node is a leaf
			//float distance;
			//DRayIntersection dummyIntersection;
			if( this->intersectRayWithLeaf(ray, currNode, intersection, minDistace) ){
				surfaceIntersectionFound = true;
				//if( distance < minDistace){
				//	minDistace = distance;
				//	intersection = dummyIntersection;
				//}
			}
			// pop 
			currNode = *--stack_ptr;
		}
	}
	return surfaceIntersectionFound;
}

DEVICE bool DScene::visibilityTest(const Ray& ray) const{
	return false;
}
