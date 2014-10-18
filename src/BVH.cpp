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
#include <iostream>
#include <algorithm>
#include <cfloat>
#include "BVH.h"

#define SURFACES_PER_LEAF 1
#define COST_TRAVERSAL    1
#define COST_INTERSECTION 2


bool by_X_compareSurfaces(Surface* a, Surface* b){
	

	glm::vec4 aCentroid = glm::vec4(a->getCentroid(),1.0f);
	glm::vec4 bCentroid = glm::vec4(b->getCentroid(),1.0f);

	// transform centroid to world coords
	aCentroid = a->transformation() * aCentroid;
	bCentroid = b->transformation() * bCentroid;

	return aCentroid.x < bCentroid.x;
}

bool by_Y_compareSurfaces(Surface* a, Surface* b){
	

	glm::vec4 aCentroid = glm::vec4(a->getCentroid(),1.0f);
	glm::vec4 bCentroid = glm::vec4(b->getCentroid(),1.0f);

	// transform centroid to world coords
	aCentroid = a->transformation() * aCentroid;
	bCentroid = b->transformation() * bCentroid;

	return aCentroid.y < bCentroid.y;
}

bool by_Z_compareSurfaces(Surface* a, Surface* b){
	

	glm::vec4 aCentroid = glm::vec4(a->getCentroid(),1.0f);
	glm::vec4 bCentroid = glm::vec4(b->getCentroid(),1.0f);

	// transform centroid to world coords
	aCentroid = a->transformation() * aCentroid;
	bCentroid = b->transformation() * bCentroid;

	return aCentroid.z < bCentroid.z;
}




void BVH::buildHierarchy(Surface** surfaces, int numSurfaces){

	// create a temp array of the surfaces
	Surface** tempSurfaces = new Surface*[numSurfaces];
	for( int i = 0 ; i < numSurfaces; i++)
		tempSurfaces[i] = surfaces[i];

	this->buildTopDown(&m_Root, tempSurfaces, numSurfaces);
	delete tempSurfaces;
}

void BVH::buildTopDown(BvhNode** tree, Surface** surfaces, int numSurfaces){

	BvhNode* node = new BvhNode;
	*tree = node;

	// find bounding box of surfaces
	node->aabb = this->computeBoxWithSurfaces(surfaces, numSurfaces);
	node->numSurfacesEncapulated = numSurfaces;

	// is it a leaf ? Only one surface per leaf 
	if( numSurfaces <= SURFACES_PER_LEAF ){
		/*
		node->type = BVH_LEAF;
		node->leftChild = NULL;
		node->rightChild = NULL;
		node->tracedObject = surfaces[0]; 	// get the first surface
		*/
		createLeaf(node, surfaces, numSurfaces);
	}
	else{

		node->tracedObject = NULL;
		node->tracedObjectArray = NULL;
		node->type = BVH_NODE;
		int splitIndex = this->topDownSplitIndex(surfaces, numSurfaces, node->aabb);

		// recursion
		buildTopDown( &(node->leftChild), &surfaces[0], splitIndex);
		buildTopDown( &(node->rightChild),&surfaces[splitIndex], numSurfaces - splitIndex);
	}
}

Box BVH::computeBoxWithSurfaces(Surface** surfaces, int numSurfaces){

	Box computedBox = surfaces[0]->getLocalBoundingBox();
	computedBox.transformBoundingBox(surfaces[0]->transformation());

	for( int i = 1; i < numSurfaces; i++){
		// get the bounding box in local surface coordinates
		Box surfaceBox = surfaces[i]->getLocalBoundingBox();
		// transform box in world coords
		surfaceBox.transformBoundingBox(surfaces[i]->transformation());
		// expand resulting bounding box
		computedBox.expandToIncludeBox(surfaceBox);
	}

	return computedBox;
}

int BVH::topDownSplitIndex(Surface** surfaces, int numSurfaces, Box parentBox){

	int split = 0;
	int biggestIndex;
	int i;
	float midAxisCoord;
	Box centroidsBox;

	for( i = 0 ; i < numSurfaces; i++){
		glm::vec4 surfaceCentroidWorldCoords = surfaces[i]->transformation() * glm::vec4(surfaces[i]->getCentroid(), 1.0f);
		glm::vec3 surfaceCentroid = glm::vec3(surfaceCentroidWorldCoords.x, surfaceCentroidWorldCoords.y, surfaceCentroidWorldCoords.z);
		centroidsBox.expandToIncludeVertex(surfaceCentroid);
	}
	

	biggestIndex = centroidsBox.getBiggestDimension();

	/*
	biggestIndex = 0;
	if( boxCentroid.y > boxCentroid.x)
		biggestIndex = 1;
	if( boxCentroid.z > boxCentroid.y && boxCentroid.z > boxCentroid.x)
		biggestIndex = 2;
	*/

	glm::vec3 boxMinVertex = centroidsBox.getMinVertex();
	glm::vec3 boxMaxVertex = centroidsBox.getMaxVertex();
	midAxisCoord = (boxMaxVertex[biggestIndex] + boxMinVertex[biggestIndex] ) * 0.5f;

	// sort surfaces along axis ( in world coords)
	if( biggestIndex == 0 )
		std::sort(surfaces, surfaces + numSurfaces , by_X_compareSurfaces);
	else if( biggestIndex == 1)
		std::sort(surfaces, surfaces + numSurfaces , by_Y_compareSurfaces);
	else if( biggestIndex == 2)
		std::sort(surfaces, surfaces + numSurfaces , by_Z_compareSurfaces);


	// find mid split
	for(i = 0 ; i < numSurfaces; i++){
		glm::vec4 surfaceCentroid = glm::vec4(surfaces[i]->getCentroid(),1.0f);
		surfaceCentroid = surfaces[i]->transformation() * surfaceCentroid;
		//std::cout << "X = " << surfaceCentroid.x << std::endl;

		if( surfaceCentroid[biggestIndex] > midAxisCoord  ){
			
			break;
		}
	}

	split = i;
	if( split == 0 || split == numSurfaces){ // bad split ?
		split = numSurfaces / 2;
	}
	
	return split;
}

Surface* BVH::intersectRay(const Ray& ray, RayIntersection& intersectionFound){
	
	Surface* intersectedSurface = NULL;
	float distance;
	intersectedSurface = this->intersectRecursive(ray, this->getRoot(), distance, intersectionFound);
	return intersectedSurface;
}

Surface* BVH::intersectRecursive(const Ray& ray, BvhNode* node, float& minDistance, RayIntersection& intersection){
	
	bool rayIntersectsBox = false;
	float distance = 999999.0f;
	int minSurfaceIndex;

	if( node->aabb.intersectWithRay(ray, distance) == true)
		rayIntersectsBox = true;

	if(rayIntersectsBox){
		// is it a leaf ?
		if( node->type == BVH_LEAF){
			RayIntersection dummyIntersection;
			if( this->intersectRayWithLeaf( ray, node, dummyIntersection, minDistance, minSurfaceIndex)){
				intersection = dummyIntersection;
				return node->tracedObjectArray[minSurfaceIndex];
			}
			return NULL;
		}
		
		// No,its an intermediate node
		float leftChildDistance  = 999999.0f;
		float rightChildDistance = 999999.0f;
		RayIntersection leftChildeIntersection;
		RayIntersection rightChildIntersection;

		Surface* leftChildSurface  = this->intersectRecursive(ray, node->leftChild, leftChildDistance, leftChildeIntersection);
		Surface* rightChildSurface = this->intersectRecursive(ray, node->rightChild, rightChildDistance, rightChildIntersection);


		if( leftChildSurface != NULL && leftChildDistance < rightChildDistance ){
			minDistance = leftChildDistance;
			intersection = leftChildeIntersection;
			return leftChildSurface;
		}
		else if( rightChildSurface != NULL){
			minDistance  = rightChildDistance;
			intersection = rightChildIntersection; 
			return rightChildSurface;
		}
		else 
			return NULL;

	}
	else{
		return NULL;
	}

}
/*
 * Returns True when ray actually intersects surface 
 * in local coordinates and fills arguments intersection
 * and distance with results
 *
 * Caution: Argument ray must be  in world coords
 */ 

bool BVH::intersectRayWithLocalSurface(const Ray& ray, Surface* surface, RayIntersection& intersection, float& distance){

	bool surfaceIntersectionFound = false;
	float distanceFromOrigin = 9999999.0f;
	RayIntersection possibleIntersection;
	glm::vec4 intersectionPointWorldCoords;
	glm::vec4 intersectionNormalWorldCoords;

	// Transform ray to local coordinates
	const glm::mat4& M = surface->transformation();
	glm::vec3 localRayOrigin    = glm::vec3(glm::inverse(M) * glm::vec4(ray.getOrigin(), 1.0f));
	glm::vec3 localRayDirection = glm::vec3(glm::inverse(M) * glm::vec4(ray.getDirection(), 0.0f)); 
	Ray localRay(localRayOrigin, localRayDirection);

	surfaceIntersectionFound = surface->hit(localRay, possibleIntersection, distanceFromOrigin);
	if(surfaceIntersectionFound){
		intersectionPointWorldCoords  = M * glm::vec4( possibleIntersection.getPoint(), 1.0f);
		intersectionNormalWorldCoords = glm::transpose(glm::inverse(M)) * glm::vec4(possibleIntersection.getNormal(), 0.0f);

		intersection.setPoint(glm::vec3(intersectionPointWorldCoords));
		intersection.setNormal(glm::normalize(glm::vec3(intersectionNormalWorldCoords)));
		intersection.setMaterial(surface->getMaterial());
		distance = distanceFromOrigin;
		return true;
	}

	return false;

}

Surface* BVH::pointInsideSurface(glm::vec3& point){
	return this->isPointInsideSurfaceRecursive(this->getRoot(), point);
}

Surface* BVH::isPointInsideSurfaceRecursive(BvhNode* node, glm::vec3& point){

	Surface* leftChildSurface  = NULL;
	Surface* rightChildSurface = NULL; 

	// is point inside the current bounding box ?
	if( node->aabb.isPointInBox(point) == false )
		return NULL;

	// point is in box

	// is it a leaf ?
	if( node->type == BVH_LEAF){
		if( node->tracedObject->isPointInside(point) )
			return node->tracedObject;
		else
			return NULL;
	}
	else{
		// this isnt a surface.Traverse BVH
		leftChildSurface  = this->isPointInsideSurfaceRecursive(node->leftChild, point);
		if( leftChildSurface)
			return leftChildSurface; 

		rightChildSurface = this->isPointInsideSurfaceRecursive(node->rightChild, point);

		if( rightChildSurface )
			return rightChildSurface;

		return NULL;
	}
}

void BVH::createLeaf(BvhNode* newNode, Surface** surfaces, int numSurfaces){


	Surface** leafSurfaces = new Surface*[numSurfaces];
	for(int i = 0 ; i < numSurfaces; i++)
		leafSurfaces[i] = surfaces[i];
	
	newNode->type = BVH_LEAF;
	newNode->numSurfacesEncapulated = numSurfaces;
	newNode->tracedObjectArray = leafSurfaces;
	newNode->leftChild = NULL;
	newNode->rightChild = NULL;
}

bool BVH::intersectRayWithLeaf(const Ray& ray, BvhNode* leaf, RayIntersection& intersection, float& distance, int& leafSurfaceIndex){

	int i;
	bool surfaceIntersectionFound = false;
	float distanceFromOrigin = 9999999.0f;
	float minDistanceFound = 99999999.0f;

	RayIntersection possibleIntersection;
	glm::vec4 intersectionPointWorldCoords;
	glm::vec4 intersectionNormalWorldCoords;

	int numSurfaces = leaf->numSurfacesEncapulated;
	Surface** surfaces = leaf->tracedObjectArray;

	for( i = 0 ; i < numSurfaces; i++){
		
		Surface* surface = surfaces[i];

		// Transform ray to local coordinates
		const glm::mat4& M = surface->transformation();
		glm::vec3 localRayOrigin    = glm::vec3(glm::inverse(M) * glm::vec4(ray.getOrigin(), 1.0f));
		glm::vec3 localRayDirection = glm::vec3(glm::inverse(M) * glm::vec4(ray.getDirection(), 0.0f)); 
		Ray localRay(localRayOrigin, localRayDirection);

		
		if(surface->hit(localRay, possibleIntersection, distanceFromOrigin)){
			
			surfaceIntersectionFound  = true;

			if( distanceFromOrigin < minDistanceFound ){
				//update distance
				minDistanceFound = distanceFromOrigin;
			
				intersectionPointWorldCoords  = M * glm::vec4( possibleIntersection.getPoint(), 1.0f);
				intersectionNormalWorldCoords = glm::transpose(glm::inverse(M)) * glm::vec4(possibleIntersection.getNormal(), 0.0f);
				
				distance = minDistanceFound;
				leafSurfaceIndex = i;
			}

		}
	}
	if(surfaceIntersectionFound){
		intersection.setPoint(glm::vec3(intersectionPointWorldCoords));
		intersection.setNormal(glm::normalize(glm::vec3(intersectionNormalWorldCoords)));
		intersection.setMaterial(surfaces[leafSurfaceIndex]->getMaterial());
	}	
	return surfaceIntersectionFound;
}

void BVH::buildTopDownSAH(BvhNode** tree, Surface** surfaces, int numSurfaces){

	int splitIndex;
	float costSplit;
	BvhNode* node = new BvhNode;
	*tree = node;

	node->aabb = this->computeBoxWithSurfaces(surfaces, numSurfaces);
	node->numSurfacesEncapulated = numSurfaces;

	std::cout << "Computing Split" << std::endl;
	splitIndex = topDownSplitIndexSAH(surfaces, numSurfaces, node->aabb, costSplit);
	std::cout << "Split Index Found: " << splitIndex << " With Cost :" << costSplit << std::endl;

	if( costSplit > node->numSurfacesEncapulated * COST_INTERSECTION){
		// create leaf.It's cheaper to intersect all surfaces than to make a split
		createLeaf(node, surfaces, numSurfaces);
	}
	else{

		node->type = BVH_NODE;
		node->tracedObject = NULL;
		node->tracedObjectArray = NULL;

		// recursion
		buildTopDownSAH(&(node->leftChild), &surfaces[0], splitIndex);
		buildTopDownSAH(&(node->rightChild),&surfaces[splitIndex], numSurfaces - splitIndex);
	}
}

int BVH::topDownSplitIndexSAH(Surface** surfaces, int numSurfaces, Box& parentBox, float& splitCost){

	float computedCost;
	float parentSurfaceArea;
	float minCost = FLT_MAX;
	int minCostSplit = 1;
	int dim;
	int nl,nr;	// left count children,right count children
	int i;
	Box leftChildBox;
	Box rightChildBox;
	float leftChildSurfaceArea,rightChildSurfaceArea;

	parentSurfaceArea = parentBox.computeSurfaceArea();
	for( dim = 0 ; dim < 3; dim++){

		/*
		if( dim == 0)
			std::sort(surfaces, surfaces + numSurfaces , by_X_compareSurfaces);
		else if( dim == 1)
			std::sort(surfaces, surfaces + numSurfaces , by_Y_compareSurfaces);
		else
			std::sort(surfaces, surfaces + numSurfaces , by_Z_compareSurfaces);

		*/

		for(i = 0; i < numSurfaces-1; i++){
			nl = i + 1;
			nr = numSurfaces - nl;

			leftChildBox  = this->computeBoxWithSurfaces(surfaces, nl);
			rightChildBox = this->computeBoxWithSurfaces(surfaces + nl, nr);

			leftChildSurfaceArea  = leftChildBox.computeSurfaceArea();
			rightChildSurfaceArea = rightChildBox.computeSurfaceArea();

			// SAH: C = CT + nl*CI* (S(Bl) / S(Bp)) + nr* CI * (S(Br) / S(Bp))  
			computedCost = COST_TRAVERSAL + nl * COST_INTERSECTION * leftChildSurfaceArea / parentSurfaceArea + nr * COST_INTERSECTION * rightChildSurfaceArea / parentSurfaceArea;
			if(computedCost < minCost){
				minCost = computedCost;
				minCostSplit = nl;
			}

		}
	}

	if( minCostSplit == 1)
		splitCost = FLT_MIN;	// create leaf 

	splitCost = minCost;
	return minCostSplit;
}