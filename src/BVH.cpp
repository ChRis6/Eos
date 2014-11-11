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
 #include <stack>
#include <cfloat>
#include "BVH.h"

#define SAH_SURFACE_CEIL  800
#define COST_TRAVERSAL    5
#define COST_INTERSECTION 10


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
	// build
	this->buildTopDownHybrid(&m_Root, surfaces, 0, numSurfaces);
	// flat
	
	m_FlatTreePointers.reserve(1000);
	m_FlatTreePointers.push_back(m_Root);
	this->makeTreeFlat(m_Root, 0, m_FlatTreePointers);
	this->copyFlatToBuffer();

	this->deallocateTreePointers(this->getRoot());
}

void BVH::buildTopDown(BvhNode** tree, Surface** surfaces, int start, int end){

	BvhNode* node = new BvhNode;
	*tree = node;

	// find bounding box of surfaces
	node->aabb = this->computeBoxWithSurfaces(surfaces, start, end);
	node->numSurfacesEncapulated = end - start;

	// is it a leaf ? Only one surface per leaf 
	if( end - start < SURFACES_PER_LEAF ){
		/*
		node->type = BVH_LEAF;
		node->leftChild = NULL;
		node->rightChild = NULL;
		node->tracedObject = surfaces[0]; 	// get the first surface
		*/
		createLeaf(node, surfaces, start, end );
	}
	else{

		node->type = BVH_NODE;
		int splitIndex = this->topDownSplitIndex(surfaces, node->aabb, start, end);

		// recursion
		buildTopDown( &(node->leftChild), surfaces, start, splitIndex );
		buildTopDown( &(node->rightChild), surfaces, splitIndex, end);
	}
}

Box BVH::computeBoxWithSurfaces(Surface** surfaces, int start, int end){

	Box computedBox = surfaces[start]->getLocalBoundingBox();
	computedBox.transformBoundingBox(surfaces[start]->transformation());

	for( int i = start + 1 ; i < end; i++){
		// get the bounding box in local surface coordinates
		Box surfaceBox = surfaces[i]->getLocalBoundingBox();
		// transform box in world coords
		surfaceBox.transformBoundingBox(surfaces[i]->transformation());
		// expand resulting bounding box
		computedBox.expandToIncludeBox(surfaceBox);
	}

	return computedBox;
}

int BVH::topDownSplitIndex(Surface** surfaces, Box parentBox, int start, int end){

	int split = 0;
	int biggestIndex;
	int i;
	float midAxisCoord;
	Box centroidsBox;

	for( i = start ; i < end; i++){
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
		std::sort(surfaces + start, surfaces + end, by_X_compareSurfaces);
	else if( biggestIndex == 1)
		std::sort(surfaces + start, surfaces + end, by_Y_compareSurfaces);
	else if( biggestIndex == 2)
		std::sort(surfaces + start, surfaces + end, by_Z_compareSurfaces);


	// find mid split
	for(i = start ; i < end; i++){
		glm::vec4 surfaceCentroid = glm::vec4(surfaces[i]->getCentroid(),1.0f);
		surfaceCentroid = surfaces[i]->transformation() * surfaceCentroid;
		//std::cout << "X = " << surfaceCentroid.x << std::endl;

		if( surfaceCentroid[biggestIndex] > midAxisCoord  ){
			
			break;
		}
	}

	split = i;
	if( split == start || split == end){ // bad split ?
		split = start + ( (end - start) / 2 );
	}
	
	return split;
}

bool BVH::intersectRay(const Ray& ray, RayIntersection& intersectionFound, bool nearest, Surface** surfaces) const{
	
	
	Surface* intersectedSurface = NULL;
	float distance;
	if(nearest)
		//intersectedSurface = this->intersectRecursiveNearestHit(ray, this->getRoot(), distance, intersectionFound, surfaces, 0);
		return  intersectStackNearest(ray, this->getRoot(), intersectionFound, surfaces);
	else
		return intersectStackVisibility(ray, this->getRoot(), surfaces);
		
}

Surface* BVH::intersectRecursiveNearestHit(const Ray& ray, BvhNode* node, float& minDistance, RayIntersection& intersection, Surface** surfaces, int depth) const{
	
	bool rayIntersectsBox = false;
	float distance = 999999.0f;
	int minSurfaceIndex;
	static int maxDepthFound = 0;
	
	if( depth > maxDepthFound){
		//std::cout << "Biggest depth = " << depth << std::endl;
		maxDepthFound = depth;
	}

	if( node->aabb.intersectWithRayOptimized(ray, 0.0001f, 999999.0f) == true)
		rayIntersectsBox = true;

	if(rayIntersectsBox){
		// is it a leaf ?
		if( node->type == BVH_LEAF){
			RayIntersection dummyIntersection;
			if( this->intersectRayWithLeaf( ray, node, dummyIntersection, minDistance, minSurfaceIndex, surfaces)){
				intersection = dummyIntersection;
				return surfaces[minSurfaceIndex];
			}
			return NULL;
		}
		
		// No,its an intermediate node
		float leftChildDistance  = 999999.0f;
		float rightChildDistance = 999999.0f;
		RayIntersection leftChildeIntersection;
		RayIntersection rightChildIntersection;

		Surface* leftChildSurface  = this->intersectRecursiveNearestHit(ray, node->leftChild, leftChildDistance, leftChildeIntersection, surfaces, depth + 1);
		Surface* rightChildSurface = this->intersectRecursiveNearestHit(ray, node->rightChild, rightChildDistance, rightChildIntersection, surfaces, depth +1);


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

bool BVH::intersectRayVisibilityTest(const Ray& ray, BvhNode* node, Surface** surfaces) const{

	bool intersectsCurrent;
	bool intersectsLeftChild;
	bool intersectsRightChild;
	bool leftChildSurfaceIntersected = false;
	bool rightChildSurfaceIntersected = false;

	intersectsCurrent = node->aabb.intersectWithRayOptimized(ray, 0.0001f, 999999.0f);
	if( intersectsCurrent){
		if( node->type == BVH_LEAF ){
			RayIntersection dummyIntersection;
			float minDistance;
			int minSurfaceIndex;
			return this->intersectRayWithLeaf( ray, node, dummyIntersection, minDistance, minSurfaceIndex, surfaces);				
		}

		leftChildSurfaceIntersected = this->intersectRayVisibilityTest(ray, node->leftChild, surfaces);
		if( leftChildSurfaceIntersected)
			return true;

		rightChildSurfaceIntersected = this->intersectRayVisibilityTest(ray, node->rightChild, surfaces);
		if( rightChildSurfaceIntersected)
			return true;

		// none of the children are intersected
		return false;
	}
	else{
		return false;
	}
}



void BVH::createLeaf(BvhNode* newNode, Surface** surfaces, int start, int end){

	
	newNode->type = BVH_LEAF;
	newNode->numSurfacesEncapulated = end - start;
	newNode->leftChild = NULL;
	newNode->rightChild = NULL;

	
	int i;
	for( i = 0; i < (end - start); i++){
		newNode->surfacesIndices[i] = i + start;
	}


}

bool BVH::intersectRayWithLeaf(const Ray& ray, BvhNode* leaf, RayIntersection& intersection, float& distance, int& leafSurfaceIndex, Surface** surfaces) const{

	int i;
	bool surfaceIntersectionFound = false;
	float distanceFromOrigin = 9999999.0f;
	float minDistanceFound = 99999999.0f;

	RayIntersection possibleIntersection;
	glm::vec4 intersectionPointWorldCoords;
	glm::vec4 intersectionNormalWorldCoords;
	Ray localRay;

	int numSurfaces = leaf->numSurfacesEncapulated;
	

	for( i = 0 ; i < numSurfaces; i++){
		int surfaceIndex = leaf->surfacesIndices[i];
		Surface* surface = surfaces[surfaceIndex];

		// Transform ray to local coordinates
		const glm::mat4& M = surface->transformation();
		glm::vec3 localRayOrigin    = glm::vec3(surface->getInverseTransformation() * glm::vec4(ray.getOrigin(), 1.0f));
		glm::vec3 localRayDirection = glm::vec3(surface->getInverseTransformation() * glm::vec4(ray.getDirection(), 0.0f)); 
		localRay.setOrigin(localRayOrigin);
		localRay.setDirection(localRayDirection);

		
		if(surface->hit(localRay, intersection, distance)){
			
			surfaceIntersectionFound  = true;
			intersectionPointWorldCoords  = M * glm::vec4( intersection.getPoint(), 1.0f);
			intersectionNormalWorldCoords = surface->getInverseTransposeTransformation() * glm::vec4(intersection.getNormal(), 0.0f);
					
			leafSurfaceIndex = surfaceIndex;
		}
	}
	if(surfaceIntersectionFound){
		intersection.setPoint(glm::vec3(intersectionPointWorldCoords));
		intersection.setNormal(glm::normalize(glm::vec3(intersectionNormalWorldCoords)));
	}	
	return surfaceIntersectionFound;
}

void BVH::buildTopDownSAH(BvhNode** tree, Surface** surfaces, int start, int end){

	int splitIndex;
	float costSplit;
	BvhNode* node = new BvhNode;
	*tree = node;

	node->aabb = this->computeBoxWithSurfaces(surfaces, start, end);
	node->numSurfacesEncapulated = end - start;

	splitIndex = topDownSplitIndexSAH(surfaces, node->aabb, costSplit, start, end);
	
	if( costSplit > node->numSurfacesEncapulated * COST_INTERSECTION){
		// create leaf.It's cheaper to intersect all surfaces than to make a split
		createLeaf(node, surfaces, start, end);
	}
	else{

		node->type = BVH_NODE;

		// recursion
		buildTopDownSAH(&(node->leftChild), surfaces, start, splitIndex);
		buildTopDownSAH(&(node->rightChild),surfaces, splitIndex, end);
	}
}

int BVH::topDownSplitIndexSAH(Surface** surfaces,  Box& parentBox, float& splitCost, int start, int end){

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

		
		if( dim == 0)
			std::sort(surfaces + start, surfaces + end , by_X_compareSurfaces);
		else if( dim == 1)
			std::sort(surfaces + start, surfaces + end , by_Y_compareSurfaces);
		else
			std::sort(surfaces + start, surfaces + end , by_Z_compareSurfaces);


		for(i = start ; i < end - 1; i++){
			nl = i + 1;
			
			leftChildBox  = this->computeBoxWithSurfaces(surfaces, start, nl);
			rightChildBox = this->computeBoxWithSurfaces(surfaces, nl, end);

			leftChildSurfaceArea  = leftChildBox.computeSurfaceArea();
			rightChildSurfaceArea = rightChildBox.computeSurfaceArea();

			// SAH: C = CT + nl*CI* (S(Bl) / S(Bp)) + nr* CI * (S(Br) / S(Bp))  
			computedCost = COST_TRAVERSAL + (nl - start) * COST_INTERSECTION * leftChildSurfaceArea / parentSurfaceArea + (end - nl) * COST_INTERSECTION * rightChildSurfaceArea / parentSurfaceArea;
			if(computedCost < minCost){
				minCost = computedCost;
				minCostSplit = nl;
			}
		}
	}
 

	splitCost = minCost;

	if( minCostSplit == start)
		splitCost = FLT_MAX;	// create leaf

	return minCostSplit;
}

void BVH::buildTopDownHybrid(BvhNode** tree, Surface** surfaces, int start, int end){

	int splitIndex;
	float costSplit;
	BvhNode* node = new BvhNode;
	*tree = node;

	node->aabb = this->computeBoxWithSurfaces(surfaces,  start,  end);
	node->numSurfacesEncapulated = end - start;

	if( end - start > SAH_SURFACE_CEIL ){

		// is it a leaf ? Only one surface per leaf 
		if( end - start < SURFACES_PER_LEAF ){
			createLeaf(node, surfaces, start, end);
		}
		else{

			node->type = BVH_NODE;
			splitIndex = this->topDownSplitIndex(surfaces, node->aabb, start, end);

			//std::cout << "median surface index = " << splitIndex << " Start = " << start << " End = " << end<< std::endl;

			// recursion
			buildTopDownHybrid( &(node->leftChild), surfaces, start, splitIndex);
			buildTopDownHybrid( &(node->rightChild),surfaces, splitIndex, end);
		}
	}
	else{
		// use SAH

		if( end - start < SURFACES_PER_LEAF){		
			createLeaf(node, surfaces, start, end);
		}
		else{
			node->type = BVH_NODE;
			splitIndex = topDownSplitIndexSAH(surfaces, node->aabb, costSplit, start, end);
			//std::cout << "SAH split index = " << splitIndex  << " Start = " << start << " End = " << end << std::endl;

			// recursion
			buildTopDownHybrid(&(node->leftChild), surfaces, start, splitIndex);
			buildTopDownHybrid(&(node->rightChild), surfaces, splitIndex, end);
		}
	}
}


bool BVH::intersectStackNearest(const Ray& ray, BvhNode* root, RayIntersection& intersection, Surface** surfaces) const {

	BvhNode* stack[1024];
	BvhNode** stack_ptr = stack;
	
	float minDistace = FLT_MAX;
	bool surfaceIntersectionFound = false;
	BvhNode* currNode = &m_NodesBuffer[0];

	if( currNode->aabb.intersectWithRayOptimized(ray, 0.0001f, 999999.0f) == false )
		return false;

	// push null
	*stack_ptr++ = NULL;

	while(currNode != NULL){

		if( currNode->type == BVH_NODE ){
			//BvhNode* leftChild  = currNode->leftChild;
			//BvhNode* rightChild = currNode->rightChild;

			int leftIndex = currNode->leftChildIndex;
			int rightIndex = currNode->rightChildIndex;

			//BvhNode* leftChild  = m_FlatTreePointers[leftIndex];	
			//BvhNode* rightChild = m_FlatTreePointers[rightIndex];

			BvhNode* leftChild  = &m_NodesBuffer[leftIndex];	
			BvhNode* rightChild = &m_NodesBuffer[rightIndex];

			bool leftChildIntersected = leftChild->aabb.intersectWithRayOptimized(ray, 0.0001f, 999999.0f);
			bool rightChildIntersected = rightChild->aabb.intersectWithRayOptimized(ray, 0.0001f, 999999.0f);

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
			
			
			int leafSurfaceIndex;
			bool leafIntersected = this->intersectRayWithLeaf(ray, currNode, intersection, minDistace, leafSurfaceIndex, surfaces);
			if( leafIntersected )
				surfaceIntersectionFound = true;
			
			// pop 
			currNode = *--stack_ptr;
		}
	}
	return surfaceIntersectionFound;
}

bool BVH::intersectStackVisibility(const Ray& ray, BvhNode* root, Surface** surfaces) const{

	BvhNode* stack[1024];
	BvhNode** stack_ptr = stack;
	BvhNode* currNode;

	currNode = &m_NodesBuffer[0];
	if( currNode->aabb.intersectWithRayOptimized(ray, 0.0001f, 999999.0f) == false )
		return false;

	// push null
	*stack_ptr++ = NULL;
	
	while(currNode != NULL){

		if( currNode->type == BVH_NODE ){

			BvhNode* leftChild  =  &m_NodesBuffer[currNode->leftChildIndex];
			BvhNode* rightChild =  &m_NodesBuffer[currNode->rightChildIndex];

			bool leftChildIntersected = leftChild->aabb.intersectWithRayOptimized(ray, 0.0001f, 999999.0f);
			bool rightChildIntersected = rightChild->aabb.intersectWithRayOptimized(ray, 0.0001f, 999999.0f);

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
			float distance;
			int leafSurfaceIndex;
			RayIntersection dummyIntersection;

			bool leafIntersected = this->intersectRayWithLeaf(ray, currNode, dummyIntersection, distance, leafSurfaceIndex, surfaces);
			if( leafIntersected )
				return true;
			
			// pop 
			currNode = *--stack_ptr;
		}
	}
	return false;
}

void BVH::makeTreeFlat(BvhNode* node, int nodeIndex, std::vector<BvhNode*>& array){
	int left;
	int right;
	if(node == NULL)
		return;


	array.push_back(node->leftChild);
	left = array.size() - 1;
	

	array.push_back(node->rightChild);
	right = array.size() - 1;
	

	node->leftChildIndex = left;
	node->rightChildIndex = right;

	makeTreeFlat(node->leftChild, left, array);
	makeTreeFlat(node->rightChild, right, array);

}

void BVH::copyFlatToBuffer(){
	int numObjects;

	numObjects = m_FlatTreePointers.size();

	m_NodesBuffer = new BvhNode[numObjects];
	for( int i = 0; i < numObjects; i++){

		if( m_FlatTreePointers[i] != NULL)
			m_NodesBuffer[i] = *m_FlatTreePointers[i];
			
	}

}

void BVH::deallocateTreePointers(BvhNode* node){
	if( node == NULL)
		return;
	deallocateTreePointers(node->leftChild);
	deallocateTreePointers(node->rightChild);

	delete(node);
}