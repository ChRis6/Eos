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

#ifndef _BVH_H
#define _BVH_H

#include "Box.h"
#include "surface.h"
#include "Ray.h"
#include "RayIntersection.h"
#include <vector>

#define SURFACES_PER_LEAF 4

enum{
	BVH_NODE = 0,
	BVH_LEAF
};

typedef struct bvhNode_t{
	int type;
	Box aabb;
	int numSurfacesEncapulated;
	struct bvhNode_t *rightChild;
	struct bvhNode_t *leftChild;
	int leftChildIndex;
	int rightChildIndex;
	int surfacesIndices[SURFACES_PER_LEAF];
	int splitAxis;
}BvhNode;

/*
 * Bounding Volume Hierarchy
 * 
 */
class BVH{
public:
	BVH():m_Root(0){}

	void buildHierarchy(Surface** surfaces, int numSurfaces);								// builds BVH tree
	BvhNode* getRoot() const { return m_Root;}												// return Root of tree
	bool intersectRay(const Ray& ray, RayIntersection& intersectionFound, bool nearest, Surface** surfaces) const;	// Return Intesected Surface with Ray.Get closest hit when nearest = true
												// return surface that has point

	BvhNode* getNodesBuffer() const	{ return m_NodesBuffer;}
	int getNodesBufferSize() const 	{ return m_FlatTreePointers.size();}
private:
	Box computeBoxWithSurfaces(Surface** surfaces, int start , int end);
	Box computeBoxWithCentroids(Surface** surfaces, int start , int end);

	void buildTopDown(BvhNode** tree, Surface** surfaces, int start, int end);
	int topDownSplitIndex(Surface** surfaces, Box parentBox, int start, int end, int* splitAxis);

	Surface* intersectRecursiveNearestHit(const Ray& ray, BvhNode* node, float& minDistance, RayIntersection& intersection, Surface** surfaces, int depth) const;
	bool 	 intersectRayVisibilityTest(const Ray& ray, BvhNode* node, Surface** surfaces) const;
	
	

	// SAH
	void buildTopDownSAH(BvhNode** tree, Surface** surfaces, int start, int end);
	int topDownSplitIndexSAH(Surface** surfaces, Box& parentBox, float& splitCost, int start, int end, int* splitAxis);	// returns best split index, sets splitCost - cost of split index returned
	void createLeaf(BvhNode* newNode, Surface** surfaces, int start, int end);
	bool intersectRayWithLeaf(const Ray& ray, BvhNode* leaf, RayIntersection& intersection, float& distance, int& leafSurfaceIndex, Surface** surfaces) const;


	void buildTopDownHybrid(BvhNode** tree, Surface** surfaces, int start, int end);
	bool intersectStackNearest(const Ray& ray, BvhNode* root, RayIntersection& intersection, Surface** surfaces) const;
	bool intersectStackVisibility(const Ray& ray, BvhNode* root, Surface** surfaces) const;
	
	void makeTreeFlat(BvhNode* node, int nodeIndex, std::vector<BvhNode*>& array);
	void copyFlatToBuffer();
	void deallocateTreePointers(BvhNode* node);
private:
	BvhNode* m_Root;
	std::vector<BvhNode*> m_FlatTreePointers;
	BvhNode*	m_NodesBuffer;	
};

#endif