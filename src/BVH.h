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
 

enum{
	BVH_NODE,
	BVH_LEAF
};

typedef struct bvhNode_t{
	char type;
	Box aabb;
	int numSurfacesEncapulated;
	struct bvhNode_t *rightChild;
	struct bvhNode_t *leftChild;
	Surface* tracedObject;
	Surface** tracedObjectArray;
}BvhNode;

/*
 * Bounding Volume Hierarchy
 * One surface per leaf
 */
class BVH{
public:
	BVH():m_Root(0){}

	void buildHierarchy(Surface** surfaces, int numSurfaces);								// builds BVH tree
	BvhNode* getRoot() const { return m_Root;}												// return Root of tree
	bool intersectRay(const Ray& ray, RayIntersection& intersectionFound, bool nearest) const;	// Return Intesected Surface with Ray.Get closest hit when nearest = true
	Surface* pointInsideSurface(glm::vec3& point);											// return surface that has point

private:
	Box computeBoxWithSurfaces(Surface** surfaces, int numSurfaces);

	void buildTopDown(BvhNode** tree, Surface** surfaces, int numSurfaces);
	int topDownSplitIndex(Surface** surfaces, int numSurfaces,Box parentBox);

	Surface* intersectRecursiveNearestHit(const Ray& ray, BvhNode* node, float& minDistance, RayIntersection& intersection) const;
	bool 	 intersectRayVisibilityTest(const Ray& ray, BvhNode* node) const;
	Surface* isPointInsideSurfaceRecursive(BvhNode* node, glm::vec3& point);
	bool    intersectRayWithLocalSurface(const Ray& ray, Surface* surface, RayIntersection& intersection, float& distance);

	// SAH
	void buildTopDownSAH(BvhNode** tree, Surface** surfaces, int numSurfaces);
	int topDownSplitIndexSAH(Surface** surfaces, int numSurfaces, Box& parentBox, float& splitCost);	// returns best split index, sets splitCost - cost of split index returned
	void createLeaf(BvhNode* newNode, Surface** surfaces, int numSurfaces);
	bool intersectRayWithLeaf(const Ray& ray, BvhNode* leaf, RayIntersection& intersection, float& distance, int& leafSurfaceIndex) const;


	void buildTopDownHybrid(BvhNode** tree, Surface** surfaces, int numSurfaces);

	bool intersectStack(const Ray& ray, BvhNode* root, RayIntersection& intersection);
private:
	BvhNode* m_Root;
};

#endif