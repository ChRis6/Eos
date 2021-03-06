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

#ifndef _SCENE_H
#define _SCENE_H

#include <vector>
#include "surface.h"
#include "LightSource.h"
#include "Ray.h"
#include "Camera.h"
#include "BVH.h"
#include "triangleMesh.h"
#include "Material.h"

#define SCENE_AA_32   32
#define SCENE_AA_16   16
#define SCENE_AA_8    8
#define SCENE_AA_4    4
#define SCENE_AA_2    2
#define SCENE_AA_1    1

class Scene{

public:

	void printFirstTriangles(int n);

	bool addSurface(Surface* surface);
	bool addTriangleMesh(TriangleMesh* mesh);			
	bool addLightSource(LightSource* light);
	int addMaterial(Material material);
	int addTransformation(const glm::mat4& trans);
	const Material& getMaterialAtIndex(int index) const;
	const glm::mat4& getTransformationAtIndex(int index)const { return m_Transformations[index];}
	const glm::mat4& getInverseTransformationAtIndex(int index)const { return m_InverseTransformations[index];}
	const glm::mat4& getInverseTransposeTransformationAtIndex(int index)const { return m_InverseTransposeTransformations[index];}
	int getNumMaterials() const { return m_Materials.size();}

	int getNumSurfaces() const;
	int getNumTransformations()const{ return m_Transformations.size();} // 3 transformations: local->world, inverse, inverse-transpose
	int getNumLightSources() const;
	float getAmbientRefractiveIndex() const;
	void setAASamples(int samples){ m_AASamples = samples; }
	int getAASamples(){ return m_AASamples; }




	bool findMinDistanceIntersectionLinear(const Ray& ray, RayIntersection& intersection) const;
	bool findMinDistanceIntersectionBVH(const Ray& ray, RayIntersection& intersection) const;
	bool shadowRayVisibilityBVH(const Ray& ray) const;

	void setAmbientRefractiveIndex(float refractiveIndex);

	Surface* getSurface( int id) const;
	const LightSource* getLightSource( int id) const;
	Surface** getFirstSurfaceAddress()const	{ return (Surface**) &(m_SurfaceObjects[0]);}

	void render(const Camera& camera, unsigned char* outputImage);
	
	void setMaxTracedDepth(int depth) { m_MaxTracedDepth = depth;}
	int getMaxTracedDepth() const     {return m_MaxTracedDepth;}

	void useBvh(bool use) { m_UsingBvh = use;}
	bool isUsingBvh() const {return m_UsingBvh;}
	void flush();

	void printScene();
	const BVH& getBVH() const	{ return m_Bvh;}

private:
	/*
	glm::vec4 rayTrace(const Ray& ray, const Camera& camera, float sourceRefactionIndex, int depth);
	glm::vec4 calcPhong( const Camera& camera, const LightSource& lightSource, RayIntersection& intersection);
	glm::vec4 findDiffuseColor(const LightSource& lightSource, const glm::vec4& intersectionToLight, const RayIntersection& intersection);
	glm::vec4 shadeIntersection(RayIntersection& intersection, const Ray& ray, const Camera& camera, float sourceRefactionIndex, int depth);
	float fresnel(const glm::vec3& incident, const glm::vec3& normal, float n1, float n2);
	float slick(const glm::vec3& incident, const glm::vec3& normal, float n1, float n2);
*/
	glm::vec3 getReflectedRay(const glm::vec3& rayDir, const glm::vec3& normal);
	glm::vec3 getRefractedRay(const glm::vec3& rayDir, const glm::vec3& normal, float n1, float n2);

private:
	std::vector<Surface*> m_SurfaceObjects;
	std::vector<LightSource*> m_LightSources;
	std::vector<Material>    m_Materials;
	std::vector<glm::mat4>   m_Transformations;
	std::vector<glm::mat4>   m_InverseTransformations;
	std::vector<glm::mat4>   m_InverseTransposeTransformations;
	int m_MaxTracedDepth;
	float m_AmbientRefractiveIndex; 

	// bounding volume hierarchy
	BVH m_Bvh;
	bool m_UsingBvh;

	int m_AASamples;
};

#endif