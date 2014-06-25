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

class Scene{

public:

	bool addSurface(Surface* surface);			
	bool addLightSource(LightSource* light);   
	int getNumSurfaces() const;
	int getNumLightSources() const;

	const Surface* getSurface( unsigned int id) const;
	const LightSource* getLightSource(unsigned int id) const;

	void render(const Camera& camera, unsigned char* outputImage);

private:
	glm::vec4 rayTrace(const Ray& ray, const Camera& camera, int depth);
	bool findMinDistanceIntersection(const Ray& ray, RayIntersection& intersection);
	glm::vec4 calcPhong( const Camera& camera, const LightSource& lightSource, const RayIntersection& intersection);
	glm::vec4 findDiffuseColor(const LightSource& lightSource, const RayIntersection& intersection);
	glm::vec4 shadeIntersection(const RayIntersection& intersection, const Ray& ray, const Camera& camera);

private:
	std::vector<Surface*> m_SurfaceObjects;
	std::vector<LightSource*> m_LightSources; 
};

#endif