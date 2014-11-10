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

#ifndef _RAYTRACER_H
#define _RAYTRACER_H

#include <glm/glm.hpp>

#include "Camera.h"
#include "Scene.h"

#define RAYTRACER_AA_SAMPLES_8   8
#define RAYTRACER_AA_SAMPLES_4   4
#define RAYTRACER_AA_SAMPLES_2   2
#define RAYTRACER_NO_AA          0

#define RAYTRACER_DEFAULT_TRACE_DEPTH   4

class RayTracer{

public:
	RayTracer():m_AASamples(RAYTRACER_NO_AA), m_TraceDepth(RAYTRACER_DEFAULT_TRACE_DEPTH) {}

	void render(const Scene& scene, const Camera& camera, unsigned char* outputImage);


	// setters - getters
	int getAASamples() const;
	void setAASamples(int samples);

	int getTracedDepth() const;
	void setTracedDepth(int);

private:
	void renderWithoutAA(const Scene& scene, const Camera& camera, unsigned char* outputImage);
	void renderWithAA(const Scene& scene, const Camera& camera, unsigned char* outputImage);

	glm::vec4 rayTrace(const Scene& scene, const Camera& camera, const Ray& ray,  int depth);
	glm::vec4 shadeIntersection(const Scene& scene, const Ray& ray, const Camera& camera, RayIntersection& intersection, int depth);
	glm::vec4 calcPhong( const Scene& scene, const Camera& camera, const LightSource* lightSource, RayIntersection& intersection);
	glm::vec4 findDiffuseColor( const Scene& scene, const LightSource* lightSource, const glm::vec4& intersectionToLight, const RayIntersection& intersection);

	float fresnel(const glm::vec3& incident, const glm::vec3& normal, float n1, float n2);
	float slick(const glm::vec3& incident, const glm::vec3& normal, float n1, float n2);

private:
	int m_AASamples;
	int m_TraceDepth;
};

#endif