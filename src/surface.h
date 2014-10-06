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

#ifndef _SURFACE_H
#define _SURFACE_H

#include "Ray.h"
#include "RayIntersection.h"
#include "Box.h"
#include "Material.h"

class Surface{

public:
	virtual bool hit(const Ray& ray, RayIntersection& intersection, float& distance) = 0;
	virtual Box getLocalBoundingBox() = 0;
	virtual glm::vec3 getCentroid() = 0;
	virtual const glm::mat4& transformation() = 0;
	virtual void setTransformation(glm::mat4& transformation) = 0;
	virtual Material& getMaterial() = 0;
	virtual void setMaterial(Material& material) = 0;    
};


#endif