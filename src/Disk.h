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

#ifndef _DISK_H
#define _DISK_H

#include "Plane.h"

class Disk:public Plane{
private:
	typedef Plane inherited; // base class = Plane
public:
	Disk():inherited(),m_RadiusSquared(1.0f){}
	Disk(float radiusSquared):inherited(),m_RadiusSquared(radiusSquared){}
	Disk(float radiusSquared, const glm::vec3& diskPoint, const glm::vec3& diskNormal):inherited(diskPoint, diskNormal),m_RadiusSquared(radiusSquared){}

	virtual bool hit(const Ray& ray, RayIntersection& intersection, float& distance);

	void setDiskNormal(const glm::vec3& normal){ inherited::setPlaneNormal(normal);}
	void setDiskPoint(const glm::vec3& point)  { inherited::setPlanePoint(point);}
	void setDiskRadius(float radiusSquared)    { m_RadiusSquared = radiusSquared;}

	const glm::vec3& getDiskNormal(){return inherited::getPlaneNormal();}
	const glm::vec3& getDiskPoint() {return inherited::getPlanePoint();}
	float getDiskRadius()			{return m_RadiusSquared;}

private:
	float m_RadiusSquared;
};

#endif