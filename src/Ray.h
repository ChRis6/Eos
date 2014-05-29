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

#ifndef _RAY_H
#define _RAY_H


#include <glm/glm.hpp>

class Ray{

public:
	Ray(glm::vec3 origin, glm::vec3 direction): m_Origin(origin), m_Direction(direction)
	{
		m_InvDirection = glm::vec3( 1.0 / direction.x, 1.0/ direction.y, 1.0/direction.z);
	}
	
	void setOrigin(glm::vec3 origin);
	void setDirection(glm::vec3 direction);

	glm::vec3 getOrigin() const;
	glm::vec3 getDirection() const;
	glm::vec3 getInvDirection() const;

private:
	glm::vec3 m_Origin;
	glm::vec3 m_Direction;
	glm::vec3 m_InvDirection;
};

#endif