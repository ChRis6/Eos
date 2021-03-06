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

#include "Ray.h"

/*
DEVICE HOST void Ray::setOrigin(const glm::vec3& origin){
	m_Origin = origin;
}

DEVICE HOST void Ray::setDirection(const glm::vec3& direction) {
	m_Direction = direction;
	m_InvDirection = glm::vec3( 1.0f/ direction.x, 1.0f/direction.y, 1.0/direction.z);

	m_sign[0] = m_InvDirection.x < 0.0f;
	m_sign[1] = m_InvDirection.y < 0.0f;
	m_sign[2] = m_InvDirection.z < 0.0f;
}

DEVICE HOST const glm::vec3& Ray::getOrigin() const{
	return m_Origin;
}

DEVICE HOST const glm::vec3& Ray::getDirection() const{
	return m_Direction;
}

DEVICE HOST const glm::vec3& Ray::getInvDirection() const{
	return m_InvDirection;
}
*/