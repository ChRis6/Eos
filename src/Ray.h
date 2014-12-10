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
#include <glm/gtx/type_aligned.hpp>

#include "cudaQualifiers.h"

class Ray{

public:
	DEVICE HOST Ray():m_Origin(0.0f),m_Direction(0.0f),m_InvDirection(0.0f){}

	DEVICE HOST Ray(const glm::vec3& origin, const glm::vec3& direction): m_Origin(origin), m_Direction(direction)
	{
		m_InvDirection = glm::vec3( 1.0 / direction.x, 1.0/ direction.y, 1.0/direction.z);
		m_sign[0] = m_InvDirection.x < 0.0f;
		m_sign[1] = m_InvDirection.y < 0.0f;
		m_sign[2] = m_InvDirection.z < 0.0f;
	}

	DEVICE HOST inline void setOrigin(const glm::vec3& origin) { m_Origin = origin; }
	DEVICE HOST inline void setDirection(const glm::vec3& direction) {
		m_Direction = direction;
		m_InvDirection = glm::vec3( 1.0f/ direction.x, 1.0f/direction.y, 1.0/direction.z);

		m_sign[0] = m_InvDirection.x < 0.0f;
		m_sign[1] = m_InvDirection.y < 0.0f;
		m_sign[2] = m_InvDirection.z < 0.0f;
	}

	DEVICE HOST inline const glm::vec3& getOrigin() const { return m_Origin;}
	DEVICE HOST inline const glm::vec3& getDirection() const { return m_Direction;}
	DEVICE HOST inline const glm::vec3& getInvDirection() const { return m_InvDirection;}

private:
	glm::vec3 m_Origin;
	glm::vec3 m_Direction;
	glm::vec3 m_InvDirection;
public:

	int m_sign[3];				// sign for evey dimension
};

class cudaRay{

public:
	DEVICE HOST cudaRay():m_Origin(0.0f),m_Direction(0.0f),m_InvDirection(0.0f){}

	DEVICE HOST cudaRay(const glm::aligned_vec3& origin, const glm::aligned_vec3& direction): m_Origin(origin), m_Direction(direction)
	{
		m_InvDirection = glm::aligned_vec3( 1.0 / direction.x, 1.0/ direction.y, 1.0/direction.z);

		sign[0] = m_InvDirection.x < 0.0f;
		sign[1] = m_InvDirection.y < 0.0f;
		sign[2] = m_InvDirection.z < 0.0f;
	}

	DEVICE inline void setOrigin(const glm::aligned_vec3& origin) { m_Origin = origin; }
	DEVICE inline void setDirection(const glm::aligned_vec3& direction) {
		m_Direction = direction;
		m_InvDirection = glm::aligned_vec3( 1.0f/ direction.x, 1.0f/direction.y, 1.0/direction.z);

		sign[0] = m_InvDirection.x < 0.0f;
		sign[1] = m_InvDirection.y < 0.0f;
		sign[2] = m_InvDirection.z < 0.0f;
	}

	DEVICE inline const glm::aligned_vec3& getOrigin() const { return m_Origin;}
	DEVICE inline const glm::aligned_vec3& getDirection() const { return m_Direction;}
	DEVICE inline const glm::aligned_vec3& getInvDirection() const { return m_InvDirection;}

private:
	glm::aligned_vec3 m_Origin;
	glm::aligned_vec3 m_Direction;
	glm::aligned_vec3 m_InvDirection;
public:
	int sign[3];
};

#endif