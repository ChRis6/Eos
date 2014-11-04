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

#ifndef _CAMERA_H
#define _CAMERA_H

#include <glm/glm.hpp>
#include "cudaQualifiers.h"
#include "Ray.h"

class Camera {

public:
	HOST DEVICE Camera(): m_Position(0.0f),m_ViewingDirection(0.0f),m_RightVector(0.0f),m_UpVector(0.0f),m_Width(0),m_Height(0)
	{}


	Ray computeRayFromPixel(int i, int j);


	HOST DEVICE void setPosition(glm::vec3 pos);
	HOST DEVICE void setViewingDirection(glm::vec3 dir);
	HOST DEVICE void setRightVector(glm::vec3 right);
	HOST DEVICE void setUpVector(glm::vec3 up);
	HOST DEVICE void setWidth(int width);
	HOST DEVICE void setHeight(int height);

	HOST DEVICE const glm::vec3& getPosition() const;
	HOST DEVICE const glm::vec3& getViewingDirection() const;
	HOST DEVICE const glm::vec3& getRightVector() const;
	HOST DEVICE const glm::vec3& getUpVector() const;
	HOST DEVICE int getWidth() const;
	HOST DEVICE int getHeight() const;


private:
	// camera position and orientation
	glm::vec3 m_Position;
	glm::vec3 m_ViewingDirection;
	glm::vec3 m_RightVector;
	glm::vec3 m_UpVector;

	int m_Width;	// width  in pixels
	int m_Height;   // height in pixels
};

#endif 