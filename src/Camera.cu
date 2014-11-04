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

#include "Camera.h"


HOST DEVICE void Camera::setPosition(glm::vec3 pos){
	m_Position = pos;
}
HOST DEVICE void Camera::setViewingDirection(glm::vec3 dir){
	m_ViewingDirection = dir;
}
HOST DEVICE void Camera::setRightVector(glm::vec3 right){
	m_RightVector = right;
}
HOST DEVICE void Camera::setUpVector(glm::vec3 up){
	m_UpVector = up;
}

HOST DEVICE void Camera::setWidth(int width){
	m_Width = width;
}
HOST DEVICE void Camera::setHeight(int height){
	m_Height = height;
}


HOST DEVICE const glm::vec3& Camera::getPosition() const{
	return m_Position;
}
HOST DEVICE const glm::vec3& Camera::getViewingDirection() const{
	return m_ViewingDirection;
}

HOST DEVICE const glm::vec3& Camera::getRightVector() const{
	return m_RightVector;
}
HOST DEVICE const glm::vec3& Camera::getUpVector() const{
	return m_UpVector;
}

HOST DEVICE int Camera::getWidth() const{
	return m_Width;
}
HOST DEVICE int Camera::getHeight() const{
	return m_Height;
}
