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


void Camera::setPosition(glm::vec3 pos){
	m_Position = pos;
}
void Camera::setViewingDirection(glm::vec3 dir){
	m_ViewingDirection = dir;
}
void Camera::setRightVector(glm::vec3 right){
	m_RightVector = right;
}
void Camera::setUpVector(glm::vec3 up){
	m_UpVector = up;
}
void Camera::setFov(float fov){
	m_Fov = fov;
	m_TanFov = tanf(fov);
}
void Camera::setWidth(int width){
	m_Width = width;
}
void Camera::setHeight(int height){
	m_Height = height;
}


glm::vec3 Camera::getPosition() const{
	return m_Position;
}
glm::vec3 Camera::getViewingDirection() const{
	return m_ViewingDirection;
}
glm::vec3 Camera::getRightVector() const{
	return m_RightVector;
}
glm::vec3 Camera::getUpVector() const{
	return m_UpVector;
}
float Camera::getFov() const{
	return m_Fov;
}
float Camera::getTanFov() const{
	return m_TanFov;
}
float Camera::getWidth() const{
	return m_Width;
}
float Camera::getHeight() const{
	return m_Height;
}

/*
Ray Camera::computeRayFromPixel(int x, int y){


    float aa = m_TanFov * ( ( x - m_Width /2.0) / (float)( m_Width / 2.0 ) );
    float bb = m_TanFov * ( (y - (m_Height/2.0)) / (float)( m_Height/ 2.0) );

    //double aa = ( ( (2*x + 0.5) / (double)WINDOW_WIDTH) - 1);
    //double bb = (1 - ((2*y + 0.5) / (double) WINDOW_HEIGHT) );

    //glm::vec4 rayDirection = (aa * glm::vec4(1.0, 0.0, 0.0, 0.0) ) + ( bb * glm::vec4(0.0 , 1.0, 0.0, 0.0) ) + glm::vec4( 0.0, 0.0, -1.0, 0.0);
    //glm::vec4 rayOrigin(0.0, 0.0, 0.0, 1.0);
	
	//return Ray();
}*/
    