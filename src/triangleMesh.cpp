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

/*
 * ONLY ONE MESH PER FILE
 */

template <class T> T MAX (const T& a, const T& b) {
  return (a<b)?b:a;     
}

template <class T> T MIN (const T& a, const T& b) {
  return (a > b)?b:a;     
}

#include <assimp/cimport.h>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#include <iostream>
#include "triangleMesh.h"
#include "Ray.h"

#ifndef EPSILON
#define EPSILON        1e-6
#endif

TriangleMesh::TriangleMesh(){
	m_Vertices    = 0;
	m_Indices     = 0;
	m_NumVertices = 0;
	m_NumIndices  = 0;

	m_BoxMin = glm::vec3(0.0f);
	m_BoxMax = glm::vec3(0.0f);
	m_LocalToWorldTransformation = glm::mat4(1.0f);
	m_Valid       = false;
}

TriangleMesh::~TriangleMesh(){
	if(m_Valid){
		delete m_Vertices;
		delete m_Indices;
		delete m_Normals;
	}
		
}

unsigned int TriangleMesh::getNumTriangles(){
	return m_NumIndices / 3;
}

unsigned int TriangleMesh::getNumVertices(){
	return m_NumVertices;
}

const Material& TriangleMesh::getMaterial(){
	return m_Material;
}

void TriangleMesh::setMaterial(const Material& material){
	m_Material = material;
} 

bool TriangleMesh::loadFromFile(const char* filename){

	int numFaces, numVertices, numIndices;
	float boxMinX,boxMinY,boxMinZ;
	float boxMaxX,boxMaxY,boxMaxZ;

	const aiScene* scene = aiImportFile (filename, aiProcess_Triangulate | aiProcess_GenSmoothNormals | aiProcess_JoinIdenticalVertices); // make sure that aiFace::mNumIndices is always 3. 
	if(!scene)
		return false;

	const aiMesh* mesh = scene->mMeshes[0];  // first mesh
	if( !(mesh->HasPositions()) || !(mesh->HasFaces()) || !(mesh->HasNormals()) ){
		aiReleaseImport(scene);
		return false;
	}

	numVertices = mesh->mNumVertices;
	numFaces  = mesh->mNumFaces;

	// allocate memory and get the vertices
	m_Vertices    = new glm::vec3[numVertices];
	m_Normals     = new glm::vec3[numVertices];
	m_NumVertices = numVertices;


	boxMinX = boxMinY = boxMinZ = 999999;
	boxMaxX = boxMaxY = boxMaxZ = -9999999;


	for( int i = 0 ; i < numVertices; i++){
		const aiVector3D* pos = &(mesh->mVertices[i]);
		glm::vec3 vertex(pos->x, pos->y, pos->z);
		m_Vertices[i] = vertex;

		// find bottom left box vertex
		if( pos->x < boxMinX)
			boxMinX = pos->x;

		if( pos->y < boxMinY)
			boxMinY = pos->y;

		if( pos->z < boxMinZ)
			boxMinZ = pos->z;

		// find upper right box vertex
		if( pos->x > boxMaxX)
			boxMaxX = pos->x;

		if( pos->y > boxMaxY)
			boxMaxY = pos->y;

		if( pos->z > boxMaxZ)
			boxMaxZ = pos->z;


		// get vertex normal
		const aiVector3D* norm = &(mesh->mNormals[i]);
		glm::vec3 v_normal = glm::vec3( norm->x, norm->y, norm->z);
		m_Normals[i] = v_normal; 

	}

	// set bounding box vertices
	m_BoxMin = glm::vec3(boxMinX, boxMinY, boxMinZ);
	m_BoxMax = glm::vec3(boxMaxX, boxMaxY, boxMaxZ);

	// allocate memory for the indices
	numIndices   = numFaces * 3;
	m_NumIndices = numIndices;
	m_Indices    = new unsigned int[numIndices];

	int k = 0;
	for( int i = 0 ; i < numFaces; i++, k += 3){

		const aiFace& face = mesh->mFaces[i];
       
        m_Indices[k]    = face.mIndices[0];
        m_Indices[k+1]  = face.mIndices[1];
        m_Indices[k+2]  = face.mIndices[2];
	} 

	aiReleaseImport(scene);
	m_Valid = true;
	return true;
}

bool TriangleMesh::RayTriangleIntersection(glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, const Ray& ray, glm::vec3& barycetricCoords){


   glm::vec3 e1, e2;  //Edge1, Edge2
   glm::vec3 P, Q, T;
   float det, inv_det, u, v;
   float t;
 
   //Find vectors for two edges sharing V1
   e1  = v2 - v1;
   e2 =  v3 -  v1;
  
   P = glm::cross(ray.getDirection(), e2);
  
   det = glm::dot(e1, P);

   // culling
   if( det < EPSILON) return false;

   T = ray.getOrigin() - v1;
   u = glm::dot( T, P);
   if( u < 0.0 || u > det) return false;

   Q = glm::cross(T, e1);
   v = glm::dot( ray.getDirection(), Q);
   if( v < 0.0 || u + v > det) return false;

   t = glm::dot(e2, Q);
   inv_det = 1.0 / (float)det;

   t *= inv_det;
   u *= inv_det;
   v *= inv_det;

   barycetricCoords.x = t;
   barycetricCoords.y = u;
   barycetricCoords.z = v;

   return true;


}

bool TriangleMesh::hit(const Ray& ray, RayIntersection& intersection, float& distance){

	if( this->m_Valid == false)
		return false;


	/* 
	 * Find ray-triangle intersections for every triangle in the mesh
	 * and fill 'intersection' as a result
	 */

	/*
	 * u and v barycentric coords will be used later for texture mapping.Right now
	 * they are used only for normals interpolation
	 */

	float u,v,t, min_t = 9999999;
	bool collided,intersectionFound;
	glm::vec3 norm(0.0f);
	glm::vec3 pos(0.0f); 
	glm::vec3 barCoords(0.0f);

	unsigned int v0_index, v1_index, v2_index;
	
	intersectionFound = false;

	// check first for ray-box intersection

	if( (this->RayBoxIntersection(ray, this->boundingBox())) == false){
	//	std::cout << "BOX-RAY intersection NOT found..." << std::endl;
		return false;
	}
	else{
	//	std::cout << "BOX-RAY intersection Found" << std::endl;
	}


	// check for the rest of the mesh
	int k = 0;
	for ( unsigned int i = 0 ; i < m_NumIndices / 3; i++, k += 3){
		v0_index = m_Indices[k];
		v1_index = m_Indices[k+1];
		v2_index = m_Indices[k+2];

		glm::vec3 v0 = m_Vertices[v0_index];
		glm::vec3 v1 = m_Vertices[v1_index];
		glm::vec3 v2 = m_Vertices[v2_index];

		collided = RayTriangleIntersection(v0, v1, v2, ray, barCoords);  // barCoords = (t_distance,u,v)
		if(collided){
			t = barCoords.x;
			if( t < min_t && t > 0.0f ){
				intersectionFound = true;
				pos  = ray.getOrigin() + t * ray.getDirection();

				// interpolate normals
				u = barCoords.y;
				v = barCoords.z;
				norm = glm::normalize(v0 * ( 1.0f - u - v) + (v1 * u) + (v2*v));

				min_t = t;				
			}
		}
	}

	intersection.setPoint(pos);
	intersection.setNormal(norm);
	intersection.setMaterial(m_Material);
	distance = min_t;
	return intersectionFound;
}

Box TriangleMesh::boundingBox(){
	return Box(m_BoxMin, m_BoxMax);
}

bool TriangleMesh::RayBoxIntersection(const Ray& ray, const Box& box){

	// lb is the corner of AABB with minimal coordinates - left bottom, rt is maximal corner
	glm::vec3 lb = box.getMinVertex();
	glm::vec3 rt = box.getMaxVertex();

	glm::vec3 rayOrigin = ray.getOrigin();
	glm::vec3 rayInvDirection = ray.getInvDirection();

	float t1 = (lb.x - rayOrigin.x) * rayInvDirection.x;
	float t2 = (rt.x - rayOrigin.x) * rayInvDirection.x;
	float t3 = (lb.y - rayOrigin.y) * rayInvDirection.y;
	float t4 = (rt.y - rayOrigin.y) * rayInvDirection.y;
	float t5 = (lb.z - rayOrigin.z) * rayInvDirection.z;
	float t6 = (rt.z - rayOrigin.z) * rayInvDirection.z;

	float tmin = MAX(MAX(MIN(t1, t2), MIN(t3, t4)), MIN(t5, t6));
	float tmax = MIN(MIN(MAX(t1, t2), MAX(t3, t4)), MAX(t5, t6));

	// if tmax < 0, ray (line) is intersecting AABB, but whole AABB is behing us
	if (tmax < 0)
	{
    	//t = tmax;
    	return false;
	}

	// if tmin > tmax, ray doesn't intersect AABB
	if (tmin > tmax)
	{
    	//t = tmax;
    	return false;
	}

	//t = tmin;
	return true;
}

const glm::mat4& TriangleMesh::transformation(){
	return m_LocalToWorldTransformation;
}
void  TriangleMesh::setTransformation(glm::mat4 transformation){
	m_LocalToWorldTransformation = transformation;
}