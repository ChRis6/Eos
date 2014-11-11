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
	m_Triangles    = 0;
	m_NumTriangles = 0;
	m_LocalToWorldTransformation = glm::mat4(1.0f);
	m_SceneIndexStart = 0;
	m_SceneIndexEnd   = 0;
	m_Valid           = false;
}

TriangleMesh::~TriangleMesh(){
	if(m_Valid){
		delete m_Triangles;
	}
		
}

Triangle* TriangleMesh::getTriangle(int index){
	if( index < m_NumTriangles && m_Valid){
		return m_Triangles[index];
	}
	return NULL;
}


Material& TriangleMesh::getMaterial(){
	return m_Material;
}

void TriangleMesh::setMaterial(const Material& material){
	m_Material = material;
} 

bool TriangleMesh::loadFromFile(const char* filename){

	int numFaces, numVertices, numIndices;
	float boxMinX,boxMinY,boxMinZ;
	float boxMaxX,boxMaxY,boxMaxZ;

	glm::vec3* Vertices;
	glm::vec3* Normals;
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
	Vertices    = new glm::vec3[numVertices];
	Normals     = new glm::vec3[numVertices];
	

	for( int i = 0 ; i < numVertices; i++){
		const aiVector3D* pos = &(mesh->mVertices[i]);
		glm::vec3 vertex(pos->x, pos->y, pos->z);
		Vertices[i] = vertex;

		// get vertex normal
		const aiVector3D* norm = &(mesh->mNormals[i]);
		glm::vec3 v_normal = glm::vec3( norm->x, norm->y, norm->z);
		Normals[i] = v_normal; 

	}



	// allocate memory for the indices
	//numIndices   = numFaces * 3;
	//m_Indices    = new unsigned int[numIndices];

	m_Triangles = new Triangle*[numFaces];
	m_NumTriangles = numFaces;

	for( int i = 0 ; i < numFaces; i++){

		const aiFace& face = mesh->mFaces[i];
       	
       	int v1_index = face.mIndices[0];
       	int v2_index = face.mIndices[1];
       	int v3_index = face.mIndices[2];

       	glm::vec3 v1 = Vertices[v1_index];
       	glm::vec3 v2 = Vertices[v2_index];
       	glm::vec3 v3 = Vertices[v3_index];

       	glm::vec3 n1 = Normals[v1_index];
       	glm::vec3 n2 = Normals[v2_index];
       	glm::vec3 n3 = Normals[v3_index];

       	Triangle* newTriangle = new Triangle(v1, v2, v3, n1, n2, n3);
       	
       	newTriangle->setTransformation(m_LocalToWorldTransformation);
       	newTriangle->setInverseTransformation(m_Inverse);
       	newTriangle->setInverseTransposeTransformation(m_InverseTranspose);

       	newTriangle->setMaterial(m_Material);
       	newTriangle->setMaterialIndex(m_MaterialIndex);

       	m_Triangles[i] = newTriangle;
	} 

	aiReleaseImport(scene);
	m_Valid = true;
	delete Vertices;
	delete Normals;
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
void  TriangleMesh::setTransformation(glm::mat4& transformation){
	m_LocalToWorldTransformation = transformation;
	 m_Inverse = glm::inverse(transformation);
  	m_InverseTranspose = glm::transpose(m_Inverse);
}

