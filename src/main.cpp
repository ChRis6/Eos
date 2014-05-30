#include <iostream>
#include <string>
#include <fstream>
#include <streambuf>


#include <omp.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <GL/glew.h>
#include <GLFW/glfw3.h>

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "stb_image.h"
#include "Ray.h"
#include "sphere.h"
#include "RayIntersection.h"
#include "triangleMesh.h"
#include "Camera.h"
#include "Scene.h"
#include "LightSource.h"
#include "Material.h"

#define WINDOW_WIDTH   640  // in pixels
#define WINDOW_HEIGHT  480  // in pixels
#define FOV            70

#ifndef EPSILON
#define EPSILON        1e-3
#endif

void glfwErrorCallback(int error, const char *description)
{
   std::cerr << "GLFW error " << error << ": " << description << std::endl;
}

std::string readShaderFromFile(const char* filename){

   std::ifstream t(filename);
   std::string str;

   t.seekg(0, std::ios::end);   
   str.reserve(t.tellg());
   t.seekg(0, std::ios::beg);

   str.assign((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());

   return str;
}

bool triangle_intersection( glm::vec3 v1, glm::vec3 v2, glm::vec3 v3, glm::vec3 rayOrigin, glm::vec3 rayDirection, float* out){

   glm::vec3 e1, e2;  //Edge1, Edge2
   glm::vec3 P, Q, T;
   float det, inv_det, u, v;
   float t;
 
   //Find vectors for two edges sharing V1
   e1  = v2 - v1;
   e2 =  v3 -  v1;
   //Begin calculating determinant - also used to calculate u parameter
   P = glm::cross(rayDirection, e2);
   //if determinant is near zero, ray lies in plane of triangle
   det = glm::dot(e1, P);
   //NOT CULLING
   if(det > -EPSILON && det < EPSILON) return false;
   inv_det = 1.f / det;
 
   //calculate distance from V1 to ray origin
   T = rayOrigin - v1;
 
   //Calculate u parameter and test bound
   u = glm::dot(T, P) * inv_det;
   //The intersection lies outside of the triangle
   if(u < 0.f || u > 1.f) return false;
 
   //Prepare to test v parameter
   Q =  glm::cross(T, e1);
 
   //Calculate V parameter and test bound
   v = glm::dot(rayDirection, Q) * inv_det;
   //The intersection lies outside of the triangle
   if(v < 0.f || u + v  > 1.f) return false;
 
   t = glm::dot(e2, Q) * inv_det;
 
   if(t > EPSILON) { //ray intersection
    *out = t;
    return false;
  }
 
   // No hit, no win
   return false;
}

bool quadSolve( float a, float b, float c, float *t){

   float dis = (b * b) - (4.0 * a * c);
   if( dis < 0.0 )
      return false;

   float t1,t2;

   t1 = ( -b + sqrt(dis)) / (float) ( 2.0 * a);
   t2 = ( -b - sqrt(dis)) / (float) ( 2.0 * a);


   if( t1 > EPSILON){
      if( t2 > EPSILON){
         if( t1 > t2 ){
            *t = t2;
            
         }
         else{
            *t = t1;
         }
         //std::cout << "Returning : " << *t << std::endl;
         return true;
      }
      *t = t1;
      //std::cout << "Returning : " << *t << std::endl;
      return true;      
   }

   if( t2 > EPSILON){
      *t = t2;
      //std::cout << "Returning: " << *t << std::endl;
      return true;
   }

   return false;

}

GLuint loadShader( const char* vertexShaderSrc, const  char* fragmentShaderSrc){

 
   // Create the shaders
   GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
   GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

 
   GLint res = GL_FALSE;
   int InfoLogLength;
 
   // Compile Vertex Shader
   glShaderSource(VertexShaderID, 1, &vertexShaderSrc , NULL);
   glCompileShader(VertexShaderID);
 
   // Check Vertex Shader
   glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &res);
   if( res == GL_FALSE){

      char log[1024];

      glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
      glGetShaderInfoLog(VertexShaderID, InfoLogLength, NULL , &log[0]);

      std::cout << "Vertex Shader: " << vertexShaderSrc << std::endl;

      std::cout << "Error: " << log << std::cout;
      return 0;
   }
 
    // Compile Fragment Shader
   glShaderSource(FragmentShaderID, 1, &fragmentShaderSrc, NULL);
   glCompileShader(FragmentShaderID);
 
   // Check Fragment Shader
   glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &res);
   if( res == GL_FALSE){

      char log[1024];

      glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
      glGetShaderInfoLog(FragmentShaderID, InfoLogLength, NULL , &log[0]);

      std::cout << " Fragment Shader: " <<  fragmentShaderSrc << std::endl;

      std::cout << "Error: " << log << std::cout;
      glDeleteShader(VertexShaderID);
      return 0;
   }
 
   // Link the program
   GLuint ProgramID = glCreateProgram();
   if( ProgramID == 0 ){

      glDeleteShader(VertexShaderID);
      glDeleteShader(FragmentShaderID);
      return 0;
   }
   glAttachShader(ProgramID, VertexShaderID);
   glAttachShader(ProgramID, FragmentShaderID);
   glLinkProgram(ProgramID);
 
   // Check the program
   glGetProgramiv(ProgramID, GL_LINK_STATUS, &res);
   if( res == GL_FALSE ){

      char log[1024];

      glGetProgramiv( ProgramID , GL_INFO_LOG_LENGTH, &InfoLogLength);
      glGetProgramInfoLog(ProgramID, InfoLogLength, NULL , &log[0]);

      std::cout << "Vertex Shader Source" << vertexShaderSrc << std::endl;
      std::cout << "Fragment Shader Source" << fragmentShaderSrc << std::endl;

      std::cout << "Error linking: " << log << std::endl;

      glDeleteProgram(ProgramID);
      glDeleteShader(VertexShaderID);
      glDeleteShader(FragmentShaderID);

      return 0;
   }

   // detach and delete shaders.Be nice to the driver
   glDetachShader(ProgramID, VertexShaderID);
   glDetachShader(ProgramID, FragmentShaderID);

   glDeleteShader(VertexShaderID);
   glDeleteShader(FragmentShaderID);

   return ProgramID;
}


int main(int argc, char **argv)
{
   GLFWwindow* window;
   glfwSetErrorCallback(glfwErrorCallback);

   if(!glfwInit())
   {
      std::cerr << "Failed to initialize GLFW...\n";
      return -1;
   }

   glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
   glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 2);
   glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

   window = glfwCreateWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "RayTracing", NULL, NULL);
   if(!window)
   {
      std::cerr << "Failed to open GLFW window...\n";
      glfwTerminate();
      return -1;
   }

   std::cout<< "Going to Create context\n";
   glfwMakeContextCurrent(window);

   std::cout << "Going to initialize GLEW...\n";
   glewExperimental = GL_TRUE;
   if (glewInit() != GLEW_OK)
   {
      std::cerr << "Failed to initialize GLEW...\n";
      glfwTerminate();
      return -1;
   }
  
   std::cout << "Going to Create Vertex Array Object...\n";
   GLuint vao;
   glGenVertexArrays(1, &vao);
   glBindVertexArray(vao);

   // set up shaders
   GLuint program;

   std::string vertexShaderSource = readShaderFromFile("shaders/simple.vs");
   std::string fragmentShaderSource = readShaderFromFile("shaders/simple.fs");

   program = loadShader( vertexShaderSource.c_str(), fragmentShaderSource.c_str() );
   if( program == 0 ){
      std::cout << "Program not created...\n";
      glfwTerminate();
      return -1;
   }

   // screen quad
   float quad[] = { -1.0f, -1.0f, 0.0f, // bottom-left
                     -1.0f, 1.0f, 0.0f, // top-left
                     1.0f, -1.0f, 0.0f, // bottom-right

                     -1.0f, 1.0f, 0.0f, // top-left
                      1.0f, 1.0f, 0.0f, // top-right
                      1.0f, -1.0f, 0.0f // bottom-right 
                   };

   GLuint vbo;
   glGenBuffers(1, &vbo);

   glBindBuffer(GL_ARRAY_BUFFER, vbo);
   glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);   // copy to device memory

   unsigned char* imageBuffer = new unsigned char[ WINDOW_WIDTH * WINDOW_HEIGHT * 4];
   unsigned char* screen = imageBuffer;
   memset(imageBuffer, 0, sizeof(unsigned char) * 4 * WINDOW_WIDTH * WINDOW_HEIGHT);

   GLuint pbo;
   glGenBuffers(1, &pbo);
   glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
   glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(GLchar) * WINDOW_WIDTH * WINDOW_HEIGHT * 4, imageBuffer, GL_DYNAMIC_DRAW);

   /*
   int texx, texy, n;
   int force_channels = 4;
   unsigned char* image_data = stbi_load (argv[1], &texx, &texy, &n, force_channels);
   if( !image_data ){
      std::cout<< "Image no found\n";
      glfwTerminate(); 
   }

   int width_in_bytes = texx * 4;
   unsigned char *top = NULL;
   unsigned char *bottom = NULL;
   unsigned char temp = 0;
   int half_height = texy / 2;

   for (int row = 0; row < half_height; row++) {
      top = image_data + row * width_in_bytes;
      bottom = image_data + (texy - row - 1) * width_in_bytes;
      for (int col = 0; col < width_in_bytes; col++) {
         temp = *top;
         *top = *bottom;
         *bottom = temp;
         top++;
         bottom++;
      }
   }

 */

   // generate texture
   GLuint texture;
   glGenTextures(1, &texture);
   glActiveTexture(GL_TEXTURE0);
   glBindTexture(GL_TEXTURE_2D, texture);


   glUseProgram(program);
   glEnableVertexAttribArray(0);
   glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);



   //float angle = tan( FOV * 0.5 * M_PI / 180 );
   //float aspectRatio = WINDOW_WIDTH/ (float) WINDOW_HEIGHT;

   glm::vec4 sphereOrigin(0.0, 0.0, 0.0, 1.0);
   glm::vec4 sphereColor( 0.6f , 0.6f , 0.6f, 0.0);

   glm::vec4 lightPosition( 120.0f, 0.0f , 0.0f, 1.0f);
   glm::vec4 lightColor( 1.0, 1.0, 1.0, 0.0);
   float lightAmbientIntesity = 0.1f;

   float fovx_rad = FOV * M_PI / (float) 180.0;
   //float fovy_rad =  (aspectRatio * FOV) * M_PI / (float)180.0;
   float fovy_rad =  FOV * M_PI / (float)180.0;

   float tan_fovx = tan( fovx_rad / (float) 2);
   float tan_fovy = tan( fovy_rad / (float) 2);
   std::cout << "FOVX = " << fovx_rad << " FOVY = " << fovy_rad << std::endl;
   std::cout << "tan_FOVX = " << tan_fovx << " tan_FOVY = " << tan_fovy << std::endl;


   glm::mat4 M = glm::translate( glm::mat4(1.0), glm::vec3(-2.0, 0.0, -20.0));
   glm::mat4 inv_M = glm::inverse(M);
   glm::mat4 identity(1.0f);
   glm::mat4 mat_rot = glm::rotate(identity, 0.1f, glm::vec3(0.0f, 0.0f,1.0f));
   /* clear buffer */
   //memset(imageBuffer, 0, sizeof(unsigned char) * 4 * WINDOW_WIDTH * WINDOW_HEIGHT);

   //Sphere mySphere(glm::vec3(0.0, 0.0, 0.0), 4);
   
   /*
   TriangleMesh monkey;
   bool loaded = monkey.loadFromFile("monkey.obj");
   if( loaded == false){
      std::cout << "Monkey not loaded...." <<std::endl;
      glfwTerminate();
      return -1;
   }
   std::cout << "Monkey loaded" << std::endl;
   std::cout <<"Num Vertices = " << monkey.getNumVertices() << std::endl;
   std::cout << "Triangles   =" << monkey.getNumTriangles() << std::endl;
   */
  

   /*
   std::cout << "Ray tracing is over..." << std::endl;
  
   screen = imageBuffer;
   FILE *imageFile;
   imageFile = fopen("RayTraced.ppm", "wb");
   fprintf(imageFile, "P6\n%d %d\n255\n", WINDOW_WIDTH, WINDOW_HEIGHT);
   for( int j = 0 ; j < WINDOW_HEIGHT; j++)
      for( int i = 0 ; i < WINDOW_WIDTH; i++, screen += 4)
         fwrite(screen, 1, 3, imageFile);

   fclose(imageFile);
   */

   // transfer pixels to texture via pixel buffer object
   glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA, WINDOW_WIDTH , WINDOW_HEIGHT, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
   // filtering
   glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
   glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
   glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
   glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

   Camera camera;

   camera.setPosition(glm::vec3(0.0f, 0.0f,0.0f));
   camera.setViewingDirection(glm::vec3(0.0f,0.0f, -1.0f));
   camera.setRightVector(glm::vec3(1.0f, 0.0f, 0.0f));
   camera.setUpVector(glm::vec3(0.0f, 1.0f, 0.0f));
   camera.setFov(FOV);
   camera.setWidth(WINDOW_WIDTH);
   camera.setHeight(WINDOW_HEIGHT);

   float horizontalAngle = 3.1415f;
   float verticalAngle = 3.1414f / 4.0f;
   float speed = 5.0f; // 3 units / second
   float mouseSpeed = 0.005f;

   glfwSetCursorPos(window, WINDOW_WIDTH/2.0, WINDOW_HEIGHT/2.0);


   double currTime = 0.0;
   double timeAcc = 0.0;
   int fps = 0;
   float deltaTime = 0.0f;

   // set up scene
   Scene scene;
   Material sphereMaterial(0.1f, glm::vec4(0.9f, 0.6f, 0.5f, 0.0f), glm::vec4(1.0f), 50);
   LightSource* lightSource = new LightSource;
   Sphere* sphere = new Sphere(glm::vec3(0.0f), 4);
   sphere->setTransformation(M);
   sphere->setMaterial(sphereMaterial);

   scene.addSurface(sphere);
   scene.addLightSource(lightSource);

   while (glfwGetKey(window, GLFW_KEY_ESCAPE) != GLFW_PRESS && !glfwWindowShouldClose(window))
   {
      currTime = glfwGetTime();
      lightPosition = mat_rot * lightPosition;
      memset(imageBuffer, 0, sizeof(unsigned char) * 4 * WINDOW_WIDTH * WINDOW_HEIGHT);
      // handle input
      double xpos, ypos;
      glfwGetCursorPos(window, &xpos, &ypos);
      glfwSetCursorPos(window, WINDOW_WIDTH/2.0, WINDOW_HEIGHT/2.0);

      horizontalAngle -= mouseSpeed * float(WINDOW_WIDTH/2 - xpos );
      verticalAngle   += mouseSpeed * float( WINDOW_HEIGHT/2 - ypos );

      glm::vec3 viewDirection(cos(verticalAngle) * sin(horizontalAngle),
                          sin(verticalAngle),
                          cos(verticalAngle) * cos(horizontalAngle));

      // Right vector
      glm::vec3 rightDirection = glm::vec3(sin(horizontalAngle - 3.1415f/2.0f),0, cos(horizontalAngle - 3.1415f/2.0f));
      glm::vec3 upDirection = glm::cross( rightDirection, viewDirection  );

      glm::vec3 cameraPos = camera.getPosition();
      // Move forward
      if (glfwGetKey(window, GLFW_KEY_W ) == GLFW_PRESS){
         cameraPos += viewDirection * speed * deltaTime;
      }
      // Move backward
      if (glfwGetKey(window, GLFW_KEY_S ) == GLFW_PRESS){
         cameraPos -= viewDirection * speed * deltaTime;   
      }
      // Strafe right
      if (glfwGetKey(window, GLFW_KEY_D ) == GLFW_PRESS){
         cameraPos += rightDirection * speed * deltaTime;
      }
      // Strafe left
      if (glfwGetKey(window, GLFW_KEY_A ) == GLFW_PRESS){
         cameraPos -= rightDirection * speed * deltaTime;
      }


      // update camera
      camera.setViewingDirection(viewDirection);
      camera.setUpVector(upDirection);
      camera.setRightVector(rightDirection);
      camera.setPosition(cameraPos);

      /* Raytracing */
      scene.render(camera, imageBuffer);

      // copy pixels to GPU buffer (async copy)
      glBufferSubData(GL_PIXEL_UNPACK_BUFFER, 0, sizeof(GLchar) * WINDOW_WIDTH * WINDOW_HEIGHT * 4, imageBuffer);
      
      // update texture data from the GL_PIXEL_UNPACK_BUFFER bounded pixel buffer object
      glTexSubImage2D(GL_TEXTURE_2D, 0, 0 , 0, WINDOW_WIDTH, WINDOW_HEIGHT, GL_RGBA, GL_UNSIGNED_BYTE, NULL);


      /* draw window sized quad */
      glDrawArrays( GL_TRIANGLES, 0, 6);


      glfwSwapBuffers(window);
      fps++;
      // measure Frames per second
      deltaTime = glfwGetTime() - currTime;
      timeAcc += glfwGetTime() - currTime;
      if( timeAcc >= 1.0){
         timeAcc = 0.0;
         std::cout << " FPS: " << fps << std::endl;
         fps = 0;
      }

      glfwPollEvents();

   }

   glDeleteProgram(program);
   glfwTerminate();
   return 0;
}

