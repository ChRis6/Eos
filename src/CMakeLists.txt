find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()



find_package(GLEW REQUIRED)
find_package(CUDA REQUIRED)
find_package(ASSIMP REQUIRED)
include_directories(${GLEW_INLUDE_PATH})
include_directories(${ASSIMP_INCLUDE_DIR})
include_directories(glm)

set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Wall -g")
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=sm_20 -maxrregcount 40)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-lineinfo -O3  -gencode arch=compute_30,code=sm_30)
SET(CUDA_SEPARABLE_COMPILATION ON)
set(BUILD_SHARED_LIBS OFF)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
FILE(GLOB CPPSources *.cpp)
FILE(GLOB CUSources *.cu)
include_directories(${Eos_SOURCE_DIR}/glfw-3.0.4/include)

cuda_add_library(cudautilities ${CUSources} STATIC OPTIONS -gencode=arch=compute_30,code=sm_30)


cuda_add_executable(Eos ${CPPSources})
target_link_libraries(Eos glfw cudautilities ${GLFW_LIBRARIES} ${GLEW_LIBRARY} ${ASSIMP_LIBRARIES})
