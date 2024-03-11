add_definitions(-w)
add_catch(test_ray_matching_gpu run.cpp)

set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)

target_compile_definitions(test_ray_matching_gpu PUBLIC BASE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/")

target_include_directories(test_ray_matching_gpu PUBLIC ../ray_matching_gpu/camera)
target_include_directories(test_ray_matching_gpu PUBLIC ../ray_matching_gpu/image_utils)
target_include_directories(test_ray_matching_gpu PUBLIC ../ray_matching_gpu/geometry)
target_include_directories(test_ray_matching_gpu PUBLIC ../ray_matching_gpu/postprocessing)
target_include_directories(test_ray_matching_gpu PUBLIC ../ray_matching_gpu/scene)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package(CUDA REQUIRED)

# set nvcc compiler
set(CUDA_NVCC_FLAGS -gencode arch=compute_60,code=sm_60;-G;-g)



target_link_libraries(test_ray_matching_gpu ${PNG_LIBRARY} ${JPEG_LIBRARIES})
target_include_directories(
  test_ray_matching
  PRIVATE ${PNG_INCLUDE_DIRS} ${JPEG_INCLUDE_DIRS}
)
