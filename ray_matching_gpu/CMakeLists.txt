add_definitions(-w)
cuda_add_executable(test_ray_matching_gpu run.cu)

set(CMAKE_CONFIGURATION_TYPES "Release" CACHE STRING "" FORCE)

target_compile_definitions(test_ray_matching_gpu PUBLIC BASE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/")

target_include_directories(test_ray_matching_gpu PUBLIC ../ray_matching_gpu/camera)
target_include_directories(test_ray_matching_gpu PUBLIC ../ray_matching_gpu/image_utils)
target_include_directories(test_ray_matching_gpu PUBLIC ../ray_matching_gpu/geometry)
target_include_directories(test_ray_matching_gpu PUBLIC ../ray_matching_gpu/postprocessing)
target_include_directories(test_ray_matching_gpu PUBLIC ../ray_matching_gpu/scene)

target_link_libraries(test_ray_matching_gpu ${PNG_LIBRARY} ${JPEG_LIBRARIES})
target_include_directories(
  test_ray_matching_gpu
  PRIVATE ${PNG_INCLUDE_DIRS} ${JPEG_INCLUDE_DIRS}
)


