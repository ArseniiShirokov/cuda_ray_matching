#include <cmath>
#include <string>
#include <optional>
#include <sphere.h>
#include <scene.h>

#include <camera_options.h>
#include <image.h>
#include "raymatching.h"
#include "render_options.h"
#include <sierpinski.h>

#include <vector>


int artifact_index = 0;
const std::string kArtifactsDir = BASE_DIR;


void SaveImage(const std::string& result_filename, Image& image) {
    image.Write(kArtifactsDir + "/results/" + result_filename);
}


__global__ void build_scene_1(Scene **scene) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
<<<<<<< HEAD
        auto center = Vector{5, 5, -16};
        auto color = Vector{1, 1, 0};
        auto sphere1 = new Sphere(center, 1, color);

        auto center2 = Vector{4.4, 5.1, -15};
        auto color2 = Vector{1, 1, 1};
        auto sphere2 = new Sphere(center2, 0.2, color2);

        auto center3 = Vector{0, -0, -20};
        auto object = new Sierpinski(center3, 5, Vector{0.5, 1, 1});

        const int num_obj = 3;
        SDF* objects[num_obj];
        objects[0] = object;
        objects[1] = sphere1;
        objects[2] = sphere2;
=======
        auto center = Vector{0, -0, -20};
        auto color = Vector{1, 1, 1};
        auto object = new Sierpinski(center, 5, color);

        int num_obj = 1;
        SDF* objects[1];
        objects[0] = object;
>>>>>>> f0c852973a14c5f53a6b3e3ed9bfec570456208e
        auto sdfs = new UnionSDF(objects, num_obj);

        auto position = Vector{-4, 5, -1.0};
        auto intensity = Vector{0.1, 0.1, 0.1};
        auto light = Light(position, intensity); 

        *scene = new Scene(sdfs, light);
    }
}


__global__ void build_scene_2(Scene **scene) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        auto center = Vector{-3, -3, -5};
        auto color = Vector{0, 1, 0};
        auto object = new Sphere(center, 1, color);

        auto center2 = Vector{1, 1, -3};
        auto color2 = Vector{1, 0, 0};
        auto object2 = new Sphere(center2, 0.5, color2);

        int num_obj = 2;
        SDF* objects[2];
        objects[0] = object;
        objects[1] = object2;
        auto sdfs = new UnionSDF(objects, num_obj);

        auto position = Vector{0, 5, 0};
        auto intensity = Vector{0.1, 0.1, 0.1};
        auto light = Light(position, intensity); 

        *scene = new Scene(sdfs, light);
    }
}


int main() {
    {
        // Base part
        CameraOptions camera_opts(1280, 720);
        RenderOptions render_opts;

        Scene **device_scene;
        checkCudaErrors(cudaMalloc((void**)&device_scene, sizeof(Scene)));
        build_scene_1<<<1,1>>>(device_scene);

        auto image = Render(device_scene, camera_opts, render_opts);
        checkCudaErrors(cudaFree(device_scene));
        SaveImage("base_gpu.png", image);
    }

    {
<<<<<<< HEAD
        // Base part camera movement
        float fov = M_PI / 3;
        Vector look_from = Vector(-12.0f, -5.0f, -6.0f);
        Vector look_to = Vector(-11.0f, -4.0f, -10.0f);

        CameraOptions camera_opts(1280, 720, fov, look_from, look_to);
        RenderOptions render_opts;

        Scene **device_scene;
        checkCudaErrors(cudaMalloc((void**)&device_scene, sizeof(Scene)));
        build_scene_1<<<1,1>>>(device_scene);

        auto image = Render(device_scene, camera_opts, render_opts);
        checkCudaErrors(cudaFree(device_scene));
        SaveImage("base_gpu_rotated.png", image);
    }

    {
=======
>>>>>>> f0c852973a14c5f53a6b3e3ed9bfec570456208e
        // Spheres
        CameraOptions camera_opts(512, 512);
        RenderOptions render_opts;

        Scene **device_scene;
        checkCudaErrors(cudaMalloc((void**)&device_scene, sizeof(Scene)));
        build_scene_2<<<1,1>>>(device_scene);

        auto image = Render(device_scene, camera_opts, render_opts);
        checkCudaErrors(cudaFree(device_scene));
        SaveImage("spheres_gpu.png", image);
    }
}
