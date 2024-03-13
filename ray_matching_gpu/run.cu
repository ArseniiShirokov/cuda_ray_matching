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
        auto center = Vector{0, -0, -20};
        auto color = Vector{1, 1, 1};
        auto object = new Sierpinski(center, 5, color);

        int num_obj = 1;
        SDF* objects[1];
        objects[0] = object;
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
