#pragma once

#include <cuda_runtime.h>
#include <image.h>
#include <camera_options.h>
#include <string>
#include <vector.h>
#include <sdf.h>
#include <initializer_list>
#include <ray.h>
#include <cmath>
#include <transformer.h>
#include "render_options.h"
#include <postprocessing.h>
#include <scene.h>
#include <ctime> 
#include <iostream>

#define checkCudaErrors(val) check_cuda((val), #val, __FILE__, __LINE__)


void check_cuda(cudaError_t result, 
                char const *const func, 
                const char *const file, 
                int const line) {
    if(result) {
        std::cerr << "CUDA error = "<< static_cast<unsigned int>(result) << " at " <<
        file << ":" << line << " '" << func << "' \n";
        cudaDeviceReset();
        exit(99);
    }
}


__global__ void helloCUDA() {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    //printf("Pixel %d, %d\n", x, y);
}


Vector RayCast(const Scene& scene, Ray& view_ray, RenderOptions opt) {
    auto object = scene.GetUnionObject();
    double dist = object->ComputeSdf(view_ray.GetOrigin());
    bool is_inf = Length(view_ray.GetOrigin()) > opt.max_range; 

    while (dist > opt.eps) {

        if (is_inf) {
            return Vector();
        }
        
        double step = std::max(dist, opt.min_step);
        view_ray.ShiftOrigin(step);
        dist = object->ComputeSdf(view_ray.GetOrigin());
        is_inf = Length(view_ray.GetOrigin()) > opt.max_range; 
    }
    
    auto hitted_object = object->GetHittedObject(view_ray.GetOrigin());
    auto base_color = hitted_object->GetColor();

    auto light_dir = scene.GetLight().position - view_ray.GetOrigin();
    light_dir.Normalize();

    auto norm = EstimateNormal(hitted_object, view_ray.GetOrigin());
    double coeff = std::max(0.1, DotProduct(norm, light_dir));

    return coeff * scene.GetLight().intensity * base_color;
}


Image Render(const Scene& scene, const CameraOptions& camera_options, const RenderOptions& render_options) {
    Image img(camera_options.screen_width, camera_options.screen_height);
    Transformer transformer(camera_options);

    std::time_t tic = std::time(NULL);
    std::cout << "Start running at: " << std::asctime(std::localtime(&tic)) << std::endl;

    std::vector<std::vector<Vector>> color_map(img.Height(), std::vector<Vector>(img.Width()));
    for (int i = 0; i < img.Height(); ++i) {
        for (int j = 0; j < img.Width(); ++j) {
            Ray view_ray = transformer.MakeRay(i, j);
            color_map[i][j] = RayCast(scene, view_ray, render_options);
        }
    }

    int tx = 8;
    int ty = 8;
    dim3 blocks(img.Width() / tx + 1, img.Height() / ty + 1);
    dim3 threads(tx, ty);
    helloCUDA<<<blocks, threads>>>();
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    std::time_t toc = std::time(NULL);
    std::cout << "Finish running at: " << std::asctime(std::localtime(&toc)) << std::endl;
    std::cout << "Time consuming: " << toc - tic << "s" << std::endl;

    PostProc(img, color_map);
    return img;
}
