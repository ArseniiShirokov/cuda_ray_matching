#pragma once

#include <cuda_runtime.h>
#include <image.h>
#include <camera_options.h>
#include <string>
#include <vector.h>
#include <initializer_list>
#include <cmath>
#include <transformer.h>
#include "render_options.h"
#include <postprocessing.h>
#include <scene.h>
#include <ctime> 
#include <iostream>
#include <union_sdf.h>
#include <sierpinski.h>
#include <cstdlib>


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


__global__ void RayCastCUDA(Scene** scene, RenderOptions* opt, Transformer* transformer, Vector* results, int width) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    auto view_ray = transformer->MakeRay(x, y);
    auto object = (*scene)->GetUnionObject();

    float dist = object->ComputeSdf(view_ray.GetOrigin());
    bool is_inf = Length(view_ray.GetOrigin()) > opt->max_range; 

    while (dist > opt->eps) {

        if (is_inf) {
            return; //Vector();
        }
        
        float step = max(dist, opt->min_step);
        view_ray.ShiftOrigin(step);
        dist = object->ComputeSdf(view_ray.GetOrigin());
        is_inf = Length(view_ray.GetOrigin()) > opt->max_range; 
    }
    
    auto hitted_object = object->GetHittedObject(view_ray.GetOrigin());
    auto base_color = hitted_object->GetColor();

    auto light_dir = (*scene)->GetLight().position - view_ray.GetOrigin();
    light_dir.Normalize();

    auto norm = EstimateNormal(hitted_object, view_ray.GetOrigin());
    float coeff = max(0.1, DotProduct(norm, light_dir));
    results[x * width + y] = coeff * (*scene)->GetLight().intensity * base_color;
}


Image Render(Scene** device_scene, const CameraOptions& camera_options, const RenderOptions& render_options) {
    Image img(camera_options.screen_width, camera_options.screen_height);
    Transformer transformer(camera_options);

    int tx = 8;
    int ty = 8;
    dim3 blocks(img.Height() / tx + 1, img.Width() / ty + 1);
    dim3 threads(tx, ty);

    RenderOptions *device_render_options;
    checkCudaErrors(cudaMalloc((void**)&device_render_options, sizeof(RenderOptions)));
    checkCudaErrors(cudaMemcpy(device_render_options, &render_options, sizeof(RenderOptions), cudaMemcpyHostToDevice));

    Transformer *device_transformer;
    checkCudaErrors(cudaMalloc((void**)&device_transformer, sizeof(Transformer)));
    checkCudaErrors(cudaMemcpy(device_transformer, &transformer, sizeof(Transformer), cudaMemcpyHostToDevice));

    Vector *device_results;
    checkCudaErrors(cudaMalloc((void**)&device_results, img.Width() * img.Height() * sizeof(Vector)));

    RayCastCUDA<<<blocks, threads>>>(device_scene, device_render_options, device_transformer, device_results, img.Width());
    checkCudaErrors(cudaGetLastError());

    checkCudaErrors(cudaDeviceSynchronize());
    Vector *results = (Vector*) malloc(img.Width() * img.Height() * sizeof(Vector));
    checkCudaErrors(cudaMemcpy(results, device_results, img.Width() * img.Height() * sizeof(Vector), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaFree(device_render_options));
    checkCudaErrors(cudaFree(device_transformer));

    std::vector<std::vector<Vector>> color_map(img.Height(), std::vector<Vector>(img.Width()));
    for (int i = 0; i < img.Width() * img.Height(); ++i) {
        color_map[i / img.Width()][i % img.Width()] = results[i];
    }
    PostProc(img, color_map);
    return img;
}
