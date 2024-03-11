#pragma once

#include <image.h>
#include <camera_options.h>
#include <string>
#include <vector.h>
#include <sdf.h>
#include <initializer_list>
#include <ray.h>
#include <cmath>
#include <transformer.h>
#include <render_options.h>
#include <postprocessing.h>
#include <scene.h>

#include <iostream>


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

    std::vector<std::vector<Vector>> color_map(img.Height(), std::vector<Vector>(img.Width()));
    for (int i = 0; i < img.Height(); ++i) {
        for (int j = 0; j < img.Width(); ++j) {
            Ray view_ray = transformer.MakeRay(i, j);
            color_map[i][j] = RayCast(scene, view_ray, render_options);
        }
    }
    std::cout << "GPU time " << "\n";
    PostProc(img, color_map);
    return img;
}
