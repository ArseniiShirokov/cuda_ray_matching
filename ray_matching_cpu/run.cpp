#include <catch.hpp>

#include <cmath>
#include <string>
#include <optional>

#include <sphere.h>
#include <scene.h>

#include <camera_options.h>
#include <image.h>
#include <raymatching.h>
#include <render_options.h>
#include <scene.h>
#include <light.h>
#include <sierpinski.h>

#include <vector>


int artifact_index = 0;
const std::string kArtifactsDir = BASE_DIR;


void SaveImage(const std::string& result_filename, Image& image) {
    image.Write(kArtifactsDir + "/results/" + result_filename);
}


TEST_CASE("sphere", "[ray_matching]") {
    CameraOptions camera_opts(512, 512);
    RenderOptions render_opts;

    auto center = Vector{-3, -3, -5};
    auto color = Vector{0, 1, 0};
    auto object = new Sphere(center, 1, color);

    auto center2 = Vector{5, 5, -25};
    auto color2 = Vector{1, 0, 0};
    auto object2 = new Sphere(center2, 5, color2);

    std::vector<SDF*> objects;
    objects.push_back(object);
    objects.push_back(object2);


    auto position = Vector{0, 5, 0};
    auto intensity = Vector{0.1, 0.1, 0.1};
    auto light = Light(position, intensity);

    auto scene = Scene(objects, light);
    auto image = Render(scene, camera_opts, render_opts);
    SaveImage("sphere.png", image);
}


TEST_CASE("base", "[ray_matching]") {
    CameraOptions camera_opts(1280, 720);
    RenderOptions render_opts;

    auto center = Vector{0, -0, -20};
    auto color = Vector{1, 1, 1};
    auto object = new Sierpinski(center, 5, color);
    
    std::vector<SDF*> objects;
    objects.push_back(object);

    auto position = Vector{-4, 5, -1.0};
    auto intensity = Vector{0.1, 0.1, 0.1};
    auto light = Light(position, intensity);

    auto scene = Scene(objects, light);
    auto image = Render(scene, camera_opts, render_opts);
    SaveImage("base.png", image);
}
