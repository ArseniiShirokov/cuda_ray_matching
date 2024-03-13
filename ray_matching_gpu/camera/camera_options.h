#pragma once

#include <array>
#include <cmath>
#include <vector.h>

struct CameraOptions {
    int screen_width;
    int screen_height;
    double fov;
    Vector look_from;
    Vector look_to;

    CameraOptions(int width, int height, float fov = M_PI / 2,
                  Vector look_from = Vector(0.0f, 0.0f, 0.0f),
                  Vector look_to = Vector(0.0f, 0.0f, -1.0f))
        : screen_width(width),
          screen_height(height),
          fov(fov),
          look_from(look_from),
          look_to(look_to)
    {
    }
};
