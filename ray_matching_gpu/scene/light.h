#pragma once

#include <vector.h>

struct Light {
    Vector position;
    Vector intensity;

    __device__ Light(Vector position, Vector intensity)
        : position(std::move(position)), intensity(std::move(intensity)) {
    }
};
