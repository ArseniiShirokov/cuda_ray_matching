#pragma once

#include <vector.h>

class Ray {
public:
    __device__ Ray(Vector origin, Vector direction) : origin_(origin), direction_(direction){};
    __device__ const Vector& GetOrigin() const {
        return origin_;
    };
    __device__ const Vector& GetDirection() const {
        return direction_;
    };
    __device__ void ShiftOrigin(float t) {
        origin_ += t * direction_;
    }

private:
    Vector origin_;
    Vector direction_;
};
