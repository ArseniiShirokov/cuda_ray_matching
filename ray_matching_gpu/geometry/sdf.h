#pragma once

#include <vector.h>

class SDF {
public:
    __device__ SDF() : color(Vector{1, 1, 1}) {};
    __device__ SDF(const Vector& color) : color(color) {};

    __device__ virtual float ComputeSdf(const Vector &point) const = 0;

    __device__ Vector GetColor() const {
        return color;
    }

protected:
    Vector color;
};


__device__ Vector EstimateNormal(const SDF* obj, const Vector& z, float eps = 1e-4) {
    Vector z1 = z + Vector{eps, 0.f, 0.f};
    Vector z2 = z - Vector{eps, 0.f, 0.f};
    Vector z3 = z + Vector{0.f, eps, 0.f};
    Vector z4 = z - Vector{0.f, eps, 0.f};
    Vector z5 = z + Vector{0.f, 0.f, eps};
    Vector z6 = z - Vector{0.f, 0.f, eps};
    float dx = obj->ComputeSdf(z1) - obj->ComputeSdf(z2);
    float dy = obj->ComputeSdf(z3) - obj->ComputeSdf(z4);
    float dz = obj->ComputeSdf(z5) - obj->ComputeSdf(z6);
    auto v = Vector{dx, dy, dz};
    return v / (2.0f * eps);
}
