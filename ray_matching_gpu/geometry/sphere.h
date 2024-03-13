#pragma once

#include <vector.h>
#include <sdf.h>

class Sphere : public SDF{
public:
    __device__ Sphere(){};
    __device__ Sphere(Vector center, float radius) : center_(center), radius_(radius){};
    __device__ Sphere(Vector center, float radius, const Vector& color) : SDF(color), center_(center), radius_(radius){};

    __device__ const Vector& GetCenter() const {
        return center_;
    }
    __device__ float GetRadius() const {
        return radius_;
    }
    __device__ float ComputeSdf(const Vector &point) const override{
        return Length(point - center_) - radius_;
    }

private:
    Vector center_;
    float radius_;
};
