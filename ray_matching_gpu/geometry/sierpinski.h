#pragma once

#include <vector.h>
#include <sdf.h>
#include <vector>

#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))


__device__ float cuda_fabs(float a) {
    return a >= 0 ? a : -a;
}


class Sierpinski : public SDF{
public:
    Sierpinski(){};

    __device__ Sierpinski(Vector center, float size, const Vector& color = {0, 0, 0}) : SDF(color), center_(center), size_(size){
        vertices[0] = Vector(1, 1, 1);
        vertices[1] = Vector(-1, -1, 1);
        vertices[2] = Vector(1, -1, -1);
        vertices[3] = Vector(-1, 1, -1);
        level_ = 3;
    };


    __device__ Vector Transform2Local(const Vector &point) const {
        return (point - center_) / size_;
    }

    __device__ Vector Fold(const Vector &point, const Vector& pointOnPlane, const Vector& planeNormal) const {
        // Center plane on origin for distance calculation
        float distToPlane = DotProduct(point - pointOnPlane, planeNormal);
        
        // We only want to reflect if the dist is negative
        distToPlane = min(distToPlane, 0.0);
        return point - 2.0 * distToPlane * planeNormal;
    }

    __device__ float ComputeSdf(const Vector& point) const override{
        float scale = 1.0;
        Vector cur_point = Transform2Local(point);

        for (int i = 0; i < level_; i++) {
            // Scale point toward corner vertex, update scale accumulator
            cur_point -= vertices[0];
            cur_point.Scale(2.0);
            cur_point += vertices[0];
            scale *= 2;
            
            // Fold point across each plane
            for (int i = 1; i <= 3; i++) {
                // The plane is defined by:
                // Point on plane: The vertex that we are reflecting across
                // Plane normal: The direction from said vertex to the corner vertex
                Vector normal = vertices[0] - vertices[i];
                normal.Normalize(); 
                cur_point = Fold(cur_point, vertices[i], normal);
            }
        }
        // Now that the space has been distorted by the IFS,
        // just return the distance to a tetrahedron
        // Divide by scale accumulator to correct the distance field
        auto len =  (max(cuda_fabs(cur_point[0] + cur_point[1]) - cur_point[2],
            cuda_fabs(cur_point[0] - cur_point[1]) + cur_point[2]) - 1.0f) / 1.73205f; // 1.73205 == std::sqrt(3.)
        return size_ * len / scale;
    }

private:
    Vector center_;
    float size_;

    Vector vertices[4];
    int level_;
};
