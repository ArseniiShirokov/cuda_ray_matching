#pragma once

#include <vector.h>
#include <sdf.h>


#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))


class UnionSDF : public SDF{
public:
     __device__ UnionSDF(){};
     __device__ UnionSDF(SDF** objects, int num_obj, const Vector& color = Vector(0, 0, 0)) : SDF(color) {
        num_obj_ = num_obj;
        objects_ = new SDF*[num_obj];
        for (int i = 0; i < num_obj_; i++) {
            objects_[i] = objects[i];
        }
    };

    __device__ float ComputeSdf(const Vector &point) const override{
        float dist = 1e+5;
        for (int i = 0; i < num_obj_; i++) {
            dist = min(dist, objects_[i]->ComputeSdf(point));
        }
        return dist;
    }

    __device__ const SDF* GetHittedObject(const Vector &point) const {
        float min_dist = 1e+5;
        SDF* hitted_object;

        for (int i = 0; i < num_obj_; i++) {
            float dist = objects_[i]->ComputeSdf(point);
            if (dist < min_dist) {
                min_dist = dist;
                hitted_object = objects_[i];
            }
        }
        return hitted_object;
    }

private:
    SDF** objects_;
    int num_obj_;
};
