#pragma once

#include <sdf.h>
#include <vector.h>
#include <light.h>

#include <vector>
#include <string>
#include <initializer_list>
#include <union_sdf.h>


class Scene {
public:
    __host__ __device__ Scene(UnionSDF* union_object, Light light)
        : union_object_(union_object),
          light_(light) {
    }

    __device__ const UnionSDF* GetUnionObject() const {
        return union_object_;
    }

    __device__ const Light& GetLight() const {
        return light_;
    }


private:
    UnionSDF* union_object_;
    Light light_;
};
