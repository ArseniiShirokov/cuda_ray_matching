#include <camera_options.h>
#include <ray.h>
#include <math.h>

__host__ inline void ComputeRotationMat(Vector forward, Vector* result);
__device__ inline Vector Rotate(const Vector& base, Vector* mat);

class Transformer {
    Vector rotation_mat_[3];
    Vector new_origin_;
    float width_;
    float height_;
    float step_;

public:
    __host__ Transformer(const CameraOptions& options) {
        height_ = 2 * tan(options.fov / 2);
        width_ = (height_ * options.screen_width) / (float)(options.screen_height);
        new_origin_ = options.look_from;
        ComputeRotationMat(options.look_from - options.look_to, rotation_mat_);
        step_ = width_ / (float)options.screen_width;
    };

    __device__ Ray MakeRay(int y, int x) {
        Vector default_dir = {-width_ / 2 + (x + 0.5f) * step_, height_ / 2 - (y + 0.5f) * step_, -1};
        default_dir.Normalize();
        Vector rotated_dir = Rotate(default_dir, rotation_mat_);
        return Ray(new_origin_, rotated_dir);
    }
};

__device__ inline Vector Rotate(const Vector& base, Vector* mat) {
    Vector result;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i] += mat[j][i] * base[j];
        }
    }
    result.Normalize();
    return result;
}

 __host__ inline void ComputeRotationMat(Vector forward, Vector* result) {
    Vector tmp(0.0f, 0.1f, 0.0f);
    forward.Normalize();
    auto val = DotProduct(forward, tmp);
    val = val > 0 ? val : -val; //abs 
    if (val == 1) {
        tmp = {0, 0, -forward[1]};
    }
    auto right = CrossProduct(tmp, forward);
    auto up = CrossProduct(forward, right);
    right.Normalize();
    up.Normalize();
    result[0] = right;
    result[1] = up;
    result[2] = forward;
}
