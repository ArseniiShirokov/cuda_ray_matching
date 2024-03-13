#pragma once

#include <array>
#include <cmath>
#include <iostream>
#include <initializer_list>
#include <algorithm>

class Vector {
public:
    __host__ __device__ Vector(float x = 0, float y = 0, float z = 0) {
        data_[0] = x;
        data_[1] = y;
        data_[2] = z;
    };
    __host__ __device__  Vector(const Vector &data) {
        data_[0] = data[0];
        data_[1] = data[1];
        data_[2] = data[2];
    };

    __host__ __device__ float& operator[](size_t ind) {
        return data_[ind];
    };
    __host__ __device__ float operator[](size_t ind) const {
        return data_[ind];
    };

    __host__ __device__ bool operator==(const Vector& gt) {
        return ((*this)[0] == gt[0]) && ((*this)[1] == gt[1]) && ((*this)[2] == gt[2]);
    }

    __host__ __device__ Vector& operator-=(const Vector& rhs) {
        for (size_t i = 0; i < (*this).Size(); ++i) {
            (*this)[i] = (*this)[i] - rhs[i];
        }
        return *this;
    }

    __host__ __device__ Vector& operator+=(const Vector& rhs) {
        for (size_t i = 0; i < (*this).Size(); ++i) {
            (*this)[i] += rhs[i];
        }
        return *this;
    }

    __host__ __device__ Vector operator-() const {
        Vector ans;
        for (size_t i = 0; i < (*this).Size(); ++i) {
            ans[i] = -(*this)[i];
        }
        return Vector(ans);
    }

    __host__ __device__ void Normalize();

    __host__ __device__ void Scale(float value) {
        for (size_t i = 0; i < (*this).Size(); ++i) {
            (*this)[i] *= value;
        }
    }

    __host__ __device__ size_t Size() const {
        return 3;
    }

private:
    float data_[3];
};

 __host__ __device__ inline Vector operator-(Vector lhs, const Vector& rhs) {
    return lhs -= rhs;
}

 __host__ __device__ inline Vector operator+(Vector lhs, const Vector& rhs) {
    return lhs += rhs;
}

 __host__ __device__ inline Vector operator*(Vector lhs, float value) {
    Vector ans;
    for (size_t i = 0; i < lhs.Size(); ++i) {
        ans[i] = lhs[i] * value;
    }
    return Vector(ans);
}

 __host__ __device__ inline Vector operator*(float value, Vector rhs) {
    return rhs * value;
}

 __host__ __device__ inline Vector operator/(Vector rhs, float value) {
    return rhs * (1 / value);
}

 __host__ __device__ inline Vector operator+(float value, Vector rhs) {
    return rhs += Vector({value, value, value});
}

 __host__ __device__ inline Vector operator*(const Vector& lhs, const Vector& rhs) {
    Vector ans;
    for (size_t i = 0; i < lhs.Size(); ++i) {
        ans[i] = lhs[i] * rhs[i];
    }
    return Vector(ans);
}

 __host__ __device__ inline Vector operator/(const Vector& lhs, const Vector& rhs) {
    Vector ans;
    for (size_t i = 0; i < lhs.Size(); ++i) {
        ans[i] = lhs[i] / rhs[i];
    }
    return Vector(ans);
}

 __host__ __device__ inline float DotProduct(const Vector& lhs, const Vector& rhs) {
    float ans = 0.0;
    for (size_t i = 0; i < lhs.Size(); ++i) {
        ans += lhs[i] * rhs[i];
    }
    return ans;
}

 __host__ __device__ inline Vector CrossProduct(const Vector& a, const Vector& b) {
    return Vector(
        {a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]});
}

 __host__ __device__ inline float Length(const Vector& vec) {
    return std::sqrt(DotProduct(vec, vec));
}

 __host__ __device__ void Vector::Normalize() {
    Scale(1.0f / Length(*this));
}
