#include "iostream"
#include "fstream"
#include "string"
#include "vector"

typedef unsigned char b8;
typedef unsigned int b32;
typedef signed long long s64;
typedef unsigned long long u64;
typedef signed int s32;
typedef unsigned int u32;
typedef float f32;

// TODO: Intrinsics for math primitives
#include "math.h"
#include "float.h"
inline f32 SquareRoot(f32 value)
{
    f32 result = sqrtf(value);
    return(result);
}
#define FLOAT_INFINITY FLT_MAX
#define FLOAT_MINUS_INFINITY FLT_MIN
#define PI32 3.14159265359f
#define DEGREES_TO_RADIANS(degrees) ((degrees) * PI32 / 180.0f)

union vec2
{
    f32 e[2];
    struct
    {
        f32 x, y;
    };
    struct
    {
        f32 u, v;
    };

    inline f32 &operator[](int i)
    {
        return(e[i]);
    }
};

vec2 Vec2(f32 x, f32 y)
{
    vec2 result;
    result.x = x;
    result.y = y;
    return(result);
}

vec2 Vec2(f32 x)
{
    vec2 result = Vec2(x, x);
    return(result);
}

inline vec2 operator*(vec2 v, f32 f)
{
    vec2 result;
    result.x = v.x * f;
    result.y = v.y * f;
    return(result);
}

inline vec2 operator/(vec2 v, f32 f)
{
    vec2 result;
    result.x = v.x / f;
    result.y = v.y / f;
    return(result);
}

inline vec2 operator+(vec2 v1, vec2 v2)
{
    vec2 result;
    result.x = v1.x + v2.x;
    result.y = v1.y + v2.y;
    return(result);
}

inline vec2 operator-(vec2 v1, vec2 v2)
{
    vec2 result;
    result.x = v1.x - v2.x;
    result.y = v1.y - v2.y;
    return(result);
}

union vec3
{
    f32 e[3];
    struct
    {
        f32 x, y, z;
    };
    struct
    {
        f32 u, v, w;
    };
    struct
    {
        vec2 xy;
        f32 ignored_z;
    };
    struct
    {
        f32 ignored_x;
        vec2 yz;
    };
    struct
    {
        f32 ignored_u;
        vec2 vw;
    };
    struct
    {
        vec2 uv;
        f32 ignored_w;
    };

    inline f32 &operator[](int i)
    {
        return(e[i]);
    }
};

vec3 Vec3(f32 x, f32 y, f32 z)
{
    vec3 result;
    result.x = x;
    result.y = y;
    result.z = z;
    return(result);
}

vec3 Vec3(f32 x)
{
    vec3 result = Vec3(x, x, x);
    return(result);
}

vec3 Vec3(vec2 v, f32 z)
{
    vec3 result;
    result.x = v.x;
    result.y = v.y;
    result.z = z;
    return(result);
}

vec3 COLOR_BLACK = Vec3(0.0f);
vec3 COLOR_WHITE = Vec3(1.0f);
vec3 COLOR_LIGHT_BLUE = Vec3(0.5f, 0.7f, 1.0f);
vec3 COLOR_RED = Vec3(1.0f, 0.0f, 0.0f);

inline vec3 operator*(vec3 v, f32 f)
{
    vec3 result;
    result.x = v.x * f;
    result.y = v.y * f;
    result.z = v.z * f;
    return(result);
}

inline vec3 operator*(f32 f, vec3 v)
{
    vec3 result;
    result.x = v.x * f;
    result.y = v.y * f;
    result.z = v.z * f;
    return(result);
}

inline vec3 operator/(vec3 v, f32 f)
{
    vec3 result;
    result.x = v.x / f;
    result.y = v.y / f;
    result.z = v.z / f;
    return(result);
}

inline vec3 operator+(vec3 v1, vec3 v2)
{
    vec3 result;
    result.x = v1.x + v2.x;
    result.y = v1.y + v2.y;
    result.z = v1.z + v2.z;
    return(result);
}

inline vec3 operator-(vec3 v1, vec3 v2)
{
    vec3 result;
    result.x = v1.x - v2.x;
    result.y = v1.y - v2.y;
    result.z = v1.z - v2.z;
    return(result);
}

inline vec3 operator-(vec3 v1)
{
    vec3 result;
    result.x = -v1.x;
    result.y = -v1.y;
    result.z = -v1.z;
    return(result);
}

inline vec3 HadamardProduct(vec3 v1, vec3 v2)
{
    vec3 result;
    result.x = v1.x * v2.x;
    result.y = v1.y * v2.y;
    result.z = v1.z * v2.z;
    return(result);
}

inline f32 DotProduct(vec3 v1, vec3 v2)
{
    f32 result = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    return(result);
}

inline f32 LengthSquared(vec3 v1)
{
    f32 result = DotProduct(v1, v1);
    return(result);
}

inline f32 VectorLength(vec3 v1)
{
    f32 result = SquareRoot(LengthSquared(v1));
    return(result);
}

inline vec3 Normalize(vec3 v1)
{
    f32 length = VectorLength(v1);
    vec3 result;
    result.x = v1.x / length;
    result.y = v1.y / length;
    result.z = v1.z / length;
    return(result);
}

inline b32 NearZero(vec3 v1)
{
    f32 zero = 1e-8;
    if((abs(v1.x) < zero) && (abs(v1.y) < zero) && (abs(v1.z) < zero))
    {
        return(1);
    }
    return(0);
}

void PrintVector(vec3 v1)
{
    std::cout << v1.x << " " << v1.y << " " << v1.z << std::endl;
}

struct ray3
{
    vec3 origin;
    vec3 direction;
        
    vec3 GetPositionAt(f32 t)
    {
        vec3 result = origin + (direction * t);
        return(result);
    }
};

ray3 Ray3(vec3 origin, vec3 direction)
{
    ray3 result;
    result.origin = origin;
    result.direction = direction;
    return(result);
}

enum material_type
{
    MaterialType_Lambertian = 0,
    MaterialType_Metal = 1,
};

struct material
{
    vec3 albedo;
    material_type type;
};

material Material(vec3 albedo, material_type type)
{
    material result;
    result.albedo = albedo;
    result.type = type;
    return(result);
}

struct hit_record
{
    vec3 point;
    vec3 normal;
    f32 t;
    b32 front_face;
    material mat;
    
    void SetFaceNormal(ray3 ray, vec3 outward_normal)
    {
        // TODO: We're always normalizing inside the function, 
        // although we could also assume the input vector to already be normalized
        f32 test = DotProduct(Normalize(ray.direction), Normalize(outward_normal));
        if(test < 0)
        {
            front_face = 1;
        }
        else
        {
            front_face = 0;
        }

        if(front_face)
        {
            normal = outward_normal; 
        }
        else
        {
            normal = -outward_normal;
        }
    }
};

struct interval
{
    f32 min;
    f32 max;
    inline f32 Size()
    {
        return(max - min);
    }
    inline b32 Contains(f32 x)
    {
        if((min <= x) && (x <= max))
        {
            return(1);
        }
        else
        {
            return(0);
        }
    }
    inline b32 Surrounds(f32 x)
    {
        if((min < x) && (x < max))
        {
            return(1);
        }
        else
        {
            return(0);
        }
    }
};

interval Interval(f32 min, f32 max)
{
    interval result;
    result.min = min;
    result.max = max;
    return(result);
}

inline f32 Clamp(interval t, f32 x)
{
    if(x < t.min)
    {
        return(t.min);
    }
    else if(x > t.max)
    {
        return(t.max);
    }
    return(x);
}

interval INTERVAL_EMPTY = Interval(FLOAT_INFINITY, FLOAT_MINUS_INFINITY);
interval INTERVAL_UNIVERSE = Interval(FLOAT_MINUS_INFINITY, FLOAT_INFINITY); 
interval INTERVAL_INTENSITY = Interval(0.0f, 0.999f);

struct hittable_sphere
{
    vec3 center;
    f32 radius;
    material mat;
};

hittable_sphere HittableSphere(vec3 center, f32 radius, material mat)
{
    hittable_sphere result;
    result.center = center;
    result.radius = radius;
    result.mat = mat;
    return(result);
}

b32 HitSphere(hit_record *record, interval ray_interval, ray3 ray, hittable_sphere sphere)
{
    // Solve the ray-sphere intersection equation and find the roots
    vec3 oc = sphere.center - ray.origin;
    f32 a = LengthSquared(ray.direction);
    f32 h = DotProduct(ray.direction, oc);
    f32 c = LengthSquared(oc) - (sphere.radius * sphere.radius);
    f32 discriminant = h*h - a*c;
    if(discriminant < 0)
    {
        return(0);
    }
    f32 sqrt_d = SquareRoot(discriminant);
    f32 root = (h - sqrt_d) / a;
    if(!ray_interval.Surrounds(root))
    {
        root = (h + sqrt_d) / a;
        if(!ray_interval.Surrounds(root))
        {
            return(0);
        }
    }

    record->t = root;
    record->point = ray.GetPositionAt(root);
    vec3 outward_normal = (record->point - sphere.center) / sphere.radius;
    record->SetFaceNormal(ray, outward_normal);
    record->mat = sphere.mat;
    return(1);
}

b32 HitSphereList(hit_record *record, interval ray_interval, ray3 ray, std::vector<hittable_sphere> &sphere_list)
{
    b32 has_ray_hit_anything = 0;
    f32 closest_so_far = ray_interval.max;
    for(auto &sphere : sphere_list)
    {
        if(HitSphere(record, Interval(ray_interval.min, closest_so_far), ray, sphere))
        {
            has_ray_hit_anything = 1;
            closest_so_far = record->t;
        }
    }
    return(has_ray_hit_anything);
}

// TODO: RNG
#include "cstdlib"
inline f32 RandomFloat()
{
    f32 result = (f32)(std::rand()) / (RAND_MAX + 1.0f);
    return(result);
}
inline f32 RandomFloat(f32 min, f32 max)
{
    f32 result = min + (max - min) * RandomFloat();
    return(result);
}

vec3 Lerp(vec3 A, vec3 B, f32 alpha)
{
    vec3 result = ((1.0f - alpha) * A) + (alpha * B);
    return(result);
}

ray3 GetRandomRayAtPosition(s32 x, s32 y, vec3 camera_center, vec3 upper_left_pixel_location, vec3 pixel_delta)
{
    vec3 sample_square = Vec3(RandomFloat() - 0.5f, RandomFloat() - 0.5f, 0.0f);
    f32 offset_x = (x + sample_square.x);
    f32 offset_y = (y + sample_square.y);
    vec3 pixel_sample = upper_left_pixel_location + Vec3(offset_x * pixel_delta.x, offset_y * pixel_delta.y, 0.0f);
    ray3 ray = Ray3(camera_center, Normalize(pixel_sample - camera_center));
    return(ray);
}

vec3 RandomVec3()
{
    vec3 result = Vec3(RandomFloat(), RandomFloat(), RandomFloat());
    return(result);
}
vec3 RandomVec3(f32 min, f32 max)
{
    vec3 result = Vec3(RandomFloat(min, max), RandomFloat(min, max), RandomFloat(min, max));
    return(result);
}
vec3 RandomUnitVec3()
{
    while(true)
    {
        vec3 point = RandomVec3(-1.0f, 1.0f);
        f32 length_squared = LengthSquared(point);
        // Reject vectors that are outside the unit sphere,
        // but also reject ones that are too close to the center 
        // and can cause floating-point issues
        if((length_squared <= 1.0f) && (length_squared > 1e-18))
        {
            return(point / (SquareRoot(length_squared)));
        }
    }
}
vec3 RandomVectorOnHemisphere(vec3 normal)
{
    vec3 unit_sphere_vector = RandomUnitVec3();
    if(DotProduct(unit_sphere_vector, normal) > 0.0f)
    {
        return(unit_sphere_vector);
    }
    else
    {
        return(-unit_sphere_vector);
    }
}

inline vec3 Reflect(vec3 v1, vec3 normal)
{
    vec3 result = v1 - 2 * DotProduct(v1, normal) * normal;
    return(result); 
}

b32 Scatter(hit_record *record, ray3 incoming_ray, ray3 *scattered_ray, vec3 *attenuation_color)
{
    switch(record->mat.type)
    {
        case(MaterialType_Lambertian):
        {
            vec3 scatter_direction = record->normal + RandomUnitVec3();
            if(NearZero(scatter_direction))
            {
                scatter_direction = record->normal;
            }
            (*scattered_ray) = Ray3(record->point, scatter_direction);
            (*attenuation_color) = record->mat.albedo;
            return(1);
        } break;
        case(MaterialType_Metal):
        {
            vec3 reflected = Reflect(incoming_ray.direction, record->normal);
            (*scattered_ray) = Ray3(record->point, reflected);
            (*attenuation_color) = record->mat.albedo;
            return(1);
        } break;
    }
    return(0);
}

vec3 GetRayColor(ray3 ray, std::vector<hittable_sphere> sphere_list, s32 depth)
{
    if(depth <= 0)
    {
        return(COLOR_BLACK);
    }
    hit_record record;
    if(HitSphereList(&record, Interval(0.001f, FLOAT_INFINITY), ray, sphere_list))
    {
        ray3 scattered_ray = ray;
        vec3 attenuation_color = COLOR_BLACK;
        if(Scatter(&record, ray, &scattered_ray, &attenuation_color))
        {
            return(HadamardProduct(attenuation_color, GetRayColor(scattered_ray, sphere_list, depth - 1)));
        }
        else
        {
            return(COLOR_BLACK);
        }
    }
    else
    {
        f32 alpha = 0.5f * (Normalize(ray.direction).y + 1.0f);
        return(Lerp(COLOR_WHITE, COLOR_LIGHT_BLUE, alpha));
    }
}

f32 LinearToGamma(f32 linear_component)
{
    if(linear_component > 0.0f)
    {
        return(pow(linear_component, 1.0f / 2.2f));
    }
    return(0.0f);
}
void WritePixel(std::ostream &out, vec3 color)
{
    b8 r = (b8)(Clamp(INTERVAL_INTENSITY, LinearToGamma(color.x)) * 255);
    b8 g = (b8)(Clamp(INTERVAL_INTENSITY, LinearToGamma(color.y)) * 255);
    b8 b = (b8)(Clamp(INTERVAL_INTENSITY, LinearToGamma(color.z)) * 255);
    out << r << g << b;
}
int main(void)
{
    std::string output_path = "bin/out.ppm";
    std::ofstream output(output_path, std::ios::binary);    

    f32 aspect_ratio = (16.0f / 9.0f);
    s32 image_height = 256;
    s32 image_width = (s32)(image_height * aspect_ratio);

    f32 focal_length = 1.0f;
    f32 viewport_height = 2.0f;
    f32 viewport_width = viewport_height * ((f32)image_width / (f32)image_height);

    vec3 camera_center = Vec3(0.0f);
    vec3 viewport = Vec3(viewport_width, -viewport_height, 0.0f);
    vec3 pixel_delta = Vec3(viewport.u / image_width, viewport.v / image_height, 0.0f);
    vec3 viewport_upper_left = camera_center - Vec3(0.0f, 0.0f, focal_length) - (viewport / 2.0f);
    vec3 upper_left_pixel_location = viewport_upper_left + (pixel_delta * 0.5f);

    s32 samples_per_pixel = 16;
    f32 pixel_samples_scale = 1.0f / ((f32)samples_per_pixel);

    s32 max_ray_recursion_depth = 32;    

    material gray_ground_mat = Material(Vec3(0.1f, 0.1f, 0.1f), MaterialType_Lambertian);
    material blue_metal_mat = Material(Vec3(0.15f, 0.15f, 0.9f), MaterialType_Metal);
    material red_sphere_mat = Material(Vec3(0.9f, 0.15f, 0.15f), MaterialType_Lambertian);

    std::vector<hittable_sphere> sphere_list;
    sphere_list.push_back(HittableSphere(Vec3(-0.25f, 0.0f, -0.75f), 0.35f, blue_metal_mat));
    sphere_list.push_back(HittableSphere(Vec3(0.5f, 0.0f, -0.75f), 0.35f, red_sphere_mat));
    sphere_list.push_back(HittableSphere(Vec3(0.0f, -100.5f, -1.0f), 100.0f, gray_ground_mat));

    output << "P6\n";
    output << std::to_string(image_width) << " " << std::to_string(image_height) << "\n";
    output << "255\n";
    for(int y = 0; y < image_height; y++)
    {
        for(int x = 0; x < image_width; x++)
        {
            vec3 pixel_color = COLOR_BLACK;
            for(s32 sample = 0; sample < samples_per_pixel; sample++)
            {
                ray3 ray = GetRandomRayAtPosition(x, y, camera_center, upper_left_pixel_location, pixel_delta);
                pixel_color = pixel_color + GetRayColor(ray, sphere_list, max_ray_recursion_depth);
            }
            pixel_color = pixel_color * pixel_samples_scale;
            WritePixel(output, pixel_color);
        }
    }

    output.close();
}