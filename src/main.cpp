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
inline f32 Tangent(f32 value)
{
    f32 result = tanf(value);
    return(result);
}
#define FLOAT_INFINITY FLT_MAX
#define FLOAT_MINUS_INFINITY FLT_MIN
#define PI32 3.14159265359f
#define DEGREES_TO_RADIANS(degrees) ((degrees) * PI32 / 180.0f)
#define Minimum(A, B) ((A < B) ? (A) : (B))
#define Maximum(A, B) ((A > B) ? (A) : (B))

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

inline vec3 CrossProduct(vec3 v1, vec3 v2)
{
    vec3 result;
    result.x = v1.y * v2.z - v1.z * v2.y;
    result.y = v1.z * v2.x - v1.x * v2.z;
    result.z = v1.x * v2.y - v1.y * v2.x;
    return(result);
}

inline f32 LengthSquared(vec3 v1)
{
    f32 result = DotProduct(v1, v1);
    return(result);
}

inline f32 Length(vec3 v1)
{
    f32 result = SquareRoot(LengthSquared(v1));
    return(result);
}

inline vec3 Normalize(vec3 v1)
{
    f32 length = Length(v1);
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
    f32 time;   
    
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
    result.time = 0.0f;
    return(result);
}

ray3 Ray3(vec3 origin, vec3 direction, f32 time)
{
    ray3 result;
    result.origin = origin;
    result.direction = direction;
    result.time = time;
    return(result);
}

enum material_type
{
    MaterialType_Lambertian = 0,
    MaterialType_Metal = 1,
    MaterialType_Dielectric = 2,
};

struct material
{
    vec3 albedo;
    material_type type;
    f32 fuzziness;
    f32 refraction_index;
};

material Material(vec3 albedo, material_type type, f32 fuzziness = 1.0f, f32 refraction_index = 1.0f)
{
    material result;
    result.albedo = albedo;
    result.type = type;
    result.fuzziness = fuzziness;
    result.refraction_index = refraction_index;
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
    ray3 center;
    f32 radius;
    material mat;
};

material GRAY_MATTE_MATERIAL = Material(Vec3(0.5f, 0.5f, 0.5f), MaterialType_Lambertian);

hittable_sphere HittableSphere(vec3 center1, vec3 center2, f32 radius, material mat)
{
    hittable_sphere result;
    result.center = Ray3(center1, center2 - center1);
    result.radius = radius;
    result.mat = mat;
    return(result);
}

hittable_sphere HittableSphere(vec3 center, f32 radius, material mat)
{
    hittable_sphere result;
    result.center = Ray3(center, Vec3(0.0f));
    result.radius = radius;
    result.mat = mat;
    return(result);
}

hittable_sphere HittableSphere(ray3 center, f32 radius, material mat)
{
    hittable_sphere result;
    result.center = center;
    result.radius = radius;
    result.mat = mat;
    return(result);
}

hittable_sphere HittableSphere(ray3 center, f32 radius)
{
    return(HittableSphere(center, radius, GRAY_MATTE_MATERIAL));
}

hittable_sphere HittableSphere(vec3 center, f32 radius)
{
    return(HittableSphere(center, radius, GRAY_MATTE_MATERIAL));
}

b32 HitSphere(hit_record *record, interval ray_interval, ray3 ray, hittable_sphere sphere)
{
    vec3 current_center = sphere.center.GetPositionAt(ray.time);
    vec3 oc = current_center - ray.origin;
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
    vec3 outward_normal = (record->point - current_center) / sphere.radius;
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
vec3 RandomVectorOnUnitDisk()
{
    while(true)
    {
        vec3 point = Vec3(RandomFloat(-1.0f, 1.0f), RandomFloat(-1.0f, 1.0f), 0.0f);
        if(LengthSquared(point) < 1.0f)
        {
            return(point);
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

inline vec3 Refract(vec3 uv, vec3 normal, f32 refractive_index_ratio)
{
    f32 cos_theta = DotProduct(Normalize(-uv), normal);
    vec3 ray_out_perp = refractive_index_ratio * (uv + cos_theta * normal);
    vec3 ray_out_parallel = -SquareRoot(abs(1.0f - LengthSquared(ray_out_perp))) * normal;
    return(ray_out_perp + ray_out_parallel);
}

inline f32 ShlickReflectance(f32 cosine, f32 refraction_index)
{
    f32 r0 = (1.0f - refraction_index) / (1.0f + refraction_index);
    r0 = r0 * r0;
    f32 result = r0 + (1 - r0) * pow((1 - cosine), 5);
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
            (*scattered_ray) = Ray3(record->point, scatter_direction, incoming_ray.time);
            (*attenuation_color) = record->mat.albedo;
            return(1);
        } break;
        case(MaterialType_Metal):
        {
            vec3 reflected = Reflect(incoming_ray.direction, record->normal);
            reflected = Normalize(reflected) + (record->mat.fuzziness * RandomUnitVec3());
            (*scattered_ray) = Ray3(record->point, reflected, incoming_ray.time);
            (*attenuation_color) = record->mat.albedo;
            b32 is_above_surface = (DotProduct(reflected, record->normal) > 0);
            return(is_above_surface);
        } break;
        case(MaterialType_Dielectric):
        {
            f32 ri = record->mat.refraction_index;
            if(record->front_face)
            {
                ri = 1.0f / (record->mat.refraction_index);
            }
            vec3 incoming_direction = Normalize(incoming_ray.direction);
            f32 cos_theta = Minimum(DotProduct(-incoming_direction, record->normal), 1.0f);
            f32 sin_theta = SquareRoot(1.0f - (cos_theta * cos_theta));
            b32 cannot_refract = ((ri * sin_theta) > 1.0f);
            vec3 refracted;
            if(cannot_refract || (ShlickReflectance(cos_theta, ri) > RandomFloat()))
            {
                refracted = Reflect(incoming_direction, record->normal);
            }
            else
            {
                refracted = Refract(incoming_direction, record->normal, ri);
            }
            (*scattered_ray) = Ray3(record->point, refracted, incoming_ray.time);
            (*attenuation_color) = COLOR_WHITE;
            return(1);
        }
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

struct camera_frame_basis
{
    vec3 u;
    vec3 v;
    vec3 w;
};

inline ray3 GetRandomRayAtPosition(s32 x, s32 y, vec3 camera_center, vec3 upper_left_pixel_location, vec3 pixel_delta_horizontal, vec3 pixel_delta_vertical, vec3 defocus_disk_u, vec3 defocus_disk_v, f32 defocus_angle)
{
    vec3 sample_square = Vec3(RandomFloat() - 0.5f, RandomFloat() - 0.5f, 0.0f);
    f32 offset_x = (x + sample_square.x);
    f32 offset_y = (y + sample_square.y);
    vec3 pixel_sample = upper_left_pixel_location + (offset_x * pixel_delta_horizontal) + (offset_y * pixel_delta_vertical);

    vec3 ray_origin = camera_center;
    if(defocus_angle > 0)
    {
        vec3 point = RandomVectorOnUnitDisk();
        ray_origin = camera_center + (defocus_disk_u * point.u) + (defocus_disk_v * point.v);
    }
    f32 ray_time = RandomFloat();
    ray3 ray = Ray3(ray_origin, Normalize(pixel_sample - ray_origin), ray_time);
    return(ray);
}

void AddBookOneScene(std::vector<hittable_sphere> &sphere_list)
{
    for(s32 a = -11; a < 11; a++)
    {
        for(s32 b = -11; b < 11; b++)
        {
            vec3 center = Vec3(a + 0.9f * RandomFloat(), 0.2f, b + 0.9f * RandomFloat());
            if(Length(center - Vec3(4.0f, 0.2f, 0.0f)) > 0.9f)
            {
                material sphere_material = GRAY_MATTE_MATERIAL;
                f32 choose_material = RandomFloat();   
                if(choose_material < 0.8f)
                {
                    vec3 albedo = HadamardProduct(RandomVec3(), RandomVec3());
                    sphere_material.albedo = albedo;
                    vec3 center2 = center + Vec3(0.0f, RandomFloat(0.0f, 0.5f), 0.0f);
                    sphere_list.push_back(HittableSphere(center, center2, 0.2f, sphere_material));
                }
                else if(choose_material < 0.95f)
                {
                    vec3 albedo = RandomVec3(0.5f, 1.0f);
                    f32 fuzziness = RandomFloat(0.0f, 0.5f);
                    sphere_material.type = MaterialType_Metal;
                    sphere_material.albedo = albedo;
                    sphere_material.fuzziness = fuzziness;
                    sphere_list.push_back(HittableSphere(center, 0.2f, sphere_material));
                }
                else
                {
                    sphere_material.type = MaterialType_Dielectric;
                    sphere_material.albedo = COLOR_WHITE;
                    sphere_material.refraction_index = 1.5f;
                    sphere_list.push_back(HittableSphere(center, 0.2f, sphere_material));
                }
            }
        }
    }
}

int main(void)
{
    std::string output_path = "bin/out.ppm";
    std::ofstream output(output_path, std::ios::binary);    

    f32 aspect_ratio = (16.0f / 9.0f);
    s32 image_height = 256;
    s32 image_width = (s32)(image_height * aspect_ratio);

    // TODO: Customizable camera settings, separate camera viewport calculations
    vec3 camera_center =  Vec3(13.0f, 2.0f, 3.0f);
    vec3 camera_look_at = Vec3(0.0f, 0.0f, 0.0f);
    f32 vertical_fov = DEGREES_TO_RADIANS(20.0f);
    f32 focus_distance = 10.0f;
    f32 defocus_angle = 0.6f;

    vec3 camera_up = Vec3(0.0f, 1.0f, 0.0f);
    camera_frame_basis camera_frame;
    camera_frame.w = Normalize(camera_center - camera_look_at);
    camera_frame.u = Normalize(CrossProduct(camera_up, camera_frame.w));
    camera_frame.v = CrossProduct(camera_frame.w, camera_frame.u);

    f32 fov_ratio = Tangent(vertical_fov / 2.0f);
    f32 viewport_height = 2.0f * fov_ratio * focus_distance;
    f32 viewport_width = viewport_height * ((f32)image_width / (f32)image_height);

    vec3 viewport_horizontal = viewport_width * camera_frame.u;
    vec3 viewport_vertical = viewport_height * (-camera_frame.v);
    vec3 viewport_upper_left = camera_center - (focus_distance * camera_frame.w) - (viewport_horizontal / 2.0f) - (viewport_vertical / 2.0f);
    vec3 pixel_delta_horizontal = viewport_horizontal / image_width;
    vec3 pixel_delta_vertical = viewport_vertical / image_height;
    vec3 upper_left_pixel_location = viewport_upper_left + 0.5f * (pixel_delta_horizontal + pixel_delta_vertical);

    f32 defocus_radius = focus_distance * Tangent(DEGREES_TO_RADIANS(defocus_angle / 2.0f));
    vec3 defocus_disk_u = camera_frame.u * defocus_radius;
    vec3 defocus_disk_v = camera_frame.v * defocus_radius;

    s32 samples_per_pixel = 8;
    f32 pixel_samples_scale = 1.0f / ((f32)samples_per_pixel);

    s32 max_ray_recursion_depth = 32;

    std::vector<hittable_sphere> sphere_list;
    material gray_ground_mat = Material(Vec3(0.5f, 0.5f, 0.5f), MaterialType_Lambertian);
    sphere_list.push_back(HittableSphere(Vec3(0.0f, -1000.0f, -1.0f), 1000.0f, gray_ground_mat));

    material lambertian_mat = Material(Vec3(0.4f, 0.2f, 0.1f), MaterialType_Lambertian);
    sphere_list.push_back(HittableSphere(Vec3(-4.0f, 1.0f, 0.0f), 1.0f, lambertian_mat));

    material metal_mat = Material(Vec3(0.7f, 0.6f, 0.5f), MaterialType_Metal, 0.0f);
    sphere_list.push_back(HittableSphere(Vec3(4.0f, 1.0f, 0.0f), 1.0f, metal_mat));

    material dielectric_mat = Material(Vec3(1.0f, 1.0f, 1.0f), MaterialType_Dielectric, 1.0f, 1.5f);
    sphere_list.push_back(HittableSphere(Vec3(0.0f, 1.0f, 0.0f), 1.0f, dielectric_mat));

    AddBookOneScene(sphere_list);

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
                ray3 ray = GetRandomRayAtPosition(x, y, camera_center, upper_left_pixel_location, pixel_delta_horizontal, pixel_delta_vertical, defocus_disk_u, defocus_disk_v, defocus_angle);
                pixel_color = pixel_color + GetRayColor(ray, sphere_list, max_ray_recursion_depth);
            }
            pixel_color = pixel_color * pixel_samples_scale;
            WritePixel(output, pixel_color);
        }
    }

    output.close();
}