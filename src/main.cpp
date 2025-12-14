#include "iostream"
#include "fstream"
#include "string"
#include "vector"

typedef unsigned int b32;
typedef signed long long s64;
typedef unsigned long long u64;
typedef signed int s32;
typedef unsigned int u32;
typedef unsigned char u8;
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
inline f32 Arctangent2(f32 a, f32 b)
{
    f32 result = atan2(a, b);
    return(result);
}
inline f32 Arccosine(f32 a)
{
    f32 result = acosf(a);
    return(result);
}
inline s32 Floor(f32 value)
{
    s32 result = (s32)floorf(value);
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
vec3 COLOR_MAGENTA = Vec3(1.0f, 0.0f, 1.0f);
vec3 COLOR_CYAN = Vec3(0.0f, 1.0f, 1.0f);

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

interval Interval(interval i1, interval i2)
{
    interval result;
    result.min = Minimum(i1.min, i2.min);
    result.max = Maximum(i1.max, i2.max);
    return(result);
}

inline f32 Clamp(f32 x, interval t)
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

inline f32 Clamp(f32 x, f32 min, f32 max)
{
    interval t = Interval(min, max);
    return(Clamp(x, t));
}

inline interval ExpandInterval(interval t, f32 delta)
{
    f32 padding = (delta / 2.0f);
    interval result = Interval(t.min - padding, t.max + padding);
    return(result);
}

interval INTERVAL_EMPTY = Interval(FLOAT_INFINITY, FLOAT_MINUS_INFINITY);
interval INTERVAL_UNIVERSE = Interval(FLOAT_MINUS_INFINITY, FLOAT_INFINITY); 
interval INTERVAL_INTENSITY = Interval(0.0f, 0.999f);
interval INTERVAL_UNIT = Interval(0, 1);

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
struct image_data
{
    s32 bytes_per_pixel;
    s32 bytes_per_scanline;
    s32 width;
    s32 height;
    u8 *data;
};
b32 LoadImage(image_data *img, std::string filename)
{
    s32 bytes_per_pixel = 3;
    img->data = stbi_load(filename.c_str(), &img->width, &img->height, &img->bytes_per_pixel, bytes_per_pixel);
    if(img->data)
    {
        img->bytes_per_scanline = img->width * img->bytes_per_pixel;
        return(1);
    }
    else
    {
        img->data = nullptr;
        return(0);
    }
}

vec3 GetImagePixel(image_data img, s32 x, s32 y)
{
    if(img.data)
    {
        x = Clamp(x, 0, img.width);
        y = Clamp(y, 0, img.height);
        u8 *pixel = img.data + (y * img.bytes_per_scanline) + (x * img.bytes_per_pixel);
        f32 byte_to_float  = 1.0f / 255.0f;
        vec3 pixel_color = Vec3(pixel[0] * byte_to_float, pixel[1] * byte_to_float, pixel[2] * byte_to_float);
        return(pixel_color);
    }
    else
    {
        return(COLOR_MAGENTA);
    }
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
inline s32 RandomInt(s32 min, s32 max)
{
    s32 result = int(RandomFloat((f32)min, (f32)(max + 1)));
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

enum texture_type : u32
{
    TextureType_SolidColor,
    TextureType_Checker,
    TextureType_Image,
};

struct texture
{
    texture_type type;
    vec3 albedo;

    vec3 even_albedo;
    vec3 odd_albedo;
    f32 scale;

    image_data img;
};

texture Texture(texture_type type, vec3 albedo)
{
    texture tex;
    tex.type;
    tex.albedo = albedo;
    return(tex);
}

texture CheckerTexture(vec3 even, vec3 odd, f32 scale)
{
    texture checker_texture;
    checker_texture.type = TextureType_Checker;
    checker_texture.albedo = COLOR_BLACK;
    checker_texture.even_albedo = even;
    checker_texture.odd_albedo = odd;   
    checker_texture.scale = scale;
    return(checker_texture);
}

texture ImageTexture(image_data img)
{
    texture image_texture;
    image_texture.type = TextureType_Image;
    image_texture.img = img;
    return(image_texture);
}

texture ImageTexture(std::string filename)
{
    image_data img;
    LoadImage(&img, filename);
    return(ImageTexture(img));
}

vec3 SampleTexture(texture tex, vec2 texture_coordinates, vec3 point)
{
    switch(tex.type)
    {
        case(TextureType_SolidColor):
        {
            return(tex.albedo);
        } break;
        case(TextureType_Checker):
        {
            f32 inv_scale = 1.0f / tex.scale;
            s32 x = (s32)(Floor(inv_scale * point.x));
            s32 y = (s32)(Floor(inv_scale * point.y));
            s32 z = (s32)(Floor(inv_scale * point.z));
            b32 is_even = ((x + y + z) % 2) == 0;
            if(is_even)
            {
                return(tex.even_albedo);
            }
            else
            {
                return(tex.odd_albedo);
            }
        } break;
        case(TextureType_Image):
        {
            if(tex.img.height <= 0)
            {
                return(COLOR_CYAN);
            }
            f32 u = Clamp(texture_coordinates.u, Interval(0.0f, 1.0f));
            f32 v = 1.0f - Clamp(texture_coordinates.v, Interval(0.0f, 1.0f));
            
            s32 x = (s32)(u * tex.img.width);
            s32 y = (s32)(v * tex.img.height);
            return(GetImagePixel(tex.img, x, y));
        } break;
    };
    return(COLOR_CYAN);
}

enum material_type : u32
{
    MaterialType_Lambertian,
    MaterialType_Metal,
    MaterialType_Dielectric,
    MaterialType_DiffuseLight,
};

struct material
{
    vec3 albedo;
    material_type type;
    f32 fuzziness;
    f32 refraction_index;
    texture tex;
    b32 has_texture;
    vec3 emitted_color;
};

material Material(vec3 albedo, material_type type, f32 fuzziness = 1.0f, f32 refraction_index = 1.0f)
{
    material result;
    result.albedo = albedo;
    result.type = type;
    result.fuzziness = fuzziness;
    result.refraction_index = refraction_index;
    result.has_texture = 0;
    result.emitted_color = COLOR_BLACK;
    return(result);
}

material Material(texture tex, material_type type, f32 fuzziness = 1.0f, f32 refraction_index = 1.0f)
{
    material result;
    result.albedo = tex.albedo;
    result.type = type;
    result.fuzziness = fuzziness;
    result.refraction_index = refraction_index;
    result.has_texture = 1;
    result.tex = tex;
    result.emitted_color = COLOR_BLACK;
    return(result);
}

material MaterialDiffuseLight(vec3 albedo, vec3 emitted_color)
{
    material result = Material(albedo, MaterialType_DiffuseLight);
    result.emitted_color = emitted_color;
    return(result);
}

material MaterialDiffuseLight(vec3 emitted_color)
{
    return(MaterialDiffuseLight(COLOR_WHITE, emitted_color));
}

material GRAY_MATTE_MATERIAL = Material(Vec3(0.5f, 0.5f, 0.5f), MaterialType_Lambertian);

struct hit_record
{
    vec3 point;
    vec3 normal;
    f32 t;
    b32 front_face;
    material mat;
    vec2 texture_coordinates;
    
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
            if(record->mat.has_texture)
            {
                (*attenuation_color) = SampleTexture(record->mat.tex, record->texture_coordinates, record->point);
            }
            else
            {
                (*attenuation_color) = record->mat.albedo;
            }
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

f32 LinearToGamma(f32 linear_component)
{
    if(linear_component > 0.0f)
    {
        return(pow(linear_component, 1.0f / 2.2f));
    }
    return(0.0f);
}

struct camera_frame_basis
{
    vec3 u;
    vec3 v;
    vec3 w;
};

struct axis_aligned_bounding_box
{
    interval x;
    interval y;
    interval z;

    inline interval GetInterval(s32 i)
    {
        if(i == 0)
        { 
            return(x);
        }
        else if(i == 1)
        {
            return(y);
        }
        else
        {
            return(z);
        }
    }

    inline s32 GetLongestAxis()
    {
        if(x.Size() > y.Size())
        {
            if(x.Size() > z.Size())
            {
                return(0);
            }
            else
            {
                return(2);
            }
        }
        else
        {
            if(y.Size() > z.Size())
            {
                return(1);
            }
            else
            {
                return(2);
            }
        }
        return(2);
    }

    inline void PadToMinimums()
    {
        f32 delta = 0.0001f;
        if(x.Size() < delta)
        {
            x = ExpandInterval(x, delta);
        }
        if(y.Size() < delta)
        {
            y = ExpandInterval(y, delta);
        }
        if(z.Size() < delta)
        {
            z = ExpandInterval(z, delta);
        }
    }
};

axis_aligned_bounding_box AxisAlignedBoundingBox(interval x, interval y, interval z)
{
    axis_aligned_bounding_box result;
    result.x = x;
    result.y = y;
    result.z = z;
    result.PadToMinimums();
    return(result);
}

axis_aligned_bounding_box AxisAlignedBoundingBox(interval i)
{
    return(AxisAlignedBoundingBox(i, i, i));
}

axis_aligned_bounding_box AxisAlignedBoundingBox(vec3 a, vec3 b)
{
    axis_aligned_bounding_box result;
    result.x = (a.x <= b.x) ? Interval(a.x, b.x) : Interval(b.x, a.x);
    result.y = (a.y <= b.y) ? Interval(a.y, b.y) : Interval(b.y, a.y);
    result.z = (a.z <= b.z) ? Interval(a.z, b.z) : Interval(b.z, a.z);
    result.PadToMinimums();
    return(result); 
}

axis_aligned_bounding_box AxisAlignedBoundingBox(axis_aligned_bounding_box box1, axis_aligned_bounding_box box2)
{
    axis_aligned_bounding_box result;
    result.x = Interval(box1.x, box2.x);
    result.y = Interval(box1.y, box2.y);
    result.z = Interval(box1.z, box2.z);
    return(result);
}

struct hittable_sphere
{
    ray3 center;
    f32 radius;
    material mat;
};

vec2 GetSphereUVCoordinates(hittable_sphere sphere, vec3 point)
{
    f32 theta = Arccosine(-point.y);
    f32 phi = Arctangent2(-point.z, point.x) + PI32;
    vec2 result;
    result.u = phi / (2 * PI32);
    result.v = theta / PI32;
    return(result);
}

struct hittable_quad
{
    vec3 Q;
    vec3 u;
    vec3 v;
    vec3 w;
    vec3 normal;
    f32 D;
    material mat;
};

enum object_type : u32
{
    ObjectType_Sphere,
    ObjectType_Quad,
    ObjectType_BVHNode,
};
struct hittable_object
{
    object_type type;

    axis_aligned_bounding_box aabb;
    hittable_object *left;
    hittable_object *right;

    hittable_sphere sphere;
    hittable_quad quad;
};

hittable_object CreateHittableSphere(ray3 center, f32 radius, material mat)
{
    hittable_object result;
    result.type = ObjectType_Sphere;
    result.sphere.center = center;
    result.sphere.radius = radius;
    result.sphere.mat = mat;
    vec3 radius_vec = Vec3(radius);
    axis_aligned_bounding_box aabb1 = AxisAlignedBoundingBox(result.sphere.center.GetPositionAt(0.0f) - radius_vec, result.sphere.center.GetPositionAt(0.0f) + radius_vec);
    axis_aligned_bounding_box aabb2 = AxisAlignedBoundingBox(result.sphere.center.GetPositionAt(1.0f) - radius_vec, result.sphere.center.GetPositionAt(1.0f) + radius_vec);
    result.aabb = AxisAlignedBoundingBox(aabb1, aabb2);
    return(result);
}

hittable_object CreateHittableSphere(vec3 center1, vec3 center2, f32 radius, material mat)
{
    return(CreateHittableSphere(Ray3(center1, center2 - center1), radius, mat));
}

hittable_object CreateHittableSphere(vec3 center, f32 radius, material mat)
{
    return(CreateHittableSphere(Ray3(center, Vec3(0.0f)), radius, mat));
}

hittable_object CreateHittableSphere(ray3 center, f32 radius)
{
    return(CreateHittableSphere(center, radius, GRAY_MATTE_MATERIAL));
}

hittable_object CreateHittableSphere(vec3 center, f32 radius)
{
    return(CreateHittableSphere(center, radius, GRAY_MATTE_MATERIAL));
}

hittable_object CreateHittableQuad(vec3 Q, vec3 u, vec3 v, material mat)
{
    hittable_object result;
    result.type = ObjectType_Quad;
    result.quad.Q = Q;
    result.quad.u = u;
    result.quad.v = v;
    result.quad.mat = mat;
    vec3 n = CrossProduct(u, v);
    result.quad.w = n / DotProduct(n, n);
    result.quad.normal = Normalize(n);
    result.quad.D = DotProduct(result.quad.normal, Q);
    axis_aligned_bounding_box aabb1 = AxisAlignedBoundingBox(Q, Q + u + v);
    axis_aligned_bounding_box aabb2 = AxisAlignedBoundingBox(Q + u, Q + v);
    result.aabb = AxisAlignedBoundingBox(aabb1, aabb2);
    return(result);  
}

hittable_object CreateHittableQuad(vec3 Q, vec3 u, vec3 v)
{
    return(CreateHittableQuad(Q, u, v, GRAY_MATTE_MATERIAL));
}

b32 AABBCompareAxis(hittable_object &a, hittable_object &b, s32 axis)
{
    return(a.aabb.GetInterval(axis).min < b.aabb.GetInterval(axis).min);
}
b32 AABBCompareX(hittable_object &a, hittable_object &b)
{
    return(AABBCompareAxis(a, b, 0));
}
b32 AABBCompareY(hittable_object &a, hittable_object &b)
{
    return(AABBCompareAxis(a, b, 1));
}
b32 AABBCompareZ(hittable_object &a, hittable_object &b)
{
    return(AABBCompareAxis(a, b, 2));
}

#include "algorithm"
void BuildBVHNode(hittable_object *root, std::vector<hittable_object> &objects, u32 start, u32 end)
{
    root->type = ObjectType_BVHNode;
    axis_aligned_bounding_box aabb = AxisAlignedBoundingBox(INTERVAL_EMPTY);
    for(u32 i = start; i < end; i++)
    {
        aabb = AxisAlignedBoundingBox(aabb, objects[i].aabb);
    }
    root->aabb = aabb;
    s32 axis = aabb.GetLongestAxis();
    auto comparison_func = (axis == 0) ? AABBCompareX 
                         : (axis == 1) ? AABBCompareY
                         : AABBCompareZ;
    u32 length = end - start;
    if(length == 1)
    {
        root->left = root->right = &objects[start];
    }
    else if(length == 2)
    {
        root->left = &objects[start];
        root->right = &objects[start + 1];
    }
    else
    {
        std::sort(std::begin(objects) + start, std::begin(objects) + end, comparison_func);
        u32 mid = start + (length / 2);
        root->left = (hittable_object*) malloc(sizeof(hittable_object));
        root->right = (hittable_object*) malloc(sizeof(hittable_object)); 
        BuildBVHNode(root->left, objects, start, mid);
        BuildBVHNode(root->right, objects, mid, end);
    }
    root->aabb = AxisAlignedBoundingBox(root->left->aabb, root->right->aabb);
}

b32 HitSphere(hittable_sphere sphere, ray3 ray, interval ray_interval, hit_record *record)
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
    record->texture_coordinates = GetSphereUVCoordinates(sphere, outward_normal);
    record->mat = sphere.mat;
    return(1);
}

b32 HitQuad(hittable_quad quad, ray3 ray, interval ray_interval, hit_record *record)
{
    f32 denominator = DotProduct(quad.normal, ray.direction);
    if(abs(denominator) < 1e-4)
    {
        return(0);
    }
    f32 t = (quad.D - DotProduct(quad.normal, ray.origin)) / denominator;
    if(!ray_interval.Contains(t))
    {
        return(0);
    }
    vec3 intersection_point = ray.GetPositionAt(t);
    vec3 planar_hitpoint_vector = intersection_point - quad.Q;
    f32 alpha = DotProduct(quad.w, CrossProduct(planar_hitpoint_vector, quad.v));
    f32 beta = DotProduct(quad.w, CrossProduct(quad.u, planar_hitpoint_vector));
    
    if(INTERVAL_UNIT.Contains(alpha) && INTERVAL_UNIT.Contains(beta))
    {
        record->texture_coordinates = Vec2(alpha, beta);
        record->t = t;
        record->point = intersection_point;
        record->mat = quad.mat;
        record->SetFaceNormal(ray, quad.normal);
        return(1);
    }
    else
    {
        return(0);
    }
}

b32 HitAABB(axis_aligned_bounding_box aabb, ray3 ray, interval ray_interval)
{
    vec3 origin = ray.origin;
    vec3 direction = ray.direction;
    for(s32 axis = 0; axis <= 2; axis++)
    {
        interval axis_interval = aabb.GetInterval(axis);
        f32 inverted_axis_direction = 1.0f / direction[axis];
        f32 t0 = (axis_interval.min - origin[axis]) * inverted_axis_direction;
        f32 t1 = (axis_interval.max - origin[axis]) * inverted_axis_direction;
        if(t0 < t1)
        {
            if(t0 > ray_interval.min)
            {
                ray_interval.min = t0;
            }
            if(t1 < ray_interval.max)
            {
                ray_interval.max = t1;
            }
        }
        else
        {
            if(t1 > ray_interval.min)
            {
                ray_interval.min = t1;
            }
            if(t0 < ray_interval.max)
            {
                ray_interval.max = t0;
            }
        }

        if(ray_interval.max <= ray_interval.min)
        {
            return(0);
        }
    }
    return(1);
}

b32 HitObject(hittable_object *object, ray3 ray, interval ray_interval, hit_record *record)
{
    switch(object->type)
    {
        case(ObjectType_Sphere):
        {
            return(HitSphere(object->sphere, ray, ray_interval, record));
        } break;
        case(ObjectType_Quad):
        {
            return(HitQuad(object->quad, ray, ray_interval, record));
        } break;
        case(ObjectType_BVHNode):
        {
            if(HitAABB(object->aabb, ray, ray_interval))
            {
                b32 hit_left = HitObject(object->left, ray, ray_interval, record);
                if(hit_left)
                {
                    ray_interval.max = record->t;
                }
                b32 hit_right = HitObject(object->right, ray, ray_interval, record);
                return(hit_left || hit_right);
            }
            else
            {
                return(0);
            }
        }
        default:
        {
            std::cout << "error. undefined object type.\n";
        } break;
    }
    return(0);
}

#if 0
b32 HitScene(std::vector<hittable_object> &scene, ray3 ray, interval ray_interval, hit_record *record)
{
    b32 has_ray_hit_anything = 0;
    f32 closest_so_far = ray_interval.max;
    for(auto &o : scene)
    {
        if(HitObject(&o, ray, Interval(ray_interval.min, closest_so_far), record))
        {
            has_ray_hit_anything = 1;
            closest_so_far = record->t;
        }
    }
    return(has_ray_hit_anything);
}
vec3 CalculateRayColor(std::vector<hittable_object> &scene, ray3 ray, s32 depth)
{
    if(depth <= 0)
    {
        return(COLOR_BLACK);
    }
    hit_record record;
    if(HitScene(scene, ray, Interval(0.001f, FLOAT_INFINITY), &record))
    {
        ray3 scattered_ray = ray;
        vec3 attenuation_color = COLOR_BLACK;
        vec3 emitted_color = record.mat.emitted_color;
        if(Scatter(&record, ray, &scattered_ray, &attenuation_color))
        {
            vec3 color_from_scatter = HadamardProduct(attenuation_color, CalculateRayColor(scene, scattered_ray, depth - 1)); 
            return(color_from_scatter + emitted_color);
        }
        else
        {
            return(emitted_color);
        }
    }
    else
    {
        f32 alpha = 0.5f * (Normalize(ray.direction).y + 1.0f);
        return(Lerp(COLOR_WHITE, COLOR_LIGHT_BLUE, alpha));
    }
}
#endif

vec3 CalculateRayColor(hittable_object *scene_root, ray3 ray, s32 depth)
{
    if(depth <= 0)
    {
        return(COLOR_BLACK);
    }
    hit_record record;
    if(HitObject(scene_root, ray, Interval(0.0001f, FLOAT_INFINITY), &record))
    {
        ray3 scattered_ray = ray;
        vec3 attenuation_color = COLOR_BLACK;
        vec3 emitted_color = record.mat.emitted_color;
        if(Scatter(&record, ray, &scattered_ray, &attenuation_color))
        {
            vec3 color_from_scatter = HadamardProduct(attenuation_color, CalculateRayColor(scene_root, scattered_ray, depth - 1)); 
            return(color_from_scatter + emitted_color);
        }
        else
        {
            return(emitted_color);
        }
    }
    else
    {
        return(COLOR_BLACK);
    }
}

enum scene_type : u32
{
    SceneType_Balls,
    SceneType_CornellBox,
    SceneType_Base,
};
void BuildScene(std::vector<hittable_object> &scene, scene_type type)
{
    // TODO: Adjust camera based on scene
    switch(type)
    {
        case(SceneType_Balls):
        {
            texture tex = CheckerTexture(COLOR_BLACK, COLOR_WHITE, 0.32f);
            material gray_ground_mat = Material(tex, MaterialType_Lambertian);
            scene.push_back(CreateHittableSphere(Vec3(0.0f, -1000.0f, -1.0f), 1000.0f, gray_ground_mat));

            material lambertian_mat = Material(Vec3(0.4f, 0.2f, 0.1f), MaterialType_Lambertian);
            scene.push_back(CreateHittableSphere(Vec3(-4.0f, 1.0f, 0.0f), 1.0f, lambertian_mat));

            material metal_mat = Material(Vec3(0.7f, 0.6f, 0.5f), MaterialType_Metal, 0.0f);
            scene.push_back(CreateHittableSphere(Vec3(4.0f, 1.0f, 0.0f), 1.0f, metal_mat));

            material dielectric_mat = Material(Vec3(1.0f, 1.0f, 1.0f), MaterialType_Dielectric, 1.0f, 1.5f);
            scene.push_back(CreateHittableSphere(Vec3(0.0f, 1.0f, 0.0f), 1.0f, dielectric_mat));

            texture earth_texture = ImageTexture("assets/earthmap.jpg");
            material earth_material = Material(earth_texture, MaterialType_Lambertian);
            scene.push_back(CreateHittableSphere(Vec3(7.0f, 1.0f, 2.0f), 0.6f, earth_material));

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
                            scene.push_back(CreateHittableSphere(center, center2, 0.2f, sphere_material));
                        }
                        else if(choose_material < 0.95f)
                        {
                            vec3 albedo = RandomVec3(0.5f, 1.0f);
                            f32 fuzziness = RandomFloat(0.0f, 0.5f);
                            sphere_material.type = MaterialType_Metal;
                            sphere_material.albedo = albedo;
                            sphere_material.fuzziness = fuzziness;
                            scene.push_back(CreateHittableSphere(center, 0.2f, sphere_material));
                        }
                        else
                        {
                            sphere_material.type = MaterialType_Dielectric;
                            sphere_material.albedo = COLOR_WHITE;
                            sphere_material.refraction_index = 1.5f;
                            scene.push_back(CreateHittableSphere(center, 0.2f, sphere_material));
                        }
                    }
                }
            }
        } break;

        case(SceneType_CornellBox):
        {
            material red = Material(Vec3(0.65f, 0.05f, 0.05f), MaterialType_Lambertian);
            material white = Material(Vec3(0.73f, 0.73f, 0.73f), MaterialType_Lambertian);
            material green = Material(Vec3(0.12f, 0.45f, 0.15f), MaterialType_Lambertian);
            material light = MaterialDiffuseLight(Vec3(15.0f)); 

            scene.push_back(CreateHittableQuad(Vec3(555, 0, 0), Vec3(0, 555, 0), Vec3(0, 0, 555), green));
            scene.push_back(CreateHittableQuad(Vec3(0, 0, 0), Vec3(0, 555, 0), Vec3(0, 0, 555), red));
            scene.push_back(CreateHittableQuad(Vec3(343, 554, 332), Vec3(-130, 0, 0), Vec3(0, 0, -105), light));
            scene.push_back(CreateHittableQuad(Vec3(0.0f), Vec3(555, 0, 0), Vec3(0, 0, 555), white));
            scene.push_back(CreateHittableQuad(Vec3(555.0f), Vec3(-555, 0, 0), Vec3(0, 0, -555), white));
            scene.push_back(CreateHittableQuad(Vec3(0, 0, 555), Vec3(555, 0, 0), Vec3(0, 555, 0), white));
        } break;

        default:
        {
        } break;
    }
}

int main(void)
{
    std::string output_path = "bin/out.ppm";
    std::ofstream output(output_path, std::ios::binary);    

    f32 aspect_ratio = (9.0f / 9.0f);
    s32 image_height = 240;
    s32 image_width = (s32)(image_height * aspect_ratio);

    // TODO: Customizable camera settings, separate camera viewport calculations
    vec3 camera_center =  Vec3(278.0f, 278.0f, -800.0f);
    vec3 camera_look_at = Vec3(278.0f, 278.0f, 0.0f);
    f32 vertical_fov = DEGREES_TO_RADIANS(40.0f);
    f32 focus_distance = 10.0f;
    f32 defocus_angle = 0.0f;

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

    s32 samples_per_pixel = 16;
    f32 pixel_samples_scale = 1.0f / ((f32)samples_per_pixel);

    s32 max_ray_recursion_depth = 64;

    std::vector<hittable_object> scene;
    BuildScene(scene, SceneType_CornellBox);
    hittable_object *scene_root = (hittable_object*)malloc(sizeof(hittable_object));
    BuildBVHNode(scene_root, scene, 0, scene.size());
    
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
                pixel_color = pixel_color + CalculateRayColor(scene_root, ray, max_ray_recursion_depth);
            }
            pixel_color = pixel_color * pixel_samples_scale;
            u8 r = (u8)(Clamp(LinearToGamma(pixel_color.x), INTERVAL_INTENSITY) * 255);
            u8 g = (u8)(Clamp(LinearToGamma(pixel_color.y), INTERVAL_INTENSITY) * 255);
            u8 b = (u8)(Clamp(LinearToGamma(pixel_color.z), INTERVAL_INTENSITY) * 255);
            output << r << g << b;
        }
        if(((y+1) % 10) == 0) std::cout << "scanlines: " << std::to_string(y + 1) << "/" << std::to_string(image_height) << "\n"; 
    }

    output.close();
}