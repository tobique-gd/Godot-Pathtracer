#[compute]
#version 450

#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require

struct Triangle {
    vec4 v0;
    vec4 edge1;
    vec4 edge2;
};

struct Normal {
    f16vec3 n0;
    f16vec3 n1;
    f16vec3 n2;
};

struct Material {
    vec4 color;
    vec4 specular;
    vec4 emission;

    float roughness;
    float metallic;
    float transmittance;

    float ior;

    int albedo_texture_id;
    int normal_texture_id;
    int orm_texture_id;
    float padding;
};

struct Ray {
    vec3 origin;
    vec3 dir;
    vec3 inv_dir;
};

struct HitInfo {
    float t;
    float u;
    float v;
    uint triangle_index;
    bool isBackface;
};

struct BVHNode {
    vec4 aabb_min;
    vec4 aabb_max;
    uint child_index;
    uint triangle_offset;
    uint triangle_count;
};

struct Light {
    uint triangle_index;
    float area;

};

struct LightSample {
    vec3 position;
    vec3 normal;
    vec3 emission;
};

struct BRDFSample {
    vec3 direction;
    vec3 weight;
    float pdf;
    bool specular;
};

struct UV {
    vec4 uv0_uv1;
    vec2 uv2;
    float padding[2];
};

struct Surface {
    vec3 albedo;
    vec3 normal;
    vec3 geometric_normal;

    float roughness;
    float metallic;
    float transmission;
    float ior;

    vec3 emission;
    float emissionStrength;

    bool backface;
    float specular;
};

layout(std430, set = 0, binding = 1) readonly buffer TriangleBuffer {
    Triangle triangles[];
};


layout(std430, set = 0, binding = 2) readonly buffer NormalBuffer {
    Normal normals[];
};

layout(std430, set = 0, binding = 3) readonly buffer MaterialBuffer {
    Material materials[];
};

layout(std430, set = 0, binding = 4) readonly buffer BVH {
    BVHNode nodes[];
};

layout(std430, set = 0, binding = 5) readonly buffer LightBuffer {
    Light lights[];
};

layout(std430, set = 0, binding = 7) readonly buffer UVBuffer {
    UV uvs[];
};


layout(rgba16f, set = 0, binding = 0) uniform image2D color_image;
layout(rgba16f, set = 0, binding = 6) uniform image2D accumulation_image;

layout(set = 2, binding = 0) uniform sampler2DArray albedo_texture_array;
layout(set = 2, binding = 1) uniform sampler2DArray normal_texture_array;
layout(set = 2, binding = 2) uniform sampler2DArray orm_texture_array;

layout(push_constant, std430) uniform Params {
    vec2 raster_size;
    vec2 padding;
    vec4 view_params;
    mat4 camera_transform;
} params;


const float EPSILON = 1e-4;
const int MAX_BOUNCES = 5;

const float INF = 999999999999.0;

uint hash(uint x)
{
    x += (x << 10u);
    x ^= (x >> 6u);
    x += (x << 3u);
    x ^= (x >> 11u);
    x += (x << 15u);
    return x;
}

uint pcg(inout uint state)
{
    uint old = state;

    state = old * 747796405u + 2891336453u;

    uint word = ((old >> ((old >> 28u) + 4u)) ^ old) * 277803737u;
    return (word >> 22u) ^ word;
}

float random(inout uint state)
{
    return float(pcg(state)) * (1.0 / 4294967296.0);
}

float random_value_normal_distribution(inout uint st) {
    float theta = 2.0 * 3.14159265359 * random(st);
    float rho = sqrt(-2.0 * log(random(st)));
    return rho * cos(theta);
}

vec3 random_direction(inout uint st) {
    float x = random_value_normal_distribution(st);
    float y = random_value_normal_distribution(st);
    float z = random_value_normal_distribution(st);

    return normalize(vec3(x, y, z));
}

vec3 random_hemisphere_direction(vec3 normal, inout uint st) {
    vec3 dir = random_direction(st);
    return dir * sign(dot(dir, normal));
}

void build_onb(vec3 n, out vec3 tangent, out vec3 bitangent)
{
    if (n.z < -0.9999999)
    {
        tangent   = vec3(0.0, -1.0, 0.0);
        bitangent = vec3(-1.0, 0.0, 0.0);
        return;
    }

    float a = 1.0 / (1.0 + n.z);
    float b = -n.x * n.y * a;

    tangent = vec3(
        1.0 - n.x * n.x * a,
        b,
        -n.x
    );

    bitangent = vec3(
        b,
        1.0 - n.y * n.y * a,
        -n.y
    );
}

vec3 sample_color(Material material, vec2 uv)
{
    if (material.albedo_texture_id < 0)
        return material.color.rgb;

    vec3 tex = texture(
        albedo_texture_array,
        vec3(uv, float(material.albedo_texture_id))
    ).rgb;

    return material.color.rgb * tex;
}

vec3 sample_normal(Material material, vec2 uv, vec3 geom_normal)
{
    if (material.normal_texture_id < 0)
        return geom_normal;

    vec3 tex = texture(
        normal_texture_array,
        vec3(uv, float(material.normal_texture_id))
    ).rgb;

    tex = tex * 2.0 - 1.0;

    vec3 tangent;
    vec3 bitangent;
    build_onb(geom_normal, tangent, bitangent);

    return normalize(tangent * tex.x + bitangent * tex.y + geom_normal * tex.z);
}

vec3 sample_orm(Material material, vec2 uv)
{
    if (material.orm_texture_id < 0)
        return vec3(material.roughness, material.metallic, material.transmittance);

    vec3 tex = texture(
        orm_texture_array,
        vec3(uv, float(material.orm_texture_id))
    ).rgb;

    return vec3(tex.r * material.roughness, tex.g * material.metallic, tex.b * material.transmittance);
}

vec3 cosine_sample_hemisphere(vec3 normal, inout uint st)
{
    float r1 = random(st);
    float r2 = random(st);

    float phi = 6.28318530718 * r1;

    float r = sqrt(r2);

    float x = cos(phi) * r;
    float y = sin(phi) * r;
    float z = sqrt(1.0 - r2);

    vec3 tangent;
    vec3 bitangent;

    build_onb(normal, tangent, bitangent);

    return tangent * x +
           bitangent * y +
           normal * z;
}

float intersect_aabb_dist(Ray ray, vec3 aabb_min, vec3 aabb_max) {

    vec3 t0s = (aabb_min - ray.origin) * ray.inv_dir;
    vec3 t1s = (aabb_max - ray.origin) * ray.inv_dir;

    vec3 tsmaller = min(t0s, t1s);
    vec3 tbigger = max(t0s, t1s);

    float tmin = max(max(tsmaller.x, tsmaller.y), tsmaller.z);
    float tmax = min(min(tbigger.x, tbigger.y), tbigger.z);

    if (tmax >= max(tmin, 0.0)) {
        return tmin;
    }

    return INF;
}
const float PI = 3.14159265359;

float saturate(float x) {
    return clamp(x, 0.0, 1.0);
}
float D_GGX(float NoH, float alpha)
{
    float a2 = alpha * alpha;
    float d = NoH * NoH * (a2 - 1.0) + 1.0;

    return a2 / (PI * d * d);
}


float G1_SmithGGX(float NoV, float alpha)
{
    float a2 = alpha * alpha;

    return 2.0 * NoV /
        (NoV + sqrt(a2 + (1.0 - a2) * NoV * NoV));
}


float G_Smith(float NoV, float NoL, float alpha)
{
    return G1_SmithGGX(NoV, alpha) *
           G1_SmithGGX(NoL, alpha);
}


vec3 FresnelSchlick(float cosTheta, vec3 F0)
{
    return F0 +
        (1.0 - F0) *
        pow(clamp(1.0 - cosTheta,0.0,1.0),5.0);
}


vec3 dielectric_f0(float ior)
{
    float f0 = (ior - 1.0) / (ior + 1.0);
    f0 *= f0;

    return vec3(f0);
}


float calculate_reflectance(vec3 in_dir, vec3 normal, float ior_a, float ior_b)
{
    float eta = ior_a / ior_b;
    float cos_angle_in = clamp(-dot(in_dir, normal), 0.0, 1.0);
    float sin2_angle_of_refraction = eta * eta * (1.0 - cos_angle_in * cos_angle_in);

    if (sin2_angle_of_refraction >= 1.0)
        return 1.0;

    float cos_angle_of_refraction = sqrt(1.0 - sin2_angle_of_refraction);
    float denom_perpendicular = ior_a * cos_angle_in + ior_b * cos_angle_of_refraction;
    float denom_parallel = ior_b * cos_angle_in + ior_a * cos_angle_of_refraction;

    if (min(denom_perpendicular, denom_parallel) < 1e-8)
        return 1.0;

    float r_perpendicular = (ior_a * cos_angle_in - ior_b * cos_angle_of_refraction) / denom_perpendicular;
    r_perpendicular *= r_perpendicular;

    float r_parallel = (ior_b * cos_angle_in - ior_a * cos_angle_of_refraction) / denom_parallel;
    r_parallel *= r_parallel;

    return 0.5 * (r_perpendicular + r_parallel);
}


vec3 get_F0(Surface surface)
{
    if(surface.metallic > 0.0)
    {
        return mix(
            dielectric_f0(1.5),
            surface.albedo,
            surface.metallic
        );
    }

    return dielectric_f0(1.5);
}

LightSample sample_light(uint light_index, inout uint st) {
    LightSample result;

    Triangle triangle = triangles[lights[light_index].triangle_index];

    float u = random(st);
    float v = random(st);

    if(u + v > 1.0)
    {
        u = 1.0 - u;
        v = 1.0 - v;
    }

    result.position = triangle.v0.xyz + u * triangle.edge1.xyz + v * triangle.edge2.xyz;

    uint material_index = uint(triangle.v0.w);

    result.emission = materials[material_index].emission.xyz * materials[material_index].emission.w;
    result.normal = normalize(cross(triangle.edge2.xyz, triangle.edge1.xyz));
    return result;
}

vec3 sampleGGX(vec3 N, vec3 V, float roughness, inout uint st)
{

    float u1 = random(st);
    float u2 = random(st);

    float phi = 2.0 * PI * u1;

    float alpha = max(roughness * roughness, 0.001);

    float cosTheta = sqrt(
        (1.0 - u2) /
        (1.0 + (alpha - 1.0) * u2)
    );

    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

    vec3 T, B;
    build_onb(N, T, B);

    vec3 H =
        normalize(T * (cos(phi) * sinTheta) +
                  B * (sin(phi) * sinTheta) +
                  N * cosTheta);

    vec3 L = reflect(-V, H);

    if (roughness < 0.001)
        return reflect(-V, N);

    if (dot(N, L) <= 0.0)
        return cosine_sample_hemisphere(N, st);

    return L;
}

float sqr(float x) { return x*x; }

float SchlickFresnel(float u)
{
    float m = clamp(1.0 - u, 0.0, 1.0);
    float m2 = m*m;
    return m2*m2*m;
}

float GTR1(float NdotH, float a)
{
    if (a >= 1.0) return 1.0/PI;
    float a2 = a*a;
    float t = 1.0 + (a2-1.0)*NdotH*NdotH;
    return (a2-1.0) / (PI*log(a2)*t);
}

float GTR2(float NdotH, float a)
{
    float a2 = a*a;
    float t = 1.0 + (a2-1.0)*NdotH*NdotH;
    return a2 / (PI * t*t);
}

float GTR2_aniso(float NdotH, float HdotX, float HdotY, float ax, float ay)
{
    return 1.0 / (PI * ax*ay * sqr( sqr(HdotX/ax) + sqr(HdotY/ay) + NdotH*NdotH ));
}

float smithG_GGX(float NdotV, float alphaG)
{
    float a = alphaG*alphaG;
    float b = NdotV*NdotV;
    return 1.0 / (NdotV + sqrt(a + b - a*b));
}

float smithG_GGX_aniso(float NdotV, float VdotX, float VdotY, float ax, float ay)
{
    return 1.0 / (NdotV + sqrt( sqr(VdotX*ax) + sqr(VdotY*ay) + sqr(NdotV) ));
}

vec3 mon2lin(vec3 x)
{
    return vec3(pow(x[0], 2.2), pow(x[1], 2.2), pow(x[2], 2.2));
}


vec3 DisneyBRDF(
    Surface surface,
    vec3 L,
    vec3 V)
{
    vec3 N = surface.normal;

    float NdotL = max(dot(N, L), 0.0);
    float NdotV = max(dot(N, V), 0.0);

    if (NdotL <= 0.0 || NdotV <= 0.0)
        return vec3(0.0);

    vec3 H = normalize(L + V);

    float NdotH = max(dot(N, H), 0.0);
    float LdotH = max(dot(L, H), 0.0);

    vec3 X, Y;
    build_onb(N, X, Y);

    const float subsurface = 0.0;
    const float specular = 0.5;
    const float specularTint = 0.0;
    const float anisotropic = 0.0;
    const float sheen = 0.0;
    const float sheenTint = 0.5;
    const float clearcoat = 0.0;
    const float clearcoatGloss = 1.0;

    vec3 Cdlin = mon2lin(surface.albedo);
    float Cdlum = dot(Cdlin, vec3(0.3, 0.6, 0.1));

    vec3 Ctint = Cdlum > 0.0 ? Cdlin / Cdlum : vec3(1.0);

    vec3 Cspec0 = mix(
        specular * 0.08 * mix(vec3(1.0), Ctint, specularTint),
        Cdlin,
        surface.metallic
    );

    vec3 Csheen = mix(vec3(1.0), Ctint, sheenTint);

    float FL = SchlickFresnel(NdotL);
    float FV = SchlickFresnel(NdotV);

    float Fd90 = 0.5 + 2.0 * LdotH * LdotH * surface.roughness;
    float Fd = mix(1.0, Fd90, FL) * mix(1.0, Fd90, FV);

    float Fss90 = LdotH * LdotH * surface.roughness;
    float Fss = mix(1.0, Fss90, FL) * mix(1.0, Fss90, FV);

    float ss = 1.25 *
        (Fss * (1.0 / (NdotL + NdotV + 1e-6) - 0.5) + 0.5);

    float aspect = sqrt(max(0.0001, 1.0 - anisotropic * 0.9));

    float ax = max(0.001, sqr(surface.roughness) / aspect);
    float ay = max(0.001, sqr(surface.roughness) * aspect);

    float Ds = GTR2_aniso(
        NdotH,
        dot(H, X),
        dot(H, Y),
        ax,
        ay
    );

    float FH = SchlickFresnel(LdotH);

    vec3 Fs = mix(Cspec0, vec3(1.0), FH);

    float Gs =
        smithG_GGX_aniso(NdotL, dot(L, X), dot(L, Y), ax, ay) *
        smithG_GGX_aniso(NdotV, dot(V, X), dot(V, Y), ax, ay);

    vec3 Fsheen = FH * sheen * Csheen;

    float Dr = GTR1(
        NdotH,
        mix(0.1, 0.001, clearcoatGloss)
    );

    float Fr = mix(0.04, 1.0, FH);

    float Gr =
        smithG_GGX(NdotL, 0.25) *
        smithG_GGX(NdotV, 0.25);

    vec3 diffuse =
        ((1.0 / PI) * mix(Fd, ss, subsurface) * Cdlin + Fsheen) *
        (1.0 - surface.metallic);

    vec3 specular_term = Gs * Fs * Ds;

    vec3 clearcoatTerm =
        vec3(0.25 * clearcoat * Gr * Fr * Dr);

    return diffuse + specular_term + clearcoatTerm;
}


float brdf_pdf(
    Surface surface,

    vec3 V,
    vec3 L)
{
    float NoL = max(dot(surface.normal, L), 0.0);

    if (NoL <= 0.0)
        return 0.0;

    float diffusePdf = NoL / PI;

    vec3 H = normalize(V + L);

    float NoH = max(dot(surface.normal, H), 0.0);
    float VoH = max(dot(V, H), 0.0);

    float alpha = max(surface.roughness * surface.roughness, 0.001);

    float specularPdf = D_GGX(NoH, alpha) * NoH / max(4.0 * VoH, 0.001);

    vec3 F0 = get_F0(surface);
    vec3 F = FresnelSchlick(max(dot(V, H), 0.0), F0);
    float specProb = clamp(max(max(F.r, F.g), F.b), 0.05, 0.95);

    return mix(diffusePdf, specularPdf, specProb);
}


vec3 evaluate_brdf(
    Surface surface,
    vec3 V,
    vec3 L)
{
    float dotNV = dot(surface.normal, V);
    float dotNL = dot(surface.normal, L);

    if (dotNV * dotNL > 0.0) {
        float NoV = max(dotNV, 0.0);
        float NoL = max(dotNL, 0.0);

        if (NoV <= 0.0 || NoL <= 0.0)
            return vec3(0.0);

        vec3 X, Y;
        build_onb(surface.normal, X, Y);

        float subsurface = 0.0;
        float specular = surface.specular.x;
        float specularTint = 0.0;
        float anisotropic = 0.0;
        float sheen = 0.0;
        float sheenTint = 0.5;
        float clearcoat = 0.0;
        float clearcoatGloss = 1.0;

        return DisneyBRDF(
            surface,
            L,
            V
        );
    }


    return vec3(0.0);
}


BRDFSample sample_brdf(
    Surface surface,
    vec3 V,
    inout uint st)
{
    BRDFSample s;


    vec3 F0 = get_F0(surface);

    float NoV = max(dot(surface.normal, V), 0.0);

    //glass path
    if (surface.transmission > 0.0)
    {
        if (surface.ior <= 1e-4 || abs(surface.ior - 1.0) < 1e-4)
        {
            s.direction = -V;
            s.weight = vec3(1.0);
            s.pdf = 1.0;
            s.specular = true;
            return s;
        }

        bool entering = !surface.backface;

        vec3 N = entering ? surface.normal : -surface.normal;

        float iorA = entering ? 1.0 : surface.ior;
        float iorB = entering ? surface.ior : 1.0;
        vec3 incident = -V;
        float reflectProb = calculate_reflectance(incident, N, iorA, iorB);
        float transmitProb = max(1.0 - reflectProb, 1e-6);
        vec3 refractDir = refract(incident, N, iorA / iorB);

        if (dot(refractDir, refractDir) < 1e-12 || random(st) < reflectProb)
        {
            s.direction = reflect(incident, N);
            s.weight = vec3(1.0);
            s.pdf = max(reflectProb, 1e-6);
        }
        else
        {
            s.direction = normalize(refractDir);
            s.weight = surface.albedo * surface.transmission / transmitProb;
            s.pdf = transmitProb;
        }

        s.specular = true;

        return s;
    }

    vec3 F = FresnelSchlick(NoV, F0);

    float specProb = clamp(max(max(F.r, F.g), F.b), 0.05, 0.95);

    if (random(st) < specProb) {
        s.direction = sampleGGX(surface.normal, V, surface.roughness, st);
        s.specular = true;
    } else {
        s.direction = cosine_sample_hemisphere(surface.normal, st);
        s.specular = false;
    }

    float NoL = max(dot(surface.normal, s.direction), 0.0);

    if (NoL <= 0.0) {
        s.pdf = 0.0;
        s.weight = vec3(0.0);
        return s;
    }

    if (surface.roughness < 0.001 && s.specular) {
        vec3 F0 = get_F0(surface);
        float specProb = clamp(max(max(F0.r, F0.g), F0.b), 0.05, 0.95);
        s.pdf = specProb;
        vec3 F = FresnelSchlick(max(dot(surface.normal, V), 0.0), F0);
        s.weight = F * NoL / max(s.pdf, 1e-6);
        return s;
    }

    s.pdf = brdf_pdf(surface, V, s.direction);

    vec3 f = evaluate_brdf(surface, V, s.direction);

    s.weight = f * NoL / max(s.pdf, 1e-6);

    return s;
}


float power_heuristic(float a, float b) {
    float aa = a * a;
    float bb = b * b;
    return aa / (aa + bb);
}

HitInfo intersect_triangle(Ray ray, uint triangle_index) {
    Triangle triangle = triangles[triangle_index];
    HitInfo hit_info;
    hit_info.t = INF;
    hit_info.u = 0.0;
    hit_info.v = 0.0;

    hit_info.triangle_index = triangle_index;
    hit_info.isBackface = false;

    vec3 v0 = triangle.v0.xyz;
    vec3 edge1 = triangle.edge1.xyz;
    vec3 edge2 = triangle.edge2.xyz;

    vec3 h = cross(ray.dir, edge2);
    float a = dot(edge1, h);



    float f = 1.0 / a;
    vec3 s = ray.origin - v0;
    float u = f * dot(s, h);

    if (u < 0.0 || u > 1.0) {
        return hit_info;
    }

    vec3 q = cross(s, edge1);
    float v = f * dot(ray.dir, q);

    if (v < 0.0 || u + v > 1.0) {
        return hit_info;
    }

    float t = f * dot(edge2, q);

    if (t > EPSILON) {
        hit_info.t = t;
        hit_info.u = u;
        hit_info.v = v;
        hit_info.isBackface = a > 0.0;
    }

    return hit_info;
}

HitInfo intersect_bvh_triangles(Ray ray)
{
    uint stack[24];
    int stack_idx = 0;
    stack[stack_idx++] = 0;

    HitInfo result;
    result.t = INF;
    result.u = 0.0;
    result.v = 0.0;
    result.triangle_index = 0;

    while (stack_idx > 0)
    {
        uint node_index = stack[--stack_idx];

        BVHNode node = nodes[node_index];

        float t = intersect_aabb_dist(ray, node.aabb_min.xyz, node.aabb_max.xyz);
        if (t >= INF || t > result.t)
            continue;


        if (node.child_index == 0xFFFFFFFFu)
        {
            uint first = node.triangle_offset;
            uint last = first + node.triangle_count;

            for (uint i = first; i < last; i++)
            {
                HitInfo hit = intersect_triangle(ray, i);

                if (hit.t < result.t)
                {
                    result = hit;
                }
            }
        }

        else
        {
            uint left = node.child_index;
            uint right = left + 1;

            float left_dist = intersect_aabb_dist(
                ray,
                nodes[left].aabb_min.xyz,
                nodes[left].aabb_max.xyz
            );

            float right_dist = intersect_aabb_dist(
                ray,
                nodes[right].aabb_min.xyz,
                nodes[right].aabb_max.xyz
            );

            if (left_dist < right_dist)
            {
                if (right_dist < result.t)
                    stack[stack_idx++] = right;

                if (left_dist < result.t)
                    stack[stack_idx++] = left;
            }
            else
            {
                if (left_dist < result.t) stack[stack_idx++] = left;

                if (right_dist < result.t) stack[stack_idx++] = right;
            }
        }
    }

    return result;
}

vec3 bvh_trace_ray(Ray ray, inout uint st) {

    vec3 incoming_light = vec3(0.0, 0.0, 0.0);
    vec3 ray_color = vec3(1.0, 1.0, 1.0);

    float prev_bsdf_pdf = 0.0;
    bool prev_specular = true;

    int num_lights = lights.length();

    int diffuse_bounces = 0;

    const int MAX_ITERATIONS = 24;

    for (int i = 0; i < MAX_ITERATIONS; i++) {

        HitInfo hit_info = intersect_bvh_triangles(ray);

        if (hit_info.t < INF)
        {
            vec3 hit_point = ray.origin + hit_info.t * ray.dir;

            Triangle triangle = triangles[hit_info.triangle_index];
            Normal normal_data = normals[hit_info.triangle_index];

            vec3 gN = normalize(
                (1.0 - hit_info.u - hit_info.v) * vec3(normal_data.n0) +
                hit_info.u * vec3(normal_data.n1) +
                hit_info.v * vec3(normal_data.n2)
            );

            vec3 geoN = normalize(cross(triangle.edge2.xyz, triangle.edge1.xyz));
            vec3 faceN = geoN;

            if (dot(faceN, gN) < 0.0)
                faceN = -faceN;

            Material material = materials[uint(triangle.v0.w)];

            UV uv_data = uvs[hit_info.triangle_index];

            vec2 uv =
                (1.0 - hit_info.u - hit_info.v) * uv_data.uv0_uv1.xy +
                hit_info.u * uv_data.uv0_uv1.zw +
                hit_info.v * uv_data.uv2;

            Surface surface;

            surface.albedo = sample_color(material, uv);

            vec3 orm = sample_orm(material, uv);
            surface.roughness = orm.r;
            surface.metallic = orm.g;
            surface.transmission = orm.b;

            surface.normal = sample_normal(material, uv, gN);
            surface.geometric_normal = faceN;

            surface.ior = material.ior;

            surface.emission = material.emission.rgb;
            surface.emissionStrength = material.emission.w;

            surface.backface = hit_info.isBackface;

            surface.specular = material.specular.x;

            vec3 V = -ray.dir;

            if (dot(surface.normal, V) < 0.0)
                surface.normal = -surface.normal;

            if (surface.transmission > 0.0 && (surface.ior <= 1e-4 || abs(surface.ior - 1.0) < 1e-4))
            {
                ray.origin = hit_point + geoN * sign(dot(geoN, ray.dir)) * EPSILON;
                i--;
                continue;
            }





            if (surface.emissionStrength > 0.0)
            {
                float mis_weight = 1.0;

                if (i > 0 && !prev_specular && num_lights > 0)
                {
                    float area = 0.5 * length(cross(triangle.edge1.xyz, triangle.edge2.xyz));
                    float dist2 = hit_info.t * hit_info.t;
                    float cosLight = max(dot(surface.normal, V), 0.0);

                    float light_pdf = 0.0;
                    if (cosLight > 1e-6 && area > 0.0)
                        light_pdf = dist2 / (float(num_lights) * area * cosLight);

                    if (light_pdf > 0.0)
                        mis_weight = power_heuristic(prev_bsdf_pdf, light_pdf);
                }

                incoming_light += ray_color * surface.emission * surface.emissionStrength * mis_weight;
            }

            bool near_mirror = (surface.roughness < 0.02 && surface.metallic > 0.98) || surface.transmission > 0.0;


            if (num_lights > 0 && !near_mirror)
            {
                int light_index = int(random(st) * float(num_lights));
                light_index = min(light_index, num_lights - 1);

                Light picked_light = lights[light_index];
                LightSample light = sample_light(light_index, st);

                vec3 to_light = light.position - hit_point;
                float dist2 = dot(to_light, to_light);
                float dist = sqrt(dist2);
                vec3 L = to_light / dist;

                float cosSurface = dot(surface.normal, L);
                float cosLight = dot(light.normal, -L);



                if (cosSurface > 0.0 && cosLight > 0.0)
                {
                    float light_pdf = dist2 / (float(num_lights) * picked_light.area * cosLight);


                    if (light_pdf > 0.0)
                    {

                        Ray shadow_ray;
                        shadow_ray.origin = hit_point + geoN * sign(dot(geoN, L)) * EPSILON;
                        shadow_ray.dir = L;
                        shadow_ray.inv_dir = 1.0 / L;

                        bool shadow_visible = false;

                        for (int skip = 0; skip < 8; skip++)
                        {
                            HitInfo shadow_hit = intersect_bvh_triangles(shadow_ray);

                            if (shadow_hit.t >= INF)
                                break;

                            Triangle shadow_triangle = triangles[shadow_hit.triangle_index];
                            Material shadow_material = materials[uint(shadow_triangle.v0.w)];

                            if (shadow_material.transmittance > 0.0)
                            {
                                shadow_ray.origin += shadow_ray.dir * (shadow_hit.t + EPSILON);
                                continue;
                            }

                            if (abs(shadow_hit.t - dist) < 1e-3)
                            {
                                shadow_visible = true;
                            }

                            break;
                        }

                        if (shadow_visible)
                        {
                            vec3 f = evaluate_brdf(surface, V, L);
                            float bsdf_pdf = brdf_pdf(surface, V, L);

                            float mis_weight = power_heuristic(light_pdf, bsdf_pdf);

                            incoming_light += ray_color * f * light.emission * cosSurface * mis_weight / light_pdf;
                        }
                        }
                }
            }

            BRDFSample brdf_sample = sample_brdf(surface, V, st);

            if (brdf_sample.pdf <= 0.0)
                break;

            ray.origin = hit_point + geoN * sign(dot(geoN, brdf_sample.direction)) * EPSILON;
            ray.dir = brdf_sample.direction;
            ray.inv_dir = 1.0 / ray.dir;

            ray_color *= brdf_sample.weight;

            prev_bsdf_pdf = brdf_sample.pdf;
            prev_specular = brdf_sample.specular;

            if (!brdf_sample.specular)
            {
                diffuse_bounces++;
                if (diffuse_bounces >= MAX_BOUNCES)
                    break;
            }

            if (diffuse_bounces > 2)
            {
                float p = clamp(max(ray_color.r, max(ray_color.g, ray_color.b)), 0.05, 0.95);

                if (random(st) > p)
                    break;

                ray_color /= max(p, 1e-6);
            }
        }
        else
        {
            break;
        }
    }

    return incoming_light;
}

vec3 tonemapACES(vec3 color)
{
    color *= 1.0;

    const mat3 ACESInputMat = mat3(
        vec3(0.59719, 0.07600, 0.02840),
        vec3(0.35458, 0.90834, 0.13383),
        vec3(0.04823, 0.01566, 0.83777)
    );

    const mat3 ACESOutputMat = mat3(
        vec3( 1.60475, -0.10208, -0.00327),
        vec3(-0.53108,  1.10813, -0.07276),
        vec3(-0.07367, -0.00605,  1.07602)
    );

    color = ACESInputMat * max(color, 0.0);

    color = (color * (color + 0.0245786) - 0.000090537) /
            (color * (0.983729 * color + 0.4329510) + 0.238081);

    color = ACESOutputMat * color;

    return clamp(color, 0.0, 1.0);
}


layout(local_size_x = 32, local_size_y = 32, local_size_z = 1) in;
void main() {
	ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
	ivec2 size = ivec2(params.raster_size);

	if (uv.x >= size.x || uv.y >= size.y) {
		return;
	}

    float frame = float(params.view_params.w);

    uint st = hash(
        uint(uv.x) * 1973u +
        uint(uv.y) * 9277u +
        uint(frame) * 26699u
    );

    
    vec2 jitter = vec2(
        random(st),
        random(st)
    ) - 0.5;

    jitter *= 1.0;

    vec2 screen = (vec2(uv) + vec2(0.5) + jitter) / params.raster_size;
    vec2 ndc = screen * 2.0 - 1.0;

    vec3 view_point_local = vec3(
        ndc.x * params.view_params.x * 0.5,
        -ndc.y * params.view_params.y * 0.5,
        -params.view_params.z
    );
    vec3 view_point = vec3(params.camera_transform * vec4(view_point_local, 1.0));

    Ray ray;
    ray.origin = params.camera_transform[3].xyz;
    ray.dir = normalize(view_point - ray.origin);
    ray.inv_dir = 1.0 / ray.dir;

    
    vec3 sample_c = bvh_trace_ray(ray, st);

    vec4 previous = imageLoad(accumulation_image, uv);

    vec3 accumulated =
        (previous.rgb * frame + sample_c) /
        (frame + 1.0);
    
    vec3 display = tonemapACES(accumulated);

    display = pow(display, vec3(1.0 / 2.2));

    imageStore(accumulation_image, uv, vec4(accumulated, 1.0));
    imageStore(color_image, uv, vec4(display, 1.0));
}
