#include "funrender.h"
#include "funsdl.h"
#include <iostream>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <algorithm>

namespace {

duration render_time;
time_point last_avg_time = clk::now();
time_point last_frame_time = clk::now();
uint64_t frame_nr;
int last_frame_nr;

constexpr size_t elap_graph_pts = 1024;
duration elap_graph[elap_graph_pts];
// The data ends before tail, and starts at tail
size_t elap_graph_tail;

#define DEG2RAD(d) ((d)/180.0*M_PI)

//stbi_load(pathname, )
class ObjFile {
public:
    bool load(std::ifstream file)
    {
        std::string line;
        while (std::getline(file, line)) {
            if (line.empty())
                continue;
            std::stringstream ss(line);
            std::string word;
            ss >> word;
            if (word == "v") {
                glm::vec3 v;
                ss >> v.x >> v.y >> v.z;
                vertices.push_back(v);
            } else if (word == "f") {
                face f;
                ss >> word;
                int phase = 0;
                unsigned index = 0;
                for (size_t i = 0; i <= word.size(); ++i) {
                    if (i < word.size() && std::isdigit(word[i])) {
                        index *= 10;
                        index += word[i] - '0';
                    } else if (i == word.size() || word[i] == '/') {
                        switch (phase) {
                        case 0:
                            f.vertex = index;
                            ++phase;
                            index = 0;
                            break;
                        case 1:
                            f.tc = index;
                            ++phase;
                            index = 0;
                            break;
                        case 2:
                            f.normal = index;
                            ++phase;
                            index = 0;
                            break;

                        default:
                            assert(!"What do I do here?");
                        }
                    }
                }
                faces.push_back(f);
            } else if (word == "vt") {
                glm::vec2 vt;
                ss >> vt.x >> vt.y;
                texcoords.push_back(vt);
            } else if (word == "vn") {
                glm::vec3 vn;
                ss >> vn.x >> vn.y >> vn.z;
                normals.push_back(vn);
            }
        }
        return true;
    }

    bool load(char const *pathname)
    {
        std::ifstream file(pathname);
        return load(std::move(file));
    }
private:
    struct face {
        unsigned vertex{-1U};
        unsigned normal{-1U};
        unsigned tc{-1U};
    };

    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;
    std::vector<face> faces;
};

}// anonymous namespace end


#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

void cleanup(int width, int height)
{

}

int setup(int width, int height)
{
    FILE *pngfile = fopen("earthmap1k.png", "rb");
    int imgw{};
    int imgh{};
    int comp{};
    uint32_t *pixels = (uint32_t *)stbi_load_from_file(
        pngfile, &imgw, &imgh, &comp, 4);
    //flip_colors(pixels, imgw, imgh);
    bind_texture(1);
    set_texture(pixels, imgw, imgh, imgw, stbi_image_free);

    float fovy = glm::radians(60.0f);
    float aspect = (float)height / width;
    float znear = 1.0f;
    float zfar = 100.0f;

    glm::mat4 pm = proj_mtx.emplace_back(
        glm::perspective(fovy, aspect, znear, zfar));
    glm::mat4 vm = view_mtx.emplace_back(1.0f);

    srand(42);
    return 0;
}

// Cube
//
//  a     d
//
//
//
//  b     c
static glm::vec3 cube[] = {
    { -1, -1,  1 }, // a
    { -1,  1,  1 }, // b
    {  1,  1,  1 }, // c
    {  1, -1,  1 }, // d
    {  1, -1, -1 }, // other side a
    {  1,  1, -1 }, // other side b
    { -1,  1, -1 }, // other side c
    { -1, -1, -1 }, // other side d
};

static unsigned int cubeIndices[] = {
    // Front face
    0, 1, 2,
    2, 3, 0,

    // Right face
    3, 2, 4,
    4, 5, 3,

    // Back face
    5, 4, 6,
    6, 7, 5,

    // Left face
    7, 6, 1,
    1, 0, 7,

    // Top face
    1, 6, 5,
    5, 2, 1,

    // Bottom face
    7, 0, 3,
    3, 5, 7
};

void new_render_frame(frame_param const& frame)
{
    parallel_clear(frame, 0xFF123456);


}

void render_frame(frame_param const& frame)
{
    time_point this_time = clk::now();

    ++frame_nr;
    size_t constexpr triangle_cnt = 400;
    static float ang[triangle_cnt];
    static float xofs[triangle_cnt];
    static float yofs[triangle_cnt];
    static float zofs[triangle_cnt];
    static float rad[triangle_cnt];
    static float spd[triangle_cnt];
    static float spacing[triangle_cnt];
    //static unsigned col[triangle_cnt];
    static bool init_done;

    if (!init_done) {
        for (size_t i = 0; i < triangle_cnt; ++i) {
            ang[i] = M_PI * (float)rand() / RAND_MAX;
            xofs[i] = (float)rand() / RAND_MAX * frame.width * 0.95;
            yofs[i] = (float)rand() / RAND_MAX * frame.height * 0.95;
            zofs[i] = (float)rand() / RAND_MAX * 20.0f;
            rad[i] = powf(2.0f, (float)rand() / RAND_MAX * 2.9f + 4.0f);
            //col[i] = (float)rand() / RAND_MAX * 0x1000000;
            spd[i] = ((float)rand() / RAND_MAX - 0.5) * 3;
            spacing[i] = (float)rand() / RAND_MAX * DEG2RAD(140) + DEG2RAD(40);
        }
        init_done = true;
    }

    parallel_clear(frame, 0xFF123456);

    duration frame_time = this_time - last_frame_time;
    last_frame_time = this_time;
    elap_graph[elap_graph_tail] = frame_time;
    if (++elap_graph_tail == elap_graph_pts)
        elap_graph_tail = 0;

    uint64_t us_since_last =
        std::chrono::duration_cast<std::chrono::microseconds>(
            frame_time).count();

    duration elap = this_time - last_avg_time;
    if (elap >= std::chrono::seconds(2)) {
        last_avg_time = this_time;
        auto us = std::chrono::duration_cast<
            std::chrono::microseconds>(elap).count();
        size_t frame_cnt = frame_nr - last_frame_nr;
        size_t fps = frame_cnt * UINT64_C(1000000) / us;
        last_frame_nr = frame_nr;

        duration min_elap = duration::max();
        duration max_elap = duration::min();
        for (size_t i = 0, n = elap_graph_tail; i < elap_graph_pts;
                ++i, (n = ((n + 1) < elap_graph_pts ? n + 1 : 0))) {
            min_elap = std::min(min_elap, elap_graph[n]);
            max_elap = std::max(max_elap, elap_graph[n]);
        }
        size_t diff = std::chrono::duration_cast<std::chrono::microseconds>(
            max_elap - min_elap).count();

        uint64_t render_us = std::chrono::duration_cast<
            std::chrono::microseconds>(render_time).count() / frame_cnt;
        render_time = std::chrono::microseconds(0);

        std::cerr << (us / frame_cnt) << "µs/frame ±" << diff << " µs"
            " (" << fps << " fps) avg (" << render_us << " render µs)\n";
    }
    for (size_t i = 0; i < triangle_cnt; ++i) {
        float ang2 = ang[i] + spacing[i];
        float ang3 = ang[i] + spacing[i] * 1.5;
        float sa1 = std::sin(ang[i]);
        float ca1 = std::cos(ang[i]);
        float sa2 = std::sin(ang2);
        float ca2 = std::cos(ang2);
        float sa3 = std::sin(ang3);
        float ca3 = std::cos(ang3);
        ang[i] += spd[i] * us_since_last / 1000000;
        while (ang[i] >= M_PI)
            ang[i] -= M_PI * 2;
        while (ang[i] < -M_PI)
            ang[i] += M_PI * 2;
        float sz = rad[i];
        float xo = -40 + xofs[i];
        float yo = 0 + yofs[i];
        glm::vec3 v1 = { ca1 * sz + sz + xo, sa1 * sz + sz + yo, zofs[i] };
        glm::vec3 v2 = { ca2 * sz + sz + xo, sa2 * sz + sz + yo, zofs[i] };
        glm::vec3 v3 = { ca3 * sz + sz + xo, sa3 * sz + sz + yo, zofs[i] };

        glm::vec2 t1{0,0};
        // glm::vec2 t1{ca1 * 0.5 + 0.5, sa1 * 0.5 + 0.5};
        glm::vec3 n1{};
        scaninfo si1{t1, v1, n1};

        glm::vec2 t2{1,1};
        // glm::vec2 t2{ca2 * 0.5 + 0.5, sa2 * 0.5 + 0.5};
        glm::vec3 n2{};
        scaninfo si2{t2, v2, n2};

        glm::vec2 t3{0,1};
        // glm::vec2 t3{ca3 * 0.5 + 0.5, sa3 * 0.5 + 0.5};
        glm::vec3 n3{};
        scaninfo si3{t3, v3, n3};

        //fill_triangle(frame, v1, v2, v3, col[i]);
        texture_triangle(frame, si1, si2, si3);
    }

    fill_box(frame, 8, 8, frame.width - 8, 108, 0x80563412);

    draw_text(frame, 10, 10, "T e s t ! The quick brown fox");

    render_time += clk::now() - this_time;
}

// 012, 213, 234
// even: n+1, n, n+2
//  odd: n, n+1, n+2
// void texture_strip(frame_param &fp, glm::vec3 *verts,
//     size_t count, size_t stride = 2)
// {
//     for (size_t i = 0; i + 2 <= count; i += 2) {

//     }
// }

void texture_fan(frame_param &fp, glm::vec3 *verts,
    size_t count, size_t stride = 2)
{
    for (size_t i = 1; i < count; i += stride)
        texture_triangle(fp, verts[0], verts[i], verts[i + 1]);
}

void texture_tris(frame_param &fp, glm::vec3 *verts,
    size_t count, size_t stride = 3)
{
    for (size_t i = 0; i < count; i += stride) {
        glm::vec4 v4[3] = {
            proj_mtx.back() * (view_mtx.back() * glm::vec4(verts[i], 1.0f)),
            proj_mtx.back() * (view_mtx.back() * glm::vec4(verts[i + 1], 1.0f)),
            proj_mtx.back() * (view_mtx.back() * glm::vec4(verts[i + 2], 1.0f)),
        };
        for (glm::vec4 &v : v4) {
            float m = 1.0f / v.w;
            v.x *= m;
            v.y *= m;
            v.z *= m;
        }
        texture_triangle(fp, glm::vec3(v4[0]),
            glm::vec3(v4[1]), glm::vec3(v4[2]));
    }
}
