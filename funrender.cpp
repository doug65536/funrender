#include "funrender.h"
#include "funsdl.h"
#include <iostream>
#include <cstdint>
#include <cassert>
#include <fstream>
#include <sstream>
#include <array>
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

#define DEG2RADf(d) ((d)/180.0f*M_PIf)

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

    // float fovy = glm::radians(60.0f);
    // float aspect = (float)width / height;
    float znear = 1.0f;
    float zfar = 4097.0f;

    glm::mat4 pm = proj_mtx_stk.emplace_back(
        glm::frustum(-1.0f, 1.0f, -1.0f, 1.0f, znear, zfar));

    print_matrix("Proj", std::cerr, pm);

    glm::mat4 vm = view_mtx_stk.emplace_back(1.0f);

    srand(44444);
    return 0;
}

// Cube
//     h_____e
//    /|    /|
//  a/_|__d/ |
//  |  |  |  |
//  |  |__|__|
//  | /g  | /f
//  |/____|/
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
    parallel_clear(frame, 0xFF561234);


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

    // view_mtx_stk.back() = glm::rotate(glm::translate(glm::mat4(1.0f),
    //     glm::vec3(1.0f, -1.0f, 0.0f)),
    //     frame_nr/60.0f, glm::vec3(0.0f, 1.0f, 0.0f));

    view_mtx_stk.back() = glm::mat4(1.0f);

    // print_matrix("wtf identity", std::cerr, view_mtx_stk.back());

    // view_mtx_stk.back() = glm::translate(view_mtx_stk.back(),
    //     glm::vec3(frame.width * -0.5f, frame.height * 0.0f, -450.0f));

    // view_mtx_stk.back() = glm::rotate(view_mtx_stk.back(),
    //     M_PI_2f32*0.75f,// frame_nr/60.0f,
    //     glm::vec3(1.0f, 0.0f, 0.0f));

    // view_mtx_stk.back() = glm::translate(view_mtx_stk.back(),
    //     glm::vec3(0.0f, frame.height * -0.5f, 0.0f));

    // view_mtx_stk.back() = glm::translate(view_mtx_stk.back(),
    //     glm::vec3(frame.width*-0.5f, frame.height*-0.5f, 0.0f));

    // view_mtx_stk.back() = glm::translate(view_mtx_stk.back(),
    //     glm::vec3(-1000.0f, 0.0f, -700.0f));

    view_mtx_stk.back() = glm::translate(view_mtx_stk.back(),
        glm::vec3(0.0f, 0.0f, -1000.0f));


    //glm::mat4 &vm =
    // view_mtx_stk.back() = glm::rotate(view_mtx_stk.back(),
    //     M_PI_2f32*0.75f,// frame_nr/60.0f,
    //     glm::vec3(1.0f, 0.0f, 0.0f));

    // print_matrix("View", std::cerr, view_mtx_stk.back());

    // view_mtx_stk.back() = glm::translate(view_mtx_stk.back(),
    //     glm::vec3(frame.width*0.5f, frame.height*0.5f, 0.0f));

    // view_mtx_stk.back() = glm::rotate(view_mtx_stk.back(),
    //     M_PI_2f32*0.75f,// frame_nr/60.0f,
    //     glm::vec3(1.0f, 0.0f, 0.0f));

    // view_mtx_stk.back() = glm::translate(view_mtx_stk.back(),
    //     glm::vec3(-1.1704f, 0.04f, -1.0f));

    set_transform(proj_mtx_stk.back() * view_mtx_stk.back());

    // print_matrix("view", std::cerr, view_mtx_stk.back());

    if (!init_done) {
        for (size_t i = 0; i < triangle_cnt; ++i) {
            ang[i] = M_PI_2f;// M_PIf * (float)rand() / (RAND_MAX+1LL);
            if (triangle_cnt == 1) {
                xofs[i] = 0.0f;
                yofs[i] = 0.0f;
                zofs[i] = triangle_cnt - i - 1;
                rad[i] = 0.5f;
            } else {
                xofs[i] = ((float)rand() / (RAND_MAX+1LL) - 0.5f) *
                    frame.width * 0.95f;
                yofs[i] = ((float)rand() / (RAND_MAX+1LL) - 0.5f) *
                    frame.height * 0.95f;
                zofs[i] = (float)rand() / RAND_MAX * 19.0f + 1.0f;
                rad[i] = powf(2.0f, (float)rand() / (RAND_MAX+1LL) *
                    5.9f + 4.0f);
            }

            //col[i] = (float)rand() / RAND_MAX * 0x1000000;
            spd[i] = ((float)rand() / (RAND_MAX+1LL) - 0.5f) * 3.0f;
            spacing[i] = DEG2RADf(90.0f);
            // (float)rand() / RAND_MAX * DEG2RADf(140) + DEG2RADf(40);
        }
        init_done = true;
    }

    parallel_clear(frame, 0xFF561234);

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
        ang[i] += spd[i] * us_since_last / 1000000.0f;
        while (ang[i] >= M_PIf)
            ang[i] -= M_PIf * 2.0f;
        while (ang[i] < -M_PIf)
            ang[i] += M_PIf * 2.0f;
        float sz = rad[i];
        float xo = xofs[i];
        float yo = yofs[i];
        glm::vec4 v[3] = {
            { ca1 * sz + sz + xo, sa1 * sz + sz + yo, zofs[i], 1.0f },
            { ca2 * sz + sz + xo, sa2 * sz + sz + yo, zofs[i], 1.0f },
            { ca3 * sz + sz + xo, sa3 * sz + sz + yo, zofs[i], 1.0f }
        };

        glm::mat4 const& vm = view_mtx_stk.back();
        glm::mat4 const& pm = proj_mtx_stk.back();

        // std::cerr << "In:\n";
        // print_vector(std::cerr, v1) << '\n';
        // print_vector(std::cerr, v2) << '\n';
        // print_vector(std::cerr, v3) << '\n';

        glm::vec2 t1{0,0};
        glm::vec3 n1{};

        glm::vec2 t2{1,1};
        glm::vec3 n2{};

        glm::vec2 t3{0,1};
        glm::vec3 n3{};

        texture_polygon(frame, std::array<scaninfo, 3>{
            scaninfo{t1, v[0], n1},
            scaninfo{t2, v[1], n2},
            scaninfo{t3, v[2], n3}
        });
    }

    fill_box(frame, 8, 8, frame.width - 8, 168, 0x80563412, 12);

    draw_text(frame, 30, 30+24*0,
        u8"The quick brown fox jumped over the lazy dog 1234567890-=`<>?;'{}\\");
    draw_text(frame, 30, 30+24*1,
        u8"THE QUICK BROWN FOX JUMPED OVER THE LAZY DOG !@#$%^&*()_+~,./:\"[]|");

    std::string fps;
    fps = std::to_string(smooth_fps);
    if (fps.length() < 4)
        fps = std::string(4 - fps.length(), ' ') + fps;
    draw_text(frame, 30, 30+24*2, fps.c_str());

    fps = std::to_string(last_fps);
    if (fps.length() < 4)
        fps = std::string(4 - fps.length(), ' ') + fps;
    draw_text(frame, 30, 30+24*3, fps.c_str());
    // draw_text(frame, 10, 10+24*2,
    //     u8"ĀāĂăĄąĆćĈĉĊċČčĎďĐđĒēĔĕĖėĘęĚěĜĝĞğĠġĢģĤĥĦħĨĩĪīĬĭĮįİıĲĳĴĵĶķĸĹĺĻļ"
    //     "ĽľĿŀŁłŃńŅņŇňŉŊŋŌōŎŏŐőŒœŔŕŖŗŘřŚśŜŝŞşŠšŢţŤťŦŧŨũŪūŬŭŮůŰűŲųŴŵŶŷŸŹźŻ"
    //     "żŽžſƀƁƂƃƄƅƆƇƈƉƊƋƌƍƎƏƐƑƒƓƔƕƖƗƘƙƚƛƜƝƞƟƠơƢƣƤƥƦƧƨƩƪƫƬƭƮƯưƱƲƳƴƵƶƷƸƹƺ"
    //     "ƻƼƽƾƿǀǁǂǃǄǅǆǇǈǉǊǋǌǍǎǏǐǑǒǓǔǕǖǗǘǙǚǛǜǝǞǟǠǡǢǣǤǥǦǧǨǩǪǫǬǭǮǯǰǱǲ"
    //     "ǳǴǵǶǷǸǹǺǻǼǽǾǿ");
    // draw_text(frame, 10, 10+24*3,
    //     u8"Hello, 你好, こんにちは, שָׁלוֹם, नमस्ते, Γειά σας, Здравствуйте");

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

// void texture_fan(frame_param &fp, glm::vec3 *verts,
//     size_t count, size_t stride = 2)
// {
//     for (size_t i = 1; i < count; i += stride)
//         texture_triangle(fp, verts[0], verts[i], verts[i + 1]);
// }

// void texture_tris(frame_param &fp, glm::vec3 *verts,
//     size_t count, size_t stride = 3)
// {
//     for (size_t i = 0; i < count; i += stride) {
//         glm::vec4 v4[3] = {
//             glm::vec4(verts[i], 1.0f) * view_mtx_stk.back() * proj_mtx_stk.back(),
//             glm::vec4(verts[i + 1], 1.0f) * view_mtx_stk.back() * proj_mtx_stk.back(),
//             glm::vec4(verts[i + 2], 1.0f) * view_mtx_stk.back() * proj_mtx_stk.back()
//         };
//         for (glm::vec4 &v : v4) {
//             float m = 1.0f / v.w;
//             v.x *= m;
//             v.y *= m;
//             v.z *= m;
//         }
//         texture_triangle(fp, glm::vec3(v4[0]),
//             glm::vec3(v4[1]), glm::vec3(v4[2]));
//     }
// }
