#include "funrender.h"
#include "funsdl.h"
#include <iostream>
#include <fstream>
#include <cstdint>
#include <cassert>
#include <fstream>
#include <sstream>
#include <array>
#include <algorithm>
#include <unordered_map>

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
class obj_file_loader {
public:
    bool load(std::ifstream &file)
    {
        return load(file, false);
    }

    bool load(std::ifstream &file, bool is_mtllib)
    {
        bool got_face = false;

        std::string line;
        while (std::getline(file, line)) {
            for (size_t i = 0; i < line.size(); ++i) {
                if (line[i] == '#') {
                    line.clear();
                    break;
                }
                if (!std::isspace(line[i]))
                    break;
            }
            if (line.empty())
                continue;
            std::stringstream ss(line);
            std::string word;
            ss >> word >> std::ws;
            if (!is_mtllib && word == "o") {
                std::string name;
                std::getline(ss, name);
                // fixme, do something
            } else if (!is_mtllib && word == "s") {
                int group{};
                ss >> group;
                current_smoothing_group = group;
            } else if (!is_mtllib && word == "v") {
                glm::vec3 &v = vertices.emplace_back();
                ss >> v.x >> v.y >> v.z;
            } else if (!is_mtllib && word == "f") {
                face f;
                face_starts_by_mtlidx.at(current_material).push_back(
                    faces_by_mtlidx.at(current_material).size());
                while (ss >> word) {
                    got_face = true;
                    int phase = 0;
                    unsigned index = 0;
                    for (size_t i = 0; i <= word.size(); ++i) {
                        if (i < word.size() && std::isdigit(word[i])) {
                            index *= 10;
                            index += word[i] - '0';
                        } else if (i == word.size() || word[i] == '/') {
                            static constexpr unsigned (face::*member[]) = {
                                &face::vertex,
                                &face::tc,
                                &face::normal
                            };
                            if (phase < sizeof(member) / sizeof(*member)) {
                                f.*(member[phase]) = index;
                                ++phase;
                                index = 0;
                            } else {
                                assert(!"What do I do here? Probably drop it");
                            }
                        }
                    }
                    faces_by_mtlidx.at(current_material).push_back(f);
                }
            } else if (!is_mtllib && word == "vt") {
                glm::vec2 &vt = texcoords.emplace_back();;
                ss >> vt.s >> vt.t;
            } else if (!is_mtllib && word == "vn") {
                glm::vec3 &vn = normals.emplace_back();
                ss >> vn.x >> vn.y >> vn.z;
            } else if (!is_mtllib && word == "l") {
                // line not supported yet
            } else if (word == "mtllib") {
                std::string mtllib;
                std::getline(ss, mtllib);
                if (!load(mtllib.c_str(), true))
                    return false;
            } else if (word == "newmtl") {
                std::string material_name;
                std::getline(ss, material_name);
                create_material(material_name);
            } else if (word == "usemtl") {
                std::string material_name;
                std::getline(ss, material_name);
                auto it = mat_lookup.find(material_name);
                if (it != mat_lookup.end()) {
                    current_material = it->second;
                } else {
                    current_material = create_material(material_name);
                }
            } else if (current_material != -1U &&
                    material_info::material_word(
                        materials.at(current_material), word, ss)) {
                // it did it already
            } else {
                std::cerr << "Ignored word " << word << " in " <<
                    (is_mtllib ? "material library" : "object") << '\n';
            }
        }

        if (got_face) {
            face_starts_by_mtlidx.at(current_material).push_back(
                faces_by_mtlidx.at(current_material).size());
        }

        return true;
    }

    size_t create_material(std::string const &material_name)
    {
        auto ins = mat_lookup.emplace(
            material_name, materials.size());

        if (ins.second) {
            materials.emplace_back();
            faces_by_mtlidx.emplace_back();
            face_starts_by_mtlidx.emplace_back();
        }

        return ins.first->second;
    }

    bool load(char const *pathname, bool is_mtllib = false)
    {
        std::ifstream file(pathname);
        return load(file, is_mtllib);
    }

    struct material_info {
        // Ambient reflectivity
        glm::vec3 Ka;

        // Diffuse reflectivity
        glm::vec3 Kd;

        // Specular reflectivity
        glm::vec3 Ks;

        // Emissive color
        glm::vec3 Ke;

        // Specular exponent
        float Ns{1.0f};

        // Ambient reflectivity
        std::string map_Ka;
        // Diffuse reflectivity
        std::string map_Kd;
        // Specular reflectivity
        std::string map_Ks;
        // Specular exponent
        std::string map_Ns;
        // "dissolve" (alpha)
        std::string map_d;
        // Illumination type (enum)
        int illum;
        // Dissolve (alpha)
        float d{};

        static bool material_word(material_info &material,
            std::string const& word, std::stringstream &ss)
        {
            static const std::unordered_map<
                std::string, glm::vec3 (material_info::*)> vec3_members{
                    { "Ka", &material_info::Ka },
                    { "Kd", &material_info::Kd },
                    { "Ke", &material_info::Ke },
                    { "Ks", &material_info::Ks }
                };

            static const std::unordered_map<
                std::string, std::string (material_info::*)> string_members{
                    { "map_Ka", &material_info::map_Ka },
                    { "map_Kd", &material_info::map_Kd },
                    { "map_Ks", &material_info::map_Ks },
                    { "map_Ns", &material_info::map_Ns },
                    { "map_d", &material_info::map_d }
                };

            static const std::unordered_map<
                std::string, float (material_info::*)> float_members{
                    { "d", &material_info::d },
                    { "Ns", &material_info::Ns }
                };

            static const std::unordered_map<
                std::string, int (material_info::*)> int_members{
                    { "illum", &material_info::illum }
                };

            auto vec3_it = vec3_members.find(word);
            if (vec3_it != vec3_members.end()) {
                glm::vec3 &field = material.*(vec3_it->second);
                ss >> field.r >> field.g >> field.b;
                return true;
            }

            auto string_it = string_members.find(word);
            if (string_it != string_members.end()) {
                std::string &field = material.*(string_it->second);
                std::getline(ss, field);
                return true;
            }

            auto float_it = float_members.find(word);
            if (float_it != float_members.end()) {
                float &field = material.*(float_it->second);
                ss >> field;
                return true;
            }

            auto int_it = int_members.find(word);
            if (int_it != int_members.end()) {
                int &field = material.*(int_it->second);
                ss >> field;
                return true;
            }

            return false;
        }
    };

    struct loaded_mesh {
        std::vector<scaninfo> vertices;
        std::vector<uint32_t> elements;
        std::vector<material_info> materials;
        std::vector<uint32_t> mtl_boundaries;
        //fixme std::vector<int> smoothing_groups;

        std::ostream & dump_info(std::ostream &s) const
        {
            s << vertices.size() << " vertices, ";
            s << elements.size() << " elements, ";
            s << materials.size() << " materials, ";
            s << mtl_boundaries.size() << " material boundaries\n";
            return s;
        }
    };

    loaded_mesh instantiate()
    {
        // Make dedup lookup table, value is its index in vertex array
        using scaninfo_lookup_t = std::unordered_map<scaninfo, size_t>;
        scaninfo_lookup_t scaninfo_lookup;

        std::vector<uint32_t> polygon_elements;

        loaded_mesh result;

        assert(face_starts_by_mtlidx.size() == faces_by_mtlidx.size());

        for (size_t m = 0; m < face_starts_by_mtlidx.size(); ++m) {
            for (face const& f : faces_by_mtlidx[m]) {
                // Piece together the vertex from the face table
                scaninfo vertex = from_face(f);

                auto ins = scaninfo_lookup.emplace(
                    vertex, scaninfo_lookup.size());

                polygon_elements.push_back(ins.first->second);
            }
        }

        // Copy the deduplicated vertices (keys)
        // into the vertex array indexes specified
        // in the associated values
        result.vertices.resize(scaninfo_lookup.size());
        for (auto &item : scaninfo_lookup)
            result.vertices[item.second] = item.first;

        for (size_t m = 0; m < face_starts_by_mtlidx.size(); ++m) {
            result.mtl_boundaries.push_back(result.elements.size());

            // Make triangle fans from the polygons
            // Makes a triangle from the first vertex
            // plus each pair of vertices
            for (size_t i = 0, e = face_starts_by_mtlidx[m].size() - 1;
                    i < e; ++i) {
                size_t st = face_starts_by_mtlidx[m][i];
                size_t en = face_starts_by_mtlidx[m][i + 1];
                size_t nr = en - st;

                for (size_t k = 1; k + 1 < nr; ++k) {
                    result.elements.push_back(polygon_elements[st]);
                    result.elements.push_back(polygon_elements[st+k]);
                    result.elements.push_back(polygon_elements[st+k+1]);
                }
            }
        }
        result.mtl_boundaries.push_back(result.elements.size());

        std::swap(result.materials, materials);
        mat_lookup.clear();
        current_material = -1U;
        vertices.clear();
        normals.clear();
        texcoords.clear();
        faces_by_mtlidx.clear();
        face_starts_by_mtlidx.clear();

        return result;
    }
private:
    struct face {
        unsigned vertex{-1U};
        unsigned normal{-1U};
        unsigned tc{-1U};
    };

    scaninfo from_face(face const &f) const
    {
        return {
            f.tc != -1U ? texcoords[f.tc] : glm::vec2{},
            f.vertex != -1U
            ? glm::vec4(vertices[f.vertex], 1.0f) : glm::vec4{},
            f.normal != -1U ? normals[f.normal] : glm::vec3{}
        };
    }

    size_t current_material{-1U};
    int current_smoothing_group{-1};
    std::vector<material_info> materials;
    std::unordered_map<std::string, size_t> mat_lookup;

    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;
    std::vector<std::vector<face>> faces_by_mtlidx;
    std::vector<std::vector<size_t>> face_starts_by_mtlidx;
};

}// anonymous namespace end


#include "stb_image.h"

void cleanup(int width, int height)
{

}

obj_file_loader::loaded_mesh mesh;

int setup(int width, int height)
{
    if (!command_line_files.empty()) {
        obj_file_loader obj;
        obj.load(command_line_files.front().c_str());
        mesh = obj.instantiate();
        mesh.dump_info(std::cerr);
    }
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

void user_frame(frame_param const& frame)
{
    time_point this_time = clk::now();

    ++frame_nr;

    parallel_clear(frame, 0xFF333333);

    view_mtx_stk.back() = glm::mat4(1.0f);

    view_mtx_stk.back() = glm::rotate(view_mtx_stk.back(),
        -mouselook_pitch, glm::vec3(1.0f, 0.0f, 0.0f));

    view_mtx_stk.back() = glm::rotate(view_mtx_stk.back(),
        -mouselook_yaw, glm::vec3(0.0f, 1.0f, 0.0f));

    view_mtx_stk.back() = glm::translate(view_mtx_stk.back(),
        -mouselook_pos);

    print_vector(std::cerr, mouselook_pos) << ' ' <<
        mouselook_pitch << ' ' << mouselook_yaw << '\n';

    set_transform(proj_mtx_stk.back() * view_mtx_stk.back());

#if 0
    time_point huge_st = clk::now();
    texture_elements(frame,
        mesh.vertices.data(), mesh.vertices.size(),
        mesh.elements.data(), mesh.elements.size());
    time_point huge_en = clk::now();
    size_t element_count = mesh.elements.size() / 3;
    size_t ns_elap = std::chrono::duration_cast<std::chrono::nanoseconds>(
            huge_en - huge_st).count();
    std::cerr << std::setprecision(3) <<
        (element_count * 1.0e+3f / ns_elap) <<
        " million elements per second\n";
#else

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
                xofs[i] = ((double)rand() / (RAND_MAX+1LL) - 0.5) *
                    frame.width * 0.95;
                yofs[i] = ((double)rand() / (RAND_MAX+1LL) - 0.5) *
                    frame.height * 0.95;
                //zofs[i] = (double)rand() / RAND_MAX * 19.0 + 1.0;
                zofs[i] = i * 8.0f;
                rad[i] = pow(2.0, (double)rand() / (RAND_MAX+1LL) *
                    5.9 + 4.0);
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

    float sec_since = us_since_last / 1.0e+6f;

    mouselook_acc.z = 0.0f;
    if (mouselook_pressed[0] && !mouselook_pressed[2])
        mouselook_acc.z = -4.0f;
    else if (mouselook_pressed[2] && !mouselook_pressed[0])
        mouselook_acc.z = 4.0f;
    // else if (mouselook_pressed[2] && mouselook_pressed[0])
    //     mouselook_acc.z *= 0.95f;
    else
        mouselook_vel.z *= 0.91f;

    mouselook_acc.x = 0.0f;
    if (mouselook_pressed[1] && !mouselook_pressed[3])
        mouselook_acc.x = -4.0f;
    else if (mouselook_pressed[3] && !mouselook_pressed[1])
        mouselook_acc.x = 4.0f;
    // else if (mouselook_pressed[3] && mouselook_pressed[1])
    //     mouselook_acc.x *= 0.91f;
    else
        mouselook_vel.x *= 0.91f;

    mouselook_acc.y = 0.0f;
    if (mouselook_pressed[4] && !mouselook_pressed[5])
        mouselook_acc.y = -4.0f;
    else if (mouselook_pressed[5] && !mouselook_pressed[4])
        mouselook_acc.y = 4.0f;
    // else if (mouselook_pressed[5] && mouselook_pressed[4])
    //     mouselook_acc.y = 0.91f;
    else
        mouselook_vel.y *= 0.91f;

    mouselook_vel += mouselook_acc;
    mouselook_pos += mouselook_vel * sec_since;

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

        // glm::mat4 const& vm = view_mtx_stk.back();
        // glm::mat4 const& pm = proj_mtx_stk.back();

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
#endif

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
