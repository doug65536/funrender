#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <array>
#include <unordered_map>
#include <set>
#include <sstream>
#include <random>
#include <glm/glm.hpp>
#include "funsdl.h"
#include "likely.h"
#include "huge_alloc.h"

class obj_file_loader {
public:
    bool load(std::ifstream &file)
    {
        return load(file, false);
    }

    bool load(std::ifstream &file, bool is_mtllib)
    {
        bool got_face = false;

        uint32_t face_nr = 0;

        // Keep these outside the loop so it
        // doesn't have to keep recreating them
        std::string line;
        std::string word;
        std::stringstream ss;
        std::string name;
        while (std::getline(file, line)) {
            // Scan for comments
            bool anything = false;
            for (size_t i = 0; i < line.size(); ++i) {
                if (line[i] == '#') {
                    // Trim off everything after and including the comment
                    line = line.substr(0, i);
                    break;
                }

                // Note whether any non-space characters were seen
                anything |= !std::isspace(line[i]);
            }

            if (line.empty() || !anything)
                continue;

            // Reset the stringstring to read from the newly read line
            ss.clear();
            ss.str(line);

            // Each line begins with a short string that says the line type
            ss >> word >> std::ws;

            // Check the line type
            if (!is_mtllib && word == "o") {
                std::getline(ss, name);
                std::cerr << "object " << name <<
                    " at polygon " << face_nr << '\n';

                // Add the new name to the object_names list
                object_names.emplace_back(std::move(name));

                // Add the entry that says where the new object starts
                object_starts.push_back(face_nr);
            } else if (!is_mtllib && word == "s") {
                std::getline(ss, name);

                if (name == "off")
                    name = "0";

                int group = std::atoi(name.c_str());

                // If first smoothing group and not at face 0
                // inject the implied range starting at 0 in group 0
                if (sg_starts.empty() && face_nr != 0)
                    sg_starts.emplace_back(0U, 0);//initial

                current_smoothing_group = group;
                sg_starts.emplace_back(face_nr, group);// explicit
            } else if (!is_mtllib && word == "v") {
                glm::vec3 &v = vertices.emplace_back();
                ss >> v.x >> v.y >> v.z;
            } else if (!is_mtllib && word == "f") {
                face_ent f;
                // Remember where this polygon starts
                face_starts_by_mtlidx.at(current_material).push_back(
                    faces_by_mtlidx.at(current_material).size());
                while (ss >> word) {
                    got_face = true;
                    int phase = 0;
                    uint32_t index = 0;
                    for (size_t i = 0; i <= word.size(); ++i) {
                        if (i < word.size() && std::isdigit(word[i])) {
                            index *= 10;
                            index += word[i] - '0';
                        } else if (i == word.size() || word[i] == '/') {
                            static constexpr uint32_t face_ent::*member[] = {
                                &face_ent::vertex,
                                &face_ent::tc,
                                &face_ent::normal
                            };
                            size_t constexpr phase_count =
                                sizeof(member) / sizeof(*member);
                            if (phase < (int)phase_count) {
                                f.*(member[phase]) = index - 1;
                                ++phase;
                                index = 0;
                            } else {
                                // There were more than 3 things in this entry?
                                // This code does not know what to do with that
                                // We'd need to know what's after normal
                                // in the "member" lookup table
                                assert(!"What do I do here? Probably drop it");
                            }
                        }
                    }
                    faces_by_mtlidx.at(current_material).push_back(f);
                }
                // If a face is defined when there is no object (yet)...
                if (object_names.empty()) {
                    // ...then make an object with an empty string as the name,
                    // starting at the beginning
                    object_names.push_back("");
                    object_starts.push_back(0);
                }
                ++face_nr;
            } else if (!is_mtllib && word == "vt") {
                glm::vec2 &vt = texcoords.emplace_back();
                ss >> vt.s >> vt.t;
            } else if (!is_mtllib && word == "vn") {
                glm::vec3 &vn = normals.emplace_back();
                ss >> vn.x >> vn.y >> vn.z;
            } else if (!is_mtllib && word == "l") {
                // line not supported yet
            } else if (word == "mtllib") {
                std::getline(ss, name);
                if (!load(name.c_str(), true))
                    return false;
            } else if (word == "newmtl") {
                std::getline(ss, name);
                current_material = create_material(name);
            } else if (word == "usemtl") {
                std::getline(ss, name);
                auto it = mat_lookup.find(name);
                if (it != mat_lookup.end()) {
                    current_material = it->second;
                } else {
                    current_material = create_material(name);
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
            object_starts.push_back(face_nr);
            face_starts_by_mtlidx.at(current_material).push_back(
                faces_by_mtlidx.at(current_material).size());

            if (!sg_starts.empty()) {
                current_smoothing_group = 0;
                sg_starts.emplace_back(face_nr, 0); // sentinel
            }
        }

        assert(object_starts.size() == (object_starts.size()
            ? object_names.size() + 1 : 0));

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
        // Ambient is constant, and is not affected by lights
        // Diffuse only looks at the angle between the normal
        // and the vector to the light source. The viewer position
        // is ignored.
        // Specular looks at the angle between the reflection
        // direction and the vector to the camera

        // Ambient reflectivity
        glm::vec3 Ka;

        // Diffuse reflectivity
        glm::vec3 Kd;

        // Specular reflectivity
        glm::vec3 Ks;

        // Emissive color
        glm::vec3 Ke;

        // Index of refraction
        float Ni;

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
                    { "Ns", &material_info::Ns },
                    { "Ni", &material_info::Ni }
                };

            static const std::unordered_map<
                std::string, int (material_info::*)> int_members{
                    { "illum", &material_info::illum }
                };

            // Handle vec3 type
            auto vec3_it = vec3_members.find(word);
            if (vec3_it != vec3_members.end()) {
                glm::vec3 &field = material.*(vec3_it->second);
                ss >> field.r >> field.g >> field.b;
                return true;
            }

            // Handle string type
            auto string_it = string_members.find(word);
            if (string_it != string_members.end()) {
                std::string &field = material.*(string_it->second);
                std::getline(ss, field);
                return true;
            }

            // Handle float type
            auto float_it = float_members.find(word);
            if (float_it != float_members.end()) {
                float &field = material.*(float_it->second);
                ss >> field;
                return true;
            }

            // Handle int type
            auto int_it = int_members.find(word);
            if (int_it != int_members.end()) {
                int &field = material.*(int_it->second);
                ss >> field;
                return true;
            }

            return false;
        }
    };

    using vertex_type = scaninfo;

    struct loaded_mesh {
        std::vector<vertex_type> vertices;
        std::vector<uint32_t> elements;
        std::vector<material_info> materials;
        std::vector<uint32_t> mtl_boundaries;
        std::vector<std::string> object_names;
        std::vector<uint32_t> object_starts;
        //fixme std::vector<int> smoothing_groups;

        void scale(float multiplier)
        {
            std::transform(vertices.begin(), vertices.end(), vertices.begin(),
                [multiplier](vertex_type item) {
                    item.p = glm::vec4(glm::vec3(item.p) * multiplier,
                        item.p.w);
                    return item;
                });
        }

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
        using vertex_type_lookup_t = std::unordered_map<vertex_type, size_t>;
            vertex_type_lookup_t vertex_type_lookup;

        std::vector<uint32_t> polygon_elements;

        loaded_mesh result;

        assert(face_starts_by_mtlidx.size() == faces_by_mtlidx.size());

        using vertex_types_fvn_key = std::pair<uint32_t, uint32_t>;

        struct hasher {
            size_t operator()(vertex_types_fvn_key const& item) const
            {
                return std::hash<uint32_t>()(item.first) * 31 +
                    std::hash<uint32_t>()(item.second);
            }
        };

        // Key is {vertex_nr, sg}, value is list of vertex_type indices
        using vertex_types_from_vertex_nr_table = std::unordered_map<
            vertex_types_fvn_key, std::vector<size_t>,
            hasher, std::equal_to<vertex_types_fvn_key>>;

        vertex_types_from_vertex_nr_table vertex_types_from_vertex_nr;

        uint32_t face_nr = 0;
        size_t sg_index = 0;

        current_smoothing_group = 0;

        for (size_t m = 0; m < face_starts_by_mtlidx.size(); ++m) {
            if (sg_index < sg_starts.size() &&
                    sg_starts[sg_index].first == face_nr) {
                current_smoothing_group = sg_starts[sg_index].second;
                ++sg_index;
            }

            for (face_ent const& f : faces_by_mtlidx[m]) {
                // Piece together the vertex from the face table
                vertex_type vertex = from_face(f, current_smoothing_group);

                auto ins = vertex_type_lookup.emplace(
                    vertex, vertex_type_lookup.size());

                // Add the resulting vertex_type index to the list of
                // vertex_types that came from vertex number {f.vertex}
                vertex_types_from_vertex_nr[{
                    f.vertex,
                    current_smoothing_group
                }].push_back(ins.first->second);

                polygon_elements.push_back(ins.first->second);
            }

            ++face_nr;
        }

        // Copy the deduplicated vertices (keys)
        // into the vertex array indices specified
        // in the associated values
        result.vertices.resize(vertex_type_lookup.size());
        for (auto &item : vertex_type_lookup)
            result.vertices[item.second] = item.first;

        face_nr = 0;
        uint32_t object_starts_index = 0;

        for (size_t m = 0; m < face_starts_by_mtlidx.size(); ++m) {
            // Remember where this material starts
            result.mtl_boundaries.push_back(result.elements.size());

            // Make triangle fans from the polygons
            // Makes a triangle from the first vertex
            // plus each pair of vertices
            for (size_t i = 0, e = face_starts_by_mtlidx[m].size() - 1;
                    i < e; ++i) {
                // This face's vertices are the range from the start
                // of this face to the start of the next face
                // there is an extra entry at the end
                // to make this work for the last one
                size_t st = face_starts_by_mtlidx[m][i];
                size_t en = face_starts_by_mtlidx[m][i + 1];

                if (face_nr == object_starts[object_starts_index]) {
                    ++object_starts_index;
                    result.object_starts.push_back(result.elements.size());
                }

                for (size_t k = st + 1; k + 1 < en; ++k) {
                    result.elements.push_back(polygon_elements[st]);
                    result.elements.push_back(polygon_elements[k]);
                    result.elements.push_back(polygon_elements[k+1]);
                }

                ++face_nr;
            }
        }
        if (!result.mtl_boundaries.empty())
            result.mtl_boundaries.push_back(result.elements.size());
        if (!result.object_starts.empty())
            result.object_starts.push_back(result.elements.size());

        std::vector<glm::vec3> triangle_normals;

        for (size_t i = 0; i + 3 <= result.elements.size(); i += 3) {
            vertex_type &v0 = result.vertices[result.elements[i]];
            vertex_type &v1 = result.vertices[result.elements[i+1]];
            vertex_type &v2 = result.vertices[result.elements[i+2]];
            glm::vec3 v10 = glm::normalize(v1.p - v0.p);
            glm::vec3 v20 = glm::normalize(v2.p - v0.p);
            glm::vec3 normal = glm::cross(v10, v20);
            triangle_normals.emplace_back(normal);
        }

        // Average together the vertices that got put into the same
        // vector(s) by smoothing group
        for (auto &item : vertex_types_from_vertex_nr) {
            auto &indices = item.second;
            if (indices.size() < 2)
                continue;
            glm::vec3 normal_sum{};
            size_t sum_count = 0;
            for (auto index : indices) {
                normal_sum += result.vertices[index].n;
                ++sum_count;
            }
            normal_sum *= 1.0f / (float)sum_count;
            for (auto index : indices) {
                result.vertices[index].n = normal_sum;
            }
        }

        result.materials.swap(materials);
        result.object_names.swap(object_names);
        mat_lookup.clear();
        current_material = -1U;
        vertices.clear();
        normals.clear();
        texcoords.clear();
        faces_by_mtlidx.clear();
        face_starts_by_mtlidx.clear();

        std::cerr << "starts size " << result.object_starts.size() <<
            ", names + 1 =" << (result.object_names.size() + 1) << '\n';
        assert(result.object_starts.size() == result.object_names.size() + 1);
        assert(result.object_names.size() == result.object_starts.size() - 1);

        std::unordered_map<uint32_t, std::vector<uint32_t>> faces_using_vertex;

        return result;
    }
private:
    struct face_ent {
        uint32_t vertex{-1U};
        uint32_t normal{-1U};
        uint32_t tc{-1U};
    };

    vertex_type from_face(face_ent const &f, uint32_t smoothing_group) const
    {
        return {
            f.tc != -1U ? texcoords[f.tc] : glm::vec2{},
            f.vertex != -1U
            ? glm::vec4(vertices[f.vertex], 1.0f) : glm::vec4{},
            f.normal != -1U ? normals[f.normal] :
                // Encode the smoothing group into the normal fields
                // low 16 bits of smoothing group number in x (as float!)
                // high 16 bits of smoothing group number in y (as float!)
                // INFINITY in z
                // Don't worry, up to 24 bit integers are exactly representable
                // This isn't sufficient to identify which vertices need to be
                // averaged together, but it ensures that they will be split
                // into separate vertices if they are not in the same smoothing
                // group
                glm::vec3{(float)(uint32_t)(smoothing_group & 0xFFFF),
                    (float)(uint32_t)(smoothing_group >> 16),
                    INFINITY},

            // color
            glm::vec3(1.0f)
        };
    }

    static bool compare_sg_starts(
        std::pair<uint32_t, uint32_t> const& a,
        std::pair<uint32_t, uint32_t> const& b)
    {
        return a.first < b.first;
    }

    size_t current_material{-1U};
    uint32_t current_smoothing_group{-1U};
    std::vector<material_info> materials;
    std::unordered_map<std::string, size_t> mat_lookup;

    std::vector<glm::vec3> vertices;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texcoords;
    std::vector<std::vector<face_ent>> faces_by_mtlidx;
    std::vector<std::vector<size_t>> face_starts_by_mtlidx;
    std::vector<std::string> object_names;
    std::vector<uint32_t> object_starts;
    std::vector<std::pair<uint32_t, uint32_t> > sg_starts;
};

class bsp_tree {
    using vertex_type = scaninfo;

    struct tri {
        tri() = default;

        tri(vertex_type const& v0,
                vertex_type const& v1,
                vertex_type const& v2)
            : plane{plane_from_tri(v0, v1, v2)}
            , v{v0, v1, v2}
        {
        }

        tri(vertex_type const& v0,
                vertex_type const& v1,
                vertex_type const& v2,
                glm::vec4 const& plane)
            : plane{plane}
            , v{v0, v1, v2}
        {
        }

        // It can do it hypothetically if you give nullptr for the vectors
        std::pair<size_t, size_t> clip_against(
            tri const& partition,
            std::vector<vertex_type> *front_out = nullptr,
            std::vector<vertex_type> *back_out = nullptr)
        {
            std::pair<size_t, size_t> counts{ 0, 0 };

            for (size_t n = 1, i = 0; i < v.size(); ++i, ++n) {
                // Wrap around n when it gets out of range
                n &= -(n < v.size());
                vertex_type *curr = &v[i];
                vertex_type *next = &v[n];
                float cd = glm::dot(glm::vec3(curr->p),
                    glm::vec3(plane)) + plane.w;
                float nd = glm::dot(glm::vec3(next->p),
                    glm::vec3(plane)) + plane.w;

                bool curr_back = cd < 0.0f;
                bool next_back = nd < 0.0f;

                if (curr_back) {
                    if (back_out)
                        back_out->push_back(*curr);
                    ++counts.second;
                } else {
                    if (front_out)
                        front_out->push_back(*curr);
                    ++counts.first;
                }

                if (next_back != curr_back) {
                    // Project point onto plane
                    vertex_type diff = *next - *curr;
                    float full_dist = nd - cd;
                    vertex_type intersection =
                        diff * (cd / full_dist) + *curr;
                    if (back_out)
                        back_out->push_back(intersection);
                    ++counts.second;
                    if (front_out)
                        front_out->push_back(intersection);
                    ++counts.first;
                }
            }

            return counts;
        }

        glm::vec4 plane;
        std::array<vertex_type, 3> v;
    };

    struct node {
        // xyz is center, w is squared radius
        glm::vec4 sphere;
        std::vector<tri> front;
        std::vector<tri> back;
        std::vector<tri> coplanar;
    };
public:
    size_t select_random_tris(size_t *random_indices, size_t guesses)
    {
        if (guesses < tris.size()) {
            for (size_t guess = 0; guess < guesses; ++guess) {
                size_t *end = random_indices + guess;
                random_indices[guess] = rand() *
                        tris.size() / (RAND_MAX+1ULL);
                if (unlikely(std::find(random_indices, end,
                        random_indices[guess]) != end)) {
                    // Retry that guess. That guess was a duplicate.
                    --guess;
                    continue;
                }
            }
        } else {
            guesses = tris.size();
            for (size_t guess = 0; guess < guesses; ++guess)
                random_indices[guess] = guess;
            std::shuffle(random_indices,
                random_indices + tris.size(), lcggen);
        }

        return guesses;
    }

    size_t select_partition()
    {
        // Randomly select `guesses` partition planes
        // and choose the best among those. This is
        // nearly as good as exhaustively checking
        // every possibility and picking the apparent
        // absolute best one.
        constexpr size_t guesses = 4;
        size_t random_indices[guesses];
        size_t available = select_random_tris(
            random_indices, guesses);

        size_t best_selection;
        float best_score;

        for (size_t check = 0; check < available; ++check) {
            size_t tri_index = random_indices[check];
            tri &candidate_tri = tris[tri_index];
            size_t front = 0;
            size_t back = 0;
            size_t split = 0;
            size_t crossed = 0;
            for (size_t i = 0; i < tris.size(); ++i) {
                // Don't check it against itself
                if (i == tri_index)
                    continue;
                std::pair<size_t, size_t> sides =
                    tris[i].clip_against(candidate_tri);
                if (sides.first && !sides.second)
                    ++front;
                else if (!sides.first && sides.second)
                    ++back;
                else
                    ++split;

                // Prefer to select partition planes that
                // are going to potentially be split more
                std::pair<size_t, size_t> crossed_sides =
                    candidate_tri.clip_against(tris[i]);
                crossed += crossed_sides.first && crossed_sides.second;
            }

            float imbalance = back - front;
            // Squared
            imbalance *= imbalance;

            // Cube splits
            float destructiveness = split;
            destructiveness *= destructiveness;
            destructiveness *= destructiveness;

            float goodness = -imbalance +
                -destructiveness +
                crossed;

            if (!check || goodness < best_score) {
                best_selection = check;
                best_score = goodness;
            }
        }

        return best_selection;
    }

    void compile(std::vector<tri> &tris)
    {
        size_t selected = select_partition();
        tri &selected_tri = tris[selected];

        std::unique_ptr<node> n = std::make_unique<node>();

        for (size_t i = 0; i < tris.size(); ++i) {
            if (i == selected)
                continue;

            std::vector<vertex_type> front_verts;
            std::vector<vertex_type> back_verts;

            using side_pair = std::pair<
                std::vector<vertex_type> *,
                std::vector<tri> node::*>;
            side_pair sides[2] = {
                { &front_verts, &node::front },
                { &back_verts, &node::back }
            };

            tris[i].clip_against(selected_tri,
                &front_verts, &back_verts);

            for (size_t side = 0; side < 2; ++side) {
                side_pair &side_info = sides[side];

                // Make fan from front and back verts
                for (size_t i = 1; i + 2 <= side_info.first->size(); ++i) {
                    std::vector<tri> &dest = (*n).*(side_info.second);
                    dest.emplace_back(front_verts[0],
                        front_verts[i], front_verts[i+1],
                        tris[i].plane);
                }
            }
        }
    }

    void compile(obj_file_loader::loaded_mesh const& mesh)
    {
        for (size_t i = 0; i + 3 <= mesh.elements.size(); i += 3) {
            tris.emplace_back(
                mesh.vertices[mesh.elements[i]],
                mesh.vertices[mesh.elements[i+1]],
                mesh.vertices[mesh.elements[i+2]]);
        }

        compile(tris);
    }

    static glm::vec4 plane_from_tri(vertex_type const &v0,
        vertex_type const &v1, vertex_type const &v2)
    {
        glm::vec3 v10 = glm::vec3(v1.p) - glm::vec3(v0.p);
        glm::vec3 v20 = glm::vec3(v2.p) - glm::vec3(v0.p);
        glm::vec3 normal = glm::normalize(glm::cross(v10, v20));
        glm::vec4 plane{normal, -glm::dot(glm::vec3(v0.p), normal)};
        return plane;
    }

private:
    std::vector<tri> tris;
    std::unique_ptr<node> root;
    std::minstd_rand lcggen;
};

#include "abstract_vector.h"

// F is a vector type, like vecf32x8, for 256-bit vector width
// or vecf32x4 for 128-bit, or vecf32x16 for 512-bit, or whatever
// a "bundle" is a whole vector, whatever number that is, like 8
// It's AoSoA, but the innermost "array" is just the vector width
template<typename F>
class simd_raytracer {
    // Vector component type
    using C = component_of_t<F>;

    // The code probably needs to be 32 bit components
    // it needs reviewing to make sure this assert
    // isn't needed, half expect it might almost
    // work though
    static_assert(std::is_same_v<C, float>,
        "Requires single precision float"
        " vector components in F");

    // Uncomment this if we ever want signed
    //using I = typename vecinfo_t<F>::as_int;

    // Unsigned vector
    using U = typename vecinfo_t<F>::as_unsigned;

    // one-bit-per-component bitmask
    using M = typename vecinfo_t<F>::bitmask;

    using D = vecinfo_t<F>;
public:
    using vertex_type = scaninfo;
    static constexpr glm::vec4 vertex_type::*pm =
        &vertex_type::p;
    using index_type = uint32_t;
    using component_type = C;

    // vec_sz would be 8 for 256-bit vector of float
    static constexpr size_t vec_sz = vecinfo_t<F>::sz;

    // Holds the volatile information about ray bundles
    // To avoid the immutable data being on dirtied lines
    // (The volatile part of the search)
    struct ray_best_match {
        // Holds the squared distance to the best match
        F min_sqdist;

        // Holds the index of the
        // nearest intersecting triangle
        // -1U means "not found"
        U best_element;
    };

    static constexpr ray_best_match cleared_best_match{
        vec_broadcast<F>(FLT_MAX),
        vec_broadcast<U>(-1U)
    };

    // Holds a bundle of rays (the immutable part of the search)
    struct ray_bundle {
        // Origin
        F ori_x;
        F ori_y;
        F ori_z;

        // Direction
        F dir_x;
        F dir_y;
        F dir_z;

        // Color
        F col_r;
        F col_g;
        F col_b;

        // Remember which pixel owns this ray
        F pixel_nr;
    };

    // Holds a bundle of ray-triangle
    // collision results
    struct collision_bundle {
        // Interpolated position
        F px;
        F py;
        F pz;

        // Interpolated texcoord
        F tu;
        F tv;

        // Interpolated normal
        F nx;
        F ny;
        F nz;
    };

    // A SIMD implementation of the
    // Möller–Trumbore intersection algorithm
    struct collision_detail {
        template<bool check>
        always_inline_method
        bool collide(
            ray_bundle const &rays,
            vertex_type const& v1,
            vertex_type const& v2,
            vertex_type const& v3)
        {
            // Calculate two edges from common vertex

            glm::vec3 e1 = glm::vec3(v2.p) -
                glm::vec3(v1.p);

            glm::vec3 e2 = glm::vec3(v3.p) -
                glm::vec3(v1.p);

            // p = ray_dir cross e2
            F p_x, p_y, p_z;
            cross(p_x, p_y, p_z,
                rays.dir_x, rays.dir_y, rays.dir_z,
                vec_broadcast<F>(e2.x),
                vec_broadcast<F>(e2.y),
                vec_broadcast<F>(e2.z));

            // det = p dot e1
            F determinants = dot(
                p_x, p_y, p_z,
                vec_broadcast<F>(e1.x),
                vec_broadcast<F>(e1.y),
                vec_broadcast<F>(e1.z));

            C epsilon = std::numeric_limits<C>::epsilon();

            if constexpr (check) {
                // See which lanes (rays) are ok
                // "not ok" means the ray and
                // plane apparently don't intersect
                lanemask = determinants >
                    std::numeric_limits<C>::epsilon();
            }

            F inv_det = 1.0f / determinants;

            // T = v1 - ray_origin
            T_x = rays.ori_x - (v1.*pm).x;
            T_y = rays.ori_y - (v1.*pm).y;
            T_z = rays.ori_z - (v1.*pm).z;

            // u = (T dot p) / determinant
            u = dot(T_x, T_y, T_z,
                p_x, p_y, p_z) * inv_det;

            if constexpr (check) {
                // Not okay if intersection is outside triangle
                lanemask &= u >= 0.0f;
                lanemask &= u <= 1.0f;

                if (vec_all_false(lanemask))
                    return false;
            }

            // Q = T cross e1
            F Q_x, Q_y, Q_z;
            cross(Q_x, Q_y, Q_z,
                T_x, T_y, T_z,
                vec_broadcast<F>(e1.x),
                vec_broadcast<F>(e1.y),
                vec_broadcast<F>(e1.z));

            // v = (ray_dir dot Q) / determinants
            v = dot(rays.dir_x, rays.dir_y, rays.dir_z,
                Q_x, Q_y, Q_z) * inv_det;

            if constexpr (check) {
                // Not okay if outside triangle
                lanemask &= v >= 0.0f;
                lanemask &= v <= 1.0f;

                if (vec_all_false(lanemask))
                    return false;
            }

            // T = (e1 dot Q) / determinants
            T = dot(
                vec_broadcast<F>(e2.x),
                vec_broadcast<F>(e2.y),
                vec_broadcast<F>(e2.z),
                Q_x, Q_y, Q_z) * inv_det;

            if constexpr (check) {
                // Okay if in front of origin
                lanemask &= T > epsilon;

                if (vec_all_false(lanemask))
                    return false;
            }

            return true;
        }

        always_inline_method
        void update_best(ray_bundle const& rays,
            uint32_t element_nr, ray_best_match *matches)
        {
            // Squared distance from the
            // ray origin to the triangle
            F sq_dist = T_x * T_x + T_y * T_y + T_z * T_z;

            // Avoid the stores below that have no effect,
            // other than dirtying cache lines for nothing
            unsigned bitmask = vec_movemask(lanemask);
            while (bitmask) {
                int lane = ffs0(bitmask);
                bitmask &= ~(1U << lane);

                uint32_t pixel_nr = rays.pixel_nr[lane];
                int match_lane = pixel_nr & ~-vec_sz;
                ray_best_match &match_bundle = matches[pixel_nr / vec_sz];
                if (sq_dist[lane] < match_bundle.min_sqdist[match_lane]) {
                    match_bundle.min_sqdist[match_lane] = sq_dist[lane];
                    match_bundle.best_element[match_lane] = element_nr;
                }
            }

            // // Write back improved minimums
            // matches.min_sqdist = vec_blend(
            //     matches.min_sqdist, sq_dist, lanemask);

            // // Write out the closer element numbers
            // matches.best_element = vec_blend(matches.best_element,
            //     vec_broadcast<U>(element_nr), lanemask);
        }

        // This expects that you have set up lanemask
        always_inline_method
        void finish(ray_bundle const& rays,
            vertex_type const& v1,
            vertex_type const& v2,
            vertex_type const& v3,
            collision_bundle &collisions) const
        {
            F px = rays.ori_x + rays.dir_x * T_x;
            F py = rays.ori_y + rays.dir_y * T_y;
            F pz = rays.ori_z + rays.dir_z * T_z;

            F w = 1.0f - u - v;

            F tu = u * v1.t.x + v * v2.t.x + w * v3.t.x;
            F tv = u * v1.t.y + v * v2.t.y + w * v3.t.y;

            F nx = u * v1.n.x + v * v2.n.x + w * v3.n.x;
            F ny = u * v1.n.y + v * v2.n.y + w * v3.n.y;
            F nz = u * v1.n.z + v * v2.n.z + w * v3.n.z;

            collisions.px = vec_blend(collisions.px, px, lanemask);
            collisions.py = vec_blend(collisions.py, py, lanemask);
            collisions.pz = vec_blend(collisions.pz, pz, lanemask);

            collisions.tu = vec_blend(collisions.tu, tu, lanemask);
            collisions.tv = vec_blend(collisions.tv, tv, lanemask);

            collisions.nx = vec_blend(collisions.nx, nx, lanemask);
            collisions.ny = vec_blend(collisions.ny, ny, lanemask);
            collisions.nz = vec_blend(collisions.nz, nz, lanemask);
        }

        U lanemask;

        F T_x;
        F T_y;
        F T_z;

        F T;
        F u;
        F v;

        vertex_type const* v1;
        vertex_type const* v2;
        vertex_type const* v3;
    };

    // AABB slab node
    struct bvh_node {
        // Cartesian AABB
        float sx;   // left
        float sy;   // bottom
        float sz;   // far

        float ex;   // right
        float ey;   // top
        float ez;   // near

        std::unique_ptr<bvh_node> children[2];
        std::vector<index_type> elements;

        // Queue of rays to be checked against children
        unsigned pending_count{};
        std::vector<ray_bundle> pending_rays;

        void deposit_pending(
            ray_bundle const& rays,
            unsigned intersect_mask)
        {
            // Queue the rays into the node
            deposit_mask<F> dep{
                pending_count,
                intersect_mask,
                pending_rays};
            do {
                ray_bundle &last_soa = pending_rays.back();
                dep.deposit_into(last_soa.ori_x, rays.ori_x);
                dep.deposit_into(last_soa.ori_y, rays.ori_y);
                dep.deposit_into(last_soa.ori_z, rays.ori_z);
                dep.deposit_into(last_soa.dir_x, rays.dir_x);
                dep.deposit_into(last_soa.dir_y, rays.dir_y);
                dep.deposit_into(last_soa.dir_z, rays.dir_z);
                dep.deposit_into(last_soa.col_r, rays.col_r);
                dep.deposit_into(last_soa.col_g, rays.col_g);
                dep.deposit_into(last_soa.col_b, rays.col_b);
                dep.deposit_into(last_soa.pixel_nr, rays.pixel_nr);
            } while (dep.continue_from(pending_count, pending_rays));
        }

        // Returns a bitmask for each
        uint32_t rays_intersect_slab(
            ray_bundle const& rays) const
        {
            C const epsilon =
                std::numeric_limits<C>::epsilon();

            // See if nearly parallel
            // and also not between the slab planes
            // on each axis

            // ray is parallel to x
            U not_par_to_x = (rays.dir_x < -epsilon) |
                (rays.dir_x > epsilon);
            // ray is parallel to y
            U not_par_to_y = (rays.dir_y < -epsilon) |
                (rays.dir_y > epsilon);
            // ray is parallel to z
            U not_par_to_z = (rays.dir_z < -epsilon) |
                (rays.dir_z > epsilon);

            // Keep the result for each ray that is
            // not parallel to the axis or originates
            // between the slab planes...
            U keep = not_par_to_x |
                ((rays.ori_x >= sx) &
                (rays.ori_x <= ex));
            // ...on each axis
            keep &= not_par_to_y |
                ((rays.ori_y >= sy) &
                (rays.ori_y <= ey));
            keep &= not_par_to_z |
                ((rays.ori_z >= sz) &
                (rays.ori_z <= ez));

            // Return zero early if done already
            if (vec_all_false(keep))
                return 0;

            int result = 0;

            F inv = 1.0f / rays.dir_x;
            F t1s = (sx - rays.ori_x) * inv;
            F t2s = (ex - rays.ori_x) * inv;
            F lo_ts = min(t1s, t2s);
            F hi_ts = max(t1s, t2s);
            result |= vec_movemask((hi_ts >= lo_ts) & keep);

            // y
            inv = 1.0f / rays.dir_y;
            t1s = (sy - rays.ori_y) * inv;
            t2s = (ey - rays.ori_y) * inv;
            lo_ts = min(t1s, t2s);
            hi_ts = max(t1s, t2s);
            result |= vec_movemask((hi_ts >= lo_ts) & keep);

            // z
            inv = 1.0f / rays.dir_z;
            t1s = (sz - rays.ori_z) * inv;
            t2s = (ez - rays.ori_z) * inv;
            lo_ts = min(t1s, t2s);
            hi_ts = max(t1s, t2s);
            result |= vec_movemask((hi_ts >= lo_ts) & keep);

            return result;
        }
    };

    // Updates the rays with the closer intersection,
    // when tested against a closer triangle.
    __attribute__((__noinline__))
    void apply_closer_intersection(
        ray_bundle const &rays,
        //ray_best_match &matches,
        vertex_type const *verts,
        uint32_t const *indices,
        uint32_t element_nr) const
    {
        // Fetch triangle vertices
        vertex_type const& v1 = verts[indices[element_nr + 0]];
        vertex_type const& v2 = verts[indices[element_nr + 1]];
        vertex_type const& v3 = verts[indices[element_nr + 2]];

        collision_detail collision;
        // The lanemask does not need to
        // be set when check=true
        if (collision.template collide<true>(rays, v1, v2, v3))
            collision.update_best(rays, element_nr, best_matches);
    }

    void actual_intersection(
        ray_bundle const& rays,
        ray_best_match const& matches,
        collision_bundle& collisions,
        vertex_type const *verts,
        uint32_t const *indices)
    {
        // Get a bitmask of lanes that found a best element
        M found_any = vec_movemask(
            matches.best_element != -1U);

        collision_detail collision;
        while (found_any) {
            // Find the first lane that found something
            int first_lane = ffs0((uint32_t)found_any);

            // Read the value of that component
            uint32_t element_nr = matches.best_element[first_lane];

            // Make a vector mask of lanes that are the same element
            collision.lanemask = matches.best_element == element_nr;

            // Bitmasks of lanes that are the same element
            M same_element_bitmask = vec_movemask(collision.lanemask);

            // Clear the bit for each lane that is this element
            found_any &= ~same_element_bitmask;

            vertex_type const& v1 = verts[indices[element_nr + 0]];
            vertex_type const& v2 = verts[indices[element_nr + 1]];
            vertex_type const& v3 = verts[indices[element_nr + 2]];

            collision.template collide<false>(rays, v1, v2, v3);
            collision.finish(rays, v1, v2, v3, collisions);
        }
    }

    using vertex_lookup_map =
        std::unordered_map<vertex_type, index_type>;

    std::vector<vertex_type> vertices;
    vertex_lookup_map vertex_lookup;
    std::vector<index_type> elements;

    std::unique_ptr<bvh_node> root;

    void clear()
    {
        vertices.clear();
        vertex_lookup.clear();
        elements.clear();
    }

    void add_triangles_indexed(
        glm::mat4 const& modelview_matrix,
        size_t material_nr,
        vertex_type const *user_vertices,
        index_type const *user_elements,
        index_type count)
    {
        constexpr size_t verts_per_tri = 3;
        for (size_t i = 0; i + verts_per_tri <= count; ) {
            for (size_t o = 0; o < verts_per_tri; ++o, ++i) {
                index_type vertex_index = user_elements[i];
                vertex_type const &vertex = user_vertices[vertex_index];
                std::pair<vertex_lookup_map::iterator, bool> ins =
                    vertex_lookup.emplace(
                        modelview_matrix * vertex,
                        index_type(vertices.size()));
                if (ins.second)
                    vertices.emplace_back(ins.first->first);

                // Use deduped index
                elements.emplace_back(ins.first->second);
            }

            // Material number follows each triangle
            elements.emplace_back(material_nr);
        }
    }

    void partition(bvh_node *node)
    {
        std::vector<bvh_node*> todo(1, node);

        do {
            node = todo.back();
            todo.pop_back();

            if (node->elements.size() < 8)
                continue;

            float dx = node->ex - node->sx;
            float dy = node->ey - node->sy;
            float dz = node->ez - node->sz;

            float glm::vec4::*field;
            float mid;

            if (dx >= dy && dx >= dz) {
                // Partition on x
                field = &glm::vec4::x;
                mid = dx * 0.5f + node->sx;
            } else if (dy >= dx && dy >= dz) {
                // Partition on y
                field = &glm::vec4::y;
                mid = dy * 0.5f + node->sy;
            } else {
                // Partition on z
                field = &glm::vec4::z;
                mid = dz * 0.5f + node->sz;
            }

            glm::vec4 tri_min;
            glm::vec4 tri_max;

            for (auto el : node->elements) {
                // Get triangle
                scaninfo &v0 = vertices.at(elements[el]);
                scaninfo &v1 = vertices.at(elements[el+1]);
                scaninfo &v2 = vertices.at(elements[el+2]);
                index_type material_nr = elements[el+3];

                // Get AABB of triangle
                tri_min.x = sane_min((v0.*pm).x, (v1.*pm).x);
                tri_max.x = sane_max((v0.*pm).x, (v1.*pm).x);
                tri_min.y = sane_min((v0.*pm).y, (v1.*pm).y);
                tri_max.y = sane_max((v0.*pm).y, (v1.*pm).y);
                tri_min.z = sane_min((v0.*pm).z, (v1.*pm).z);
                tri_max.z = sane_max((v0.*pm).z, (v1.*pm).z);
                tri_min.x = sane_min(tri_min.x, (v2.*pm).x);
                tri_max.x = sane_max(tri_max.x, (v2.*pm).x);
                tri_min.y = sane_min(tri_min.y, (v2.*pm).y);
                tri_max.y = sane_max(tri_max.y, (v2.*pm).y);
                tri_min.z = sane_min(tri_min.z, (v2.*pm).z);
                tri_max.z = sane_max(tri_max.z, (v2.*pm).z);

                int back0 = (tri_min.*field <= mid);
                int back1 = (tri_max.*field <= mid) << 1;

                if (back0 == back1) {
                    // All on one side of partition
                    auto &child_ptr = node->children[back0];
                    if (!child_ptr)
                        child_ptr = std::make_unique<bvh_node>();

                    bvh_node &child = *child_ptr;
                    child.elements.emplace_back(el);
                } else {
                    // Spanning partition
                    vertex_type frontv[4];
                    vertex_type backv[4];
                    size_t front_count{};
                    size_t back_count{};

                    if (!node->children[0])
                        node->children[0] = std::make_unique<bvh_node>();
                    if (!node->children[1])
                        node->children[1] = std::make_unique<bvh_node>();

                    for (size_t v = 0; v < 3; ++v) {
                        size_t next = (v + 1) & -(v < 2);
                        // start vert, end vert
                        vertex_type sv = vertices[elements[v]];
                        vertex_type ev = vertices[elements[next]];
                        // start is on back, end is on back
                        bool sb = (sv.*pm).*field < mid;
                        bool eb = (ev.*pm).*field < mid;
                        // start and end aren't both back or front
                        if (sb != eb) {
                            // Get full distance
                            vertex_type dv = ev - sv;
                            // Get axis position coordinate
                            float sf = (sv.*pm).*field;
                            float ef = (ev.*pm).*field;
                            // Calculate proportion to partition
                            float t = (mid - sf) / (ef - sf);
                            // Interpolate new vertex
                            vertex_type mv = sv + dv * t;
                            // If starting on back,
                            if (sb) {
                                // starting on back
                                backv[back_count++] = sv;
                                backv[back_count++] = mv;
                                frontv[front_count++] = ev;
                            } else {
                                // starting on front
                                frontv[front_count++] = sv;
                                frontv[front_count++] = mv;
                                backv[back_count++] = mv;
                            }
                        }
                    }
                    index_type front_indices[4];
                    index_type back_indices[4];
                    for (size_t i = 0; i < front_count; ++i) {
                        auto ins = vertex_lookup.emplace(
                            frontv[i], vertices.size());
                        if (ins.second)
                            vertices.emplace_back(frontv[i]);
                        front_indices[i] = ins.first->second;
                    }
                    for (size_t i = 0; i < back_count; ++i) {
                        auto ins = vertex_lookup.emplace(
                            backv[i], vertices.size());
                        if (ins.second)
                            vertices.emplace_back(backv[i]);
                        back_indices[i] = ins.first->second;
                    }

                    // Make a triangle fan from front and back polygon
                    for (size_t i = 1; i + 2 <= front_count; ++i) {
                        node->children[0]->elements.emplace_back(front_indices[0]);
                        node->children[0]->elements.emplace_back(front_indices[i]);
                        node->children[0]->elements.emplace_back(
                            front_indices[i + 1]);
                        node->children[0]->elements.emplace_back(material_nr);
                    }
                    for (size_t i = 1; i + 2 <= back_count; ++i) {
                        node->children[1]->elements.emplace_back(back_indices[0]);
                        node->children[1]->elements.emplace_back(back_indices[i]);
                        node->children[1]->elements.emplace_back(
                            back_indices[i + 1]);
                        node->children[1]->elements.emplace_back(material_nr);
                    }
                }
            }

            if (node->children[0])
                todo.emplace_back(node->children[0].get());
            if (node->children[1])
                todo.emplace_back(node->children[1].get());
        } while (!todo.empty());
    }

    void compile()
    {
        root = std::make_unique<bvh_node>();

        glm::vec3 st{std::numeric_limits<float>::max()};
        glm::vec3 en{-std::numeric_limits<float>::max()};
        for (size_t i = 0; i + 4 <= elements.size(); i += 4) {
            root->elements.push_back(i);

            vertex_type v0 = vertices[elements[i]];
            vertex_type v1 = vertices[elements[i+1]];
            vertex_type v2 = vertices[elements[i+2]];

            st.x = sane_min(st.x, (v0.*pm).x);
            st.y = sane_min(st.y, (v0.*pm).y);
            st.z = sane_min(st.z, (v0.*pm).z);
            st.x = sane_min(st.x, (v1.*pm).x);
            st.y = sane_min(st.y, (v1.*pm).y);
            st.z = sane_min(st.z, (v1.*pm).z);
            st.x = sane_min(st.x, (v2.*pm).x);
            st.y = sane_min(st.y, (v2.*pm).y);
            st.z = sane_min(st.z, (v2.*pm).z);

            en.x = sane_max(en.x, (v0.*pm).x);
            en.y = sane_max(en.y, (v0.*pm).y);
            en.z = sane_max(en.z, (v0.*pm).z);
            en.x = sane_max(en.x, (v1.*pm).x);
            en.y = sane_max(en.y, (v1.*pm).y);
            en.z = sane_max(en.z, (v1.*pm).z);
            en.x = sane_max(en.x, (v2.*pm).x);
            en.y = sane_max(en.y, (v2.*pm).y);
            en.z = sane_max(en.z, (v2.*pm).z);
        }

        root->sx = st.x;
        root->sy = st.y;
        root->sz = st.z;
        root->ex = en.x;
        root->ey = en.y;
        root->ez = en.z;

        partition(root.get());
    }

    __attribute__((__noinline__))
    void create_rays(render_target const& target,
        glm::mat4 const& camera_mtx)
    {
        glm::vec3 campos = glm::vec3(camera_mtx[3]);

        glm::vec3 pts[] = {
            // top left
            camera_mtx * glm::vec4{ -1.0f, 1.0f, -1.0f, 1.0f },
            // top right
            camera_mtx * glm::vec4{  1.0f, 1.0f, -1.0f, 1.0f },
            // bottom left
            camera_mtx * glm::vec4{ -1.0f,-1.0f, -1.0f, 1.0f }
        };

        glm::vec3 xedge = pts[1] - pts[0];
        glm::vec3 yedge = pts[2] - pts[0];

        // Figure out how many bundles to create
        int bundles_across = (target.pitch +
            (vec_sz - 1)) / vec_sz;

        int bundles = bundles_across * target.height;

        if (unlikely(ray_count != (size_t)bundles)) {
            ray_count = bundles;
            huge_free(aligned_memory,
                aligned_memory_size);
            size_t best_match_sz = bundles *
                sizeof(*best_matches);
            size_t collisions_sz = bundles *
                sizeof(*collisions);
            aligned_memory_size = best_match_sz + collisions_sz;
            aligned_memory = huge_alloc(
                aligned_memory_size, &aligned_memory_size);
            std::cerr << "Raytracer allocated " <<
                (aligned_memory_size >> 20) <<
                " MB of huge pages\n";

            // These assume one lane per pixel
            best_matches = reinterpret_cast<
                ray_best_match *>(aligned_memory);
            collisions = reinterpret_cast<
                collision_bundle *>(best_matches + bundles);

            // Check alignment here,
            // before the details go out of scope
            assert(!(uintptr_t(best_matches) & ~-sizeof(F)));
            assert(!(uintptr_t(collisions) & ~-sizeof(F)));
        }

        float fw = (float)target.width;
        float fh = (float)target.height;
        float sw = 1.0f / (fw - 1.0f);
        float sh = 1.0f / (fh - 1.0f);

        // laneoffs is vector sized
        // and is like {0.0f, 1.0f, 2.0f, 3.0f, etc...}
        F ofs = D::laneoffs * sw;

        root->pending_rays.resize(target.width * target.height);

        size_t i = 0;
        for (float y = 0; y < fh; ++y) {
            for (float x = 0; x < fw; x += vec_sz, ++i) {
                ray_bundle &bundle = root->pending_rays[i];

                bundle.ori_x = pts[0].x +
                    xedge.x * ((x + ofs) * sw) +
                    yedge.x * (y * sh);
                bundle.ori_y = pts[0].y +
                    xedge.y * ((x + ofs) * sw) +
                    yedge.y * (y * sh);
                bundle.ori_z = pts[0].z +
                    xedge.z * ((x + ofs) * sw) +
                    yedge.z * (y * sh);

                bundle.dir_x = bundle.ori_x - campos.x;
                bundle.dir_y = bundle.ori_y - campos.y;
                bundle.dir_z = bundle.ori_z - campos.z;

                bundle.col_r = vec_broadcast<F>(1.0f);
                bundle.col_g = bundle.col_r;
                bundle.col_b = bundle.col_r;

                vec_normalize(bundle.dir_x,
                    bundle.dir_y, bundle.dir_z);

                // should be all nearly 1.0f
                // F hack = bundle.dir_x * bundle.dir_x +
                //     bundle.dir_y * bundle.dir_y +
                //     bundle.dir_z * bundle.dir_z;

                bundle.pixel_nr = D::laneoffs + (y * target.pitch + x);
            }
        }
    }

    __attribute__((__noinline__))
    void trace(render_target const& target)
    {
        std::fill(best_matches,
            best_matches + ray_count,
            cleared_best_match);

        // todo is an instance member so we don't
        // have to keep allocating a new one
        // it will be empty by the time this returns
        todo.emplace_back(root.get());

        while (!todo.empty()) {
            // Get the next node with pending rays
            bvh_node *item = todo.back();
            todo.pop_back();

            ray_bundle const* input = item->pending_rays.data();
            size_t input_count = item->pending_rays.size();

            bool leaf = true;

            for (std::unique_ptr<bvh_node> &child_node
                    : item->children) {
                if (!child_node)
                    continue;

                leaf = false;

                // Collide each camera ray against the node
                bool queued = false;

                for (size_t i = 0; i * vec_sz <
                        input_count + vec_sz - 1; ++i) {
                    ray_bundle const& rays = input[i];

                    // Intersect all of the rays
                    uint32_t intersect_mask =
                        child_node->rays_intersect_slab(rays);

                    // See if any rays intersect
                    if (intersect_mask) {
                        child_node->deposit_pending(
                            rays, intersect_mask);
                        if (!queued) {
                            queued = true;
                            todo.emplace_back(child_node.get());
                        }
                    }
                }
            }

            if (leaf) {
                for (auto element_nr : item->elements) {
                    for (size_t i = 0; i * vec_sz <
                            input_count + vec_sz - 1; ++i) {
                        ray_bundle const& rays = input[i];
                        apply_closer_intersection(rays,
                            vertices.data(), elements.data(),
                            element_nr);
                    }
                }
            }

            item->pending_rays.clear();
            item->pending_count = 0;
        }

        size_t pixel_count = target.pitch * target.height;

        for (size_t i = 0; i * vec_sz < pixel_count; ++i) {
            size_t pixel_nr = i * vec_sz;
            ray_best_match &best_bundle = best_matches[i];
            collision_bundle &coll_bundle = collisions[i];

            unsigned bitmask = vec_movemask(
                best_bundle.best_element != -1U);
            while (bitmask) {
                int lane = ffs0(bitmask);
                uint32_t element_nr = best_bundle.best_element[lane];

            }
        }
    }

    std::vector<bvh_node*> todo;
    void *aligned_memory{};
    size_t aligned_memory_size{};
    size_t ray_count{};
    ray_best_match * best_matches{};
    collision_bundle * collisions{};
};

extern template class simd_raytracer<vecf32auto>;
extern template bool simd_raytracer<vecf32auto>::
    collision_detail::collide<true>(
        ray_bundle const &rays,
        vertex_type const& v1_,
        vertex_type const& v2_,
        vertex_type const& v3_);
extern template bool simd_raytracer<vecf32auto>::
    collision_detail::collide<false>(
        ray_bundle const &rays,
        vertex_type const& v1_,
        vertex_type const& v2_,
        vertex_type const& v3_);

void setup_raytrace();
void test_raytrace(render_target const &target);
