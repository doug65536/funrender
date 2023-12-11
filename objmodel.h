#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <array>
#include <unordered_map>
#include <sstream>
#include <random>
#include <glm/glm.hpp>
#include "funsdl.h"
#include "likely.h"

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
                                f.*(member[phase]) = index;
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

    struct loaded_mesh {
        std::vector<scaninfo> vertices;
        std::vector<uint32_t> elements;
        std::vector<material_info> materials;
        std::vector<uint32_t> mtl_boundaries;
        std::vector<std::string> object_names;
        std::vector<uint32_t> object_starts;
        //fixme std::vector<int> smoothing_groups;

        void scale(float multiplier)
        {
            std::transform(vertices.begin(), vertices.end(), vertices.begin(),
                [multiplier](scaninfo item) {
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
        using scaninfo_lookup_t = std::unordered_map<scaninfo, size_t>;
            scaninfo_lookup_t scaninfo_lookup;

        std::vector<uint32_t> polygon_elements;

        loaded_mesh result;

        assert(face_starts_by_mtlidx.size() == faces_by_mtlidx.size());

        using scaninfos_fvn_key = std::pair<uint32_t, uint32_t>;

        struct hasher {
            size_t operator()(scaninfos_fvn_key const& item) const
            {
                return std::hash<uint32_t>()(item.first) * 31 +
                    std::hash<uint32_t>()(item.second);
            }
        };

        // Key is {vertex_nr, sg}, value is list of scaninfo indices
        using scaninfos_from_vertex_nr_table = std::unordered_map<
            scaninfos_fvn_key, std::vector<size_t>,
            hasher, std::equal_to<scaninfos_fvn_key>>;

        scaninfos_from_vertex_nr_table scaninfos_from_vertex_nr;

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
                scaninfo vertex = from_face(f, current_smoothing_group);

                auto ins = scaninfo_lookup.emplace(
                    vertex, scaninfo_lookup.size());

                // Add the resulting scaninfo index to the list of
                // scaninfos that came from vertex number {f.vertex}
                scaninfos_from_vertex_nr[{
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
        result.vertices.resize(scaninfo_lookup.size());
        for (auto &item : scaninfo_lookup)
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
            scaninfo &v0 = result.vertices[result.elements[i]];
            scaninfo &v1 = result.vertices[result.elements[i+1]];
            scaninfo &v2 = result.vertices[result.elements[i+2]];
            glm::vec3 v10 = glm::normalize(v1.p - v0.p);
            glm::vec3 v20 = glm::normalize(v2.p - v0.p);
            glm::vec3 normal = glm::cross(v10, v20);
            triangle_normals.emplace_back(normal);
        }

        // Average together the vertices that got put into the same
        // vector(s) by smoothing group
        for (auto &item : scaninfos_from_vertex_nr) {
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

    scaninfo from_face(face_ent const &f, uint32_t smoothing_group) const
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
    struct tri {
        tri() = default;

        tri(scaninfo const& v0,
                scaninfo const& v1, scaninfo const& v2)
            : plane{plane_from_tri(v0, v1, v2)}
            , v{v0, v1, v2}
        {
        }

        tri(scaninfo const& v0, scaninfo const& v1,
                scaninfo const& v2, glm::vec4 const& plane)
            : plane{plane}
            , v{v0, v1, v2}
        {
        }

        // It can do it hypothetically if you give nullptr for the vectors
        std::pair<size_t, size_t> clip_against(
            tri const& partition,
            std::vector<scaninfo> *front_out = nullptr,
            std::vector<scaninfo> *back_out = nullptr)
        {
            std::pair<size_t, size_t> counts{ 0, 0 };

            for (size_t n = 1, i = 0; i < v.size(); ++i, ++n) {
                // Wrap around n when it gets out of range
                n &= -(n < v.size());
                scaninfo *curr = &v[i];
                scaninfo *next = &v[n];
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
                    scaninfo diff = *next - *curr;
                    float full_dist = nd - cd;
                    scaninfo intersection =
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
        std::array<scaninfo, 3> v;
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

            std::vector<scaninfo> front_verts;
            std::vector<scaninfo> back_verts;

            using side_pair = std::pair<
                std::vector<scaninfo> *,
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

    static glm::vec4 plane_from_tri(scaninfo const &v0,
        scaninfo const &v1, scaninfo const &v2)
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
