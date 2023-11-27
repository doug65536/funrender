#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <cassert>
#include <unordered_map>
#include <sstream>
#include <glm/glm.hpp>
#include "funsdl.h"

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
