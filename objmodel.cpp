// #include "objmodel.h"
// #include <iostream>
// #include <fstream>
// #include <string>
// #include <unordered_map>
// #include <glm/glm.hpp>

#include "objmodel.h"
#include "abstract_vector.h"

template class simd_raytracer<vecf32auto>;

simd_raytracer<vecf32auto> raytracer;

std::vector<scaninfo> const raytrace_vertices{
  { {0.0f, 0.0f}, // tc
    {0.0f, 0.0f, 0.1f, 1.0f}, // pos
    {0.0f, 0.0f, 1.0f}, // normal
    {1.0f, 0.0f, 0.0f}// color
  },
  { {0.0f, 0.0f},
    {1.0f, 0.0f, 0.0f, 1.0f},
    {0.0f, 0.0f, 1.0f},
    {0.0f, 1.0f, 0.0f}
  },
  { {0.0f, 0.0f},
    {0.0f, 1.0f, 0.0f, 1.0f},
    {0.0f, 0.0f, 1.0f},
    {0.0f, 0.0f, 1.0f}
  }
};

std::vector<uint32_t> const raytrace_elements{
    0, 1, 2
};

__attribute__((__noinline__))
void setup_raytrace()
{
}

__attribute__((__noinline__))
void test_raytrace(render_target const &target)
{
    glm::mat4 mtx(1.0f);
    raytracer.clear();
    raytracer.add_triangles_indexed(mtx, 0,
        raytrace_vertices.data(),
        raytrace_elements.data(), 3);
    raytracer.compile();
    mtx = glm::translate(mtx,
      glm::vec3(0.0f, 0.0f, 10.0f));
    raytracer.create_rays(target, mtx);
    raytracer.trace(target);
}
