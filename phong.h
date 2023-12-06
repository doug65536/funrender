#pragma once
#include <glm/glm.hpp>

// Ambient is a constant factor

// Diffuse is dot of normal and normalized vector to light
// viewpoint position has no effect on diffuse

// Specular is dependent upon the reflection off
// of the plane toward the light

// ks is the shininess parameter (rgb components)
// ilight the incoming light color
// v is the direction from the surface to the camera
// r is the direction of a perfectly reflected ray
// resulting from v bounding off plane with normal n
// n the surface normal
// shininess exponent
glm::vec3 specular(glm::vec3 ks, glm::vec3 ilight,
    glm::vec3 v, glm::vec3 n, float shininess)
{
    // Reflect the view direction over the normal
    glm::vec3 r = glm::reflect(-v, n);
    // Get cosine of angle between that perfect
    // reflection and the vector to the camera
    float dotProduct = glm::max(glm::dot(v, r), 0.0f);
    float shiny = glm::pow(dotProduct, shininess);
    glm::vec3 specular = ks * ilight * shiny;
    return specular;
}
