// Text input is the grid center followed by x y z channel radius per atom.
#include "libmolgrid/grid_maker.h"
#include <iostream>

int main() {
    float3 center;
    if (!(std::cin >> center.x >> center.y >> center.z)) return 1;
    std::vector<float> coords, types, radii;
    float x, y, z, type, radius;
    while (std::cin >> x >> y >> z >> type >> radius) {
        coords.insert(coords.end(), {x,y,z});
        types.push_back(type);
        radii.push_back(radius);
    }
    if (!std::cin.eof() || types.empty()) return 2;
    unsigned n = types.size();
    std::vector<float> density(28*48*48*48);
    libmolgrid::Grid4f out(density.data(),28,48,48,48);
    libmolgrid::GridMaker maker(.5,23.5);
    maker.forward(center,libmolgrid::Grid2f(coords.data(),n,3),
                  libmolgrid::Grid1f(types.data(),n),libmolgrid::Grid1f(radii.data(),n),out);
    std::cout.write(reinterpret_cast<char*>(density.data()),density.size()*sizeof(float));
}
