"""Link the upstream libmolgrid CPU forward and its host-callable CUDA density functions."""
import os
import subprocess
from download_weights import ROOT, UP, fetch

def build():
    fetch()
    src = UP/"libmolgrid"
    cu = (src/"src/grid_maker.cu").read_text()
    host = cu[cu.index("    uint2 GridMaker::get_bounds_1d"):cu.index("    /*", cu.index("float GridMaker::calc_point<true>"))]
    cpp = ROOT/".cache/grid_host.cpp"
    cpp.write_text('#include "libmolgrid/grid_maker.h"\nnamespace libmolgrid { using std::min;\n' + host + '\n}\n')
    includes = [src/"include", ROOT/".cache/sysroot/usr/include", ROOT/".cache/sysroot/usr/include/openbabel3",
                "/usr/include/openbabel3", os.getenv("CUDA_HOME","/usr/local/cuda")+"/include"]
    out = ROOT/".cache/grid_oracle"
    subprocess.run(["c++","-std=c++17","-O2","-ffunction-sections","-fdata-sections","-Wl,--gc-sections",
                    *["-I"+str(path) for path in includes], str(ROOT/"tools/grid_oracle.cpp"),
                    str(src/"src/grid_maker.cpp"),str(cpp),"-o",str(out)],check=True)
    return out

if __name__=="__main__":
    build()
