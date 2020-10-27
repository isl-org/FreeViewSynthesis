#include "preprocess.h"

PYBIND11_MODULE(preprocess, m) {
    m.def("count_nbs", &count_nbs, "tgt_dm"_a, "tgt_K"_a, "tgt_R"_a, "tgt_t"_a,
          "src_dms"_a, "src_Ks"_a, "src_Rs"_a, "src_ts"_a,
          "bwd_depth_thresh"_a = 0.01);
    m.def("get_sampling_map", &get_sampling_map, "tgt_dm"_a, "tgt_K"_a, "tgt_R"_a, "tgt_t"_a,
          "src_dms"_a, "src_Ks"_a, "src_Rs"_a, "src_ts"_a, "patch"_a,
          "bwd_depth_thresh"_a = 0.01, "invalid_depth_to_inf"_a=true);
}
