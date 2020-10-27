#pragma once

#include "common.h"

py::array_t<int> count_nbs(array_f32_d tgt_dm_np, array_f32_d tgt_K_np,
                           array_f32_d tgt_R_np, array_f32_d tgt_t_np,
                           array_f32_d src_dms_np, array_f32_d src_Ks_np,
                           array_f32_d src_Rs_np, array_f32_d src_ts_np,
                           float bwd_depth_thresh);

std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>>
get_sampling_map(array_f32_d tgt_dm_np, array_f32_d tgt_K_np,
                 array_f32_d tgt_R_np, array_f32_d tgt_t_np,
                 array_f32_d src_dms_np, array_f32_d src_Ks_np,
                 array_f32_d src_Rs_np, array_f32_d src_ts_np, array_i32_d patch,
                 float bwd_depth_thresh, bool invalid_depth_to_inf);
