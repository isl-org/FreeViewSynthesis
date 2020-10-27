#include "preprocess.h"

#include <iostream>
#include <unordered_set>

bool is_valid_projection(int h, int w, float proj_d, const depthmap_t& dm,
                         int height, int width, float bwd_depth_thresh,
                         bool invalid_depth_to_inf) {
    bool in_domain = proj_d > 0 && w >= 0 && h >= 0 && w < width && h < height;
    if (!in_domain) {
        return false;
    }

    float ds = dm(h, w);
    if (ds <= 0) {
        if (invalid_depth_to_inf) {
            ds = 1e9;
        } else {
            return false;
        }
    }

    bool valid_depth_diff =
        (bwd_depth_thresh <= 0) ||
        std::abs(ds - proj_d) < bwd_depth_thresh * std::min(ds, proj_d);
    return valid_depth_diff;
}

py::array_t<int> count_nbs(array_f32_d tgt_dm_np, array_f32_d tgt_K_np,
                           array_f32_d tgt_R_np, array_f32_d tgt_t_np,
                           array_f32_d src_dms_np, array_f32_d src_Ks_np,
                           array_f32_d src_Rs_np, array_f32_d src_ts_np,
                           float bwd_depth_thresh) {
    if (tgt_dm_np.ndim() != 2) {
        throw std::invalid_argument("tgt_dm has to be height x width");
    }
    int height = tgt_dm_np.shape(0);
    int width = tgt_dm_np.shape(1);
    if (tgt_K_np.ndim() != 2 || tgt_K_np.shape(0) != 3 ||
        tgt_K_np.shape(1) != 3) {
        throw std::invalid_argument("tgt_K has to be 3 x 3");
    }
    if (tgt_R_np.ndim() != 2 || tgt_R_np.shape(0) != 3 ||
        tgt_R_np.shape(1) != 3) {
        throw std::invalid_argument("tgt_R has to be 3 x 3");
    }
    if (tgt_t_np.ndim() != 1 || tgt_R_np.shape(0) != 3) {
        throw std::invalid_argument("tgt_R has to be 3");
    }
    if (src_dms_np.ndim() != 3 || src_dms_np.shape(1) != height ||
        src_dms_np.shape(2) != width) {
        throw std::invalid_argument(
            "src_dms has to be n_views x height x width");
    }
    int n_views = src_dms_np.shape(0);
    if (src_Ks_np.ndim() != 3 || src_Ks_np.shape(0) != n_views ||
        src_Ks_np.shape(1) != 3 || src_Ks_np.shape(2) != 3) {
        throw std::invalid_argument("Ks has to be n_views x 3 x 3");
    }
    if (src_Rs_np.ndim() != 3 || src_Rs_np.shape(0) != n_views ||
        src_Rs_np.shape(1) != 3 || src_Rs_np.shape(2) != 3) {
        throw std::invalid_argument("Rs has to be n_views x 3 x 3");
    }
    if (src_ts_np.ndim() != 2 || src_ts_np.shape(0) != n_views ||
        src_ts_np.shape(1) != 3) {
        throw std::invalid_argument("ts has to be n_views x 3");
    }

    mat3_t tgt_K(tgt_K_np.data(), 3, 3);
    mat3_t tgt_R(tgt_R_np.data(), 3, 3);
    vec3_t tgt_t(tgt_t_np.data(), 3, 1);
    proj_t tgt_Pi;
    tgt_Pi.leftCols<3>() = tgt_K.inverse();
    tgt_Pi.rightCols<1>() = -tgt_t;
    tgt_Pi = tgt_R.transpose() * tgt_Pi;

    depthmap_t tgt_dm(tgt_dm_np.data(), height, width);

    std::vector<proj_t> src_Ps;
    std::vector<Eigen::Vector3f> src_Cs;
    for (int vidx = 0; vidx < n_views; ++vidx) {
        mat3_t K(src_Ks_np.data() + vidx * 3 * 3, 3, 3);
        mat3_t R(src_Rs_np.data() + vidx * 3 * 3, 3, 3);
        vec3_t t(src_ts_np.data() + vidx * 3 * 1, 3, 1);
        proj_t P;
        P.leftCols<3>() = R;
        P.rightCols<1>() = t;
        P = K * P;
        src_Ps.push_back(P);
        Eigen::Vector3f C = -R.transpose() * t;
        src_Cs.push_back(C);
    }

    std::vector<depthmap_t> src_dms;
    for (int vidx = 0; vidx < n_views; ++vidx) {
        src_dms.push_back(depthmap_t(src_dms_np.data() + vidx * height * width,
                                     height, width));
    }

    std::vector<int> count(n_views, 0);
    for (int tgt_h = 0; tgt_h < height; ++tgt_h) {
        for (int tgt_w = 0; tgt_w < width; ++tgt_w) {
            float dt = tgt_dm(tgt_h, tgt_w);
            if (dt <= 0) {
                continue;
            }

            Eigen::Vector4f tgt_uvd(dt * (float(tgt_w) + 0.5),
                                    dt * (float(tgt_h) + 0.5), dt, 1);
            Eigen::Vector3f xyz = tgt_Pi * tgt_uvd;
            Eigen::Vector4f xyzh(xyz(0), xyz(1), xyz(2), 1);

            for (int vidx = 0; vidx < n_views; ++vidx) {
                Eigen::Vector3f src_uvd = src_Ps[vidx] * xyzh;
                float proj_d = src_uvd(2);
                float src_wf = src_uvd(0) / proj_d;
                float src_hf = src_uvd(1) / proj_d;
                int src_w = int(src_wf);
                int src_h = int(src_hf);

                if (is_valid_projection(src_h, src_w, proj_d, src_dms[vidx],
                                        height, width, bwd_depth_thresh,
                                        false)) {
                    count[vidx]++;
                }
            }
        }
    }

    return create_array1<int>(count);
}

std::tuple<py::array_t<float>, py::array_t<float>, py::array_t<float>>
get_sampling_map(array_f32_d tgt_dm_np, array_f32_d tgt_K_np,
                 array_f32_d tgt_R_np, array_f32_d tgt_t_np,
                 array_f32_d src_dms_np, array_f32_d src_Ks_np,
                 array_f32_d src_Rs_np, array_f32_d src_ts_np,
                 array_i32_d patch_np, float bwd_depth_thresh,
                 bool invalid_depth_to_inf) {
    if (tgt_dm_np.ndim() != 2) {
        throw std::invalid_argument("tgt_dm has to be height x width");
    }
    int tgt_height = tgt_dm_np.shape(0);
    int tgt_width = tgt_dm_np.shape(1);
    if (tgt_K_np.ndim() != 2 || tgt_K_np.shape(0) != 3 ||
        tgt_K_np.shape(1) != 3) {
        throw std::invalid_argument("tgt_K has to be 3 x 3");
    }
    if (tgt_R_np.ndim() != 2 || tgt_R_np.shape(0) != 3 ||
        tgt_R_np.shape(1) != 3) {
        throw std::invalid_argument("tgt_R has to be 3 x 3");
    }
    if (tgt_t_np.ndim() != 1 || tgt_R_np.shape(0) != 3) {
        throw std::invalid_argument("tgt_R has to be 3");
    }
    if (src_dms_np.ndim() != 3) {
        throw std::invalid_argument(
            "src_dms has to be n_views x height x width");
    }
    int n_views = src_dms_np.shape(0);
    int src_height = src_dms_np.shape(1);
    int src_width = src_dms_np.shape(2);
    if (src_Ks_np.ndim() != 3 || src_Ks_np.shape(0) != n_views ||
        src_Ks_np.shape(1) != 3 || src_Ks_np.shape(2) != 3) {
        throw std::invalid_argument("Ks has to be n_views x 3 x 3");
    }
    if (src_Rs_np.ndim() != 3 || src_Rs_np.shape(0) != n_views ||
        src_Rs_np.shape(1) != 3 || src_Rs_np.shape(2) != 3) {
        throw std::invalid_argument("Rs has to be n_views x 3 x 3");
    }
    if (src_ts_np.ndim() != 2 || src_ts_np.shape(0) != n_views ||
        src_ts_np.shape(1) != 3) {
        throw std::invalid_argument("ts has to be n_views x 3");
    }
    if (patch_np.ndim() != 1 || patch_np.shape(0) != 4) {
        throw std::invalid_argument("patch hast to be a 4 vector");
    }

    mat3_t tgt_K(tgt_K_np.data(), 3, 3);
    mat3_t tgt_R(tgt_R_np.data(), 3, 3);
    vec3_t tgt_t(tgt_t_np.data(), 3, 1);
    proj_t tgt_Pi;
    tgt_Pi.leftCols<3>() = tgt_K.inverse();
    tgt_Pi.rightCols<1>() = -tgt_t;
    tgt_Pi = tgt_R.transpose() * tgt_Pi;

    depthmap_t tgt_dm(tgt_dm_np.data(), tgt_height, tgt_width);

    std::vector<proj_t> src_Ps;
    std::vector<Eigen::Vector3f> src_Cs;
    for (int vidx = 0; vidx < n_views; ++vidx) {
        mat3_t K(src_Ks_np.data() + vidx * 3 * 3, 3, 3);
        mat3_t R(src_Rs_np.data() + vidx * 3 * 3, 3, 3);
        vec3_t t(src_ts_np.data() + vidx * 3 * 1, 3, 1);
        proj_t P;
        P.leftCols<3>() = R;
        P.rightCols<1>() = t;
        P = K * P;
        src_Ps.push_back(P);
        Eigen::Vector3f C = -R.transpose() * t;
        src_Cs.push_back(C);
    }

    std::vector<depthmap_t> src_dms;
    for (int vidx = 0; vidx < n_views; ++vidx) {
        src_dms.push_back(
            depthmap_t(src_dms_np.data() + vidx * src_height * src_width,
                       src_height, src_width));
    }

    int h_from = patch_np.data()[0];
    int h_to = patch_np.data()[1];
    int w_from = patch_np.data()[2];
    int w_to = patch_np.data()[3];
    int patch_height = h_to - h_from;
    int patch_width = w_to - w_from;

    std::vector<float> maps(n_views * patch_height * patch_width * 2,
                            float(-10));
    std::vector<float> valid_depth_masks(n_views * patch_height * patch_width,
                                         0);
    std::vector<float> valid_map_masks(n_views * patch_height * patch_width, 0);
    for (int tgt_h = 0; tgt_h < patch_height; ++tgt_h) {
        for (int tgt_w = 0; tgt_w < patch_width; ++tgt_w) {
            float dt = tgt_dm(tgt_h + h_from, tgt_w + w_from);

            if (dt <= 0) {
                if (invalid_depth_to_inf) {
                    dt = 1e9;
                } else {
                    continue;
                }
            } else {
                for (int vidx = 0; vidx < n_views; ++vidx) {
                    int idx =
                        (vidx * patch_height + tgt_h) * patch_width + tgt_w;
                    valid_depth_masks[idx] = 1;
                }
            }

            Eigen::Vector4f tgt_uvd(dt * (float(tgt_w + w_from) + 0.5),
                                    dt * (float(tgt_h + h_from) + 0.5), dt, 1);
            Eigen::Vector3f xyz = tgt_Pi * tgt_uvd;
            Eigen::Vector4f xyzh(xyz(0), xyz(1), xyz(2), 1);

            for (int vidx = 0; vidx < n_views; ++vidx) {
                Eigen::Vector3f src_uvd = src_Ps[vidx] * xyzh;
                float proj_d = src_uvd(2);
                float src_wf = src_uvd(0) / proj_d;
                float src_hf = src_uvd(1) / proj_d;
                int src_w = int(src_wf);
                int src_h = int(src_hf);

                if (!is_valid_projection(
                        src_h, src_w, proj_d, src_dms[vidx], src_height,
                        src_width, bwd_depth_thresh, invalid_depth_to_inf)) {
                    continue;
                }

                int idx = (vidx * patch_height + tgt_h) * patch_width + tgt_w;
                valid_map_masks[idx] = 1;
                maps[idx * 2 + 0] = (2 * src_wf / (src_width - 1)) - 1;
                maps[idx * 2 + 1] = (2 * src_hf / (src_height - 1)) - 1;
            }
        }
    }

    return {create_arrayN<float>(maps, {n_views, patch_height, patch_width, 2}),
            create_arrayN<float>(valid_depth_masks,
                                 {n_views, 1, patch_height, patch_width}),
            create_arrayN<float>(valid_map_masks,
                                 {n_views, 1, patch_height, patch_width})};
}
