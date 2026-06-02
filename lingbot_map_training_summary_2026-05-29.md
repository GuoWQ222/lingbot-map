# LingBot-MAP 当前训练配置总结

基于 `/cpfs/user/guowenqi/lingbot-map/train.sh`、`train.py` 与 `lingbot_map/` 当前代码整理；生成日期：2026-05-29。

## 总览

- 当前脚本是在 LingBot-MAP / GCTStream 上做 RGB-D 轨迹微调，监督相机位姿和稠密深度；point-map head 在 `build_model` 中被显式关闭。
- 默认训练不是单一 Manip 数据集，而是 `Manip_long3/4/5` 与五个外部 3D 数据集的跨数据集 curriculum 混合。
- 当前 manifest 文件为 `${OUTPUT_DIR}/manip_trajectory_manifest.txt`，已检查为 6929 行；按 `val_fraction=0.02` 与 `split_scenes` 的 `round` 逻辑，对应约 6790 train / 139 val。

## 数据集

Manip 主数据源：

- `Manip_long3`: `/oss-guowenqi/Manip_long3/data`
- `Manip_long4`: `/oss-guowenqi/Manip_long4/data`
- `Manip_long5`: `/oss-guowenqi/Manip_long5/data`

外部 mix-in 默认启用：

- DL3DV: `/cpfs/shared/landmark/renkerui/data/dl3dv`
- ScanNet++ v2: `/shared/smartbot/renkerui/data/scannetppv2`
- TartanAir: `/cpfs/shared/landmark/renkerui/data/tartanair`
- DynamicReplica: `/shared/smartbot/renkerui/data/dynamic_replica`
- MapFree: `/cpfs/shared/landmark/renkerui/data/mapfree`

`*_VAL=0`，所以默认验证集只来自 Manip split。

## 采样策略

- 外层使用 `CurriculumMixtureSampler`：`p_manip` 从 0.40 在 step 2000 后线性升到 0.90，在 step 15000 后保持 0.90。
- 若 5 个 external 全启用，则 step<=2000 时每个 external 概率为 0.12，step>=15000 时每个 external 概率为 0.02。
- `limit_train_batches=1000`，因此一个 epoch 产生 1000 个样本索引；sampler 权重在每个 epoch 开始按 `global_step` 更新。
- Manip_long3/4：`S=0,W=1,M=0`，实际走 W 单 realsense 随机轨迹，stride 10-60，最多 64 帧，至少 16 帧。
- Manip_long5：路径含 `Manip_long5` 时自动走 T，单 surround_cam，stride 15-60。
- 外部数据集：`num_views/min_views=0` 会继承 64/16；每次样本在 16 到 64 帧之间随机长度。

## 模型

- 主模型：`lingbot_map.models.gct_stream.GCTStream`
- Aggregator：`AggregatorStream`，默认 `dinov2_vitl14_reg`，embed_dim=1024，3D RoPE 开启。
- Camera：`CameraCausalHead`，`camera_num_iterations=4`。
- Depth：`DPTHead` 输出 `depth` 与 `depth_conf`，activation checkpoint 开启。
- Point/local point：关闭。
- 默认仅冻结 `aggregator.patch_embed`，也就是 DINO patch embed。

## 损失函数

`objective = 5.0 * loss_camera + 1.0 * loss_relative_pose + 1.0 * (loss_conf_depth + loss_reg_depth + loss_grad_depth)`

- Camera loss：GT w2c 转 c2w 后编码为 `absT_quaR_FoV`，对 `pose_enc_list` 各阶段用 `gamma=0.6` 加权平均；L1，T/R/FL 权重为 1/1/0.5。
- Relative pose：有效帧对内比较相对位姿，旋转 geodesic，平移 L1，窗口 64。
- Depth：逐像素 depth regression + confidence loss + 4-scale gradient loss；`valid_range=0.98` 过滤高分位 outlier；`min_valid_pixels=100`。
- Point loss：关闭。

## 优化

- AdamW，lr=5e-5，min_lr=1e-8，weight_decay=0.05。
- warmup_ratio=0.05，之后 cosine decay。
- AMP bf16，grad_clip_norm=1.0。
- save_every=10000，val_every=10000，TensorBoard 默认开启。
