# `/cpfs/user/guowenqi/lingbot-map/train.sh` 训练损失函数公式

本文档总结当前 `train.sh` 启动训练时实际采用的损失函数。`train.sh` 没有显式传入 loss 相关参数，因此 loss 超参数来自 `train.py` 中 `build_argparser()` 和 `VGGTStyleLoss` 的默认值。

## 1. 当前启用的 Loss 项

当前总目标包含三类损失：

| Loss 项 | 日志 key | 权重 |
| --- | --- | ---: |
| Camera loss | `loss_camera` | `camera_weight = 5.0` |
| Relative pose loss | `loss_relative_pose` | `relative_pose_weight = 1.0` |
| Depth loss | `loss_conf_depth + loss_reg_depth + loss_grad_depth` | `depth_weight = 1.0` |

Point-map loss 相关参数仍保留为兼容项，但在 LingBot-MAP fine-tuning 中已禁用，不进入总损失。

相关默认超参数如下：

| 参数 | 当前默认值 |
| --- | ---: |
| `camera_loss_type` | `l1` |
| `camera_gamma` | `0.6` |
| `weight_trans` | `1.0` |
| `weight_rot` | `1.0` |
| `weight_focal` | `0.5` |
| `relative_trans_weight` | `1.0` |
| `relative_pose_window` | `64` |
| `depth_gradient_loss_fn` | `grad` |
| `loss_gamma` | `1.0` |
| `loss_alpha` | `0.2` |
| `valid_range` | `0.98` |
| `min_valid_pixels` | `100` |

## 2. 有效帧和有效像素

训练样本包含 `point_masks`，记第 $b$ 个样本、第 $t$ 帧、第 $(u,v)$ 个像素的 mask 为 $M_{b,t,u,v}$。一帧是否有效由有效像素数决定：

$$
m_{b,t}^{\mathrm{frame}}
= \mathbb{I}\left[
\sum_{u,v} M_{b,t,u,v} > 100
\right].
$$

Camera loss、relative pose loss 和 depth loss 都只使用有效帧。Depth loss 进一步只在有效像素上计算。

## 3. Camera Loss

数据中的 `extrinsics` 是 world-to-camera 矩阵，训练监督前先转为 camera-to-world：

$$
T_{c2w,t}^{gt} = \left(T_{w2c,t}^{gt}\right)^{-1}.
$$

随后 GT 位姿被编码为 `absT_quaR_FoV`：

$$
y_t^{gt}
= \left[
\mathbf{t}_t^{gt},
\mathbf{q}_t^{gt},
\mathbf{f}_t^{gt}
\right],
$$

其中 $\mathbf{t}$ 是平移，$\mathbf{q}$ 是四元数旋转，$\mathbf{f}$ 是 focal/FoV 编码。模型每个 refinement stage 都预测一个 pose encoding：

$$
\hat{y}_t^{(s)}
= \left[
\hat{\mathbf{t}}_t^{(s)},
\hat{\mathbf{q}}_t^{(s)},
\hat{\mathbf{f}}_t^{(s)}
\right].
$$

当前 `camera_loss_type = l1`，因此第 $s$ 个 stage 的三项误差为：

$$
\mathcal{L}_{T}^{(s)}
=
\operatorname{mean}_{(b,t):m_{b,t}^{\mathrm{frame}}=1}
\left(
\left|
\hat{\mathbf{t}}_{b,t}^{(s)} - \mathbf{t}_{b,t}^{gt}
\right|
\right),
$$

$$
\mathcal{L}_{R}^{(s)}
=
\operatorname{mean}_{(b,t):m_{b,t}^{\mathrm{frame}}=1}
\left(
\left|
\hat{\mathbf{q}}_{b,t}^{(s)} - \mathbf{q}_{b,t}^{gt}
\right|
\right),
$$

$$
\mathcal{L}_{FL}^{(s)}
=
\operatorname{mean}_{(b,t):m_{b,t}^{\mathrm{frame}}=1}
\left(
\left|
\hat{\mathbf{f}}_{b,t}^{(s)} - \mathbf{f}_{b,t}^{gt}
\right|
\right).
$$

代码中会把 translation loss 截断到最大 100：

$$
\mathcal{L}_{T}^{(s)}
\leftarrow
\min\left(\mathcal{L}_{T}^{(s)}, 100\right).
$$

若共有 $S$ 个 stage，第 $s$ 个 stage 的权重为：

$$
w_s = \gamma_c^{S-s-1}, \qquad \gamma_c = 0.6.
$$

注意代码是除以 stage 数 $S$，不是除以 $\sum_s w_s$：

$$
\mathcal{L}_{T}
=
\frac{1}{S}
\sum_{s=0}^{S-1}
w_s \mathcal{L}_{T}^{(s)},
$$

$$
\mathcal{L}_{R}
=
\frac{1}{S}
\sum_{s=0}^{S-1}
w_s \mathcal{L}_{R}^{(s)},
$$

$$
\mathcal{L}_{FL}
=
\frac{1}{S}
\sum_{s=0}^{S-1}
w_s \mathcal{L}_{FL}^{(s)}.
$$

Camera loss 合成为：

$$
\mathcal{L}_{\mathrm{camera}}
=
1.0 \mathcal{L}_{T}
+
1.0 \mathcal{L}_{R}
+
0.5 \mathcal{L}_{FL}.
$$

进入总目标时：

$$
\mathcal{L}_{\mathrm{camera,total}}
=
5.0 \mathcal{L}_{\mathrm{camera}}.
$$

## 4. Relative Pose Loss

Relative pose loss 监督有效帧对之间的相对位姿。预测 pose encoding 会被解码为 camera-to-world 矩阵：

$$
\hat{T}_{c2w,t}^{(s)}
=
\operatorname{PoseDecode}\left(\hat{y}_t^{(s)}\right).
$$

GT 同样使用 camera-to-world：

$$
T_{c2w,t}^{gt}
=
\left(T_{w2c,t}^{gt}\right)^{-1}.
$$

对每个样本内的有效帧对 $(i,j)$，代码使用有序帧对，并要求：

$$
i \ne j,\qquad |i-j| < W,\qquad W=64.
$$

相对位姿为：

$$
\hat{T}_{i\rightarrow j}^{(s)}
=
\left(\hat{T}_{c2w,i}^{(s)}\right)^{-1}
\hat{T}_{c2w,j}^{(s)},
$$

$$
T_{i\rightarrow j}^{gt}
=
\left(T_{c2w,i}^{gt}\right)^{-1}
T_{c2w,j}^{gt}.
$$

### 4.1 相对旋转 Loss

设相对旋转矩阵为 $\hat{R}_{i\rightarrow j}^{(s)}$ 和 $R_{i\rightarrow j}^{gt}$，代码先计算：

$$
R_{\mathrm{err}}
=
\left(\hat{R}_{i\rightarrow j}^{(s)}\right)^T
R_{i\rightarrow j}^{gt}.
$$

旋转 geodesic angle 为：

$$
\cos \theta
=
\operatorname{clamp}
\left(
\frac{\operatorname{tr}(R_{\mathrm{err}})-1}{2},
-1,
1
\right),
$$

$$
\sin \theta
=
\frac{1}{2}
\left\|
\begin{bmatrix}
R_{\mathrm{err},32} - R_{\mathrm{err},23} \\
R_{\mathrm{err},13} - R_{\mathrm{err},31} \\
R_{\mathrm{err},21} - R_{\mathrm{err},12}
\end{bmatrix}
\right\|_2,
$$

$$
\ell_{\mathrm{rot}}^{(s)}(i,j)
=
\operatorname{atan2}(\sin\theta,\cos\theta).
$$

### 4.2 相对平移 Loss

设相对平移为 $\hat{\mathbf{p}}_{i\rightarrow j}^{(s)}$ 和 $\mathbf{p}_{i\rightarrow j}^{gt}$：

$$
\ell_{\mathrm{trans}}^{(s)}(i,j)
=
\left\|
\hat{\mathbf{p}}_{i\rightarrow j}^{(s)}
-
\mathbf{p}_{i\rightarrow j}^{gt}
\right\|_1.
$$

代码中 `F.l1_loss(..., reduction="mean")` 会对所有有效帧对和 xyz 维度取平均。

### 4.3 相对位姿总 Loss

单个 stage 的 relative pose loss 为：

$$
\mathcal{L}_{\mathrm{rel}}^{(s)}
=
\operatorname{mean}_{(i,j)}
\left[
\ell_{\mathrm{rot}}^{(s)}(i,j)
+
\lambda_{\mathrm{trans}}
\ell_{\mathrm{trans}}^{(s)}(i,j)
\right],
$$

其中：

$$
\lambda_{\mathrm{trans}} = 1.0.
$$

多 stage 加权方式与 camera loss 相同：

$$
\mathcal{L}_{\mathrm{relative\ pose}}
=
\frac{1}{S}
\sum_{s=0}^{S-1}
w_s \mathcal{L}_{\mathrm{rel}}^{(s)},
\qquad
w_s = 0.6^{S-s-1}.
$$

进入总目标时：

$$
\mathcal{L}_{\mathrm{relative,total}}
=
1.0 \mathcal{L}_{\mathrm{relative\ pose}}.
$$

## 5. Depth Loss

Depth loss 使用预测深度 $\hat{D}_{b,t,u,v}$、GT 深度 $D_{b,t,u,v}^{gt}$ 和预测 confidence $C_{b,t,u,v}$。代码先做：

$$
C_{b,t,u,v}
\leftarrow
\max(C_{b,t,u,v}, 10^{-6}).
$$

### 5.1 普通深度回归 Loss

对每个有效像素计算深度误差：

$$
e_{b,t,u,v}
=
\left\|
D_{b,t,u,v}^{gt}
-
\hat{D}_{b,t,u,v}
\right\|_2.
$$

这里的 $\|\cdot\|_2$ 是沿 depth tensor 最后一维的范数。当前深度通常为单通道，因此等价于绝对误差：

$$
e_{b,t,u,v}
=
\left|
D_{b,t,u,v}^{gt}
-
\hat{D}_{b,t,u,v}
\right|.
$$

若有效元素数大于 1000，代码会先将元素级误差 clamp 到 100，再按 `valid_range = 0.98` 过滤最高约 2% 的大误差。记过滤后的有效像素集合为 $\Omega_D'$，则：

$$
\mathcal{L}_{\mathrm{reg-depth}}
=
\operatorname{mean}_{(b,t,u,v)\in\Omega_D'}
e_{b,t,u,v}.
$$

### 5.2 Confidence Depth Loss

元素级 confidence depth loss 为：

$$
\ell_{\mathrm{conf}}(b,t,u,v)
=
\gamma_d e_{b,t,u,v} C_{b,t,u,v}
-
\alpha_d \log C_{b,t,u,v}.
$$

当前：

$$
\gamma_d = 1.0,\qquad \alpha_d = 0.2.
$$

因此：

$$
\ell_{\mathrm{conf}}(b,t,u,v)
=
e_{b,t,u,v} C_{b,t,u,v}
-
0.2 \log C_{b,t,u,v}.
$$

同样地，若有效元素数大于 1000，会按 `valid_range = 0.98` 过滤高 loss 元素。记过滤后的集合为 $\Omega_C'$：

$$
\mathcal{L}_{\mathrm{conf-depth}}
=
\operatorname{mean}_{(b,t,u,v)\in\Omega_C'}
\ell_{\mathrm{conf}}(b,t,u,v).
$$

### 5.3 多尺度深度梯度 Loss

当前 `depth_gradient_loss_fn = grad`，使用误差图：

$$
\Delta D_{b,t,u,v}
=
\hat{D}_{b,t,u,v}
-
D_{b,t,u,v}^{gt}.
$$

在某个尺度 $k$ 下，代码通过 stride slicing 取子采样图：

$$
k \in \{1,2,4,8\}.
$$

x 方向和 y 方向的相邻误差差分为：

$$
g_x^{(k)}(u,v)
=
\left|
\Delta D^{(k)}(u,v+1)
-
\Delta D^{(k)}(u,v)
\right|,
$$

$$
g_y^{(k)}(u,v)
=
\left|
\Delta D^{(k)}(u+1,v)
-
\Delta D^{(k)}(u,v)
\right|.
$$

只有相邻两个像素都有效时，该差分才计入 loss；并且差分值会被截断：

$$
g_x^{(k)}
\leftarrow
\min(g_x^{(k)}, 100),
\qquad
g_y^{(k)}
\leftarrow
\min(g_y^{(k)}, 100).
$$

令 $N_{\mathrm{valid}}^{(k)}$ 为该尺度下有效像素数量，单尺度梯度 loss 为：

$$
\mathcal{L}_{\mathrm{grad}}^{(k)}
=
\frac{
\sum g_x^{(k)}
+
\sum g_y^{(k)}
}{
N_{\mathrm{valid}}^{(k)}
}.
$$

最终多尺度深度梯度 loss 为：

$$
\mathcal{L}_{\mathrm{grad-depth}}
=
\frac{1}{4}
\sum_{k\in\{1,2,4,8\}}
\mathcal{L}_{\mathrm{grad}}^{(k)}.
$$

### 5.4 Depth 总项

Depth loss 合成为：

$$
\mathcal{L}_{\mathrm{depth}}
=
\mathcal{L}_{\mathrm{conf-depth}}
+
\mathcal{L}_{\mathrm{reg-depth}}
+
\mathcal{L}_{\mathrm{grad-depth}}.
$$

进入总目标时：

$$
\mathcal{L}_{\mathrm{depth,total}}
=
1.0 \mathcal{L}_{\mathrm{depth}}.
$$

## 6. 总损失函数

训练实际反传的是 `loss_objective`：

$$
\mathcal{L}_{\mathrm{objective}}
=
5.0\mathcal{L}_{\mathrm{camera}}
+
1.0\mathcal{L}_{\mathrm{relative\ pose}}
+
1.0
\left(
\mathcal{L}_{\mathrm{conf-depth}}
+
\mathcal{L}_{\mathrm{reg-depth}}
+
\mathcal{L}_{\mathrm{grad-depth}}
\right).
$$

展开 camera 项后：

$$
\mathcal{L}_{\mathrm{objective}}
=
5.0
\left(
\mathcal{L}_{T}
+
\mathcal{L}_{R}
+
0.5\mathcal{L}_{FL}
\right)
+
\mathcal{L}_{\mathrm{relative\ pose}}
+
\mathcal{L}_{\mathrm{conf-depth}}
+
\mathcal{L}_{\mathrm{reg-depth}}
+
\mathcal{L}_{\mathrm{grad-depth}}.
$$

若使用梯度累积，backward 前还会除以 `accum_steps`：

$$
\mathcal{L}_{\mathrm{backward}}
=
\frac{
\mathcal{L}_{\mathrm{objective}}
}{
\mathrm{accum\_steps}
}.
$$

当前 `train.sh` 默认 `ACCUM_STEPS=1`，因此默认情况下：

$$
\mathcal{L}_{\mathrm{backward}}
=
\mathcal{L}_{\mathrm{objective}}.
$$

## 7. 日志 Key 对照

| 日志 key | 含义 | 是否直接进入总目标 |
| --- | --- | --- |
| `loss_objective` | 最终训练目标 | 是 |
| `loss_camera` | camera 子项合成后的 loss | 是，乘 5.0 |
| `loss_T` | absolute translation loss | 通过 `loss_camera` 进入 |
| `loss_R` | absolute quaternion rotation loss | 通过 `loss_camera` 进入 |
| `loss_FL` | focal/FoV encoding loss | 通过 `loss_camera` 进入，内部乘 0.5 |
| `loss_relative_pose` | 相对位姿总 loss | 是，乘 1.0 |
| `loss_relative_rot` | 相对旋转 geodesic loss | 通过 `loss_relative_pose` 进入 |
| `loss_relative_trans` | 相对平移 L1 loss | 通过 `loss_relative_pose` 进入 |
| `loss_conf_depth` | confidence 加权深度 loss | 是 |
| `loss_reg_depth` | 普通深度回归 loss | 是 |
| `loss_grad_depth` | 多尺度深度梯度一致性 loss | 是 |
