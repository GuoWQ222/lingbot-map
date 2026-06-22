# LingBot-MAP 当前训练 Loss 公式说明

本文档使用 Markdown + LaTeX 数学公式。行内公式使用 `$...$`，块级公式使用 `$$...$$`。

## 当前 Loss 权重

| 参数 | 当前值 |
| --- | ---: |
| `camera_weight` | `5.0` |
| `relative_pose_weight` | `1.0` |
| `depth_weight` | `1.0` |
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

## 总目标函数

训练实际反传的是 `loss_objective`：

$$
\mathcal{L}_{\text{objective}}
= 5.0 \cdot \mathcal{L}_{\text{camera}}
+ 1.0 \cdot \mathcal{L}_{\text{relative-pose}}
+ 1.0 \cdot
\left(
  \mathcal{L}_{\text{conf-depth}}
  + \mathcal{L}_{\text{reg-depth}}
  + \mathcal{L}_{\text{grad-depth}}
\right)
$$

当前 `accum_steps = 1`，所以每个 optimizer step 直接对 $\mathcal{L}_{\text{objective}}$ 做 backward。

## 有效帧过滤

Camera、relative pose、depth loss 都会使用 `point_masks` 判断有效像素。

对第 $b$ 个样本、第 $t$ 帧：

$$
m_{b,t}^{\text{valid}}
= \left[
  \sum_{u,v} M_{b,t,u,v} > 100
\right]
$$

其中 $M$ 是 `point_masks`。如果某帧有效像素不超过 `min_valid_pixels = 100`，该帧不会参与对应 loss 的监督。

## Camera Loss

相关日志 key：`loss_camera`, `loss_T`, `loss_R`, `loss_FL`。

数据中的 `extrinsics` 是 world-to-camera，训练时先转为 camera-to-world：

$$
T_{c2w}^{gt} = \left(T_{w2c}^{gt}\right)^{-1}
$$

GT pose 编码为 `absT_quaR_FoV`：

$$
y^{gt} = \left[\mathbf{t}^{gt}, \mathbf{q}^{gt}, \mathbf{f}^{gt}\right]
$$

预测 pose encoding：

$$
\hat{y} = \left[\hat{\mathbf{t}}, \hat{\mathbf{q}}, \hat{\mathbf{f}}\right]
$$

其中 $\mathbf{t}$ 是 translation，$\mathbf{q}$ 是 quaternion rotation，$\mathbf{f}$ 是 focal/FoV encoding。

### 单个 Stage

当前 `camera_loss_type = l1`。第 $s$ 个 stage：

$$
\mathcal{L}_{T}^{(s)}
= \operatorname{mean}\left(\left|\hat{\mathbf{t}}^{(s)} - \mathbf{t}^{gt}\right|\right)
$$

$$
\mathcal{L}_{R}^{(s)}
= \operatorname{mean}\left(\left|\hat{\mathbf{q}}^{(s)} - \mathbf{q}^{gt}\right|\right)
$$

$$
\mathcal{L}_{FL}^{(s)}
= \operatorname{mean}\left(\left|\hat{\mathbf{f}}^{(s)} - \mathbf{f}^{gt}\right|\right)
$$

translation loss 会截断：

$$
\mathcal{L}_{T}^{(s)} \leftarrow \min\left(\mathcal{L}_{T}^{(s)}, 100\right)
$$

### 多 Stage 加权

如果一共有 $S$ 个 stage，第 $s$ 个 stage 权重：

$$
w_s = \gamma^{S - s - 1}, \quad \gamma = 0.6
$$

最终：

$$
\mathcal{L}_T
= \frac{1}{S}\sum_{s=0}^{S-1} w_s \mathcal{L}_{T}^{(s)}
$$

$$
\mathcal{L}_R
= \frac{1}{S}\sum_{s=0}^{S-1} w_s \mathcal{L}_{R}^{(s)}
$$

$$
\mathcal{L}_{FL}
= \frac{1}{S}\sum_{s=0}^{S-1} w_s \mathcal{L}_{FL}^{(s)}
$$

Camera loss 合成为：

$$
\mathcal{L}_{\text{camera}}
= 1.0 \cdot \mathcal{L}_T
+ 1.0 \cdot \mathcal{L}_R
+ 0.5 \cdot \mathcal{L}_{FL}
$$

进入总目标时：

$$
5.0 \cdot \mathcal{L}_{\text{camera}}
$$

## Relative Pose Loss

相关日志 key：`loss_relative_pose`, `loss_relative_rot`, `loss_relative_trans`。

这个 loss 监督帧对之间的相对位姿。预测和 GT 的 camera-to-world 矩阵为：

$$
\hat{T}_{c2w,t} = \operatorname{PoseDecode}\left(\hat{y}_t\right)
$$

$$
T_{c2w,t}^{gt} = \left(T_{w2c,t}^{gt}\right)^{-1}
$$

对每个有效帧对 $(i, j)$：

$$
i \ne j, \quad |i-j| < W, \quad W = 64
$$

相对位姿为：

$$
\hat{T}_{i \rightarrow j}
= \left(\hat{T}_{c2w,i}\right)^{-1}\hat{T}_{c2w,j}
$$

$$
T_{i \rightarrow j}^{gt}
= \left(T_{c2w,i}^{gt}\right)^{-1}T_{c2w,j}^{gt}
$$

### 相对旋转 Loss

设相对旋转为 $\hat{R}_{i \rightarrow j}$ 和 $R_{i \rightarrow j}^{gt}$：

$$
R_{\text{err}} = \hat{R}_{i \rightarrow j}^{T}R_{i \rightarrow j}^{gt}
$$

代码中使用 geodesic angle：

$$
\cos\theta
= \operatorname{clamp}\left(
  \frac{\operatorname{tr}(R_{\text{err}})-1}{2}, -1, 1
\right)
$$

$$
\sin\theta
= \frac{1}{2}
\left\|
\begin{bmatrix}
R_{\text{err},32} - R_{\text{err},23} \\
R_{\text{err},13} - R_{\text{err},31} \\
R_{\text{err},21} - R_{\text{err},12}
\end{bmatrix}
\right\|_2
$$

$$
\mathcal{L}_{\text{relative-rot}}
= \operatorname{mean}_{(i,j)}\left[\operatorname{atan2}(\sin\theta, \cos\theta)\right]
$$

### 相对平移 Loss

$$
\mathcal{L}_{\text{relative-trans}}
= \operatorname{mean}_{(i,j)}
\left(
  \left|\hat{\mathbf{t}}_{i\rightarrow j} - \mathbf{t}_{i\rightarrow j}^{gt}\right|
\right)
$$

### 相对位姿总 Loss

$$
\mathcal{L}_{\text{relative-pose}}
= \mathcal{L}_{\text{relative-rot}}
+ \lambda_{\text{trans}}\mathcal{L}_{\text{relative-trans}}
$$

当前 $\lambda_{\text{trans}} = 1.0$，所以：

$$
\mathcal{L}_{\text{relative-pose}}
= \mathcal{L}_{\text{relative-rot}} + \mathcal{L}_{\text{relative-trans}}
$$

Relative pose loss 也会对多个 stage 使用 $w_s = \gamma^{S-s-1}$ 并加权平均。进入总目标时为 $1.0 \cdot \mathcal{L}_{\text{relative-pose}}$。

## Depth Loss

相关日志 key：`loss_conf_depth`, `loss_reg_depth`, `loss_grad_depth`。

记预测深度、GT 深度、confidence 为：

$$
\hat{D}_{b,t,u,v}, \quad D_{b,t,u,v}^{gt}, \quad C_{b,t,u,v}
$$

### 普通深度回归 Loss

代码中的普通深度回归项就是 `loss_reg_depth`。它先对每个有效像素计算最后一维 `channels` 上的范数：

$$
e_{b,t,u,v}
= \left\|D_{b,t,u,v}^{gt} - \hat{D}_{b,t,u,v}\right\|_2
$$

这里的 $\|\cdot\|_2$ 是对 depth tensor 最后一维取范数，**不是平方后的 MSE**。当前 depth 通常为单通道，所以它等价于绝对误差：

$$
e_{b,t,u,v}
= \left|D_{b,t,u,v}^{gt} - \hat{D}_{b,t,u,v}\right|
$$

普通深度回归 loss：

$$
\mathcal{L}_{\text{reg-depth}}
= \operatorname{mean}_{M_{b,t,u,v}=1}\left(e_{b,t,u,v}\right)
$$

### Confidence Depth Loss

带 confidence 的深度 loss：

$$
\ell_{\text{conf}}(b,t,u,v)
= \gamma_d e_{b,t,u,v} C_{b,t,u,v}
- \alpha_d \log C_{b,t,u,v}
$$

当前 $\gamma_d = 1.0$，$\alpha_d = 0.2$，所以：

$$
\ell_{\text{conf}}(b,t,u,v)
= e_{b,t,u,v} C_{b,t,u,v} - 0.2 \log C_{b,t,u,v}
$$

最终：

$$
\mathcal{L}_{\text{conf-depth}}
= \operatorname{mean}_{M_{b,t,u,v}=1}\left(\ell_{\text{conf}}(b,t,u,v)\right)
$$

### valid_range 分位过滤

`loss_conf_depth` 和 `loss_reg_depth` 在元素足够多时会做分位过滤。当前：

$$
\text{valid\_range}=0.98
$$

含义是过滤最高约 $2\%$ 的极端 loss 值。

### Depth Gradient Loss

当前 `depth_gradient_loss_fn = grad`。先定义深度误差图：

$$
\Delta D = \hat{D} - D^{gt}
$$

x 方向和 y 方向梯度误差：

$$
g_x(u,v)
= \left|\Delta D(u,v+1)-\Delta D(u,v)\right|
$$

$$
g_y(u,v)
= \left|\Delta D(u+1,v)-\Delta D(u,v)\right|
$$

梯度误差会截断：

$$
g_x \leftarrow \min(g_x, 100), \quad g_y \leftarrow \min(g_y, 100)
$$

单尺度梯度 loss：

$$
\mathcal{L}_{\text{grad}}^{(k)}
= \frac{\sum(g_x^{(k)} + g_y^{(k)})}{\text{number of valid pixels}}
$$

多尺度使用 $k \in \{1,2,4,8\}$，最终：

$$
\mathcal{L}_{\text{grad-depth}}
= \frac{1}{4}\sum_{k\in\{1,2,4,8\}}\mathcal{L}_{\text{grad}}^{(k)}
$$

Depth 总项：

$$
\mathcal{L}_{\text{depth}}
= \mathcal{L}_{\text{conf-depth}}
+ \mathcal{L}_{\text{reg-depth}}
+ \mathcal{L}_{\text{grad-depth}}
$$

进入总目标时为 $1.0 \cdot \mathcal{L}_{\text{depth}}$。

## 日志 Key 对照

| Key | 含义 | 是否进入总目标 |
| --- | --- | --- |
| `loss_objective` | 最终 backward 的总 loss | 是 |
| `loss_camera` | camera 子项总和 | 是，乘 `5.0` |
| `loss_T` | 绝对 translation loss | 通过 `loss_camera` 进入 |
| `loss_R` | 绝对 quaternion rotation loss | 通过 `loss_camera` 进入 |
| `loss_FL` | focal/FoV encoding loss | 通过 `loss_camera` 进入，内部权重 `0.5` |
| `loss_relative_pose` | 相对位姿总 loss | 是，乘 `1.0` |
| `loss_relative_rot` | 相对旋转 geodesic loss | 通过 `loss_relative_pose` 进入 |
| `loss_relative_trans` | 相对平移 L1 loss | 通过 `loss_relative_pose` 进入 |
| `loss_conf_depth` | confidence 加权深度 loss | 是 |
| `loss_reg_depth` | 普通深度回归 loss | 是 |
| `loss_grad_depth` | 多尺度深度梯度一致性 loss | 是 |

## DDP 下的 Loss 和梯度同步

每个 rank 在自己的 local batch 上独立计算：

$$
\mathcal{L}_{\text{objective}}^{(r)}
$$

然后做：

$$
\mathcal{L}_{\text{backward}}^{(r)}
= \frac{\mathcal{L}_{\text{objective}}^{(r)}}{\text{accum\_steps}}
$$

当前 `accum_steps = 1`，所以：

$$
\mathcal{L}_{\text{backward}}^{(r)}
= \mathcal{L}_{\text{objective}}^{(r)}
$$

DDP 在 backward 过程中对各 rank 的梯度做 all-reduce 平均：

$$
\nabla \theta
= \frac{1}{R}\sum_{r=0}^{R-1}\nabla_{\theta}\mathcal{L}_{\text{objective}}^{(r)}
$$

其中 $R$ 是 `world_size`。
