## 推荐方案

针对你的场景，我不建议直接把当前 IPOPT 换成另一个“每帧只迭代一步”的速度 IK，而是设计成：

> **单臂 Pinocchio 分层位置 IK + TRAC-IK Distance 失败回退 + 独立关节指令整形器**

正常遥操作时，Pinocchio 求解器从上一时刻附近直接收敛，确定性高、精度高、不会换解支；只有遇到大幅跳变、奇异位形或初始化失败时才调用 TRAC-IK 搜索其他解。

---

# 1. 首先把当前双臂 IK 真正拆成单臂

你当前 A2D 模型保留左右共 14 个活动关节；即使调用 `solve_from_ee_pose(side=...)`，内部依然进入双臂 `solve_ik()`，同时优化左右末端目标。([GitHub][1])

这会带来两个不必要的问题：

* 优化变量从 7 维扩大到 14 维；
* 未控制侧的末端约束仍然参与优化，可能影响受控侧收敛；
* 当前平滑项和零位正则项同时作用于两臂；
* 单侧机械臂的唯一冗余自由度没有被明确利用。

应建立两个独立模型：

```text
LeftArmPrecisionIK
    active_joints = Joint1_l ... Joint7_l
    base_frame    = torso / shoulder base
    ee_frame      = left_tcp

RightArmPrecisionIK
    active_joints = Joint1_r ... Joint7_r
    base_frame    = torso / shoulder base
    ee_frame      = right_tcp
```

一次只实例化或调用正在控制的一侧，另一侧不进入 IK。

---

# 2. 明确区分三个关节状态

这是整个方案最重要的架构调整：

```text
q_measured：机器人当前实测关节角
q_ref：     IK 计算出的精确目标关节角
q_cmd：     经过速度、加速度限制后实际发送的关节角
```

当前实现把“靠近上一帧”直接放进 IK 目标：

[
0.1|q-q_{\mathrm{last}}|^2
]

并且求解后又执行四帧加权移动平均。([GitHub][1])

这会让 `q_ref` 本身就不是精确 IK 解。新的方案中：

* IK 只负责求准确的 `q_ref`；
* 跳变抑制只发生在 `q_ref → q_cmd`；
* 不再对 IK 结果做移动平均；
* 降低外部控制频率不会改变最终 IK 解。

---

# 3. 主求解器：分层阻尼最小二乘位置 IK

## 3.1 位姿误差

设当前末端位姿为：

[
{}^0T_e(q)
]

目标位姿为：

[
{}^0T_e^*
]

使用李群误差：

[
T_{\mathrm{err}}
================

{}^0T_e(q)^{-1}{}^0T_e^*
]

[
e(q)=\log_{SE(3)}(T_{\mathrm{err}})
]

Pinocchio 官方 IK 示例正是通过 `log6`、`Jlog6` 和阻尼伪逆完成迭代，而不是简单地把欧拉角相减。([gepettoweb.laas.fr][2])

位置和旋转必须分别判断精度：

[
e_p = |p^*-p(q)|
]

[
e_R =
\left|
\log_{SO(3)}\left(R(q)^TR^*\right)
\right|
]

推荐初始停止条件：

```python
position_tolerance = 5e-4             # 0.5 mm
rotation_tolerance = np.deg2rad(0.2) # 0.2°
```

数值求解可以设得更严格，但真机精度最终还会受到关节零位、TCP、连杆参数和机械间隙影响。

## 3.2 对位置和旋转归一化

不能再使用当前这种不具备明确量纲意义的：

```python
50 * position_cost + rotation_cost
```

改为：

[
\bar e =
\begin{bmatrix}
e_p / \sigma_p\
e_R / \sigma_R
\end{bmatrix}
]

推荐：

```python
sigma_position = 1e-3              # 1 mm
sigma_rotation = np.deg2rad(0.5)  # 0.5°
```

相应的 Jacobian 也乘以同样的归一化矩阵：

[
\bar J=W_eJ
]

这样“误差为 1”具有清楚含义：位置误差达到 1 mm，或姿态误差达到 0.5°。

---

# 4. 每次调用内部迭代到收敛

外部控制可能只有 20～60 Hz，但每次 IK 调用内部运行若干次迭代：

```python
for iteration in range(30):
    compute_fk()
    compute_pose_error()
    compute_jacobian()
    compute_damped_update()
    line_search()
```

因此一次外部调用就应该产生完整的目标解，而不是依靠之后几十个控制周期逐渐逼近。

基本更新为：

[
\Delta q_{\mathrm{task}}
========================

-\bar J^#
\bar e
]

阻尼伪逆采用 SVD：

[
\bar J=U\Sigma V^T
]

[
\bar J^#
========

V
\operatorname{diag}
\left(
\frac{\sigma_i}{\sigma_i^2+\lambda^2}
\right)
U^T
]

阻尼建议根据最小奇异值自动调节：

```python
if sigma_min > sigma_safe:
    damping = 1e-4
else:
    ratio = np.clip(sigma_min / sigma_safe, 0.0, 1.0)
    damping = 1e-4 + (1.0 - ratio) ** 2 * 0.1
```

远离奇异位形时阻尼很小，可以获得高精度；接近奇异位形时增大阻尼，避免关节更新爆炸。

---

# 5. 7DoF 冗余自由度使用“严格任务优先级”

7DoF 手臂完成 6D 末端位姿任务后，通常还剩一个冗余自由度。你的需求是精度优先，因此不要把关节平滑直接加入主任务，而应只在末端任务的零空间里优化。

对归一化 Jacobian 做完整 SVD：

[
\bar J=U\Sigma V^T
]

取零空间基：

[
Z=V[:,r:]
]

则零空间投影矩阵为：

[
N=ZZ^T
]

二级更新：

[
\Delta q_{\mathrm{secondary}}
=============================

-N
\left(
k_c\nabla H_{\mathrm{continuity}}
+
k_l\nabla H_{\mathrm{limit}}
\right)
]

最终：

[
\Delta q=
\Delta q_{\mathrm{task}}
+
\Delta q_{\mathrm{secondary}}
]

这样连续性和关节限位只利用剩余自由度，理论上不与末端精度竞争。

## 连续性目标

[
H_{\mathrm{continuity}}
=======================

\frac{1}{2}
(q-q_{\mathrm{anchor}})^T
W_q
(q-q_{\mathrm{anchor}})
]

其中：

```python
q_anchor = previous_valid_ik_solution
```

第一帧才使用：

```python
q_anchor = q_measured
```

关节权重用关节行程归一化：

[
W_{q,ii}=
\frac{1}{(q_{\max,i}-q_{\min,i})^2}
]

这样不会因为不同关节的活动范围不同而产生不合理偏置。

## 关节限位目标

建议使用平滑的高阶限位代价，而不是奇异 barrier：

[
u_i=
\frac{q_i-q_{\mathrm{mid},i}}
{0.5(q_{\max,i}-q_{\min,i})}
]

[
H_{\mathrm{limit}}=\sum_i u_i^8
]

它在关节中部几乎没有影响，接近限位时才快速增大。

推荐初值：

```python
k_continuity = 0.05
k_joint_limit = 0.005
```

这两个量只进入零空间，所以可以比当前软目标更放心地调整。

---

# 6. 增加步长限制和回溯线搜索

即使阻尼存在，也不要直接接受完整的 `Δq`。

先限制单次内部迭代：

```python
dq_max_per_iteration = np.deg2rad(8.0)
dq = np.clip(dq, -dq_max_per_iteration, dq_max_per_iteration)
```

然后回溯：

```python
for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
    q_try = integrate(q, alpha * dq)
    q_try = np.clip(q_try, q_lower, q_upper)

    if normalized_pose_cost(q_try) < current_cost:
        q = q_try
        accepted = True
        break
```

这比固定 `DT=0.1` 更可靠：

* 目标较远时可以快速前进；
* 接近目标时自动减小步长；
* 非线性较强时不会一步越过；
* 零空间更新不会破坏主任务收敛。

---

# 7. 正常求解流程

```python
def solve(target_pose, q_measured):
    q_seed = choose_seed(q_measured, previous_solution)

    result = local_hierarchical_ik(
        target_pose=target_pose,
        q_init=q_seed,
        max_iterations=30,
    )

    if result.meets_tolerance:
        return accept(result, mode="local")

    fallback = trac_ik_distance(
        target_pose=target_pose,
        q_seed=q_measured,
        timeout=0.010,
    )

    if fallback.success:
        polished = local_hierarchical_ik(
            target_pose=target_pose,
            q_init=fallback.q,
            max_iterations=15,
        )
        return accept_best(fallback, polished)

    return hold_last_valid_solution()
```

正常连续遥操作中，目标变化很小，局部求解器会一直停留在同一解支，不应频繁进入 TRAC-IK。

---

# 8. TRAC-IK 只作为失败回退

TRAC-IK 同时运行改进的 Newton/KDL 路线和 SQP 路线，用于改善普通 Jacobian 方法在关节限位附近的失败问题。([MoveIt][3])

回退时使用：

```yaml
solve_type: Distance
timeout: 0.010
epsilon: 1.0e-5
```

`Distance` 会运行完整 timeout，并返回与 seed 关节距离平方和最小的有效解，因此比 `Speed` 更适合你的连续遥操作场景。([MoveIt][3])

不要每帧都从多个随机 seed 启动，否则可能出现：

* 不同冗余解之间切换；
* 肘部突然翻向另一侧；
* 相近目标产生不连续关节解。

推荐回退顺序：

```text
Seed 1：当前实测 q_measured
Seed 2：上一帧有效 q_ref
Seed 3：预设自然姿态 q_nominal
```

只有前一个 seed 失败才尝试下一个。

TRAC-IK 输出后再运行 5～15 次本地精修，使最终误差判据完全由你自己的 FK 和 TCP 定义，而不是直接相信求解器的 success 标记。

---

# 9. 解的验收必须采用字典序，而不是一个总分

候选解不能简单使用：

[
w_pE_p+w_RE_R+w_qE_q
]

因为高权重仍然只是软约束。应按照以下顺序判断：

```python
def candidate_key(candidate):
    return (
        not candidate.pose_valid,       # 有效解永远优先
        candidate.position_error,
        candidate.rotation_error,
        candidate.joint_distance,
        candidate.joint_limit_cost,
    )
```

准确地说：

1. 首先必须满足位置和姿态容差；
2. 有多个有效解时，选择最接近 `q_anchor` 的；
3. 关节距离接近时，再选择远离限位的；
4. 操作度只作为最后的平局项。

这体现了你的要求：

> 精度第一，连续性第二，姿态舒适性第三。

---

# 10. 解决跳变不能再用移动平均

四帧关节移动平均的问题是：平均后的关节角通常不再满足末端目标。

应使用独立的加速度受限关节目标生成器。

设精确 IK 结果为 `q_ref`，上一时刻指令为 `q_cmd`：

```python
error = q_ref - q_cmd
```

根据剩余距离计算防过冲速度：

[
v_{\mathrm{stop},i}
===================

\sqrt{2a_{\max,i}|q_{\mathrm{ref},i}-q_{\mathrm{cmd},i}|}
]

目标速度：

[
v_{\mathrm{target},i}
=====================

\operatorname{sign}(e_i)
\min(v_{\max,i},v_{\mathrm{stop},i})
]

再限制加速度：

```python
dv = np.clip(
    v_target - v_cmd,
    -a_max * dt,
     a_max * dt,
)

v_cmd += dv
q_cmd += v_cmd * dt
```

接近目标时吸附：

```python
finished = (
    np.abs(q_ref - q_cmd) < np.deg2rad(0.03)
) & (
    np.abs(v_cmd) < np.deg2rad(0.1)
)

q_cmd[finished] = q_ref[finished]
v_cmd[finished] = 0.0
```

推荐初始限制：

```python
v_max = np.deg2rad([80, 80, 100, 100, 120, 120, 150])
a_max = np.deg2rad([250, 250, 300, 300, 400, 400, 500])
```

这只是初始调参值，最终应按照电机减速比、控制器能力和负载修正。

关键区别是：

```text
q_ref 始终是精确 IK 解
q_cmd 平滑接近 q_ref
```

而不是让 IK 自己每一帧只前进一点。

---

# 11. Pinocchio 求解核心框架

下面是接近实际实现的结构，省略了模型加载和日志代码：

```python
def solve_local(self, target: pin.SE3, q_init: np.ndarray) -> IKResult:
    q = np.clip(q_init.copy(), self.q_lb, self.q_ub)
    q_anchor = self.q_previous_valid.copy()

    for iteration in range(self.max_iterations):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        current = self.data.oMf[self.ee_frame_id]
        current_to_target = current.actInv(target)

        error6 = pin.log6(current_to_target).vector

        position_error = np.linalg.norm(
            current.translation - target.translation
        )
        rotation_error = np.linalg.norm(
            pin.log3(current.rotation.T @ target.rotation)
        )

        if (
            position_error < self.position_tolerance
            and rotation_error < self.rotation_tolerance
        ):
            return IKResult(
                success=True,
                q=q,
                position_error=position_error,
                rotation_error=rotation_error,
                iterations=iteration,
                mode="local",
            )

        J_frame = pin.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.ee_frame_id,
            pin.LOCAL,
        )

        J_error = (
            -pin.Jlog6(current_to_target.inverse()) @ J_frame
        )

        error_normalized = self.W_task @ error6
        J_normalized = self.W_task @ J_error

        U, singular_values, Vt = np.linalg.svd(
            J_normalized,
            full_matrices=True,
        )
        V = Vt.T

        damping = self.compute_damping(singular_values)

        J_damped_inverse = (
            V[:, :6]
            @ np.diag(
                singular_values
                / (singular_values**2 + damping**2)
            )
            @ U.T
        )

        dq_task = -J_damped_inverse @ error_normalized

        rank = np.count_nonzero(
            singular_values > self.rank_tolerance
        )
        Z = V[:, rank:]
        N = Z @ Z.T

        grad_continuity = self.W_joint @ (q - q_anchor)
        grad_limit = self.joint_limit_gradient(q)

        dq_secondary = -N @ (
            self.k_continuity * grad_continuity
            + self.k_joint_limit * grad_limit
        )

        dq = dq_task + dq_secondary
        dq = np.clip(
            dq,
            -self.max_iteration_step,
             self.max_iteration_step,
        )

        accepted = self.backtracking_update(
            q=q,
            dq=dq,
            target=target,
        )

        if accepted is None:
            break

        q = accepted

    return self.make_failure_result(q, target)
```

Pinocchio 的 frame Jacobian 可以在 `LOCAL`、`LOCAL_WORLD_ALIGNED` 或 `WORLD` 坐标系表达；误差和 Jacobian 必须始终使用一致的参考系。([gepettoweb.laas.fr][4])

实现后必须对 `J_error` 做数值差分检查：

[
J_{\mathrm{numeric}}[:,i]
\approx
\frac{e(q+\epsilon e_i)-e(q-\epsilon e_i)}
{2\epsilon}
]

建议：

```python
epsilon = 1e-7
relative_jacobian_error < 1e-4
```

这一步非常重要，很多“必须高频才能收敛”的 IK 问题，本质上是旋转误差和 Jacobian 不在同一个坐标系。

---

# 12. TCP 必须重新定义

当前 A2D 分支优先将 `left_base_link/right_base_link` 作为末端 frame；不存在时，直接在第七关节处添加零平移的 `L_ee/R_ee`。([GitHub][1])

这不一定是你真正控制的工具中心点。应该在 URDF 或运行时显式添加：

```text
arm_link_7
    └── ee_control_frame
            translation = 实际工具偏移
            rotation    = 实际工具安装方向
```

最终链路应为：

[
{}^{base}T_{tcp}(q)
===================

{}^{base}T_{joint7}(q)
{}^{joint7}T_{tcp}
]

在仿真里 IK 误差已经很小、真机仍然存在固定方向误差时，首先检查：

* TCP 偏移；
* 机械臂基座坐标；
* XR 到机器人坐标变换；
* 关节零位；
* 左右乘变换顺序。

仅仅更换求解器无法修复这些系统误差。

---

# 13. 建议的状态机

```text
TRACKING
  │
  ├─ 本地 IK 成功
  │     └─ 更新 q_ref 和 previous_valid
  │
  ├─ 本地 IK 失败
  │     └─ RECOVERY：TRAC-IK Distance
  │
  ├─ 回退成功
  │     └─ Pinocchio 精修 → 更新 q_ref
  │
  └─ 全部失败
        └─ HOLD：保持上一有效 q_ref
```

不要在求解失败时把未收敛的 debug 解直接发送到机器人。

---

# 14. 验收指标

建议在接入真机前建立自动测试。

### 随机可达目标

从随机关节角正向生成目标：

```python
q_ground_truth -> FK -> target_pose -> IK
```

因为 7DoF 存在冗余，不比较 `q_ik == q_ground_truth`，而比较：

```text
FK(q_ik) 与 target_pose 的误差
```

建议目标：

```text
可达目标成功率：      > 99.5%
位置误差 P99：        < 0.5 mm
姿态误差 P99：        < 0.2°
连续目标最大解支跳变：< 15°
正常局部求解迭代数：  < 10
```

### 连续轨迹

测试：

* 末端直线；
* 空间圆；
* 八字轨迹；
* 固定位置旋转手腕；
* 肘部接近伸直；
* 关节接近上下限；
* 目标突然跳变 5～20 cm。

同时记录：

```text
target pose
FK(q_ref)
FK(q_cmd)
FK(q_measured)
```

这样可以区分：

* IK 数值误差；
* 指令整形滞后；
* 下位机跟踪误差；
* 真机模型误差。

---

## 最终落地顺序

**P0：先修当前结构**

* 真正构建单侧 7DoF reduced model；
* 定义准确 TCP；
* 关闭 `WeightedMovingFilter`；
* 移除主目标中的 `regularization_cost` 和 `smooth_cost`；
* 分离 `q_ref` 与 `q_cmd`。

**P1：实现 Pinocchio 分层 DLS**

* `log6 + Jlog6`；
* 自适应阻尼；
* SVD 零空间；
* 关节限位；
* 回溯线搜索；
* 单次调用内部完全收敛。

**P2：增加 TRAC-IK Distance 回退**

* 只在本地 IK 失败时调用；
* seed 优先使用当前关节角；
* 输出再交给 Pinocchio 精修和 FK 验证。

这套方案比单纯把当前 IPOPT 权重继续调大更符合你的需求：**末端精度不再与平滑性竞争，同时通过解支保持和关节指令整形避免明显跳变。**

[1]: https://github.com/moqingx52/xr_teleoperate/blob/main/teleop/robot_control/robot_arm_ik.py "xr_teleoperate/teleop/robot_control/robot_arm_ik.py at main · moqingx52/xr_teleoperate · GitHub"
[2]: https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/master/doxygen-html/md_doc_b_examples_d_inverse_kinematics.html "pinocchio: Inverse kinematics (clik)"
[3]: https://moveit.picknik.ai/main/doc/how_to_guides/trac_ik/trac_ik_tutorial.html "TRAC-IK Kinematics Solver — MoveIt Documentation: Rolling  documentation"
[4]: https://gepettoweb.laas.fr/doc/stack-of-tasks/pinocchio/jnrh2023/template/frame.html "Frame — pinocchio 2.9.2 documentation"
