> 🌐 [English](../en/cosmos3-perception-adapter.md) · **中文**

# 设计 — `Cosmos3PerceptionAdapter`

**状态：** 草稿 / GPU 待验证。**仓库：** `worldsense`（`tempomem` 记忆核心的配套模块）。**依赖：** 一台 CUDA 机器（Cosmos 3 Nano 16B / Super 64B）或 Cosmos NIM 端点。

## 目标

使 `tempomem.TempoMem.add_frame(rgb, depth, pose)` 能够以 **NVIDIA Cosmos 3** 作为感知后端。该适配器将一帧带位姿的 RGB 图像转换为世界坐标系下的 `list[tempomem.Detection]`，从而使核心的融合 → 记忆 → 查询流程保持不变。核心库仅提供 `PerceptionAdapter` 协议；本适配器（以及其所使用的几何提升模块和图像编码器）位于此仓库中。

## 为何切实可行（2026-06 调研结论）

Cosmos 3 技术报告指出：Reasoner 输出**结构化的相机坐标系 3D 包围盒**——每个物体以 `{label, center, dimensions, orientation}` 形式呈现于统一的 **JSON** 格式中——且 `Cosmos3-Nano` 能够**估计度量尺度的自身位姿**。因此 Cosmos 提供了 `Detection` 约 70% 的内容（标签 + 度量 3D 包围盒，相机坐标系）。适配器只需完成：(1) 将相机坐标系提升至世界坐标系，(2) 为每个物体添加特征向量（Cosmos 不输出该项），(3) 健壮地解析 JSON。

## 接口约定

```python
# tempomem (core) — the seam
class PerceptionAdapter(Protocol):
    def process_frame(self, rgb, depth, pose, intrinsics=None) -> list[Detection]: ...

# this repo provides the pieces:
from worldsense.geometry import world_aabb_from_obb, transform_point
from worldsense.encoders import OpenClipImageEncoder  # ImageEncoder
```

## 处理流程（`process_frame`）

```
rgb ──► Cosmos 3 Reasoner (NIM or local Cosmos3OmniPipeline)
          prompt: "detect objects; per object JSON
                   {label, center[x,y,z] m (camera), size[l,w,h] m,
                    orient[roll,pitch,yaw] rad, box2d[x1,y1,x2,y2]}"
          ──► parse JSON (robust)
   pose (arg, preferred) or Cosmos ego-pose ──► cam→world 4×4
   per object:
     center_world = transform_point(cam2world, center_cam)
     bbox_min/max = world_aabb_from_obb(cam2world, center, size, rpy)
     feature      = encoder.encode_image(crop(rgb, box2d))   # OpenClipImageEncoder
     Detection(label, feature, center_world, bbox_min, bbox_max, confidence, ts,
               aux={"source":"cosmos3", "orient": rpy, "box2d": (...)})
```

纯 numpy 几何模块（`worldsense.geometry`）可在 CPU 上配合伪 reasoner 进行测试；仅 Cosmos 调用部分需要 GPU。

## 接口草图

```python
class Cosmos3PerceptionAdapter:  # implements tempomem.PerceptionAdapter
    def __init__(self, *, reasoner, encoder, pose_source="arg", confidence=0.8,
                 grounding_prompt=_DEFAULT, transport=None): ...
    def process_frame(self, rgb, depth, pose, intrinsics=None) -> list[Detection]:
        boxes = self._detect(rgb)                       # Cosmos -> [_Box3D]
        c2w = pose if self.pose_source == "arg" else self._estimate_pose(rgb)
        return [self._to_detection(b, c2w, rgb) for b in boxes]
```

打包方式：置于 `[cosmos]` extra 之后；默认指向托管的 NIM，无需本地权重文件。

## 待解决问题（在真实权重上验证后再实现）

1. **JSON 包围盒 schema** —— 精确字段名、2D vs 3D、度量值 vs 归一化值 —— 尚未文档化。
2. **逐包围盒置信度** —— 尚未文档化；使用配置先验值。
3. **延迟** —— System-2（秒级/片段）：对关键帧采样，不要逐帧调用。
4. **相机坐标系度量尺度** —— 确认单位为米，而非归一化值。
5. **位姿来源** —— 除非经过基准测试，否则优先信任调用方的 SLAM/ARKit 位姿而非 Cosmos 估计值；为（未文档化的）相机坐标约定提供 `axis_remap` 配置项。
6. **`aux` 持久化** —— 核心存储目前尚无 `aux` 列，因此 orient/box2d 在持久化时会被丢弃；需要对核心 schema 进行版本升级才能在会话间保留。

## 离线测试方案（无 GPU）

- 注入伪 `reasoner`/`transport`，返回预置的 JSON 包围盒。
- 断言：解析 → 世界坐标系变换（已知位姿）→ `Detection` 具有正确的 `center_xyz` / AABB；特征 L2 范数；非法 JSON → `tempomem.AdapterError`。
- 几何辅助函数已有单元测试（`tests/test_geometry.py`）。

## 本仓库中的备选方案

- **`ConceptGraphsAdapter`**（SAM + Grounding DINO + OpenCLIP）—— 开放词汇 RGB-D 感知，使用相同接口。
- **Isaac nvblox + cuVSLAM** 馈送器 —— 度量 3D + 6-DoF 位姿（CUDA）；附加标签与特征。
