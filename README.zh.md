> 🌐 [English](README.md) · **中文**

# worldsense

**[Chronotope](https://github.com/wikieden/tempomem)** 的感知适配器 ——
将 RGB(-D) 帧转换为世界坐标系下的 `Detection`，供记忆核心摄入。

> **为什么单独建一个仓库？** Chronotope 核心是一个*记忆*层（类似 Mem0 之于文本），
> 保持纯 numpy 依赖。识别不是它的职责。所有具体的感知能力——开放词汇检测器、
> 相机→世界坐标的几何提升、图像裁剪特征编码——都放在这里，位于重型 GPU 依赖之后，
> 从而保持核心轻量。

## 包含内容

| 模块 | 状态 |
|---|---|
| `geometry` — 相机坐标系下的有向 3D 包围盒 → 世界坐标系 AABB 提升（纯 numpy） | ✅ |
| `ImageEncoder` 协议 + `OpenClipImageEncoder`（文本 + 图像双塔） | ✅ (`[clip]`) |
| `Cosmos3PerceptionAdapter` — Cosmos 3 Reasoner 3D 包围盒 + 自车位姿 → `Detection` | ⬜ 设计阶段（需要 GPU）— 见 [docs/cosmos3-perception-adapter.md](docs/zh/cosmos3-perception-adapter.md) |
| `ConceptGraphsAdapter`（SAM + Grounding DINO + OpenCLIP） | ⬜ 规划中 |

所有适配器均实现 `tempomem.PerceptionAdapter` 协议，因此：

```python
mem.add_frame(rgb, depth, pose, adapter=my_adapter)  # adapter -> Detections -> fusion
```

## 安装

```bash
pip install worldsense          # 核心（numpy）
pip install worldsense[clip]    # + open-clip / torch / pillow
```

## 与核心的关系

`worldsense` **依赖** `tempomem`。核心拥有接缝
（`PerceptionAdapter`）、记忆和查询；本仓库负责填充"眼睛"。完整的系统设计——
Cosmos 3 + Chronotope + LLM "大脑"——请参阅核心仓库中的
`docs/design/cosmos3-tempomem-llm-brain.md`。

## 许可证

Apache-2.0 — 见 [LICENSE](LICENSE)。
