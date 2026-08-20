# Lace Studio Refine v0.1 for ComfyUI

用于蕾丝扫描图确定性精修的 ComfyUI 自定义节点。节点保持原始花型和网孔拓扑，处理扫描背景、位置偏移、重复结构和外轮廓，并同时输出精修图、软蒙版和 JSON 执行报告。

默认流程不调用 FLUX 或其他生成模型。

## 节点

在 ComfyUI 中搜索：`Lace Studio 精修 v0.1`

分类：`Lace Studio / Refine v0.1`

输入：

- `image`：ComfyUI `IMAGE`；
- `background`：`auto`、`white`、`black`；
- `repeat_reconstruction`：`auto`、`off`；
- `edge_cleanup`：`0–100`，默认 `50`。

输出：

- `refined_image`：保持原分辨率的精修图；
- `foreground_mask`：前景软蒙版；
- `report_json`：检测、几何和输出设置报告。

## 通过 ComfyUI Manager 安装

在 Manager 中搜索：

```text
Lace Studio Refine
```

Registry 节点 ID：

```text
lace-studio-refine
```

通过 Registry 安装不需要开启旧版 Manager 的 `Install via Git URL` 高风险功能。

### `0.1.2` 依赖修复

`0.1.2` 修复 NumPy 2 与旧 OpenCV 轮子混用时的
`numpy.core.multiarray failed to import` 故障，并在运行时导入失败后清理半初始化的模块缓存。

从 `0.1.1` 更新时，请在 Manager 完成节点和依赖更新后完整重启 ComfyUI。
如果 Manager 显示依赖安装失败，不要继续运行工作流；应先保留安装日志。

## 通过 Git 安装

GitHub 仓库为公开仓库，可以直接克隆。旧版 Manager 只有在界面仍提供 `Install via Git URL` 且安全等级允许高风险操作时，才能使用该入口。

## Windows Portable 手动安装

在 `ComfyUI_windows_portable` 根目录打开 PowerShell：

```powershell
git clone https://github.com/lin-sein/lace-studio-refine-comfyui.git ComfyUI/custom_nodes/lace-studio-refine-comfyui
./python_embeded/python.exe -m pip install -r ComfyUI/custom_nodes/lace-studio-refine-comfyui/requirements.txt
```

然后完全关闭并重新启动 ComfyUI。

## Linux / Python venv 手动安装

```bash
cd /path/to/ComfyUI/custom_nodes
git clone https://github.com/lin-sein/lace-studio-refine-comfyui.git
cd ..
source venv/bin/activate
python -m pip install -r custom_nodes/lace-studio-refine-comfyui/requirements.txt
```

重启 ComfyUI 后验证：

```text
GET /object_info/LaceStudioRefineV01
```

正常响应应包含 `LaceStudioRefineV01`、`background`、`repeat_reconstruction` 和 `edge_cleanup`。

## 工作流

- `workflows/lace-studio-refine-v01-ui.json`：拖入 ComfyUI 页面直接使用；
- `workflows/lace-studio-refine-v01-api.json`：用于 `/prompt` API 调用。

UI 工作流默认是：

```text
LoadImage → LaceStudioRefineV01 → SaveImage
                              ↘ MaskToImage → SaveImage
```

## 更新

```powershell
Set-Location ComfyUI/custom_nodes/lace-studio-refine-comfyui
git pull --ff-only
```

依赖文件发生变化时重新执行 `pip install -r requirements.txt`，然后重启 ComfyUI。

`0.1.2` 使用已验证的 NumPy 2 / OpenCV 4.13 ABI 组合。请不要在同一 Python
环境中手工混装多个 `opencv-python*` 发行包；如果导入仍失败，新的错误信息会列出
实际检测到的 NumPy 和 OpenCV 发行包版本。

## v0.1 边界

- 固定保持输入尺寸；
- 推荐输出 PNG；
- 软蒙版可用于后续透明 PNG 或遮罩内 FLUX 修复；
- FLUX 不属于 v0.1 默认链路；
- 仅验证过本地 ComfyUI 节点契约，部署后应先用一张测试图验证，再接入批量任务。
