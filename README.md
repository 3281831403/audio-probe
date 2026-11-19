# audio-probe

## 题面

素材库接入自动转码后，时长和声道数经常读错：①只认 44 字节 WAV 头，带 LIST 块的文件解析失败；②MP3 不跳 ID3v2，同步字搜索从头开始，误判帧头；③帧时长按固定 26.12 ms，MPEG2 层 III 算错；④VBR 文件按首帧比特率估算，时长偏差 10%；⑤大端/小端混用，AIFF 的采样率读成 0；⑥声道掩码硬编码 stereo，5.1 文件报 2 声道；⑦编码延迟忽略，时长比实际短 1105 个采样；⑧未知魔数直接抛异常，调用方拿不到 `"unknown"`；⑨损坏文件只跳过，批量结果里缺条目；⑩浮点采样率四舍五入，44100.5 变成 44100。probe/container.py 是核心，tests/test_container.py 15个红。要修的是：按 `probe/container.py` 的 docstring 实现 `detect_format`、`parse_wav`、`parse_mp3`、`frame_duration_ms`、`estimate_duration`、`channel_mask`、`encoder_delay`、`probe_batch`，修掉上面十个缺陷并让 15 个测试全部通过。

## 运行方式

在仓库根目录执行：

```powershell
python -m pytest tests -q
```

初始桩实现应恰好得到 `15 failed`，且没有 collection error、import error 或 syntax error。实现完成后应得到 `15 passed`。

## 字段表

| 字段 | 取值 |
| --- | --- |
| 所属领域 | 游戏、图形与多媒体 |
| 子方向 | 音视频处理 |
| 任务类型 | Bugfix |
| 语言/框架 | Python 3, pytest |
| 环境可复现等级 | 无外部依赖 |

## tests 不可修改

`tests/test_container.py` 不可修改；只能实现/修复 `probe/container.py`。全部测试只使用内联构造的 bytes，离线运行，不使用随机数，也不依赖外部素材。实现仅使用 Python 标准库（struct、math、codecs）与 pytest，不引入外部依赖。
