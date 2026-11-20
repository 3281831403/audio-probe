"""音频容器探测与基础解析。

本模块是练习的桩实现：所有公开函数都只抛出 ``NotImplementedError``。
字符串字段、字节偏移、大小端和返回键名由各函数 docstring 固定，
实现时不要改变这些契约。
"""

from __future__ import annotations

import codecs
import math
import struct


class ProbeError(Exception):
    """容器损坏、字段非法或声道布局未知时抛出的异常。"""


# ---------------------------------------------------------------------------
# MPEG Audio 常量表
# ---------------------------------------------------------------------------

# 比特率表（kbps），按 (是否 MPEG1, layer) 索引；layer 取值 1/2/3。
_BITRATE_MPEG1 = {
    1: [0, 32, 64, 96, 128, 160, 192, 224, 256, 288, 320, 352, 384, 416, 448],
    2: [0, 32, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384],
    3: [0, 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320],
}
_BITRATE_MPEG2 = {
    1: [0, 32, 48, 56, 64, 80, 96, 112, 128, 144, 160, 176, 192, 224, 256],
    2: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
    3: [0, 8, 16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128, 144, 160],
}

_SAMPLE_RATES = {
    3: [44100, 48000, 32000],  # MPEG1
    2: [22050, 24000, 16000],  # MPEG2
    0: [11025, 12000, 8000],   # MPEG2.5
}

_VERSION_NAMES = {3: "MPEG1", 2: "MPEG2", 0: "MPEG2.5"}


def _parse_frame_header(data: bytes, i: int) -> dict:
    """解析偏移 i 处的 4 字节 MPEG 帧头，返回位域字典。"""
    if i + 4 > len(data):
        raise ProbeError("帧头越界")
    b1, b2, b3 = data[i + 1], data[i + 2], data[i + 3]

    version_bits = (b1 >> 3) & 0x03
    if version_bits == 1:
        raise ProbeError("保留的 MPEG 版本")
    layer_bits = (b1 >> 1) & 0x03
    if layer_bits == 0:
        raise ProbeError("保留的层")
    layer = 4 - layer_bits  # 3->Layer I, 2->Layer II, 1->Layer III

    bitrate_index = (b2 >> 4) & 0x0F
    if bitrate_index == 0 or bitrate_index == 0x0F:
        raise ProbeError("非法比特率索引")
    sample_rate_index = (b2 >> 2) & 0x03
    if sample_rate_index == 3:
        raise ProbeError("非法采样率索引")
    padding = (b2 >> 1) & 0x01
    channel_mode = (b3 >> 6) & 0x03

    is_mpeg1 = version_bits == 3
    table = _BITRATE_MPEG1 if is_mpeg1 else _BITRATE_MPEG2
    bitrate_kbps = table[layer][bitrate_index]
    sample_rate = _SAMPLE_RATES[version_bits][sample_rate_index]

    if layer == 1:
        samples_per_frame = 384
    elif layer == 2:
        samples_per_frame = 1152
    else:
        samples_per_frame = 1152 if is_mpeg1 else 576

    if layer == 1:
        frame_size = (12 * bitrate_kbps * 1000 // sample_rate + padding) * 4
    elif layer == 3 and not is_mpeg1:
        frame_size = 72 * bitrate_kbps * 1000 // sample_rate + padding
    else:
        frame_size = 144 * bitrate_kbps * 1000 // sample_rate + padding

    return {
        "version": _VERSION_NAMES[version_bits],
        "version_bits": version_bits,
        "is_mpeg1": is_mpeg1,
        "layer": layer,
        "bitrate_kbps": bitrate_kbps,
        "sample_rate": sample_rate,
        "padding": padding,
        "channel_mode": channel_mode,
        "channels": 1 if channel_mode == 3 else 2,
        "samples_per_frame": samples_per_frame,
        "frame_size": frame_size,
    }


def detect_format(data: bytes) -> str:
    """按魔数识别容器格式，不解析容器内容。

    参数:
        data: 原始字节；允许为空或长度小于 4，不得因输入过短抛异常。

    识别规则（偏移从 0 开始）:
        - WAV: data[0:4] == b"RIFF"（也接受大端容器 b"RIFX"），
          且 data[8:12] == b"WAVE"。
        - FLAC: data[0:4] == b"fLaC"。
        - OGG: data[0:4] == b"OggS"。
        - MP3: data[0:3] == b"ID3"（ID3v2 标签），或者
          data[0] == 0xFF 且 (data[1] & 0xE0) == 0xE0（MPEG Audio 同步字）。
        - 其他或过短: "unknown"。

    返回:
        "wav"、"mp3"、"flac"、"ogg" 或 "unknown"。
        未知魔数必须返回 "unknown"，不能抛异常。
    """
    if len(data) >= 12 and data[0:4] in (b"RIFF", b"RIFX") and data[8:12] == b"WAVE":
        return "wav"
    if data[0:4] == b"fLaC":
        return "flac"
    if data[0:4] == b"OggS":
        return "ogg"
    if data[0:3] == b"ID3":
        return "mp3"
    if len(data) >= 2 and data[0] == 0xFF and (data[1] & 0xE0) == 0xE0:
        return "mp3"
    return "unknown"


def _decode_extended80(buf: bytes) -> float:
    """解码 80 位 IEEE extended（大端）为 float，不做取整。"""
    se, mantissa = struct.unpack(">HQ", buf)
    sign = -1.0 if (se & 0x8000) else 1.0
    exponent = se & 0x7FFF
    if exponent == 0 and mantissa == 0:
        return 0.0
    # 尾数字段只含小数部分，整数位恒为 1，需要补回。
    return sign * (1.0 + mantissa / float(1 << 63)) * (2.0 ** (exponent - 16383))


def _parse_wave(data: bytes) -> dict:
    """解析 RIFF/RIFX + WAVE 容器。"""
    big_endian = data[0:4] == b"RIFX"
    bo = ">" if big_endian else "<"
    if len(data) < 12 or data[8:12] != b"WAVE":
        raise ProbeError("不是有效的 WAVE 容器")

    fmt = None
    data_offset = None
    data_size = None
    pos = 12
    while pos + 8 <= len(data):
        tag = data[pos:pos + 4]
        (size,) = struct.unpack_from(bo + "I", data, pos + 4)
        payload_start = pos + 8
        if payload_start + size > len(data):
            raise ProbeError("块越界: %r" % tag)
        if tag == b"fmt ":
            fmt = data[payload_start:payload_start + size]
        elif tag == b"data":
            data_offset = payload_start
            data_size = size
        pos = payload_start + size + (size & 1)

    if fmt is None:
        raise ProbeError("缺少 fmt 块")
    if data_offset is None:
        raise ProbeError("缺少 data 块")
    if len(fmt) < 16:
        raise ProbeError("fmt 块过短")

    (audio_format, channels, sample_rate, _byte_rate,
     block_align, bits_per_sample) = struct.unpack_from(bo + "HHIIHH", fmt, 0)

    mask = None
    if audio_format == 0xFFFE and len(fmt) >= 18:
        (cb_size,) = struct.unpack_from(bo + "H", fmt, 16)
        if cb_size >= 22 and len(fmt) >= 40:
            (mask,) = struct.unpack_from(bo + "I", fmt, 20)

    if block_align <= 0:
        raise ProbeError("block_align 非法")
    frame_count = data_size // block_align

    if mask is None:
        if channels == 1:
            mask = 0x4
        elif channels == 2:
            mask = 0x3
        else:
            mask = 0x0

    return {
        "format": "wav",
        "audio_format": audio_format,
        "channels": channels,
        "sample_rate": sample_rate,
        "bits_per_sample": bits_per_sample,
        "block_align": block_align,
        "frame_count": frame_count,
        "data_offset": data_offset,
        "data_size": data_size,
        "duration_samples": frame_count,
        "channel_mask": mask,
    }


def _parse_aiff(data: bytes) -> dict:
    """解析 FORM + AIFF/AIFC 容器（大端）。"""
    form_type = data[8:12]
    if form_type not in (b"AIFF", b"AIFC"):
        raise ProbeError("不是有效的 AIFF/AIFC 容器")

    comm = None
    ssnd_offset = None
    ssnd_size = None
    pos = 12
    while pos + 8 <= len(data):
        tag = data[pos:pos + 4]
        (size,) = struct.unpack_from(">I", data, pos + 4)
        payload_start = pos + 8
        if payload_start + size > len(data):
            raise ProbeError("块越界: %r" % tag)
        if tag == b"COMM":
            comm = data[payload_start:payload_start + size]
        elif tag == b"SSND":
            if size < 8:
                raise ProbeError("SSND 块过短")
            (offset,) = struct.unpack_from(">I", data, payload_start)
            ssnd_offset = payload_start + 8 + offset
            ssnd_size = size - 8 - offset
            if ssnd_offset > len(data):
                raise ProbeError("SSND 偏移越界")
        pos = payload_start + size + (size & 1)

    if comm is None:
        raise ProbeError("缺少 COMM 块")
    if ssnd_offset is None:
        raise ProbeError("缺少 SSND 块")
    if len(comm) < 18:
        raise ProbeError("COMM 块过短")

    channels, frame_count, bits_per_sample = struct.unpack_from(">hIh", comm, 0)
    sample_rate = _decode_extended80(comm[8:18])

    if form_type == b"AIFF":
        audio_format = 1
    else:
        if len(comm) < 22:
            raise ProbeError("AIFC 缺少压缩类型")
        (audio_format,) = struct.unpack_from(">I", comm, 18)

    block_align = channels * ((bits_per_sample + 7) // 8)
    data_size = min(ssnd_size, frame_count * block_align)

    if channels == 1:
        mask = 0x4
    elif channels == 2:
        mask = 0x3
    else:
        mask = 0x0

    return {
        "format": "aiff" if form_type == b"AIFF" else "aifc",
        "audio_format": audio_format,
        "channels": channels,
        "sample_rate": sample_rate,
        "bits_per_sample": bits_per_sample,
        "block_align": block_align,
        "frame_count": frame_count,
        "data_offset": ssnd_offset,
        "data_size": data_size,
        "duration_samples": frame_count,
        "channel_mask": mask,
    }


def parse_wav(data: bytes) -> dict:
    """解析 WAV（RIFF/WAVE）或 AIFF/AIFC 容器。

    WAV（小端，偏移均从 data 起点计）:
        - 0..3 为 b"RIFF"（也接受 b"RIFX"；此时所有整数字段按大端读取），
          8..11 为 b"WAVE"。
        - 从偏移 12 开始遍历块：4 字节 tag；偏移 +4 的 u32 size；
          偏移 +8 的负载。负载长度按偶数字节对齐，奇数长度补 1 字节。
        - "fmt " 负载：wFormatTag u16@0、channels u16@2、
          sample_rate u32@4、byte_rate u32@8、block_align u16@12、
          bits_per_sample u16@14。
        - wFormatTag == 0xFFFE 且 cbSize u16@16 >= 22 时，扩展字段为：
          valid_bits u16@18、channel_mask u32@20、subformat GUID 前两个
          字节 u16@24。audio_format 仍报告原始 wFormatTag（0xFFFE）。
        - "data" 负载起点为 data_offset，声明长度为 data_size；
          frame_count = data_size // block_align。
        - 必须接受 16、18、40 字节等非 44 字节头，并忽略 LIST 等未知块。

    AIFF/AIFC（大端，函数名沿用历史命名）:
        - 0..3 为 b"FORM"，8..11 为 b"AIFF" 或 b"AIFC"；块长度为 u32 大端。
        - "COMM": channels i16@0、frame_count u32@2、bits_per_sample i16@6、
          sample_rate 为 80 位 IEEE extended（10 字节，大端，偏移 8..17）。
        - "SSND": offset u32@0；音频起点 = SSND 负载起点 + 8 + offset。
        - 采样率不得取整：44100.5 必须保留为 44100.5。

    返回 dict:
        format: "wav"、"aiff" 或 "aifc"。
        audio_format: WAV 的原始 wFormatTag；AIFF 为 1，AIFC 为压缩类型
            四字符码的整数。
        channels, sample_rate, bits_per_sample, block_align, frame_count,
        data_offset, data_size。
        duration_samples: 每声道采样点（即 frame_count）。
        channel_mask: WAVE 声道掩码；无扩展字段时按声道数推断
            （mono 0x4，stereo 0x3）。

    异常:
        数据为空、魔数不符、必需块缺失或块越界时抛 ProbeError。
    """
    if not data:
        raise ProbeError("数据为空")
    if len(data) < 12:
        raise ProbeError("数据过短")
    if data[0:4] in (b"RIFF", b"RIFX"):
        return _parse_wave(data)
    if data[0:4] == b"FORM":
        return _parse_aiff(data)
    raise ProbeError("无法识别的容器魔数")


def parse_mp3(data: bytes) -> dict:
    """跳过 ID3v2 并解析第一帧 MPEG Audio 帧头。

    ID3v2（仅当 data[0:3] == b"ID3"）:
        - 头长 10 字节；size 为 data[6:10] 的 syncsafe 整数：
          ((b0 & 0x7F) << 21) | ((b1 & 0x7F) << 14) |
          ((b2 & 0x7F) << 7) | (b3 & 0x7F)。
        - 帧起点 = 10 + size；若 flags 的 footer 位有效再加 10 字节。
        - 同步字搜索必须从帧起点之后开始，不能把 ID3 标签里的 0xFF
          误判为帧头。

    MPEG 帧头（4 字节，按大端位域读取）:
        - data[i] == 0xFF 且 (data[i+1] & 0xE0) == 0xE0。
        - version = (data[i+1] >> 3) & 0x03：
          0=MPEG2.5，2=MPEG2，3=MPEG1，1=保留。
        - layer = (data[i+1] >> 1) & 0x03：
          1=Layer III，2=Layer II，3=Layer I，0=保留。
        - bitrate_index = (data[i+2] >> 4) & 0x0F。
        - sample_rate_index = (data[i+2] >> 2) & 0x03。
        - padding = (data[i+2] >> 1) & 0x01。
        - channel_mode = (data[i+3] >> 6) & 0x03；3=单声道，其余双声道。
        - 比特率、采样率按 MPEG 版本和层查标准表；保留索引用 ProbeError。

    Xing/Info:
        - 偏移 = frame_offset + 4 + side_info_size。
        - side_info_size：MPEG1 双声道 32、单声道 17；
          MPEG2/2.5 双声道 17、单声道 9。
        - 负载以 b"Xing" 或 b"Info" 开头；flags 为 u32 大端@+4；
          flags & 1 时 frame_count 为 u32 大端@+8。
        - "Xing" 通常为 VBR，"Info" 通常为 CBR。

    返回 dict:
        format="mp3"、version、layer、bitrate_kbps、sample_rate、channels、
        samples_per_frame、frame_size、frame_offset、channel_mode、
        is_vbr、frame_count（无 Xing/Info 时为 None）、
        encoder_delay、encoder_padding（无 Gapless 信息时为 0）。

    异常:
        数据为空、找不到有效帧或出现保留字段时抛 ProbeError。
    """
    if not data:
        raise ProbeError("数据为空")

    start = 0
    if data[0:3] == b"ID3":
        if len(data) < 10:
            raise ProbeError("ID3v2 头不完整")
        size = ((data[6] & 0x7F) << 21) | ((data[7] & 0x7F) << 14) \
            | ((data[8] & 0x7F) << 7) | (data[9] & 0x7F)
        start = 10 + size
        if data[5] & 0x10:  # footer 位
            start += 10
        if start > len(data):
            raise ProbeError("ID3v2 标签越界")

    frame_offset = None
    for i in range(start, len(data) - 1):
        if data[i] == 0xFF and (data[i + 1] & 0xE0) == 0xE0:
            frame_offset = i
            break
    if frame_offset is None:
        raise ProbeError("找不到有效的 MPEG 帧头")

    hdr = _parse_frame_header(data, frame_offset)

    # Xing/Info 头
    is_vbr = False
    frame_count = None
    if hdr["is_mpeg1"]:
        side_info = 17 if hdr["channels"] == 1 else 32
    else:
        side_info = 9 if hdr["channels"] == 1 else 17
    xoff = frame_offset + 4 + side_info
    if data[xoff:xoff + 4] in (b"Xing", b"Info"):
        is_vbr = data[xoff:xoff + 4] == b"Xing"
        if xoff + 8 <= len(data):
            (flags,) = struct.unpack_from(">I", data, xoff + 4)
            if flags & 1 and xoff + 12 <= len(data):
                (frame_count,) = struct.unpack_from(">I", data, xoff + 8)

    return {
        "format": "mp3",
        "version": hdr["version"],
        "layer": hdr["layer"],
        "bitrate_kbps": hdr["bitrate_kbps"],
        "sample_rate": hdr["sample_rate"],
        "channels": hdr["channels"],
        "samples_per_frame": hdr["samples_per_frame"],
        "frame_size": hdr["frame_size"],
        "frame_offset": frame_offset,
        "channel_mode": hdr["channel_mode"],
        "is_vbr": is_vbr,
        "frame_count": frame_count,
        "encoder_delay": 0,
        "encoder_padding": 0,
    }


def frame_duration_ms(header: bytes) -> float:
    """按 MPEG 版本与层计算单帧时长，返回毫秒（float）。

    参数:
        header: 至少 4 字节的 MPEG Audio 帧头；不要求后面跟着整帧数据。
        位域与 parse_mp3 相同，所有整数按大端读取。

    每帧采样点数:
        - MPEG1: Layer I=384，Layer II=1152，Layer III=1152。
        - MPEG2/2.5: Layer I=384，Layer II=1152，Layer III=576。

    公式:
        duration_ms = samples_per_frame * 1000.0 / sample_rate

    不得固定写成 26.12 ms；例如 MPEG2 Layer III 在 16000 Hz 时是 36.0 ms，
    MPEG1 Layer III 在 44100 Hz 时约为 26.122449 ms。测试容差为 1 个采样点，
    即 1000.0 / sample_rate 毫秒。

    异常:
        header 长度不足 4、保留版本/层或采样率索引非法时抛 ProbeError。
    """
    if len(header) < 4:
        raise ProbeError("帧头长度不足 4 字节")
    if header[0] != 0xFF or (header[1] & 0xE0) != 0xE0:
        raise ProbeError("无效的 MPEG 同步字")
    hdr = _parse_frame_header(header, 0)
    return hdr["samples_per_frame"] * 1000.0 / hdr["sample_rate"]


def estimate_duration(data: bytes, info: dict) -> int:
    """估算 MP3 时长，返回整数采样点（每声道帧数）。

    info 来自 parse_mp3，至少包含 frame_offset、samples_per_frame、
    sample_rate、bitrate_kbps、frame_size；VBR 时包含 is_vbr 与 frame_count。

    规则:
        - info["is_vbr"] 为真且 frame_count 非 None：
          duration = frame_count * samples_per_frame。
        - 否则按 CBR：音频字节数 = max(0, len(data) - frame_offset)。
          优先用帧大小：frame_count = audio_bytes // frame_size，
          duration = frame_count * samples_per_frame。
          没有 frame_size 时用比特率：
          duration = round(audio_bytes * 8 * sample_rate /
                           (bitrate_kbps * 1000))。
        - 返回值与真实时长的误差不得超过 1 帧（samples_per_frame 个采样点）。
        - 本函数不自行加入或扣除 Gapless 延迟；延迟由 encoder_delay(info)
          单独取得。

    异常:
        关键字段缺失、为负或非法时抛 ProbeError。
    """
    try:
        frame_offset = info["frame_offset"]
        samples_per_frame = info["samples_per_frame"]
        sample_rate = info["sample_rate"]
        bitrate_kbps = info["bitrate_kbps"]
    except KeyError as exc:
        raise ProbeError("info 缺少关键字段: %s" % exc) from exc

    for name, value in (("frame_offset", frame_offset),
                        ("samples_per_frame", samples_per_frame),
                        ("sample_rate", sample_rate),
                        ("bitrate_kbps", bitrate_kbps)):
        if not isinstance(value, (int, float)) or value < 0:
            raise ProbeError("字段非法: %s" % name)
    if samples_per_frame <= 0 or sample_rate <= 0:
        raise ProbeError("samples_per_frame / sample_rate 必须为正")

    if info.get("is_vbr") and info.get("frame_count") is not None:
        frame_count = info["frame_count"]
        if not isinstance(frame_count, int) or frame_count < 0:
            raise ProbeError("frame_count 非法")
        return frame_count * samples_per_frame

    audio_bytes = max(0, len(data) - frame_offset)
    frame_size = info.get("frame_size")
    if frame_size:
        if not isinstance(frame_size, int) or frame_size <= 0:
            raise ProbeError("frame_size 非法")
        return (audio_bytes // frame_size) * samples_per_frame
    if bitrate_kbps <= 0:
        raise ProbeError("bitrate_kbps 必须为正")
    return round(audio_bytes * 8 * sample_rate / (bitrate_kbps * 1000))


_LAYOUTS = {
    "mono": (1, 0x4),
    "stereo": (2, 0x3),
    "2.1": (3, 0xB),
    "3.0": (3, 0x7),
    "4.0": (4, 0x33),
    "quad": (4, 0x33),
    "5.0": (5, 0x37),
    "5.1": (6, 0x3F),
    "7.1": (8, 0x63F),
}


def channel_mask(info: dict) -> int:
    """按 info["layout"] 返回 WAVE dwChannelMask。

    layout 大小写不敏感，支持：
        "mono" -> 0x4
        "stereo" -> 0x3
        "2.1" -> 0xB
        "3.0" -> 0x7
        "4.0" / "quad" -> 0x33
        "5.0" -> 0x37
        "5.1" -> 0x3F
        "7.1" -> 0x63F
    位定义：FL=0x1、FR=0x2、FC=0x4、LFE=0x8、BL=0x10、BR=0x20、
    SL=0x200、SR=0x400。

    未知布局必须抛 ProbeError，禁止对所有输入硬编码为 stereo（0x3）。

    异常:
        layout 缺失、无法识别或与 info["channels"] 明显冲突时抛 ProbeError。
    """
    layout = info.get("layout")
    if not isinstance(layout, str):
        raise ProbeError("layout 缺失或不是字符串")
    entry = _LAYOUTS.get(layout.lower())
    if entry is None:
        raise ProbeError("未知声道布局: %r" % layout)
    expected_channels, mask = entry
    channels = info.get("channels")
    if channels is not None and channels != expected_channels:
        raise ProbeError(
            "布局 %r 与声道数 %r 冲突" % (layout, channels))
    return mask


def encoder_delay(info: dict) -> int:
    """返回 Gapless 编码延迟（采样点，每声道），int。

    info["encoder_delay"] 可为 int；缺失时为 0。
    encoder_padding 是尾部填充，单独保留，不从延迟中扣除。
    已有的 1105 等非零延迟不得被忽略或清零。

    异常:
        延迟为负数或不是整数时抛 ProbeError。
    """
    delay = info.get("encoder_delay", 0)
    if not isinstance(delay, int) or isinstance(delay, bool):
        raise ProbeError("encoder_delay 不是整数")
    if delay < 0:
        raise ProbeError("encoder_delay 为负数")
    return delay


def probe_batch(blobs) -> dict:
    """批量探测，返回 {索引: 单文件解析结果}，索引从 0 开始。

    每个 blob 的处理:
        - WAV/AIFF -> parse_wav；MP3 -> parse_mp3。
        - FLAC/OGG -> {"format": "flac"} 或 {"format": "ogg"}（仅容器级）。
        - 其他格式或损坏数据 -> ProbeError。

    原子性:
        先在局部 dict 中构建全部结果；任一文件损坏或未知时抛 ProbeError，
        不返回部分结果，也不能静默跳过坏文件。
        成功时 batch[i] 必须与该文件单独解析的结果一致；空 blobs 返回 {}。

    异常:
        任一 blob 无法解析、blobs 不是可迭代对象或含非 bytes 项时抛
        ProbeError。
    """
    try:
        iterator = iter(blobs)
    except TypeError as exc:
        raise ProbeError("blobs 不是可迭代对象") from exc

    results = {}
    for index, blob in enumerate(iterator):
        if not isinstance(blob, (bytes, bytearray)):
            raise ProbeError("blob %d 不是 bytes" % index)
        blob = bytes(blob)
        # AIFF/AIFC 不在 detect_format 的魔数集合里，单独识别。
        if len(blob) >= 12 and blob[0:4] == b"FORM" \
                and blob[8:12] in (b"AIFF", b"AIFC"):
            results[index] = parse_wav(blob)
            continue
        fmt = detect_format(blob)
        if fmt == "wav":
            results[index] = parse_wav(blob)
        elif fmt == "mp3":
            results[index] = parse_mp3(blob)
        elif fmt == "flac":
            results[index] = {"format": "flac"}
        elif fmt == "ogg":
            results[index] = {"format": "ogg"}
        else:
            raise ProbeError("blob %d 格式未知或数据损坏" % index)
    return results
