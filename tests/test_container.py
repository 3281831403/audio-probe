"""audio-probe 的 15 个契约测试。

测试只使用内联构造的 bytes；不读取外部素材，不使用随机数。
当前 probe/container.py 是桩实现，因此 15 条测试都应失败。
"""

import math
import struct

import pytest

from probe.container import (
    ProbeError,
    channel_mask,
    detect_format,
    encoder_delay,
    estimate_duration,
    frame_duration_ms,
    parse_mp3,
    parse_wav,
    probe_batch,
)


# ---------------------------------------------------------------------------
# 字节构造辅助：全部在测试进程内现场生成
# ---------------------------------------------------------------------------


def _syncsafe(value):
    """Encode an ID3v2 syncsafe 28-bit integer."""
    return bytes(
        (
            (value >> 21) & 0x7F,
            (value >> 14) & 0x7F,
            (value >> 7) & 0x7F,
            value & 0x7F,
        )
    )


def _riff_chunk(tag, payload):
    """Build a little-endian RIFF chunk with even-byte padding."""
    chunk = tag + struct.pack("<I", len(payload)) + payload
    if len(payload) % 2:
        chunk += b"\x00"
    return chunk


def _wav_bytes(sample_rate=44100, channels=2, bits=16, frame_count=4):
    """Build a minimal PCM RIFF/WAVE file."""
    block_align = channels * ((bits + 7) // 8)
    data = b"\x00" * (frame_count * block_align)
    fmt = struct.pack(
        "<HHIIHH",
        1,
        channels,
        sample_rate,
        sample_rate * block_align,
        block_align,
        bits,
    )
    body = b"WAVE" + _riff_chunk(b"fmt ", fmt) + _riff_chunk(b"data", data)
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _wav_extensible_bytes():
    """Build a 40-byte WAVE_FORMAT_EXTENSIBLE header plus a LIST chunk."""
    subformat = struct.pack("<IHH", 1, 0, 0x0010) + bytes.fromhex(
        "800000aa00389b71"
    )
    fmt = struct.pack(
        "<HHIIHHHHI",
        0xFFFE,
        2,
        48000,
        48000 * 2 * 3,
        6,
        24,
        22,
        24,
        0x3,
    ) + subformat
    list_payload = b"INFO" + b"INAM" + struct.pack("<I", 4) + b"test"
    data = b"\x00" * 12
    body = (
        b"WAVE"
        + _riff_chunk(b"fmt ", fmt)
        + _riff_chunk(b"LIST", list_payload)
        + _riff_chunk(b"data", data)
    )
    return b"RIFF" + struct.pack("<I", len(body)) + body


def _mp3_header(
    version_bits,
    layer_bits,
    bitrate_index,
    sample_rate_index,
    padding=0,
    channel_mode=0,
):
    """Build a 4-byte MPEG Audio frame header."""
    word = (
        (0x7FF << 21)
        | (version_bits << 19)
        | (layer_bits << 17)
        | (1 << 16)  # no CRC
        | (bitrate_index << 12)
        | (sample_rate_index << 10)
        | (padding << 9)
        | (channel_mode << 6)
    )
    return struct.pack(">I", word)


def _id3v2(body):
    """Wrap a payload in an ID3v2.4 header."""
    return b"ID3\x04\x00\x00" + _syncsafe(len(body)) + body


def _aiff_chunk(tag, payload):
    """Build a big-endian AIFF chunk with even-byte padding."""
    chunk = tag + struct.pack(">I", len(payload)) + payload
    if len(payload) % 2:
        chunk += b"\x00"
    return chunk


def _extended80(value):
    """Encode a non-negative number as an 80-bit IEEE extended value."""
    if value == 0:
        return b"\x00" * 10
    sign = 0
    if value < 0:
        sign = 0x8000
        value = -value
    exponent = int(math.floor(math.log2(value)))
    mantissa = value / (2.0**exponent)
    fraction = int((mantissa - 1.0) * (1 << 63))
    if fraction >= (1 << 63):
        fraction //= 2
        exponent += 1
    biased = exponent + 16383
    return struct.pack(">HQ", sign | biased, fraction)


def _aiff_bytes(sample_rate, channels=2, bits=16, frame_count=4):
    """Build a minimal uncompressed AIFF file with a COMM and SSND chunk."""
    block_align = channels * ((bits + 7) // 8)
    audio = b"\x00" * (frame_count * block_align)
    comm = (
        struct.pack(">hIh", channels, frame_count, bits)
        + _extended80(sample_rate)
    )
    ssnd = struct.pack(">II", 0, 0) + audio
    body = b"AIFF" + _aiff_chunk(b"COMM", comm) + _aiff_chunk(b"SSND", ssnd)
    return b"FORM" + struct.pack(">I", len(body)) + body


# ---------------------------------------------------------------------------
# 十个缺陷各一条测试
# ---------------------------------------------------------------------------


def test_wav_with_list_chunk_parses():
    """缺陷 1：不能只认 44 字节头；要跳过 LIST 并解析扩展 fmt。"""
    info = parse_wav(_wav_extensible_bytes())

    assert info["format"] == "wav"
    assert info["audio_format"] == 0xFFFE
    assert info["channels"] == 2
    assert info["sample_rate"] == 48000
    assert info["bits_per_sample"] == 24
    assert info["data_size"] == 12
    assert info["frame_count"] == 2
    assert info["channel_mask"] == 0x3


def test_mp3_skips_id3v2_tag():
    """缺陷 2：ID3v2 里的伪同步字不能被当成第一帧。"""
    fake_header = _mp3_header(3, 1, 5, 1)  # MPEG1 L3, 64 kbps, 48000 Hz
    tag_body = b"\x00" * 12 + fake_header + b"\x00" * 64
    id3 = _id3v2(tag_body)
    real_header = _mp3_header(3, 1, 9, 0)  # MPEG1 L3, 128 kbps, 44100 Hz
    data = id3 + real_header + b"\x00" * 500

    info = parse_mp3(data)

    assert info["frame_offset"] == len(id3)
    assert info["version"] == "MPEG1"
    assert info["layer"] == 3
    assert info["bitrate_kbps"] == 128
    assert info["sample_rate"] == 44100
    assert info["channels"] == 2


def test_frame_duration_mpeg2_layer3():
    """缺陷 3：MPEG2 Layer III 是 576 个采样点，不能固定成 26.12 ms。"""
    mpeg1 = _mp3_header(3, 1, 9, 0)  # 44100 Hz
    mpeg2 = _mp3_header(2, 1, 8, 2)  # MPEG2, 16000 Hz

    assert frame_duration_ms(mpeg1) == pytest.approx(
        1152 * 1000.0 / 44100, abs=1000.0 / 44100
    )
    assert frame_duration_ms(mpeg2) == pytest.approx(36.0, abs=1000.0 / 16000)


def test_vbr_duration_uses_xing_frame_count():
    """缺陷 4：VBR 必须用 Xing/Info 的帧数，不能按首帧比特率估算。"""
    header = _mp3_header(3, 1, 9, 0)  # 128 kbps, 44100 Hz
    frame_size = (144 * 128000) // 44100
    side_info = b"\x00" * 32  # MPEG1 stereo side information
    xing = b"Xing" + struct.pack(">I", 1) + struct.pack(">I", 1000)
    first_frame = header + side_info + xing
    assert len(first_frame) <= frame_size
    data = first_frame + b"\x00" * (frame_size - len(first_frame))
    data += b"\x00" * 800

    info = parse_mp3(data)
    actual = estimate_duration(data, info)

    assert info["is_vbr"] is True
    assert info["frame_count"] == 1000
    expected = 1000 * info["samples_per_frame"]
    assert abs(actual - expected) <= info["samples_per_frame"]


def test_aiff_big_endian_sample_rate():
    """缺陷 5：AIFF 的 80 位大端采样率不能读成 0。"""
    info = parse_wav(_aiff_bytes(44100.0))

    assert info["format"] == "aiff"
    assert info["sample_rate"] == pytest.approx(44100.0, abs=1e-9)
    assert info["channels"] == 2
    assert info["bits_per_sample"] == 16
    assert info["frame_count"] == 4


def test_channel_mask_5_1():
    """缺陷 6：5.1 必须是 0x3F，不能一律硬编码成立体声。"""
    assert channel_mask({"layout": "mono", "channels": 1}) == 0x4
    assert channel_mask({"layout": "stereo", "channels": 2}) == 0x3
    assert channel_mask({"layout": "5.1", "channels": 6}) == 0x3F
    assert channel_mask({"layout": "7.1", "channels": 8}) == 0x63F
    with pytest.raises(ProbeError):
        channel_mask({"layout": "no-such-layout"})


def test_encoder_delay_samples():
    """缺陷 7：Gapless 的 1105 个采样点延迟不能被忽略。"""
    assert encoder_delay(
        {"encoder_delay": 1105, "encoder_padding": 529}
    ) == 1105
    assert encoder_delay({}) == 0


def test_unknown_magic_returns_unknown():
    """缺陷 8：未知魔数返回 "unknown"，不能抛异常。"""
    assert detect_format(b"not-a-container") == "unknown"
    assert detect_format(b"RIFF\x00\x00\x00\x00NOPE") == "unknown"


def test_corrupt_batch_does_not_drop_entry():
    """缺陷 9：坏文件不能被静默跳过，批量必须报错。"""
    with pytest.raises(ProbeError):
        probe_batch([b"\x00\x01\x02\x03"])


def test_fractional_sample_rate_not_rounded():
    """缺陷 10：44100.5 不能被四舍五入成 44100。"""
    info = parse_wav(_aiff_bytes(44100.5))

    assert info["sample_rate"] == pytest.approx(44100.5, abs=1e-9)
    assert info["sample_rate"] != 44100


# ---------------------------------------------------------------------------
# 五条组合 / 边界测试
# ---------------------------------------------------------------------------


def test_empty_data_detect_unknown():
    """空数据只能返回 unknown，不能抛异常。"""
    assert detect_format(b"") == "unknown"


def test_wav_roundtrip_sample_rate():
    """WAV 往返后采样率、声道数和帧数保持正确。"""
    wav = _wav_bytes(sample_rate=32000, channels=1, bits=16, frame_count=3)

    assert detect_format(wav) == "wav"
    info = parse_wav(wav)
    assert info["sample_rate"] == 32000
    assert info["channels"] == 1
    assert info["frame_count"] == 3


def test_zero_length_wav_raises_probe_error():
    """parse_wav 收到长度为 0 的数据必须抛 ProbeError。"""
    with pytest.raises(ProbeError):
        parse_wav(b"")


def test_batch_single_matches_direct():
    """单个文件的批量结果必须与直接调用 parse_wav 一致。"""
    wav = _wav_bytes()
    direct = parse_wav(wav)

    assert probe_batch([wav]) == {0: direct}


def test_corrupt_batch_rolls_back():
    """好文件加坏文件时，整个批量回滚，不返回部分结果。"""
    good = _wav_bytes()
    bad = good[:20]

    with pytest.raises(ProbeError):
        probe_batch([good, bad])
