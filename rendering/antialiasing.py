"""
游戏内抗锯齿与超分辨率系统 - 核心模块

功能：
- apply_antialiasing: 基于 Sobel 边缘检测的抗锯齿，边缘像素做平滑混合
- super_resolution:   超分辨率（bicubic 上采样 + 高频细节增强 + 锐化）
- dlss:               时序超分辨率（运动矢量重投影 + 深度一致性校验）
- taa:                时间抗锯齿（历史缓冲 + 重投影 + 邻域裁剪）
- fsr:                空间超分辨率（边缘自适应 EASU 风格上采样 + RCAS 风格锐化）

图像表示：List[List[float]]，灰度值 0.0 ~ 1.0。
运动矢量：List[List[Tuple[dx, dy]]]，表示该像素从上一帧到当前帧的位移。
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import math

Image = List[List[float]]
MotionField = List[List[Tuple[float, float]]]


@dataclass
class Config:
    pass


def _clamp(value, lo=0.0, hi=1.0):
    return lo if value < lo else hi if value > hi else value


def _image_size(image):
    return len(image), len(image[0])


def _make_image(height, width, fill=0.0):
    return [[fill] * width for _ in range(height)]


def _copy_image(image):
    return [row[:] for row in image]


def _sample_bilinear(image, x, y):
    height, width = _image_size(image)
    x = min(max(x, 0.0), width - 1.0)
    y = min(max(y, 0.0), height - 1.0)
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    fx = x - x0
    fy = y - y0
    top = image[y0][x0] * (1.0 - fx) + image[y0][x1] * fx
    bottom = image[y1][x0] * (1.0 - fx) + image[y1][x1] * fx
    return top * (1.0 - fy) + bottom * fy


def _cubic_interp(p0, p1, p2, p3, t):
    # Catmull-Rom 三次插值
    return p1 + 0.5 * t * (
        p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
                       + t * (3.0 * (p1 - p2) + p3 - p0))
    )


def _sample_bicubic(image, x, y):
    height, width = _image_size(image)
    x = min(max(x, 0.0), width - 1.0)
    y = min(max(y, 0.0), height - 1.0)
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    fx = x - x0
    fy = y - y0
    rows = []
    for j in range(-1, 3):
        yy = min(max(y0 + j, 0), height - 1)
        p = [image[yy][min(max(x0 + i, 0), width - 1)] for i in range(-1, 3)]
        rows.append(_cubic_interp(p[0], p[1], p[2], p[3], fx))
    return _cubic_interp(rows[0], rows[1], rows[2], rows[3], fy)


def _upscale_bicubic(image, scale, clamp=True):
    height, width = _image_size(image)
    out_h, out_w = height * scale, width * scale
    out = _make_image(out_h, out_w)
    for oy in range(out_h):
        for ox in range(out_w):
            sx = (ox + 0.5) / scale - 0.5
            sy = (oy + 0.5) / scale - 0.5
            value = _sample_bicubic(image, sx, sy)
            out[oy][ox] = _clamp(value) if clamp else value
    return out


def _gaussian_blur(image):
    # 3x3 高斯核: [1 2 1; 2 4 2; 1 2 1] / 16
    height, width = _image_size(image)
    out = _make_image(height, width)
    for y in range(height):
        for x in range(width):
            total = 0.0
            weight = 0.0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    yy = min(max(y + dy, 0), height - 1)
                    xx = min(max(x + dx, 0), width - 1)
                    k = (2 if dx == 0 else 1) * (2 if dy == 0 else 1)
                    total += image[yy][xx] * k
                    weight += k
            out[y][x] = total / weight
    return out


def _unsharp_mask(image, amount):
    # 锐化: out = img + amount * (img - blur(img))
    if amount <= 0.0:
        return _copy_image(image)
    blurred = _gaussian_blur(image)
    height, width = _image_size(image)
    out = _make_image(height, width)
    for y in range(height):
        for x in range(width):
            out[y][x] = _clamp(image[y][x] + amount * (image[y][x] - blurred[y][x]))
    return out


def _gradient_at(image, x, y):
    # 在最近整数像素处用 Sobel 估计梯度，按单位阶跃归一化
    height, width = _image_size(image)
    cx = min(max(int(round(x)), 0), width - 1)
    cy = min(max(int(round(y)), 0), height - 1)

    def px(dx, dy):
        return image[min(max(cy + dy, 0), height - 1)][min(max(cx + dx, 0), width - 1)]

    gx = (px(1, -1) + 2.0 * px(1, 0) + px(1, 1)) - (px(-1, -1) + 2.0 * px(-1, 0) + px(-1, 1))
    gy = (px(-1, 1) + 2.0 * px(0, 1) + px(1, 1)) - (px(-1, -1) + 2.0 * px(0, -1) + px(1, -1))
    return gx / 4.0, gy / 4.0


class Antialiasing:
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []

    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []

    # ------------------------------------------------------------------
    # 边缘检测
    # ------------------------------------------------------------------
    def detect_edges(self, image: Image, threshold: float = 0.2) -> List[List[bool]]:
        """Sobel 边缘检测，返回与图像同尺寸的布尔边缘图。"""
        height, width = _image_size(image)
        edges = [[False] * width for _ in range(height)]
        for y in range(height):
            for x in range(width):
                gx, gy = _gradient_at(image, x, y)
                edges[y][x] = math.hypot(gx, gy) >= threshold
        return edges

    # ------------------------------------------------------------------
    # 抗锯齿：边缘检测 + 边缘平滑
    # ------------------------------------------------------------------
    def apply_antialiasing(self, image: Image, threshold: float = 0.2,
                           blend: float = 0.75) -> Image:
        """FXAA 风格抗锯齿：检测边缘，仅对边缘像素做平滑混合。"""
        height, width = _image_size(image)
        edges = self.detect_edges(image, threshold)
        blurred = _gaussian_blur(image)
        out = _copy_image(image)
        for y in range(height):
            for x in range(width):
                if edges[y][x]:
                    out[y][x] = _clamp((1.0 - blend) * image[y][x] + blend * blurred[y][x])
        return out

    # ------------------------------------------------------------------
    # 超分辨率：bicubic 上采样 + 细节增强 + 锐化
    # ------------------------------------------------------------------
    def super_resolution(self, image: Image, scale: int = 2,
                         detail_strength: float = 1.0,
                         sharpen_strength: float = 0.5) -> Image:
        if scale < 1:
            raise ValueError("scale 必须 >= 1")
        # 1) bicubic 上采样（优于简单双线性）
        up = _upscale_bicubic(image, scale)
        # 2) 细节增强：提取源图高频细节并注入放大结果
        if detail_strength > 0.0:
            height, width = _image_size(image)
            blurred = _gaussian_blur(image)
            high_freq = [[image[y][x] - blurred[y][x] for x in range(width)]
                         for y in range(height)]
            high_up = _upscale_bicubic(high_freq, scale, clamp=False)
            out_h, out_w = _image_size(up)
            for y in range(out_h):
                for x in range(out_w):
                    up[y][x] = _clamp(up[y][x] + detail_strength * high_up[y][x])
        # 3) 锐化（unsharp mask），让边缘清晰
        if sharpen_strength > 0.0:
            up = _unsharp_mask(up, sharpen_strength)
        return up

    # ------------------------------------------------------------------
    # DLSS：运动矢量 + 深度的时序超分辨率
    # ------------------------------------------------------------------
    def dlss(self, frame: Image, motion_vectors: MotionField, depth: Image,
             scale: int = 2, history_weight: float = 0.7,
             depth_threshold: float = 0.1, detail_strength: float = 1.0,
             sharpen_strength: float = 0.5) -> Image:
        height, width = _image_size(frame)
        prev_frame = self._state.get("dlss_history")
        prev_depth = self._state.get("dlss_depth")

        if prev_frame is not None and prev_depth is not None \
                and _image_size(prev_frame) == (height, width):
            accumulated = _make_image(height, width)
            for y in range(height):
                for x in range(width):
                    dx, dy = motion_vectors[y][x]
                    hx = x - dx
                    hy = y - dy
                    in_bounds = 0.0 <= hx <= width - 1 and 0.0 <= hy <= height - 1
                    # 深度一致性校验：深度不匹配视为遮挡/去遮挡，丢弃历史
                    depth_ok = in_bounds and abs(
                        _sample_bilinear(prev_depth, hx, hy) - depth[y][x]
                    ) <= depth_threshold
                    if depth_ok:
                        hist = _sample_bilinear(prev_frame, hx, hy)
                        accumulated[y][x] = history_weight * hist \
                            + (1.0 - history_weight) * frame[y][x]
                    else:
                        accumulated[y][x] = frame[y][x]
        else:
            accumulated = _copy_image(frame)

        self._state["dlss_history"] = _copy_image(accumulated)
        self._state["dlss_depth"] = _copy_image(depth)
        # AI 超分辨率：时序累积结果 + 细节增强 + 锐化
        return self.super_resolution(accumulated, scale,
                                     detail_strength, sharpen_strength)

    # ------------------------------------------------------------------
    # TAA：历史缓冲 + 重投影的时间抗锯齿
    # ------------------------------------------------------------------
    def taa(self, frame: Image, motion_vectors: Optional[MotionField] = None,
            blend_factor: float = 0.1) -> Image:
        height, width = _image_size(frame)
        if motion_vectors is None:
            motion_vectors = [[(0.0, 0.0)] * width for _ in range(height)]

        history = self._state.get("taa_history")
        if history is None or _image_size(history) != (height, width):
            result = _copy_image(frame)
        else:
            result = _make_image(height, width)
            for y in range(height):
                for x in range(width):
                    dx, dy = motion_vectors[y][x]
                    # 重投影：按运动矢量回采历史帧
                    hist = _sample_bilinear(history, x - dx, y - dy)
                    # 邻域裁剪：把历史样本夹到当前帧 3x3 邻域 [min, max]，抑制鬼影
                    lo = 1.0
                    hi = 0.0
                    for ny in range(max(y - 1, 0), min(y + 2, height)):
                        for nx in range(max(x - 1, 0), min(x + 2, width)):
                            v = frame[ny][nx]
                            lo = min(lo, v)
                            hi = max(hi, v)
                    hist = min(max(hist, lo), hi)
                    result[y][x] = blend_factor * frame[y][x] + (1.0 - blend_factor) * hist

        self._state["taa_history"] = _copy_image(result)
        self._history.append(result)
        return result

    # ------------------------------------------------------------------
    # FSR：边缘自适应空间超分辨率 + 锐化
    # ------------------------------------------------------------------
    def fsr(self, image: Image, scale: int = 2,
            sharpen_strength: float = 0.5) -> Image:
        if scale < 1:
            raise ValueError("scale 必须 >= 1")
        up = self._easu_upscale(image, scale)
        # RCAS 风格锐化
        if sharpen_strength > 0.0:
            up = _unsharp_mask(up, sharpen_strength)
        return up

    def _easu_upscale(self, image: Image, scale: int) -> Image:
        """EASU 风格边缘自适应上采样：沿边缘方向加宽滤波，跨边缘方向收窄。"""
        height, width = _image_size(image)
        out_h, out_w = height * scale, width * scale
        out = _make_image(out_h, out_w)
        offset = 0.75
        for oy in range(out_h):
            for ox in range(out_w):
                sx = (ox + 0.5) / scale - 0.5
                sy = (oy + 0.5) / scale - 0.5
                center = _sample_bilinear(image, sx, sy)
                gx, gy = _gradient_at(image, sx, sy)
                mag = math.hypot(gx, gy)
                if mag < 1e-3:
                    out[oy][ox] = _clamp(center)
                    continue
                strength = min(mag, 1.0)
                # 边缘方向（沿边缘）与梯度方向（跨边缘）
                ex, ey = -gy / mag, gx / mag
                nx, ny = gx / mag, gy / mag
                along = 0.5 * (
                    _sample_bilinear(image, sx + ex * offset, sy + ey * offset)
                    + _sample_bilinear(image, sx - ex * offset, sy - ey * offset)
                )
                across = 0.5 * (
                    _sample_bilinear(image, sx + nx * offset * 0.5, sy + ny * offset * 0.5)
                    + _sample_bilinear(image, sx - nx * offset * 0.5, sy - ny * offset * 0.5)
                )
                value = (center * (1.0 - 0.55 * strength)
                         + along * 0.5 * strength
                         + across * 0.05 * strength)
                out[oy][ox] = _clamp(value)
        return out
