"""
游戏内抗锯齿与超分辨率系统 - 核心模块

图像表示：二维列表（行主序）的灰度浮点像素，取值范围 [0, 1]。
运动矢量：与源图同尺寸的二维列表，元素为 (dx, dy)，表示上一帧像素
移动到当前帧的位移（单位：源分辨率像素）。
深度图：与源图同尺寸的二维浮点列表，取值范围 [0, 1]。
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
import time


@dataclass
class Config:
    pass


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if value < lo else hi if value > hi else value


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
    # 基础工具
    # ------------------------------------------------------------------
    @staticmethod
    def _shape(image: List[List[float]]) -> Tuple[int, int]:
        return len(image), len(image[0])

    @staticmethod
    def _sample(image: List[List[float]], x: float, y: float) -> float:
        """双线性采样，x 为列坐标，y 为行坐标，边界钳制。"""
        rows, cols = len(image), len(image[0])
        x = min(max(x, 0.0), cols - 1.0)
        y = min(max(y, 0.0), rows - 1.0)
        x0 = int(math.floor(x))
        y0 = int(math.floor(y))
        x1 = min(x0 + 1, cols - 1)
        y1 = min(y0 + 1, rows - 1)
        fx = x - x0
        fy = y - y0
        top = image[y0][x0] * (1.0 - fx) + image[y0][x1] * fx
        bottom = image[y1][x0] * (1.0 - fx) + image[y1][x1] * fx
        return top * (1.0 - fy) + bottom * fy

    @staticmethod
    def _gradient(image: List[List[float]], x: int, y: int) -> Tuple[float, float]:
        rows, cols = len(image), len(image[0])
        xm, xp = max(x - 1, 0), min(x + 1, cols - 1)
        ym, yp = max(y - 1, 0), min(y + 1, rows - 1)
        gx = (image[y][xp] - image[y][xm]) / float(xp - xm)
        gy = (image[yp][x] - image[ym][x]) / float(yp - ym)
        return gx, gy

    @staticmethod
    def _box_blur(image: List[List[float]]) -> List[List[float]]:
        rows, cols = len(image), len(image[0])
        out = [[0.0] * cols for _ in range(rows)]
        for y in range(rows):
            for x in range(cols):
                total = 0.0
                count = 0
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < rows and 0 <= nx < cols:
                            total += image[ny][nx]
                            count += 1
                out[y][x] = total / count
        return out

    def _unsharp_mask(self, image: List[List[float]], strength: float) -> List[List[float]]:
        """锐化：原图 + strength * (原图 - 模糊图)。"""
        blurred = self._box_blur(image)
        rows, cols = self._shape(image)
        return [
            [_clamp(image[y][x] + strength * (image[y][x] - blurred[y][x]))
             for x in range(cols)]
            for y in range(rows)
        ]

    # ------------------------------------------------------------------
    # 抗锯齿：边缘检测 + 沿边缘方向的模糊（FXAA 风格）
    # ------------------------------------------------------------------
    def detect_edges(self, image: List[List[float]], threshold: float = 0.1) -> List[List[float]]:
        """基于亮度梯度的边缘检测，返回边缘强度图（低于阈值处为 0）。"""
        rows, cols = self._shape(image)
        edges = [[0.0] * cols for _ in range(rows)]
        for y in range(rows):
            for x in range(cols):
                gx, gy = self._gradient(image, x, y)
                magnitude = math.hypot(gx, gy)
                if magnitude >= threshold:
                    edges[y][x] = magnitude
        return edges

    def apply_antialiasing(self, image: List[List[float]],
                           threshold: float = 0.1,
                           strength: float = 1.0) -> List[List[float]]:
        """FXAA 风格抗锯齿：先检测边缘，再沿边缘切线方向模糊，平滑锯齿。"""
        rows, cols = self._shape(image)
        blend = min(max(strength, 0.0), 1.0)
        result = [row[:] for row in image]
        for y in range(rows):
            for x in range(cols):
                gx, gy = self._gradient(image, x, y)
                magnitude = math.hypot(gx, gy)
                if magnitude < threshold:
                    continue
                # 边缘切线方向（垂直于梯度），沿切线采样做方向性模糊
                tx, ty = -gy / magnitude, gx / magnitude
                s1 = self._sample(image, x + tx, y + ty)
                s2 = self._sample(image, x - tx, y - ty)
                smoothed = (2.0 * image[y][x] + s1 + s2) / 4.0
                result[y][x] = image[y][x] * (1.0 - blend) + smoothed * blend
        return result

    # ------------------------------------------------------------------
    # 超分辨率：细节增强 + 锐化
    # ------------------------------------------------------------------
    def super_resolution(self, image: List[List[float]], scale: int = 2,
                         detail_strength: float = 1.0,
                         sharpen_strength: float = 0.5) -> List[List[float]]:
        """超分辨率：双线性放大 + 高频细节增强 + 反锐化掩模锐化。"""
        rows, cols = self._shape(image)
        out_rows, out_cols = rows * scale, cols * scale

        # 从源图提取高频细节（原图 - 低频）
        blurred_src = self._box_blur(image)
        detail = [[image[y][x] - blurred_src[y][x] for x in range(cols)]
                  for y in range(rows)]

        base = [[0.0] * out_cols for _ in range(out_rows)]
        for y in range(out_rows):
            sy = (y + 0.5) / scale - 0.5
            for x in range(out_cols):
                sx = (x + 0.5) / scale - 0.5
                value = self._sample(image, sx, sy)
                # 细节增强：把源图高频细节叠加回放大结果
                value += detail_strength * self._sample(detail, sx, sy)
                base[y][x] = value

        # 锐化，恢复放大损失的边缘对比度
        return self._unsharp_mask(base, sharpen_strength)

    # ------------------------------------------------------------------
    # DLSS：AI 风格时间超分辨率，利用运动矢量与深度信息
    # ------------------------------------------------------------------
    def dlss(self, image: List[List[float]], scale: int = 2,
             motion_vectors: Optional[List[List[Tuple[float, float]]]] = None,
             depth: Optional[List[List[float]]] = None,
             history_weight: float = 0.5,
             depth_threshold: float = 0.1) -> List[List[float]]:
        """DLSS 风格超分辨率：空间放大 + 运动矢量重投影历史累积 + 深度校验。"""
        current = self.super_resolution(image, scale=scale)
        out_rows, out_cols = self._shape(current)
        history = self._state.get("dlss_history")

        if history is None or motion_vectors is None:
            output = current
        else:
            prev_color = history["color"]
            prev_depth = history["depth"]
            output = [[0.0] * out_cols for _ in range(out_rows)]
            for y in range(out_rows):
                sy = (y + 0.5) / scale - 0.5
                iy = min(max(int(round(sy)), 0), len(image) - 1)
                for x in range(out_cols):
                    sx = (x + 0.5) / scale - 0.5
                    ix = min(max(int(round(sx)), 0), len(image[0]) - 1)
                    mvx, mvy = motion_vectors[iy][ix]
                    # 运动矢量重投影：当前像素在历史帧中的对应位置
                    hx = x - mvx * scale
                    hy = y - mvy * scale
                    weight = history_weight
                    if depth is not None and prev_depth is not None:
                        # 深度一致性校验：深度突变视为遮挡/去遮挡，拒绝历史
                        d_now = depth[iy][ix]
                        d_prev = self._sample(prev_depth, sx - mvx, sy - mvy)
                        if abs(d_now - d_prev) > depth_threshold:
                            weight = 0.0
                    accumulated = self._sample(prev_color, hx, hy)
                    output[y][x] = current[y][x] * (1.0 - weight) + accumulated * weight

        self._state["dlss_history"] = {"color": [row[:] for row in output],
                                       "depth": depth}
        return output

    # ------------------------------------------------------------------
    # TAA：历史缓冲 + 重投影的时间抗锯齿
    # ------------------------------------------------------------------
    def taa(self, image: List[List[float]],
            motion_vectors: Optional[List[List[Tuple[float, float]]]] = None,
            current_weight: float = 0.1,
            max_history: int = 8) -> List[List[float]]:
        """TAA：把历史帧按运动矢量重投影到当前帧，与当前帧指数混合累积。"""
        rows, cols = self._shape(image)
        current_weight = min(max(current_weight, 0.0), 1.0)

        if not self._history:
            output = [row[:] for row in image]
        else:
            prev = self._history[-1]
            output = [[0.0] * cols for _ in range(rows)]
            for y in range(rows):
                for x in range(cols):
                    if motion_vectors is not None:
                        mvx, mvy = motion_vectors[y][x]
                        history_value = self._sample(prev, x - mvx, y - mvy)
                    else:
                        history_value = prev[y][x]
                    output[y][x] = (image[y][x] * current_weight +
                                    history_value * (1.0 - current_weight))

        self._history.append([row[:] for row in output])
        if len(self._history) > max_history:
            del self._history[:len(self._history) - max_history]
        return output

    # ------------------------------------------------------------------
    # FSR：边缘自适应空间超分辨率 + RCAS 风格锐化
    # ------------------------------------------------------------------
    def fsr(self, image: List[List[float]], scale: int = 2,
            sharpen_strength: float = 0.5,
            edge_threshold: float = 1e-3) -> List[List[float]]:
        """FSR 风格空间超分辨率：按边缘方向自适应插值，再做锐化。"""
        rows, cols = self._shape(image)
        out_rows, out_cols = rows * scale, cols * scale
        upscaled = [[0.0] * out_cols for _ in range(out_rows)]

        for y in range(out_rows):
            sy = (y + 0.5) / scale - 0.5
            iy = min(max(int(round(sy)), 0), rows - 1)
            for x in range(out_cols):
                sx = (x + 0.5) / scale - 0.5
                ix = min(max(int(round(sx)), 0), cols - 1)
                gx, gy = self._gradient(image, ix, iy)
                magnitude = math.hypot(gx, gy)
                if magnitude > edge_threshold:
                    # 边缘自适应：沿边缘切线方向做各向异性插值，
                    # 避免跨越边缘采样造成的模糊
                    tx, ty = -gy / magnitude, gx / magnitude
                    center = self._sample(image, sx, sy)
                    s1 = self._sample(image, sx + tx * 0.5, sy + ty * 0.5)
                    s2 = self._sample(image, sx - tx * 0.5, sy - ty * 0.5)
                    upscaled[y][x] = (2.0 * center + s1 + s2) / 4.0
                else:
                    upscaled[y][x] = self._sample(image, sx, sy)

        # RCAS 风格锐化
        return self._unsharp_mask(upscaled, sharpen_strength)
