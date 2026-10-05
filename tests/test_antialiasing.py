"""
游戏内抗锯齿与超分辨率系统 - 验收测试
运行方式：python -m pytest tests/test_antialiasing.py -q
共 15 个测试用例
"""
import random

import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rendering.antialiasing import Antialiasing


# ----------------------------------------------------------------------
# 测试辅助函数
# ----------------------------------------------------------------------
def bilinear_upscale(image, scale):
    """简单的双线性放大基线，用于对比。"""
    h, w = len(image), len(image[0])
    oh, ow = h * scale, w * scale
    out = [[0.0] * ow for _ in range(oh)]
    for oy in range(oh):
        for ox in range(ow):
            sx = min(max((ox + 0.5) / scale - 0.5, 0.0), w - 1.0)
            sy = min(max((oy + 0.5) / scale - 0.5, 0.0), h - 1.0)
            x0, y0 = int(sx), int(sy)
            x1, y1 = min(x0 + 1, w - 1), min(y0 + 1, h - 1)
            fx, fy = sx - x0, sy - y0
            top = image[y0][x0] * (1 - fx) + image[y0][x1] * fx
            bot = image[y1][x0] * (1 - fx) + image[y1][x1] * fx
            out[oy][ox] = top * (1 - fy) + bot * fy
    return out


def mse(a, b):
    h, w = len(a), len(a[0])
    return sum((a[y][x] - b[y][x]) ** 2 for y in range(h) for x in range(w)) / (h * w)


def max_horizontal_gradient(a):
    return max(abs(a[y][x + 1] - a[y][x])
               for y in range(len(a)) for x in range(len(a[0]) - 1))


def region_std(a, y0, x0, size):
    vals = [a[y0 + dy][x0 + dx] for dy in range(size) for dx in range(size)]
    mean = sum(vals) / len(vals)
    return (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5


def along_edge_variation(a):
    """沿 (1,1) 对角方向的相邻像素差异总和，用于度量对角边缘的锯齿程度。"""
    h, w = len(a), len(a[0])
    return sum(abs(a[y][x] - a[y + 1][x + 1])
               for y in range(h - 1) for x in range(w - 1))


def make_step_image(h=8, w=8, lo=0.2, hi=0.8):
    return [[lo if x < w // 2 else hi for x in range(w)] for _ in range(h)]


def make_flat_image(h=8, w=8, value=0.5):
    return [[value] * w for _ in range(h)]


def zero_motion(h, w):
    return [[(0.0, 0.0)] * w for _ in range(h)]


def constant_motion(h, w, dx, dy):
    return [[(dx, dy)] * w for _ in range(h)]


def constant_depth(h, w, value=0.5):
    return [[value] * w for _ in range(h)]


def stripes(x):
    return ((x // 2) % 2) * 0.8 + 0.1


class TestAntialiasing:
    def setup_method(self):
        self.system = Antialiasing()

    # ------------------------------------------------------------------
    # 抗锯齿：边缘检测 + 边缘平滑
    # ------------------------------------------------------------------
    def test_case_01(self):
        """边缘检测：阶跃边缘能被检测出来"""
        image = make_step_image(lo=0.0, hi=1.0)
        edges = self.system.detect_edges(image)
        assert len(edges) == 8 and len(edges[0]) == 8
        assert any(any(row) for row in edges)
        # 边缘应出现在明暗交界列（第 3、4 列）附近
        assert any(edges[y][3] or edges[y][4] for y in range(8))

    def test_case_02(self):
        """边缘检测：平坦图像没有边缘"""
        edges = self.system.detect_edges(make_flat_image())
        assert not any(any(row) for row in edges)

    def test_case_03(self):
        """抗锯齿：阶跃边缘被平滑，产生中间过渡值"""
        image = make_step_image(lo=0.0, hi=1.0)
        out = self.system.apply_antialiasing(image)
        assert out != image
        # 平滑后边缘处应出现介于 0 和 1 之间的过渡像素
        assert any(0.05 < v < 0.95 for row in out for v in row)
        # 远离边缘的平坦区域保持不变
        assert out[0][0] == 0.0
        assert out[0][7] == 1.0

    def test_case_04(self):
        """抗锯齿：平坦区域不被模糊"""
        image = make_flat_image()
        out = self.system.apply_antialiasing(image)
        assert out == image

    # ------------------------------------------------------------------
    # 超分辨率：细节增强 + 锐化
    # ------------------------------------------------------------------
    def test_case_05(self):
        """超分辨率：输出尺寸按 scale 放大"""
        image = make_step_image(h=6, w=8)
        out = self.system.super_resolution(image, scale=2)
        assert len(out) == 12 and len(out[0]) == 16
        out3 = self.system.super_resolution(image, scale=3)
        assert len(out3) == 18 and len(out3[0]) == 24

    def test_case_06(self):
        """超分辨率：细节增强，纹理区域的高频细节强于双线性插值"""
        rng = random.Random(42)
        texture = [[0.5 + 0.3 * rng.uniform(-1.0, 1.0) for _ in range(12)]
                   for _ in range(12)]
        sr = self.system.super_resolution(texture, scale=2,
                                          detail_strength=1.0,
                                          sharpen_strength=0.0)
        baseline = bilinear_upscale(texture, 2)
        # 中心 8x8 区域的局部标准差：细节增强后应明显高于双线性
        assert region_std(sr, 8, 8, 8) > region_std(baseline, 8, 8, 8)

    def test_case_07(self):
        """超分辨率：锐化使边缘过渡比双线性插值更陡"""
        image = make_step_image()
        sr = self.system.super_resolution(image, scale=2,
                                          detail_strength=0.0,
                                          sharpen_strength=0.8)
        baseline = bilinear_upscale(image, 2)
        assert max_horizontal_gradient(sr) > max_horizontal_gradient(baseline)

    # ------------------------------------------------------------------
    # DLSS：运动矢量 + 深度
    # ------------------------------------------------------------------
    def test_case_08(self):
        """DLSS：输出尺寸按 scale 放大"""
        frame = make_step_image()
        out = self.system.dlss(frame, zero_motion(8, 8), constant_depth(8, 8),
                               scale=2)
        assert len(out) == 16 and len(out[0]) == 16

    def test_case_09(self):
        """DLSS：使用运动矢量重投影历史帧，运动场景无鬼影"""
        h, w = 12, 12
        frame0 = [[stripes(x) for x in range(w)] for _ in range(h)]
        frame1 = [[stripes(x - 1) for x in range(w)] for _ in range(h)]  # 右移 1 像素
        depth = constant_depth(h, w)
        mv_correct = constant_motion(h, w, 1.0, 0.0)
        mv_zero = zero_motion(h, w)
        kwargs = dict(scale=1, detail_strength=0.0, sharpen_strength=0.0)

        good = Antialiasing()
        good.dlss(frame0, mv_zero, depth, **kwargs)
        out_correct = good.dlss(frame1, mv_correct, depth, **kwargs)

        bad = Antialiasing()
        bad.dlss(frame0, mv_zero, depth, **kwargs)
        out_wrong = bad.dlss(frame1, mv_zero, depth, **kwargs)

        # 正确的运动矢量让历史帧精确对齐，结果应等于当前帧
        assert mse(out_correct, frame1) < 1e-9
        # 错误的（零）运动矢量产生鬼影，误差明显更大
        assert mse(out_wrong, frame1) > 0.01

    def test_case_10(self):
        """DLSS：深度不一致时丢弃历史（去遮挡），深度一致时累积历史"""
        h, w = 8, 8
        frame0 = make_flat_image(h, w, 0.4)
        frame1 = [[0.9 if (2 <= y < 6 and 2 <= x < 6) else 0.6
                   for x in range(w)] for y in range(h)]
        depth0 = constant_depth(h, w, 0.5)
        depth1 = [[0.9 if (2 <= y < 6 and 2 <= x < 6) else 0.5
                   for x in range(w)] for y in range(h)]
        mv = zero_motion(h, w)
        kwargs = dict(scale=1, detail_strength=0.0, sharpen_strength=0.0)

        with_depth = Antialiasing()
        with_depth.dlss(frame0, mv, depth0, **kwargs)
        out_depth = with_depth.dlss(frame1, mv, depth1, **kwargs)

        no_depth = Antialiasing()
        no_depth.dlss(frame0, mv, depth0, **kwargs)
        out_flat = no_depth.dlss(frame1, mv, constant_depth(h, w, 0.5), **kwargs)

        # 深度不匹配区域：历史被丢弃，结果等于当前帧新值 0.9
        assert out_depth[4][4] == pytest.approx(0.9, abs=1e-6)
        # 不做深度校验时：旧值 0.4 与新值 0.9 混合（0.7*0.4 + 0.3*0.9 = 0.55）
        assert out_flat[4][4] == pytest.approx(0.55, abs=1e-6)
        # 深度一致区域：两帧均为 0.5，历史正常累积
        assert out_depth[0][0] == pytest.approx(0.7 * 0.4 + 0.3 * 0.6, abs=1e-6)

    # ------------------------------------------------------------------
    # TAA：历史缓冲 + 重投影
    # ------------------------------------------------------------------
    def test_case_11(self):
        """TAA：维护历史缓冲，每帧累积"""
        frame = make_step_image()
        self.system.taa(frame)
        assert len(self.system._history) == 1
        self.system.taa(frame)
        assert len(self.system._history) == 2
        assert "taa_history" in self.system._state

    def test_case_12(self):
        """TAA：累积历史帧，静态场景的时序噪声被抑制"""
        rng = random.Random(7)
        clean = [[0.3 + 0.4 * (x / 15.0) for x in range(16)] for _ in range(16)]
        last_frame = None
        out = None
        for _ in range(10):
            noisy = [[min(1.0, max(0.0, clean[y][x] + rng.uniform(-0.2, 0.2)))
                      for x in range(16)] for y in range(16)]
            last_frame = noisy
            out = self.system.taa(noisy)
        # 累积结果比单帧更接近干净画面
        assert mse(out, clean) < mse(last_frame, clean) * 0.5

    def test_case_13(self):
        """TAA：按运动矢量重投影历史帧，运动场景跟踪正确"""
        h, w = 16, 16
        frames = [[[stripes(x - t) for x in range(w)] for y in range(h)]
                  for t in range(6)]
        mv_correct = constant_motion(h, w, 1.0, 0.0)
        mv_zero = zero_motion(h, w)

        good = Antialiasing()
        bad = Antialiasing()
        out_correct = out_wrong = None
        for frame in frames:
            out_correct = good.taa(frame, mv_correct)
            out_wrong = bad.taa(frame, mv_zero)

        # 正确重投影的误差应显著小于不重投影（鬼影）的误差
        assert mse(out_correct, frames[-1]) < mse(out_wrong, frames[-1]) * 0.5

    # ------------------------------------------------------------------
    # FSR：边缘自适应 + 锐化
    # ------------------------------------------------------------------
    def test_case_14(self):
        """FSR：边缘自适应插值，对角边缘的锯齿少于双线性"""
        n = 16
        diagonal = [[1.0 if x >= y else 0.0 for x in range(n)] for y in range(n)]
        fsr_out = self.system.fsr(diagonal, scale=2, sharpen_strength=0.0)
        baseline = bilinear_upscale(diagonal, 2)
        assert len(fsr_out) == 2 * n and len(fsr_out[0]) == 2 * n
        # 沿边缘方向的像素差异：边缘自适应平滑后明显更小
        assert along_edge_variation(fsr_out) < along_edge_variation(baseline) * 0.8

    def test_case_15(self):
        """FSR：锐化提升边缘对比度"""
        image = make_step_image()
        soft = self.system.fsr(image, scale=2, sharpen_strength=0.0)
        sharp = self.system.fsr(image, scale=2, sharpen_strength=0.8)
        assert len(sharp) == len(soft) and len(sharp[0]) == len(soft[0])
        assert max_horizontal_gradient(sharp) > max_horizontal_gradient(soft)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
