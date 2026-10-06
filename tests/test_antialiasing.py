"""
游戏内抗锯齿与超分辨率系统 - 验收测试
运行方式：python -m pytest tests/test_antialiasing.py -q
共 15 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rendering.antialiasing import Antialiasing


def make_flat(rows=16, cols=16, value=0.5):
    return [[value] * cols for _ in range(rows)]


def make_step(rows=16, cols=16):
    """垂直硬边缘：左黑右白。"""
    return [[0.0 if x < cols // 2 else 1.0 for x in range(cols)]
            for _ in range(rows)]


def make_diagonal(n=16):
    """对角阶梯边缘（锯齿明显）。"""
    return [[1.0 if x > y else 0.0 for x in range(n)] for y in range(n)]


def make_pattern(rows=16, cols=16):
    """确定性的非均匀图案。"""
    return [[((x * 7 + y * 13) % 16) / 15.0 for x in range(cols)]
            for y in range(rows)]


def shift(image, dx, dy, fill=0.0):
    rows, cols = len(image), len(image[0])
    out = [[fill] * cols for _ in range(rows)]
    for y in range(rows):
        for x in range(cols):
            sx, sy = x - dx, y - dy
            if 0 <= sx < cols and 0 <= sy < rows:
                out[y][x] = image[sy][sx]
    return out


def make_motion(rows, cols, mvx, mvy):
    return [[(mvx, mvy)] * cols for _ in range(rows)]


def total_laplacian(image):
    rows, cols = len(image), len(image[0])
    total = 0.0
    for y in range(1, rows - 1):
        for x in range(1, cols - 1):
            lap = (4.0 * image[y][x] - image[y][x - 1] - image[y][x + 1]
                   - image[y - 1][x] - image[y + 1][x])
            total += abs(lap)
    return total


def max_gradient(image):
    rows, cols = len(image), len(image[0])
    best = 0.0
    for y in range(rows):
        for x in range(cols - 1):
            best = max(best, abs(image[y][x + 1] - image[y][x]))
    for y in range(rows - 1):
        for x in range(cols):
            best = max(best, abs(image[y + 1][x] - image[y][x]))
    return best


def mse(a, b):
    rows, cols = len(a), len(a[0])
    return sum((a[y][x] - b[y][x]) ** 2
               for y in range(rows) for x in range(cols)) / (rows * cols)


def bilinear_upscale(image, scale):
    sys_ref = Antialiasing()
    rows, cols = len(image), len(image[0])
    return [[sys_ref._sample(image, (x + 0.5) / scale - 0.5,
                             (y + 0.5) / scale - 0.5)
             for x in range(cols * scale)] for y in range(rows * scale)]


class TestAntialiasing:
    def setup_method(self):
        self.system = Antialiasing()

    # ---- 抗锯齿：边缘检测 ----
    def test_case_01(self):
        edges = self.system.detect_edges(make_step(), threshold=0.1)
        assert sum(sum(row) for row in edges) > 0.0
        # 边缘应集中在明暗交界列附近
        strong = [(x, y) for y, row in enumerate(edges)
                  for x, v in enumerate(row) if v > 0.0]
        assert all(abs(x - 8) <= 1 for x, _ in strong)

    def test_case_02(self):
        edges = self.system.detect_edges(make_flat(), threshold=0.1)
        assert all(v == 0.0 for row in edges for v in row)

    # ---- 抗锯齿：边缘模糊平滑 ----
    def test_case_03(self):
        image = make_diagonal()
        smoothed = self.system.apply_antialiasing(image)
        assert total_laplacian(smoothed) < total_laplacian(image)

    def test_case_04(self):
        image = make_flat(value=0.3)
        result = self.system.apply_antialiasing(image)
        assert result == image

    def test_case_05(self):
        image = make_diagonal()
        result = self.system.apply_antialiasing(image)
        assert len(result) == len(image)
        assert all(len(row) == len(image[0]) for row in result)

    # ---- 超分辨率：细节增强与锐化 ----
    def test_case_06(self):
        image = make_pattern(8, 8)
        result = self.system.super_resolution(image, scale=2)
        assert len(result) == 16
        assert all(len(row) == 16 for row in result)

    def test_case_07(self):
        image = make_step(16, 16)
        result = self.system.super_resolution(image, scale=2)
        reference = bilinear_upscale(image, 2)
        # 细节增强 + 锐化后，边缘对比度必须高于纯双线性插值
        assert max_gradient(result) > max_gradient(reference)

    def test_case_08(self):
        image = make_flat(8, 8, value=0.6)
        result = self.system.super_resolution(image, scale=2)
        for row in result:
            for v in row:
                assert abs(v - 0.6) < 1e-6

    # ---- DLSS：运动矢量与深度 ----
    def test_case_09(self):
        image = make_pattern(8, 8)
        result = self.system.dlss(image, scale=2)
        assert len(result) == 16
        assert all(len(row) == 16 for row in result)

    def test_case_10(self):
        frame1 = make_pattern(8, 8)
        frame2 = shift(frame1, 2, 0)
        mv = make_motion(8, 8, 2.0, 0.0)

        self.system.dlss(frame1, scale=2)
        with_mv = self.system.dlss(frame2, scale=2, motion_vectors=mv)

        self.system.reset()
        self.system.dlss(frame1, scale=2)
        without_mv = self.system.dlss(frame2, scale=2)

        assert with_mv != without_mv

    def test_case_11(self):
        frame1 = make_pattern(8, 8)
        frame2 = shift(frame1, 1, 0)
        mv = make_motion(8, 8, 1.0, 0.0)
        depth_near = make_flat(8, 8, value=0.2)
        depth_far = make_flat(8, 8, value=0.9)

        self.system.dlss(frame1, scale=2, depth=depth_near)
        consistent = self.system.dlss(frame2, scale=2, motion_vectors=mv,
                                      depth=depth_near)

        self.system.reset()
        self.system.dlss(frame1, scale=2, depth=depth_near)
        rejected = self.system.dlss(frame2, scale=2, motion_vectors=mv,
                                    depth=depth_far)

        # 深度突变应触发历史拒绝，两次结果必须不同
        assert consistent != rejected

    # ---- TAA：历史缓冲与重投影 ----
    def test_case_12(self):
        image = make_pattern(8, 8)
        assert len(self.system._history) == 0
        for _ in range(3):
            self.system.taa(image)
        assert len(self.system._history) == 3
        for _ in range(10):
            self.system.taa(image)
        assert len(self.system._history) == 8  # 历史缓冲有上限

    def test_case_13(self):
        frame1 = make_pattern(16, 16)
        frame2 = shift(frame1, 1, 0)

        mv = make_motion(16, 16, 1.0, 0.0)
        self.system.taa(frame1)
        reproj = self.system.taa(frame2, motion_vectors=mv)

        self.system.reset()
        no_mv = make_motion(16, 16, 0.0, 0.0)
        self.system.taa(frame1)
        no_reproj = self.system.taa(frame2, motion_vectors=no_mv)

        # 正确的重投影应让累积结果更贴近当前帧（无拖影）
        assert mse(reproj, frame2) < mse(no_reproj, frame2)

    # ---- FSR：边缘自适应与锐化 ----
    def test_case_14(self):
        image = make_pattern(8, 8)
        result = self.system.fsr(image, scale=2)
        assert len(result) == 16
        assert all(len(row) == 16 for row in result)
        flat = self.system.fsr(make_flat(8, 8, value=0.4), scale=2)
        for row in flat:
            for v in row:
                assert abs(v - 0.4) < 1e-6

    def test_case_15(self):
        image = make_step(16, 16)
        result = self.system.fsr(image, scale=2)
        reference = bilinear_upscale(image, 2)
        # 边缘自适应插值的结果不同于朴素双线性
        assert result != reference
        # 锐化后边缘对比度高于纯双线性插值
        assert max_gradient(result) > max_gradient(reference)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
