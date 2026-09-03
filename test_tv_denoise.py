"""数值验证：python -m unittest -v test_tv_denoise.py。

用伴随恒等式和已知解析解验证算法，再检查指标、未裁剪观测和可复现性。
测试只依赖 NumPy，不需要 scikit-image 或真实图片。
"""

import unittest

import numpy as np

from tv_denoise import (
    gradient, gradient_adjoint, psnr, run_experiments, ssim, synthetic_image, tv_pdhg,
)


class TVNumericalTests(unittest.TestCase):
    """针对数值正确性检查，避免把某个固定 PSNR 分数当作通用门槛。"""

    def test_synthetic_denoising_improves_quality(self):
        """保留原有裁剪输入的回归场景，确认兼容旧调用并改善两项指标。"""
        clean = synthetic_image()
        self.assertEqual(clean.shape, (256, 256))
        noisy = np.clip(clean + np.random.default_rng(42).normal(0.0, 0.10, clean.shape),
                        0.0, 1.0)
        result, info = tv_pdhg(noisy)
        self.assertTrue(info["converged"])
        self.assertLessEqual(info["relative_gap"], 1e-4)
        self.assertTrue(np.isfinite(result).all())
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)
        self.assertGreater(psnr(clean, result), psnr(clean, noisy))
        self.assertGreater(ssim(clean, result), ssim(clean, noisy))

    def test_gradient_adjoint_identity_including_boundaries(self):
        """在非方形图像上验证内积恒等式，随机边界值也必须正确配对。"""
        rng = np.random.default_rng(7)
        u = rng.normal(size=(13, 17))
        p = rng.normal(size=(2, 13, 17))
        np.testing.assert_allclose(np.sum(gradient(u) * p),
                                   np.sum(u * gradient_adjoint(p)), atol=1e-12)

    def test_constant_is_exact_minimizer(self):
        """常量图的 TV 为零，无噪声时应原样保留。"""
        u = np.full((16, 18), 0.37)
        result, info = tv_pdhg(u)
        np.testing.assert_allclose(result, u, atol=1e-14)
        self.assertTrue(info["converged"])

    def test_two_plateaus_against_analytic_solution(self):
        """用两块常量区域的解析最优解检验符号、TV 权重和收敛。"""
        # 12 行、每块 6 列；最优值满足 72*a-12*weight=0，
        # 72*(b-1)+12*weight=0，因此 a=weight/6、b=1-weight/6。
        f = np.zeros((12, 12))
        f[:, 6:] = 1.0
        weight = 0.2
        expected = np.empty_like(f)
        expected[:, :6] = weight / 6
        expected[:, 6:] = 1 - weight / 6
        result, info = tv_pdhg(f, weight=weight, max_iter=5000, tol=1e-9)
        self.assertTrue(info["converged"])
        np.testing.assert_allclose(result, expected, atol=2e-6)
        for row in info["history"]:
            self.assertGreaterEqual(row["primal"] + 1e-10, row["dual"])

    def test_psnr_known_error(self):
        """均方误差为 0.01 时 PSNR 必须为 20 dB。"""
        reference = np.zeros((16, 16))
        self.assertAlmostEqual(psnr(reference, reference + 0.1), 20.0)
        self.assertTrue(np.isinf(psnr(reference, reference)))

    def test_ssim_identity_and_constant_formula(self):
        """检查相同图像、常量图的解析 SSIM，以及输入交换对称性。"""
        rng = np.random.default_rng(10)
        u = rng.random((21, 23))
        self.assertAlmostEqual(ssim(u, u), 1.0, places=12)
        a, b = np.full((21, 23), 0.3), np.full((21, 23), 0.7)
        expected = (2 * 0.3 * 0.7 + 0.01**2) / (0.3**2 + 0.7**2 + 0.01**2)
        self.assertAlmostEqual(ssim(a, b), expected, places=11)
        self.assertAlmostEqual(ssim(a, u), ssim(u, a), places=12)

    def test_invalid_step_sizes_are_rejected(self):
        """不满足 PDHG 稳定条件的步长必须显式报错。"""
        with self.assertRaises(ValueError):
            tv_pdhg(np.zeros((12, 12)), tau=1.0, sigma=1.0)

    def test_unclipped_observation_against_analytic_solution(self):
        """超范围观测必须保留在保真项中：预先裁剪会得到不同的解析解。"""
        observed = np.full((12, 12), -0.01)
        observed[:, 6:] = 1.01
        before = observed.copy()
        expected = np.full_like(observed, -0.01 + 0.2 / 6)
        expected[:, 6:] = 1.01 - 0.2 / 6
        result, info = tv_pdhg(observed, weight=0.2, max_iter=5000, tol=1e-9)
        np.testing.assert_array_equal(observed, before)
        np.testing.assert_allclose(result, expected, atol=2e-6)
        self.assertTrue(info["converged"])
        self.assertLessEqual(info["relative_gap"], 1e-9)
        for row in info["history"]:
            self.assertGreaterEqual(row["primal"] + 1e-10, row["dual"])

    def test_out_of_range_constants_obey_box_constraint(self):
        """全负或全大于 1 的常量观测，其约束最优解分别是 0 或 1。"""
        for value in (-0.2, 1.2):
            with self.subTest(value=value):
                result, info = tv_pdhg(np.full((12, 13), value))
                np.testing.assert_allclose(result, np.clip(value, 0, 1), atol=1e-14)
                self.assertTrue(info["converged"])

    def test_iteration_limit_is_reported_truthfully(self):
        """故意只允许一次迭代，不能把未达到容差的结果标记为收敛。"""
        observed = np.random.default_rng(9).normal(0.5, 0.3, (17, 19))
        _, info = tv_pdhg(observed, max_iter=1, tol=1e-12)
        self.assertFalse(info["converged"])
        self.assertEqual(info["iterations"], 1)
        self.assertGreater(info["relative_gap"], 1e-12)

    def test_independent_noise_and_reproducibility(self):
        """同一配置产生相同数组，各组的标准化噪声不同，且噪声没有被裁剪。"""
        clean = synthetic_image(32)
        first = run_experiments(clean, [10, 25, 50], seed=42)
        second = run_experiments(clean, [10, 25, 50], seed=42)
        np.testing.assert_array_equal(first[1], second[1])
        np.testing.assert_array_equal(first[2], second[2])
        self.assertEqual(first[1].shape, (3, 32, 32))
        self.assertLess(first[1][2].min(), 0)
        self.assertGreater(first[1][2].max(), 1)
        self.assertGreaterEqual(first[2].min(), 0)
        self.assertLessEqual(first[2].max(), 1)
        z0 = (first[1][0] - clean) / (10 / 255)
        z1 = (first[1][1] - clean) / (25 / 255)
        self.assertFalse(np.allclose(z0, z1))
        for row in first[0]:
            self.assertTrue(row["converged"])

    def test_zero_noise_default_is_identity(self):
        """兼容旧参数允许零噪声的情形：默认 λ=0，应精确保留参考图。"""
        clean = synthetic_image(16)
        records, noisy, result = run_experiments(clean, [0])
        np.testing.assert_array_equal(noisy[0], clean)
        np.testing.assert_allclose(result[0], clean, atol=1e-15)
        self.assertTrue(records[0]["converged"])

    def test_metrics_do_not_clip_observation(self):
        """观测全部为 2、参考为 0 时，MSE=4，不能偷偷裁剪成 MSE=1。"""
        reference = np.zeros((16, 16))
        self.assertAlmostEqual(psnr(reference, reference + 2), -10 * np.log10(4))


if __name__ == "__main__":
    unittest.main()
