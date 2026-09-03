"""使用 NumPy 手写 PDHG，完成各向同性 TV 图像去噪实验。

运行 python tv_denoise.py：读取 test_image.png，分别加入 sigma=10、25、50
的高斯噪声，计算 PSNR/SSIM，并将对比图和实验数据保存到 results/。
sigma 使用 0–255 灰度单位；算法内部使用归一化强度。只需 NumPy、Matplotlib。

阅读顺序：
1. load_image / add_gaussian_noise：准备清晰图和带噪观测。
2. gradient / gradient_adjoint：定义离散梯度及其伴随。
3. tv_pdhg / primal_dual_values：交替更新图像和对偶变量，并检查收敛。
4. psnr / ssim：与清晰图比较，衡量像素误差和局部结构相似性。
5. run_experiments / save_results / main：组织实验、保存数据和解析参数。

清晰图只用于模拟加噪与评价，tv_pdhg 只接收噪声图和算法参数。
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import platform
from time import perf_counter

import numpy as np


def gradient(u: np.ndarray) -> np.ndarray:
    """计算前向差分 Ku，返回形状 (2, H, W) 的水平、垂直梯度。

    最后一列的水平差分、最后一行的垂直差分为零（Neumann 边界），
    不把图像的两侧连起来，因此不会引入周期边界产生的虚假边缘。
    例如一行 [0.2, 0.5, 0.4] 的水平差分为 [0.3, -0.1, 0]。
    """
    g = np.zeros((2, *u.shape), dtype=np.float64)
    g[0, :, :-1] = u[:, 1:] - u[:, :-1]
    g[1, :-1, :] = u[1:, :] - u[:-1, :]
    return g


def gradient_adjoint(p: np.ndarray) -> np.ndarray:
    """计算与 gradient 严格配对的伴随算子 K*，它等于负散度。

    满足内积关系 <Ku,p>=<u,K*p>；正负号和边界必须与前向差分一致。
    每条有效差分边对起点贡献 -p，对终点贡献 +p。
    """
    out = np.zeros(p.shape[1:], dtype=np.float64)
    out[:, :-1] -= p[0, :, :-1]
    out[:, 1:] += p[0, :, :-1]
    out[:-1, :] -= p[1, :-1, :]
    out[1:, :] += p[1, :-1, :]
    return out


def primal_dual_values(
    u: np.ndarray, p: np.ndarray, noisy: np.ndarray, weight: float
) -> tuple[float, float]:
    """返回原始目标 P(u) 和对偶下界 D(p)，用于判断收敛。

    调用时要求 u 在 [0,1] 内、每个像素处 ||p||₂ <= weight。
    P-D 越接近零，当前解与最优解的目标值差距上界越小。
    """
    g = gradient(u)
    primal = 0.5 * np.sum((u - noisy) ** 2)
    primal += weight * np.sum(np.hypot(g[0], g[1]))
    # F(u)=0.5||u-noisy||²+I_[0,1](u)，I 是可行集的示性函数。
    # D(p)=-F*(-K*p)；F*(s)=max_v <s,v>-F(v)，最大点是 clip(noisy+s)。
    s = -gradient_adjoint(p)
    v = np.clip(noisy + s, 0.0, 1.0)
    dual = -np.sum(s * v - 0.5 * (v - noisy) ** 2)
    return float(primal), float(dual)


def tv_pdhg(
    noisy: np.ndarray,
    weight: float = 0.10,
    max_iter: int = 3000,
    tol: float = 1e-4,
    tau: float = 0.35,
    sigma: float = 0.35,
    check_every: int = 25,
) -> tuple[np.ndarray, dict]:
    """求解 min_{0<=u<=1} 0.5||u-noisy||² + weight*TV(u)。

    参数：noisy 是二维浮点观测，原图归一化后加噪，可超出 [0,1]；
    weight 是 TV 权重 λ（越大通常越平滑）；max_iter 是最大迭代次数；
    tol 是相对原始—对偶间隙容差；check_every 是间隙检查间隔。
    tau、sigma 是原始/对偶步长，这里的 sigma 不是高斯噪声标准差。
    返回去噪图，以及迭代次数、收敛状态、间隙、耗时和历史记录。

    TV(u)=sum(sqrt(dx²+dy²)) 为各向同性 TV。采用外推系数 theta=1，
    步长须满足 tau*sigma*||K||²<1；此处 ||K||²<=8。
    默认步长满足 8*0.35*0.35=0.98<1。TV 项抑制灰度波动，
    二次保真项限制图像偏离观测的程度；lambda 越大，平滑作用越强。

    PDHG（原始—对偶混合梯度法）使用 TV 的对偶表示：
    weight*TV(u) = max_{||p[i,j]||₂<=weight} <Ku,p>。
    u 是待恢复的 (H,W) 图像；p 是 (2,H,W) 向量场，每个像素有两个分量。
    将非光滑 TV 转为向量场的球约束后，可交替做对偶投影和图像近端更新。
    求解器和停止条件都不需要清晰参考图。
    """
    f = np.asarray(noisy, dtype=np.float64)
    if f.ndim != 2 or min(f.shape) < 2 or not np.isfinite(f).all():
        raise ValueError("noisy must be a finite 2D image of size at least 2x2")
    if not np.isfinite(weight) or weight < 0:
        raise ValueError("weight must be finite and nonnegative")
    if (not isinstance(max_iter, (int, np.integer))
            or not isinstance(check_every, (int, np.integer))
            or max_iter < 1 or check_every < 1 or not np.isfinite(tol) or tol <= 0):
        raise ValueError("iteration counts and tolerance must be positive")
    if not (tau > 0 and sigma > 0 and 8 * tau * sigma < 1):
        raise ValueError("step sizes must satisfy tau>0, sigma>0, 8*tau*sigma<1")

    # 只约束待求解的图像 u；不能裁剪观测 f，否则会改变数据保真项。
    u = np.clip(f, 0.0, 1.0)
    u_bar = u.copy()
    p = np.zeros((2, *f.shape), dtype=np.float64)
    history = []
    converged = False
    started = perf_counter()

    for iteration in range(1, max_iter + 1):
        # 1. 对偶上升，再逐像素投影到半径 λ 的二维欧氏球。
        #    TV 的对偶约束是 ||p[i,j]||₂<=λ，不是逐分量裁剪。
        p += sigma * gradient(u_bar)
        if weight == 0:
            p.fill(0.0)  # λ=0 时只有二次保真项，精确解为 clip(f)。
        else:
            # hypot 得到 (H,W) 的向量长度；[None,:,:] 让两个分量共用缩放因子。
            # 球内向量不动，球外向量按原方向缩短到长度 λ。
            p /= np.maximum(1.0, np.hypot(p[0], p[1]) / weight)[None, :, :]

        # 2. 原始近端更新：二次项给出加权平均，区间约束给出 clip。
        #    令 z=u-tau*K*p，最小化 0.5||v-z||² + tau/2*||v-f||²，
        #    对 v 求导得到 (1+tau)*v=z+tau*f，再逐像素限制到 [0,1]。
        u_new = (u - tau * gradient_adjoint(p) + tau * f) / (1.0 + tau)
        u_new = np.clip(u_new, 0.0, 1.0)
        # 3. 外推，下一次对偶更新使用 u_bar；此处 theta=1。
        u_bar = 2.0 * u_new - u
        u = u_new

        if iteration == 1 or iteration % check_every == 0 or iteration == max_iter:
            primal, dual = primal_dual_values(u, p, f, weight)
            # 理论上 P>=D；max(0, ...) 仅消除舍入导致的微小负间隙。
            # 分母至少为 1，避免常量图等低目标值场景除零；不使用 PSNR 停止。
            gap = max(0.0, primal - dual) / max(1.0, abs(primal))
            history.append({"iteration": iteration, "primal": primal,
                            "dual": dual, "relative_gap": gap})
            if gap <= tol:
                converged = True
                break

    return u, {"iterations": iteration, "converged": converged,
               "relative_gap": gap, "seconds": perf_counter() - started,
               "history": history}


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """计算 PSNR=10*log10(1/MSE)，单位 dB，固定 data_range=1。

    即使观测超出 [0,1]，峰值仍用参考图的标称动态范围 1；不裁剪观测。
    两图完全一致时返回正无穷。均方误差越小，PSNR 越高；
    例如 MSE=0.01 时 PSNR=20 dB，MSE=0.001 时 PSNR=30 dB。
    """
    a, b = _metric_inputs(reference, estimate)
    mse = float(np.mean((a - b) ** 2))
    return float("inf") if mse == 0 else float(-10.0 * np.log10(mse))


def _metric_inputs(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """检查指标输入同形、二维、有限，并转为 float64，防止整数运算溢出。"""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.size == 0:
        raise ValueError("metrics require nonempty 2D images of identical shape")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("metrics require finite images")
    return a, b


def _gaussian_valid(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """先水平、再垂直做可分离高斯滤波，只保留完整窗口。

    两次一维加权求和等价于二维高斯卷积，无需 SciPy；不填充图像边缘。
    """
    n = len(kernel)
    h, w = image.shape
    horizontal = np.zeros((h, w - n + 1), dtype=np.float64)
    for offset, coefficient in enumerate(kernel):
        horizontal += coefficient * image[:, offset:offset + w - n + 1]
    output = np.zeros((h - n + 1, w - n + 1), dtype=np.float64)
    for offset, coefficient in enumerate(kernel):
        output += coefficient * horizontal[offset:offset + h - n + 1, :]
    return output


def ssim(reference: np.ndarray, estimate: np.ndarray) -> float:
    """计算局部 SSIM，再对所有有效位置取平均，固定 data_range=1。

    使用 11×11 高斯窗口（标准差 1.5）、总体方差/协方差，
    C1=(0.01*1)²、C2=(0.03*1)²，排除周围 5 像素的不完整窗口。
    SSIM=((2μaμb+C1)(2cov+C2))/((μa²+μb²+C1)(var_a+var_b+C2))。
    输入不裁剪；这不是只对整张图计算一次均值方差的近似。
    μ 表示局部平均亮度，方差反映局部对比度，协方差反映共同变化；
    C1、C2 保证常量窗口的分母非零。值越接近 1，局部结构通常越相似。
    """
    a, b = _metric_inputs(reference, estimate)
    if min(a.shape) < 11:
        raise ValueError("SSIM requires images of size at least 11x11")
    x = np.arange(-5, 6, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / 1.5) ** 2)
    kernel /= kernel.sum()
    mu_a, mu_b = _gaussian_valid(a, kernel), _gaussian_valid(b, kernel)
    # Var(a)=E[a²]-E[a]²；消除浮点舍入造成的微小负方差。
    var_a = np.maximum(0.0, _gaussian_valid(a * a, kernel) - mu_a * mu_a)
    var_b = np.maximum(0.0, _gaussian_valid(b * b, kernel) - mu_b * mu_b)
    covariance = _gaussian_valid(a * b, kernel) - mu_a * mu_b
    numerator = (2 * mu_a * mu_b + 0.01**2) * (2 * covariance + 0.03**2)
    denominator = (mu_a**2 + mu_b**2 + 0.01**2) * (var_a + var_b + 0.03**2)
    return float(np.mean(numerator / denominator))


def synthetic_image(size: int = 256) -> np.ndarray:
    """生成含几何边缘、渐变和条纹的灰度图，用于可选演示和数值测试。"""
    if size < 2:
        raise ValueError("synthetic image size must be at least 2")
    y, x = np.mgrid[0:size, 0:size].astype(np.float64) / (size - 1)
    u = 0.15 + 0.12 * x + 0.06 * y
    u[(x > 0.10) & (x < 0.44) & (y > 0.12) & (y < 0.43)] = 0.75
    u[((x - 0.72) / 0.17)**2 + ((y - 0.28) / 0.18)**2 < 1] = 0.48
    u[((x - 0.30) / 0.19)**2 + ((y - 0.72) / 0.17)**2 < 1] = 0.86
    u[((x - 0.30) / 0.07)**2 + ((y - 0.72) / 0.06)**2 < 1] = 0.30
    mask = (x > 0.57) & (x < 0.90) & (y > 0.56) & (y < 0.88)
    u[mask] = (0.50 + 0.16 * np.sin(18 * np.pi * x))[mask]
    return u


def load_image(path: Path) -> np.ndarray:
    """从磁盘读取参考图片并转换成 [0,1] 内的二维 float64 灰度图。

    整数图按 dtype 的最大值归一化；PNG 浮点读数已经在 [0,1] 内。
    RGB 使用亮度权重转灰度；RGBA 先合成到白色背景，不使用透明通道作灰度。
    不缩放图像尺寸；至少需要 11×11 像素才能计算 SSIM。
    """
    from matplotlib.image import imread

    raw = imread(path)
    clean = np.asarray(raw, dtype=np.float64)
    if np.issubdtype(raw.dtype, np.integer):
        clean /= np.iinfo(raw.dtype).max
    if clean.ndim == 3 and clean.shape[2] in (3, 4):
        if clean.shape[2] == 4:
            alpha = clean[:, :, 3:4]
            clean = clean[:, :, :3] * alpha + (1.0 - alpha)
        clean = clean[:, :, :3] @ np.array([0.2126, 0.7152, 0.0722])
    if (clean.ndim != 2 or min(clean.shape) < 11 or not np.isfinite(clean).all()
            or clean.min() < 0 or clean.max() > 1):
        raise ValueError("input must be a grayscale/RGB/RGBA image >=11x11 in [0,1]")
    return clean


def add_gaussian_noise(
    clean: np.ndarray, noise_sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """加入独立高斯噪声：f=clean+N(0,(noise_sigma/255)²)，不裁剪。

    noise_sigma 的单位是 0–255 灰度强度，输入 clean 使用 [0,1] 强度。
    例如 noise_sigma=25 对应归一化标准差约 0.098，而非方差 25/255。
    不裁剪是为了保留设定的高斯噪声分布；只有显示时才将超范围值饱和。
    rng 由调用方提供，使随机种子可控，不改变 NumPy 的全局随机状态。
    """
    if not np.isfinite(noise_sigma) or noise_sigma < 0:
        raise ValueError("noise sigma must be finite and nonnegative")
    return clean + rng.normal(0.0, noise_sigma / 255.0, clean.shape)


def run_experiments(
    clean: np.ndarray, sigmas: list[float], seed: int = 42,
    weight: float | None = None, max_iter: int = 3000, tol: float = 1e-4,
) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """依次执行多组实验，返回记录列表、噪声图堆叠、去噪图堆叠。

    后两者形状均为 (实验数, H, W)，单组实验也保留实验维度。
    每组使用独立随机子流；相同种子、相同参数及顺序得到相同数组。
    未指定 weight 时固定用 λ=0.8*sigma/255，不根据参考图指标选择 λ。
    """
    clean = np.asarray(clean, dtype=np.float64)
    if (clean.ndim != 2 or min(clean.shape) < 11 or not np.isfinite(clean).all()
            or clean.min() < 0 or clean.max() > 1):
        raise ValueError("clean must be a finite 2D reference >=11x11 in [0,1]")
    if len(sigmas) == 0:
        raise ValueError("at least one noise sigma is required")
    records, noisy_images, denoised_images = [], [], []
    # SeedSequence.spawn 为不同实验分配子流，而不是重复使用同一幅噪声。
    child_seeds = np.random.SeedSequence(seed).spawn(len(sigmas))
    for noise_sigma, child_seed in zip(sigmas, child_seeds):
        noise_sigma = float(noise_sigma)
        noisy = add_gaussian_noise(clean, noise_sigma, np.random.default_rng(child_seed))
        actual_weight = 0.8 * noise_sigma / 255.0 if weight is None else weight
        # 求解完成后才用 clean 评价；参数规则不依赖评价分数。
        restored, info = tv_pdhg(noisy, actual_weight, max_iter, tol)
        records.append({
            "sigma": noise_sigma, "noise_std": noise_sigma / 255.0,
            "weight": actual_weight, "seed_spawn_key": list(child_seed.spawn_key),
            "noisy_psnr_db": psnr(clean, noisy), "noisy_ssim": ssim(clean, noisy),
            "denoised_psnr_db": psnr(clean, restored), "denoised_ssim": ssim(clean, restored),
            **info,
        })
        noisy_images.append(noisy)
        denoised_images.append(restored)
    return records, np.stack(noisy_images), np.stack(denoised_images)


def save_results(
    clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray,
    records: list[dict], metadata: dict, paths: dict[str, Path],
) -> None:
    """保存对比图、CSV、JSON 和 NPZ，展示数据与计算数据明确区分。

    图中统一使用灰度 [0,1]，超范围值仅在显示时饱和；NPZ 保留全部精度。
    CSV 每行对应一个 sigma；JSON 额外保留运行环境、完整迭代记录等信息。
    paths 的 figure、metrics、report、arrays 分别指向 PNG、CSV、JSON、NPZ。
    """
    import matplotlib
    matplotlib.use("Agg")  # 无窗口后端，支持服务器和批量运行。
    import matplotlib.pyplot as plt

    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig, axes = plt.subplots(len(records), 3, squeeze=False,
                                 figsize=(12, 4.25 * len(records) + 0.65),
                                 layout="constrained")
        for index, row in enumerate(records):
            titles = [
                f"Original\nNoise sigma = {row['sigma']:g} (0-255 units)",
                f"Gaussian noise\nPSNR {row['noisy_psnr_db']:.2f} dB | SSIM {row['noisy_ssim']:.4f}",
                f"TV-PDHG | lambda = {row['weight']:.4f}\n"
                f"PSNR {row['denoised_psnr_db']:.2f} dB | SSIM {row['denoised_ssim']:.4f}",
            ]
            for ax, im, title in zip(axes[index], [clean, noisy[index], denoised[index]], titles):
                ax.imshow(im, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(title, fontsize=11, pad=8)
                ax.axis("off")
        fig.suptitle(f"Isotropic TV denoising | independent Gaussian noise | seed = {metadata['seed']}\n"
                     "Unclipped observations for computation; display range [0, 1]", fontsize=13)
        fig.savefig(paths["figure"], dpi=160, facecolor="white")
        plt.close(fig)

    # PNG 是显示图，不能用重新读取 PNG 的方式复核原始浮点指标。
    np.savez_compressed(paths["arrays"], original=clean, noisy=noisy, denoised=denoised,
                        sigmas=np.array([row["sigma"] for row in records]))
    fields = ["sigma", "noise_std", "weight", "noisy_psnr_db", "noisy_ssim",
              "denoised_psnr_db", "denoised_ssim", "iterations", "converged",
              "relative_gap", "seconds"]
    with paths["metrics"].open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    # 标准 JSON 不支持无穷大：零噪声等情况下将无限 PSNR 写作字符串 Infinity。
    json_records = [dict(row) for row in records]
    for row in json_records:
        for key in ("noisy_psnr_db", "denoised_psnr_db"):
            if np.isposinf(row[key]):
                row[key] = "Infinity"
    report = {
        **metadata, "shape": list(clean.shape), "noise_clipped": False,
        "data_range": 1.0, "display_range": [0, 1],
        "weight_rule": "0.8 * sigma / 255" if metadata["weight_override"] is None else "fixed override",
        "solver": {"method": "PDHG", "tau": 0.35, "sigma_dual": 0.35,
                   "theta": 1.0, "check_every": 25},
        "ssim": {"window_size": 11, "gaussian_sigma": 1.5, "K1": 0.01,
                 "K2": 0.03, "use_sample_covariance": False, "border_excluded": 5},
        "random_generator": "PCG64; SeedSequence(seed).spawn(number_of_experiments)",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "matplotlib": matplotlib.__version__},
        "experiments": json_records,
    }
    paths["report"].write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
                               encoding="utf-8")


def main() -> int:
    """解析命令行，读取参考图，运行全部实验并保存结果。正常完成返回 0。"""
    base = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="清晰参考图路径；默认读取脚本旁的 test_image.png")
    source.add_argument("--synthetic", action="store_true", help="改用 256×256 合成参考图")
    noise = parser.add_mutually_exclusive_group()
    noise.add_argument("--sigmas", type=float, nargs="+", help="0–255 单位的噪声标准差；默认 10 25 50")
    noise.add_argument("--noise-std", type=float, help="旧式单组参数：归一化后的噪声标准差")
    parser.add_argument("--seed", type=int, default=42, help="非负整数随机种子，默认 42")
    parser.add_argument("--weight", type=float, help="所有组共用的 TV 权重；默认各取 0.8*sigma/255")
    parser.add_argument("--max-iter", type=int, default=3000, help="最大迭代次数，默认 3000")
    parser.add_argument("--tol", type=float, default=1e-4, help="相对原始—对偶间隙容差，默认 1e-4")
    destination = parser.add_mutually_exclusive_group()
    destination.add_argument("--output-dir", type=Path, help="输出目录，默认脚本旁的 results/")
    destination.add_argument("--output", type=Path, help="旧式输出参数：PNG 路径，旁边保存同名 CSV/JSON/NPZ")
    args = parser.parse_args()

    # --noise-std 使用归一化单位，--sigmas 使用 0–255 单位；先统一为后者。
    # 之后 add_gaussian_noise 再除以 255，避免把 25 当成归一化标准差。
    sigmas = ([args.noise_std * 255.0] if args.noise_std is not None
              else args.sigmas if args.sigmas is not None else [10.0, 25.0, 50.0])
    if any(not np.isfinite(value) or value < 0 for value in sigmas):
        parser.error("noise standard deviations must be finite and nonnegative")
    if args.seed < 0:
        parser.error("--seed must be nonnegative")
    if args.weight is not None and (not np.isfinite(args.weight) or args.weight < 0):
        parser.error("--weight must be finite and nonnegative")
    if args.max_iter < 1 or not np.isfinite(args.tol) or args.tol <= 0:
        parser.error("--max-iter and --tol must be positive")

    if args.output is not None:
        output = args.output.resolve()
        if output.suffix.lower() != ".png":
            parser.error("--output must be a .png path")
        paths = {"figure": output, "metrics": output.with_suffix(".csv"),
                 "report": output.with_suffix(".json"), "arrays": output.with_suffix(".npz")}
    else:
        directory = (args.output_dir or base / "results").resolve()
        paths = {"figure": directory / "comparison.png", "metrics": directory / "metrics.csv",
                 "report": directory / "report.json", "arrays": directory / "images.npz"}

    # 字体缓存放在项目的忽略目录内；在导入绘图库之前设置。
    os.environ.setdefault("MPLCONFIGDIR", str(base / ".setup" / "matplotlib-cache"))
    image_path = (args.image or base / "test_image.png").resolve()
    if not args.synthetic and image_path in paths.values():
        parser.error("output must not overwrite the reference image")
    try:
        clean = synthetic_image() if args.synthetic else load_image(image_path)
        image_source = "NumPy synthetic phantom" if args.synthetic else str(image_path)
        print(f"Input: {image_source} | shape={clean.shape}", flush=True)
        print(f"Noise sigmas (0-255): {sigmas} | seed={args.seed} | observations NOT clipped", flush=True)
        records, noisy, denoised = run_experiments(clean, sigmas, args.seed, args.weight,
                                                  args.max_iter, args.tol)
        metadata = {"image_source": image_source, "seed": args.seed,
                    "max_iter": args.max_iter, "tol": args.tol, "weight_override": args.weight}
        save_results(clean, noisy, denoised, records, metadata, paths)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(f"{'sigma':>6} {'lambda':>9} {'noisy PSNR':>12} {'TV PSNR':>10} "
          f"{'noisy SSIM':>12} {'TV SSIM':>10} {'iterations':>11} {'converged':>10}")
    for row in records:
        print(f"{row['sigma']:6g} {row['weight']:9.5f} {row['noisy_psnr_db']:12.4f} "
              f"{row['denoised_psnr_db']:10.4f} {row['noisy_ssim']:12.6f} "
              f"{row['denoised_ssim']:10.6f} {row['iterations']:11d} {str(row['converged']):>10}")
        if not row["converged"]:
            print(f"NOTE: sigma={row['sigma']:g} reached max_iter; relative gap="
                  f"{row['relative_gap']:.3e}. Increase --max-iter to meet the tolerance.")
    for label, path in paths.items():
        print(f"Saved {label}: {path}")
    # 完成实验即可正常退出；图像质量不是通用的程序成功/失败判据。
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
