# Copyright (c) MyoSuite Authors. All rights reserved.
#
# This source code is licensed under the Apache 2 license found in the
# LICENSE file in the root directory of this source tree.
"""Golden values of :class:`~myosuite.utils.colored_noise.ColoredNoiseProcess`.

The goalkeeper (Soccer) and opponent (ChaseTag) velocities are seeded draws of
this process. The values below were drawn with ``pink.ColoredNoiseProcess``
(pink-noise-rl 2.0.1, numpy 2.5.3), which these envs used before, so seeded
episodes stay the same without pink installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from myosuite.utils.colored_noise import ColoredNoiseProcess

pytestmark = pytest.mark.tier1

# FFT round-off may differ in the last bits between platforms (e.g. FMA on
# arm64); a change in buffering, draw order or scaling moves samples by ~0.1.
_ATOL = 1e-12

# beta=2, size=(2, 10), scale=0.7: 25 single samples (refills after 10 and 20).
_SEED0_BUFFER10 = np.array(
    [
        [0.13399628763850166, 0.55276881924472],
        [0.6785042385399466, 0.5513082960731157],
        [0.04201253447262993, 0.5578538465123545],
        [-0.21380444712990557, -0.04943070131031364],
        [0.0896338820521966, -0.6759220024471473],
        [0.1877834500449066, -0.06824823339381215],
        [0.4540905409602213, 0.7328514625258186],
        [-0.012518648203123865, 0.6560748498465185],
        [-0.3103684132149982, 1.544880708021496],
        [-0.5295644123462349, 1.5885605475390547],
        [-0.017200092871927034, -0.42005320670046875],
        [0.10018356700753657, -1.0271102079304726],
        [0.6425723130872565, -0.7768489655261491],
        [0.9009908756563878, -0.9039605856686597],
        [0.15622316176532658, -0.30777827146642045],
        [0.195710146174071, -0.4450328556279881],
        [0.4011527107863344, -0.28472392416992076],
        [-0.20471221122508318, 0.3512684151972561],
        [0.7757394152285254, -0.1505514507482884],
        [0.784259105702575, -0.20894063685143602],
        [1.4362619924841296, -0.26808248393975315],
        [1.4001804693100803, -0.3799596019441558],
        [1.4829383734171389, -1.8689441108153828],
        [1.0244598159346139, -1.479637919837943],
        [1.142818658033002, -0.41821843567352945],
    ]
)
_SEED1_BUFFER10 = np.array(
    [
        [0.6350747342088027, 0.33101124679892674],
        [0.5714493000584041, 0.04556229671522371],
        [0.7439395045157803, 0.08682227596875236],
        [-0.21436787203423718, -0.47777046596263417],
        [-0.5501468114873936, -0.1514474473474772],
        [0.1084860903362112, -0.5535121515985059],
        [-0.17888953869607543, -0.657241402617848],
        [-0.35600280014417945, -0.3436756982935975],
        [0.2488694494984914, -0.49300953371786715],
        [0.4202227656259846, -0.006487902939023315],
        [-2.309333858431005, 0.6624378895608102],
        [-1.442818261338718, 0.39235013669429236],
        [-0.5970815889259367, 1.0839446687096161],
        [0.31516264154142576, 1.9765592097019196],
        [0.21141951159125502, 1.7514214509560453],
        [0.028034698100607543, 1.056860335681564],
        [-0.7630845024479102, 1.1860355070533477],
        [-1.941291850500934, 0.9477499767276922],
        [-2.0560120421394634, -0.03218971429437731],
        [-2.6528601943272303, -0.27008672821761204],
        [0.7662707657413204, -0.20822272494456126],
        [1.0675658815010896, 0.15895556517653567],
        [0.36914358260995433, -0.06528178475520896],
        [-0.5209433903567642, -0.5394900181622481],
        [-0.4687244897448159, 0.4121570625207977],
    ]
)
# size=(2, 2000) (the ChaseTag opponent's), seed 0: samples 1995-2004.
_SEED0_BUFFER2000_FROM_1995 = np.array(
    [
        [0.12941294853148697, -0.08355144877173515],
        [0.1550848273990122, -0.1094454620298465],
        [0.15092072679016136, -0.07871727903896462],
        [0.10816940727411581, -0.11102322967512766],
        [0.08057594985657764, -0.05904591079025319],
        [0.4065407152253373, 0.6307737970709519],
        [0.36902911308689124, 0.6484902932269014],
        [0.3535689144704259, 0.6972404622077056],
        [0.338742499861999, 0.6458484044023763],
        [0.35201511017716974, 0.709413550369212],
    ]
)
# size=(2, 10), seed 2: sample(T=13) (transposed), then one sample.
_SEED2_BUFFER10_T13 = np.array(
    [
        [-0.4663521786078889, 0.40934049632249536],
        [-0.24951496871480902, 0.20973706991614366],
        [1.0689131897902118, 0.463585595481024],
        [0.30350306285715106, 0.25797257833683407],
        [0.4867824457055932, -0.11563658493059657],
        [0.9072533478907872, -0.22823503837017634],
        [-0.4410671370193866, -1.1838422142745235],
        [-0.5594125590920639, -0.9746027925252317],
        [0.2154890077901201, -0.3744746379482473],
        [-0.4840531256008941, 0.19086715178734223],
        [-0.05524579248652344, 0.9333706325247279],
        [0.6388962075647393, 0.7822052881434288],
        [-1.0102784374861487, 0.8103115041020598],
    ]
)
_SEED2_BUFFER10_AFTER_T13 = np.array([-2.0168047923602797, 0.8163977048041833])
# size=4 (a single series), seed 3: six scalar samples.
_SEED3_SCALAR = np.array(
    [
        -0.6551255023987536,
        1.1870505930094528,
        2.81597746581964,
        0.572262550697593,
        -1.3352305023352904,
        -0.9155850236577585,
    ]
)


def _process(seed: int, size: int | tuple[int, ...]) -> ColoredNoiseProcess:
    return ColoredNoiseProcess(
        beta=2, size=size, scale=0.7, rng=np.random.default_rng(seed)
    )


@pytest.mark.parametrize("seed, golden", [(0, _SEED0_BUFFER10), (1, _SEED1_BUFFER10)])
def test_single_samples_across_refills(seed: int, golden: np.ndarray) -> None:
    """One step at a time through two buffer refills."""
    process = _process(seed, (2, 10))
    samples = [process.sample() for _ in range(25)]
    assert samples[0].shape == (2,)
    np.testing.assert_allclose(np.stack(samples), golden, rtol=0, atol=_ATOL)


def test_refill_of_the_opponent_buffer() -> None:
    """The 2000-step buffer refills after its last sample."""
    process = _process(0, (2, 2000))
    for _ in range(1995):
        process.sample()
    samples = np.stack([process.sample() for _ in range(10)])
    np.testing.assert_allclose(samples, _SEED0_BUFFER2000_FROM_1995, rtol=0, atol=_ATOL)
    assert process.idx == 5


def test_multi_step_sample_refills_inside_one_call() -> None:
    """``sample(T)`` returns ``(*size[:-1], T)`` and refills mid-call."""
    process = _process(2, (2, 10))
    chunk = process.sample(T=13)
    assert chunk.shape == (2, 13)
    np.testing.assert_allclose(chunk.T, _SEED2_BUFFER10_T13, rtol=0, atol=_ATOL)
    np.testing.assert_allclose(
        process.sample(), _SEED2_BUFFER10_AFTER_T13, rtol=0, atol=_ATOL
    )


def test_chunked_and_single_samples_are_one_stream() -> None:
    """Any split of the stream into ``sample(T)`` calls gives the same values."""
    chunked, single = _process(5, (2, 10)), _process(5, (2, 10))
    got = np.concatenate([chunked.sample(T=t) for t in (3, 10, 2, 24)], axis=-1)
    expected = np.stack([single.sample() for _ in range(39)], axis=-1)
    np.testing.assert_array_equal(got, expected)


def test_scalar_series_and_scale_read_at_sampling_time() -> None:
    """An int size gives scalar samples; ``scale`` applies when sampling."""
    process = _process(3, 4)
    samples = [process.sample() for _ in range(6)]
    assert samples[0].shape == ()
    np.testing.assert_allclose(samples, _SEED3_SCALAR, rtol=0, atol=_ATOL)

    process = _process(3, 4)
    process.scale = 1.4
    np.testing.assert_allclose(
        process.sample(T=6), 2.0 * _SEED3_SCALAR, rtol=0, atol=2 * _ATOL
    )
