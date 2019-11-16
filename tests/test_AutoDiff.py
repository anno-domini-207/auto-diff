import sys

sys.path.append("../")

import AnnoDomini.AutoDiff as AD
import numpy as np


def test_on_AutoDiff_scaler():
    def test_neg():
        x = AD(10)
        f = -x
        assert f == AD(-10, 1)

    def test_add():
        x = AD(2, 1)
        f = x + 4
        assert f == AD(6, 1)
        y = AD(4, 2)
        assert x + y == AD(6, 3)
        f2 = x + 2 + x + 4
        assert f2 == AD(10, 2)

    def test_radd():
        x = AD(2)
        f = 4 + x
        assert f == AD(6, 1)
        assert x == AD(2, 1)

    def test_sub():
        x = AD(5)
        f = x - 6
        assert f == AD(-1, 1)
        y = AD(4, 2)
        assert x - y == AD(1, -1)
        f2 = x - x - x - 1
        assert f2 == AD(-6, -1)
        assert x == AD(5, 1)

    def test_rsub():
        x = AD(5)
        f = 6 - x
        assert f == AD(1, -1)
        assert x == AD(5)

    def test_mul():
        x = AD(5, 1)
        f1 = x * 3
        assert f1 == AD(15, 3)
        y = AD(10, 2)
        f2 = x * y
        assert f2 == AD(50, 20)  # 1 * 10 + 5 * 2 = 20
        f3 = x * x
        assert f3 == AD(25, 10)

    def test_rmul():
        x = AD(5, 1)
        f = 2 * x
        assert f == AD(10, 2)
        assert x == AD(5, 1)

    def test_truediv():
        x = AD(3)
        f = x / 2
        assert np.round(f.val, 2) == 1.5
        assert np.round(f.der, 2) == 0.5
        y = AD(3)
        f2 = x / y
        assert f2 == AD(1.0, 0.0)

    def test_rtruediv():
        x = AD(2)
        f = 1 / x
        assert np.round(f.val, 2) == 0.5
        assert np.round(f.der, 2) == -0.25

        f3 = 2 / x / x
        assert f3 == AD(0.50, -0.50)

    def test_sin():
        x = AD(np.pi / 2)
        f = 2 * np.sin(x)
        assert np.round(f.val, 2) == 2.0
        assert np.round(f.der, 2) == 0.0

    def test_cos():
        x = AD(np.pi / 2)
        f = 2 * np.cos(x)
        assert np.round(f.val, 2) == 0.0
        assert np.round(f.der, 2) == -2.0

    def test_tan():
        x = AD(np.pi / 4)
        f = 3 * np.tan(x)
        assert np.round(f.val, 2) == 3.0
        assert np.round(f.der, 2) == 6.0

    def test_cosh():
        x = AD(0.5)
        f = np.cosh(x)
        assert np.round(f.val, 2) == 1.13
        assert np.round(f.der, 2) == 0.52

    test_neg()
    test_add()
    test_radd()
    test_sub()
    test_rsub()
    test_mul()
    test_rmul()
    test_truediv()
    test_rtruediv()
    test_sin()
    test_cos()
    test_tan()