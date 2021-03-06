# coding: utf-8

# Copyright 2017 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)


import pytest

pytest.importorskip("torch")
import torch  # NOQA

from espnet.espnet.nets.pytorch_backend.nets_utils import pad_list  # NOQA


def test_pad_list():
    xs = [[1, 2, 3], [1, 2], [1, 2, 3, 4]]
    xs = list(map(lambda x: torch.LongTensor(x), xs))
    xpad = pad_list(xs, -1)

    es = [[1, 2, 3, -1], [1, 2, -1, -1], [1, 2, 3, 4]]
    assert xpad.data.tolist() == es


def test_bmm_attention():
    b, t, h = 3, 2, 5
    enc_h = torch.randn(b, t, h)
    w = torch.randn(b, t)
    naive = torch.sum(enc_h * w.view(b, t, 1), dim=1)
    # (b, 1, t) x (b, t, h) -> (b, 1, h)
    fast = torch.matmul(w.unsqueeze(1), enc_h).squeeze(1)
    import numpy

    numpy.testing.assert_allclose(naive.numpy(), fast.numpy(), 1e-6, 1e-6)
