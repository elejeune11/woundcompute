from woundcompute import image_analysis as ia
import pytest


def test_fcn_1():
    x = 7
    y = 3
    known = 10
    found = ia.fcn_1(x , y)
    assert known == found

