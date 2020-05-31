import pytest


def test_ok():
    print("ok")


def test_skip():
    pytest.skip("skipping this test")
