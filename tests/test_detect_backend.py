from hybrik import backend as detect


def test_pick_cuda():
    assert detect._pick_cuda((12, 4)) == ("cu118", [])
    assert detect._pick_cuda((12, 6)) == ("cu126", [])
    assert detect._pick_cuda((12, 8)) == ("cu128", [])
    assert detect._pick_cuda((13, 0)) == ("cu130", [])

    backend, warnings = detect._pick_cuda((11, 7))
    assert backend == "cu118"
    assert warnings

    backend, warnings = detect._pick_cuda(None)
    assert backend == "cu130"
    assert warnings


def test_recommend_nvidia():
    backend, warnings = detect.recommend(
        "Linux", {"kind": "nvidia", "cuda": (12, 8)}, None, None
    )
    assert backend == "cu128"
    assert warnings == []


def test_recommend_amd_linux():
    backend, _ = detect.recommend("Linux", None, {"kind": "amd"}, None)
    assert backend == detect.NEWEST_ROCM


def test_recommend_gfx1152_warns_about_override():
    _, warnings = detect.recommend(
        "Linux", None, {"kind": "amd", "gfx": "gfx1152"}, None
    )
    assert any("HSA_OVERRIDE_GFX_VERSION=11.5.0" in warning for warning in warnings)


def test_recommend_amd_windows_is_cpu():
    backend, warnings = detect.recommend("Windows", None, {"kind": "amd"}, None)
    assert backend == "cpu"
    assert warnings


def test_recommend_darwin_is_cpu():
    backend, warnings = detect.recommend(
        "Darwin", {"kind": "nvidia", "cuda": (13, 0)}, None, None
    )
    assert backend == "cpu"
    assert warnings


def test_recommend_cpu():
    backend, warnings = detect.recommend("Linux", None, None, None)
    assert backend == "cpu"
    assert warnings == []
