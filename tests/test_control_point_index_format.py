import csv
import json

import torch

import torch_bsf
from torch_bsf.bezier_simplex import BezierSimplex, load, save


def test_backward_compatibility(tmp_path):
    # 1. JSON with old [] keys
    old_json_data = {"[1, 0]": [1.0, 2.0], "[0, 1]": [3.0, 4.0]}
    json_path = tmp_path / "old.json"
    with open(json_path, "w") as f:
        json.dump(old_json_data, f)

    bs = load(json_path)
    assert "(1, 0)" in bs.control_points
    assert "(0, 1)" in bs.control_points
    assert torch.allclose(bs.control_points["(1, 0)"], torch.tensor([1.0, 2.0]))

    # 2. YAML with old [] keys (quoted)
    old_yaml = "'[1, 0]': [1.0, 2.0]\n'[0, 1]': [3.0, 4.0]\n"
    yaml_path = tmp_path / "old.yml"
    with open(yaml_path, "w") as f:
        f.write(old_yaml)

    bs = load(yaml_path)
    assert "(1, 0)" in bs.control_points
    assert torch.allclose(bs.control_points["(1, 0)"], torch.tensor([1.0, 2.0]))

    # 3. CSV with old [] keys
    csv_path = tmp_path / "old.csv"
    with open(csv_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(["[1, 0]", 1.0, 2.0])
        writer.writerow(["[0, 1]", 3.0, 4.0])

    bs = load(csv_path)
    assert "(1, 0)" in bs.control_points
    assert torch.allclose(bs.control_points["(1, 0)"], torch.tensor([1.0, 2.0]))

    # 4. PT round-trip with new format (old [] key format is not supported; no backward compat needed)
    bs_pt = BezierSimplex({(1, 0): [1.0, 2.0], (0, 1): [3.0, 4.0]})
    pt_path = tmp_path / "new.pt"
    torch.save(bs_pt, pt_path)

    bs_loaded = load(pt_path, pt_weights_only=False)
    assert "(1, 0)" in bs_loaded.control_points
    assert "(0, 1)" in bs_loaded.control_points
    assert torch.allclose(bs_loaded.control_points["(1, 0)"], torch.tensor([1.0, 2.0]))


def test_save_load_cycle(tmp_path):
    bs = torch_bsf.bezier_simplex.randn(n_params=2, n_values=2, degree=2)
    formats = [".json", ".yml", ".csv", ".pt"]
    for ext in formats:
        path = tmp_path / f"cycle{ext}"
        save(path, bs)
        bs_loaded = load(path)
        for k in bs.control_points.keys():
            assert "(" in k and ")" in k
            assert k in bs_loaded.control_points
            assert torch.allclose(bs.control_points[k], bs_loaded.control_points[k])


def test_yaml_aesthetics(tmp_path):
    bs = BezierSimplex({(1, 0): [1.0, 2.0]})
    yaml_path = tmp_path / "test.yml"
    save(yaml_path, bs)
    with open(yaml_path, "r") as f:
        content = f.read()
        assert "(1, 0):" in content
        assert "'(1, 0)':" not in content


def test_training():
    ts = torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    xs = torch.tensor([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]])
    bs = torch_bsf.bezier_simplex.fit(params=ts, values=xs, degree=2, max_epochs=1)
    assert isinstance(bs, BezierSimplex)
