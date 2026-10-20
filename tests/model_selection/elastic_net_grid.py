import numpy as np
import pytest

from torch_bsf.model_selection.elastic_net_grid import elastic_net_grid, reverse_logspace


class TestReverseLogspace:
    def test_empty(self):
        result = reverse_logspace(0)
        assert result.shape == (0,)

    def test_single(self):
        result = reverse_logspace(1)
        assert result.shape == (1,)
        assert result[0] == 0.0

    def test_two(self):
        result = reverse_logspace(2)
        assert result.shape == (2,)
        assert result[0] == 0.0
        assert 0.0 < result[1] < 1.0

    def test_base_one_is_linspace(self):
        result = reverse_logspace(5, base=1)
        expected = np.linspace(0.0, 1.0, 5, endpoint=False)
        np.testing.assert_allclose(result, expected)

    def test_invalid_base_raises(self):
        with pytest.raises(ValueError):
            reverse_logspace(5, base=0)

    def test_base_gt1_monotonically_increasing(self):
        result = reverse_logspace(10, base=10)
        assert np.all(np.diff(result) > 0)

    def test_base_lt1_monotonically_increasing(self):
        result = reverse_logspace(10, base=0.1)
        assert np.all(np.diff(result) > 0)


class TestElasticNetGrid:
    def test_empty_when_n_lambdas_zero(self):
        result = elastic_net_grid(0, 5)
        assert result.shape[1] == 3
        assert result.shape[0] == 0

    def test_empty_when_n_alphas_zero(self):
        result = elastic_net_grid(5, 0)
        assert result.shape[1] == 3
        assert result.shape[0] == 0

    def test_shape_default_vertex_copies(self):
        n_lambdas, n_alphas = 3, 3
        result = elastic_net_grid(n_lambdas, n_alphas, n_vertex_copies=1)
        expected_rows = (n_lambdas - 1) * n_alphas + 3 * 1 - 2
        assert result.shape == (expected_rows, 3)

    def test_shape_extra_vertex_copies(self):
        n_lambdas, n_alphas, n_copies = 3, 3, 2
        result = elastic_net_grid(n_lambdas, n_alphas, n_vertex_copies=n_copies)
        expected_rows = (n_lambdas - 1) * n_alphas + 3 * n_copies - 2
        assert result.shape == (expected_rows, 3)

    def test_rows_sum_to_one(self):
        result = elastic_net_grid(5, 4, n_vertex_copies=2)
        np.testing.assert_allclose(result.sum(axis=1), np.ones(len(result)), atol=1e-10)

    def test_all_values_in_unit_interval(self):
        result = elastic_net_grid(5, 4)
        assert (result >= 0).all()
        assert (result <= 1).all()

    def test_known_small_output(self):
        result = elastic_net_grid(1, 1)
        expected = np.array([[1.0, 0.0, 0.0]])
        np.testing.assert_allclose(result, expected)

    def test_vertex_copies_increase_row_count(self):
        r1 = elastic_net_grid(3, 3, n_vertex_copies=1)
        r2 = elastic_net_grid(3, 3, n_vertex_copies=2)
        assert len(r2) == len(r1) + 3

    def test_n_vertex_copies_below_one_raises(self):
        with pytest.raises(ValueError, match="n_vertex_copies"):
            elastic_net_grid(3, 3, n_vertex_copies=0)

    def test_negative_n_lambdas_raises(self):
        with pytest.raises(ValueError, match="n_lambdas"):
            elastic_net_grid(-1, 3)

    def test_negative_n_alphas_raises(self):
        with pytest.raises(ValueError, match="n_alphas"):
            elastic_net_grid(3, -1)


class TestElasticNetGridCliMain:
    """Tests for the _cli_main() entry point of elastic_net_grid."""

    def test_cli_main_default(self, monkeypatch, capsys):
        """_cli_main() should print grid rows to stdout with default arguments."""
        from torch_bsf.model_selection.elastic_net_grid import _cli_main

        monkeypatch.setattr("sys.argv", ["elastic_net_grid", "--n_lambdas=2", "--n_alphas=2"])
        _cli_main()
        captured = capsys.readouterr()
        lines = [l for l in captured.out.strip().splitlines() if l]
        assert len(lines) > 0
        # Each line should have 3 comma-separated float values.
        for line in lines:
            parts = line.split(",")
            assert len(parts) == 3
            parsed_parts = [float(p) for p in parts]
            assert np.isfinite(parsed_parts).all()
