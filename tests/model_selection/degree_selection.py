import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from torch_bsf.model_selection.degree_selection import select_degree

_REPO_ROOT = Path(__file__).parent.parent.parent
_PARAMS_CSV = _REPO_ROOT / "params.csv"
_VALUES_CSV = _REPO_ROOT / "values.csv"
_CLI_TIMEOUT = 300  # seconds


def _make_simplex_data(n_params: int = 3, n_values: int = 2, n_samples: int = 20, *, seed: int = 0):
    """Return (params, values) tensors on the standard simplex."""
    generator = torch.Generator()
    generator.manual_seed(seed)
    raw = torch.rand(n_samples, n_params, generator=generator)
    params = raw / raw.sum(dim=1, keepdim=True)
    values = (1.0 - params**2)[:, :n_values]
    return params, values


def _fake_fit(mse_by_degree):
    """A stand-in for torch_bsf.fit whose models predict sqrt(mse) everywhere, recording its calls."""
    calls = []

    def fake_fit(params, values, degree, **kwargs):
        calls.append((params, values, degree, kwargs))
        level = mse_by_degree[degree] ** 0.5
        return lambda t: torch.full((len(t), values.shape[1]), level)

    return fake_fit, calls


class TestSelectDegree:
    def test_returns_int_in_range(self):
        params, values = _make_simplex_data(n_params=3, n_values=2, n_samples=20)
        best = select_degree(
            params,
            values,
            min_degree=1,
            max_degree=3,
            num_folds=2,
            solver="trainer",
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            log_every_n_steps=1,
        )
        assert isinstance(best, int)
        assert 1 <= best <= 3

    def test_min_equals_max(self):
        params, values = _make_simplex_data(n_params=3, n_values=2, n_samples=20)
        best = select_degree(
            params,
            values,
            min_degree=2,
            max_degree=2,
            num_folds=2,
            solver="trainer",
            max_epochs=1,
            accelerator="cpu",
            devices=1,
            enable_progress_bar=False,
            enable_model_summary=False,
            log_every_n_steps=1,
        )
        assert best == 2


class TestSelectDegreeKwargForwarding:
    """Verify how select_degree calls torch_bsf.fit."""

    _PATCH = "torch_bsf.model_selection.degree_selection.fit"

    def test_fits_every_fold_and_forwards_kwargs(self):
        """Each fold is fitted to the other folds, and the extra kwargs and seed reach fit."""
        params, values = _make_simplex_data(n_samples=9)
        fake_fit, calls = _fake_fit({1: 0.1})
        with patch(self._PATCH, side_effect=fake_fit):
            select_degree(params, values, min_degree=1, max_degree=1, num_folds=3, seed=4, ridge=0.5)
        assert len(calls) == 3
        for train_params, _, degree, kwargs in calls:
            assert len(train_params) == 6
            assert degree == 1
            assert kwargs == {"seed": 4, "ridge": 0.5}

    def test_freeze_iterator_reaches_every_fold(self):
        params, values = _make_simplex_data(n_samples=9)
        fake_fit, calls = _fake_fit({1: 0.1, 2: 0.2})
        with patch(self._PATCH, side_effect=fake_fit):
            select_degree(params, values, min_degree=1, max_degree=2, num_folds=3, freeze=iter([(1, 0, 0)]))
        assert [c[3]["freeze"] for c in calls] == [[(1, 0, 0)]] * 6

    def test_rejects_mismatched_sample_counts(self):
        params, values = _make_simplex_data(n_samples=9)
        with pytest.raises(ValueError, match="params has 9 samples, but values has 10 samples"):
            select_degree(params, torch.cat([values, values[:1]]), num_folds=3)

    def test_folds_are_reproducible(self):
        params, values = _make_simplex_data(n_samples=10)
        runs = []
        for _ in range(2):
            fake_fit, calls = _fake_fit({1: 0.1})
            with patch(self._PATCH, side_effect=fake_fit):
                select_degree(params, values, min_degree=1, max_degree=1, num_folds=2, seed=0)
            runs.append([c[0] for c in calls])
        for a, b in zip(*runs):
            torch.testing.assert_close(a, b)

    def test_simploid_type(self):
        """A simploid type is forwarded to fit, which fits Bézier simploids."""
        from torch_bsf.simploidal import sample_simploid

        params = sample_simploid((1, 1), 30, seed=0)
        values = params[:, :1] * params[:, 2:3]
        assert select_degree(params, values, min_degree=1, max_degree=3, num_folds=3, seed=0, simploid_type=(1, 1)) == 1


def _run_cli(*args, cwd=None):
    """Run `python -m torch_bsf.model_selection.degree_selection` with the given arguments."""
    workdir = cwd or _REPO_ROOT
    return subprocess.run(
        [sys.executable, "-m", "torch_bsf.model_selection.degree_selection"] + list(args),
        capture_output=True,
        text=True,
        cwd=workdir,
        timeout=_CLI_TIMEOUT,
    )


class TestDegreeSelectionCLI:
    def test_cli_basic_run(self, tmp_path):
        """CLI should complete successfully with minimal required arguments."""
        result = _run_cli(
            f"--params={_PARAMS_CSV}",
            f"--values={_VALUES_CSV}",
            "--min_degree=1",
            "--max_degree=2",
            "--num_folds=2",
            "--max_epochs=1",
            "--loglevel=WARNING",
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr
        assert any(line.startswith("Best degree:") for line in result.stdout.splitlines())

    def test_cli_output_is_valid_degree(self, tmp_path):
        """CLI stdout should contain 'Best degree: N' where N is in [min_degree, max_degree]."""
        result = _run_cli(
            f"--params={_PARAMS_CSV}",
            f"--values={_VALUES_CSV}",
            "--min_degree=1",
            "--max_degree=3",
            "--num_folds=2",
            "--max_epochs=1",
            "--loglevel=WARNING",
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr
        # "Best degree: N" may appear among other Lightning stdout lines
        matching = [line for line in result.stdout.splitlines() if line.startswith("Best degree:")]
        assert matching, f"'Best degree:' not found in stdout:\n{result.stdout}"
        degree = int(matching[0].split(":")[1].strip())
        assert 1 <= degree <= 3

    def test_cli_missing_params_fails(self):
        """CLI should exit non-zero when --params is missing."""
        result = _run_cli(f"--values={_VALUES_CSV}")
        assert result.returncode != 0

    def test_cli_missing_values_fails(self):
        """CLI should exit non-zero when --values is missing."""
        result = _run_cli(f"--params={_PARAMS_CSV}")
        assert result.returncode != 0

    def test_cli_inverted_degree_range_fails(self):
        """CLI should exit non-zero when --min_degree > --max_degree."""
        result = _run_cli(
            f"--params={_PARAMS_CSV}",
            f"--values={_VALUES_CSV}",
            "--min_degree=5",
            "--max_degree=1",
        )
        assert result.returncode != 0
        assert "min_degree" in result.stderr.lower() or "max_degree" in result.stderr.lower()

    def test_cli_devices_auto(self, tmp_path):
        """CLI should accept --devices auto (string) without error."""
        result = _run_cli(
            f"--params={_PARAMS_CSV}",
            f"--values={_VALUES_CSV}",
            "--min_degree=1",
            "--max_degree=1",
            "--num_folds=2",
            "--max_epochs=1",
            "--devices=auto",
            "--loglevel=WARNING",
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr

    def test_cli_devices_integer(self, tmp_path):
        """CLI should accept --devices 1 (integer) without error."""
        result = _run_cli(
            f"--params={_PARAMS_CSV}",
            f"--values={_VALUES_CSV}",
            "--min_degree=1",
            "--max_degree=1",
            "--num_folds=2",
            "--max_epochs=1",
            "--devices=1",
            "--loglevel=WARNING",
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr

    def test_cli_loglevel_info_shows_progress(self, tmp_path):
        """With --loglevel INFO, degree progress messages should appear in stderr."""
        result = _run_cli(
            f"--params={_PARAMS_CSV}",
            f"--values={_VALUES_CSV}",
            "--min_degree=1",
            "--max_degree=1",
            "--num_folds=2",
            "--max_epochs=1",
            "--loglevel=INFO",
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr
        assert "Checking degree" in result.stderr

    def test_cli_loglevel_warning_suppresses_progress(self, tmp_path):
        """With --loglevel WARNING, degree progress messages should not appear."""
        result = _run_cli(
            f"--params={_PARAMS_CSV}",
            f"--values={_VALUES_CSV}",
            "--min_degree=1",
            "--max_degree=1",
            "--num_folds=2",
            "--max_epochs=1",
            "--loglevel=WARNING",
            cwd=tmp_path,
        )
        assert result.returncode == 0, result.stderr
        assert "Checking degree" not in result.stderr

    def test_cli_help(self):
        """CLI should print help and exit 0."""
        result = _run_cli("--help")
        assert result.returncode == 0
        assert "--params" in result.stdout
        assert "--loglevel" in result.stdout
        assert "--devices" in result.stdout
        assert "--patience" in result.stdout


class TestSelectDegreePatience:
    """Tests for the patience-based early-stopping logic in select_degree."""

    _PATCH = "torch_bsf.model_selection.degree_selection.fit"

    def _select_with_mse_sequence(self, mse_sequence, patience=1, min_degree=1):
        """Run select_degree with a predetermined sequence of per-degree MSE values."""
        params, _ = _make_simplex_data(n_params=3, n_values=2, n_samples=20)
        values = torch.zeros(20, 2)
        fake_fit, _ = _fake_fit({min_degree + k: mse for k, mse in enumerate(mse_sequence)})

        max_degree = min_degree + len(mse_sequence) - 1
        with patch(self._PATCH, side_effect=fake_fit):
            return select_degree(
                params,
                values,
                min_degree=min_degree,
                max_degree=max_degree,
                num_folds=2,
                patience=patience,
            )

    def test_patience_1_stops_after_first_increase(self):
        """With patience=1, stops as soon as MSE increases."""
        # MSE decreases at degree 2, increases at degree 3 → should return 2
        best = self._select_with_mse_sequence([0.5, 0.3, 0.4], patience=1)
        assert best == 2

    def test_patience_2_tolerates_one_increase(self):
        """With patience=2, one transient increase is tolerated."""
        # MSE: 0.5, 0.3, 0.35 (up), 0.2 (new best) → should return 4
        best = self._select_with_mse_sequence([0.5, 0.3, 0.35, 0.2], patience=2, min_degree=1)
        assert best == 4

    def test_patience_2_stops_after_two_increases(self):
        """With patience=2, two consecutive increases trigger early stop."""
        # MSE: 0.5, 0.3, 0.35 (up 1), 0.4 (up 2) → stops, best is 2
        best = self._select_with_mse_sequence([0.5, 0.3, 0.35, 0.4], patience=2, min_degree=1)
        assert best == 2

    def test_patience_high_traverses_all_degrees(self):
        """Large patience causes all degrees to be checked."""
        params, values = _make_simplex_data(n_params=3, n_values=2, n_samples=20)
        fake_fit, calls = _fake_fit({d: 1.0 + d for d in range(1, 5)})

        with patch(self._PATCH, side_effect=fake_fit):
            select_degree(
                params,
                values,
                min_degree=1,
                max_degree=4,
                num_folds=2,
                patience=10,
            )
        # All 4 degrees (1..4) should have been evaluated, each on 2 folds
        assert [c[2] for c in calls] == [1, 1, 2, 2, 3, 3, 4, 4]

    def test_patience_zero_raises(self):
        """patience=0 is invalid and should raise ValueError."""
        params, values = _make_simplex_data()
        with pytest.raises(ValueError, match="patience"):
            select_degree(params, values, patience=0)

    def test_patience_negative_raises(self):
        """Negative patience is invalid and should raise ValueError."""
        params, values = _make_simplex_data()
        with pytest.raises(ValueError, match="patience"):
            select_degree(params, values, patience=-1)


class TestSelectDegreeEdgeCases:
    """Tests for edge-case validation in select_degree."""

    def test_empty_dataset_raises(self):
        """select_degree with zero samples should raise ValueError."""
        params = torch.empty(0, 3)
        values = torch.empty(0, 2)
        with pytest.raises(ValueError, match="At least 2 samples"):
            select_degree(params, values, min_degree=1, max_degree=2, num_folds=2)

    def test_invalid_batch_size_raises(self):
        """Passing a non-positive batch_size should raise ValueError."""
        params, values = _make_simplex_data()
        with pytest.raises(ValueError, match="batch_size"):
            select_degree(params, values, min_degree=1, max_degree=1, num_folds=2, batch_size=-1)


class TestDegreeSelectionCliMain:
    """Tests for the _cli_main() entry point of degree_selection."""

    def test_cli_main_basic(self, tmp_path, monkeypatch):
        """_cli_main() should run successfully with minimal arguments."""
        from torch_bsf.model_selection.degree_selection import _cli_main

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "degree_selection",
                f"--params={_PARAMS_CSV}",
                f"--values={_VALUES_CSV}",
                "--min_degree=1",
                "--max_degree=2",
                "--num_folds=2",
                "--max_epochs=1",
                "--loglevel=WARNING",
            ],
        )
        _cli_main()

    def test_cli_main_with_batch_size(self, tmp_path, monkeypatch):
        """_cli_main() should accept --batch_size."""
        from torch_bsf.model_selection.degree_selection import _cli_main

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "degree_selection",
                f"--params={_PARAMS_CSV}",
                f"--values={_VALUES_CSV}",
                "--min_degree=1",
                "--max_degree=1",
                "--num_folds=2",
                "--max_epochs=1",
                "--batch_size=10",
                "--loglevel=WARNING",
            ],
        )
        _cli_main()

    def test_cli_main_trainer_and_simploid_type(self, tmp_path, monkeypatch, capsys):
        """_cli_main() forwards --simploid-type and the trainer options to select_degree."""
        import numpy as np

        from torch_bsf.model_selection.degree_selection import _cli_main

        n_params = np.loadtxt(_PARAMS_CSV, delimiter=",", ndmin=2).shape[1]
        monkeypatch.setattr(
            "sys.argv",
            [
                "degree_selection",
                f"--params={_PARAMS_CSV}",
                f"--values={_VALUES_CSV}",
                f"--simploid-type={n_params - 1}",
                "--solver=trainer",
                "--max_epochs=3",
                "--accelerator=cpu",
                "--min_degree=1",
                "--max_degree=1",
            ],
        )
        with patch("torch_bsf.model_selection.degree_selection.select_degree", return_value=1) as select:
            _cli_main()
        kwargs = select.call_args.kwargs
        assert kwargs["simploid_type"] == (n_params - 1,)
        assert (kwargs["solver"], kwargs["max_epochs"], kwargs["accelerator"]) == ("trainer", 3, "cpu")
        assert "Best degree: 1" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "args, message",
        [
            (["--simploid-type=1,1"], "--params has"),
            (["--num_folds=1"], "n_folds must be at least 2"),
        ],
    )
    def test_cli_main_errors_exit(self, monkeypatch, capsys, args, message):
        """_cli_main() reports invalid columns and errors from select_degree with parser.error."""
        from torch_bsf.model_selection.degree_selection import _cli_main

        monkeypatch.setattr("sys.argv", ["degree_selection", f"--params={_PARAMS_CSV}", f"--values={_VALUES_CSV}", *args])
        with pytest.raises(SystemExit) as exc_info:
            _cli_main()
        assert exc_info.value.code == 2
        assert message in capsys.readouterr().err

    def test_cli_main_inverted_degrees_exits(self, monkeypatch):
        """_cli_main() should call parser.error (SystemExit 2) when min > max."""
        from torch_bsf.model_selection.degree_selection import _cli_main

        monkeypatch.setattr(
            "sys.argv",
            [
                "degree_selection",
                f"--params={_PARAMS_CSV}",
                f"--values={_VALUES_CSV}",
                "--min_degree=5",
                "--max_degree=1",
            ],
        )
        with pytest.raises(SystemExit) as exc_info:
            _cli_main()
        assert exc_info.value.code != 0
