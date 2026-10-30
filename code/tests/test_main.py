# code/tests/test_main.py

import runpy
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestParseArgs:
    def test_no_arguments_all_false(self, main_module):
        assert main_module.parse_args([]) == {"eda": False, "eval": False, "sample": False}

    def test_all_arguments_true(self, main_module):
        out = main_module.parse_args(["--eda", "--eval", "--sample"])
        assert out == {"eda": True, "eval": True, "sample": True}

    def test_individual_flags(self, main_module):
        out = main_module.parse_args(["--sample"])
        assert out["sample"] is True
        assert out["eda"] is False and out["eval"] is False

    def test_keys_have_no_dashes(self, main_module):
        assert all("-" not in k for k in main_module.parse_args(["--eda"]))

    def test_values_are_real_bools(self, main_module):
        out = main_module.parse_args(["--eda"])
        assert all(isinstance(v, bool) for v in out.values())

    def test_unknown_arguments_are_ignored(self, main_module):
        out = main_module.parse_args(["--bogus", "eda", "-eda", "--EDA"])
        assert set(out) == {"eda", "eval", "sample"}
        assert not any(out.values())

    def test_order_independent_and_duplicates_ok(self, main_module):
        out = main_module.parse_args(["--eval", "--eda", "--eval"])
        assert out == {"eda": True, "eval": True, "sample": False}


class TestRunMainPipeline:
    @pytest.fixture
    def stubs(self, main_module, monkeypatch):
        s = types.SimpleNamespace(
            data=MagicMock(name="DataHandler()"),
            image=MagicMock(name="ImageHandler()"),
            claims=MagicMock(name="run_claims_pipeline"),
            evaluation=MagicMock(name="run_evaluation_report_pipeline"),
            banner=MagicMock(name="show_banner"),
        )
        s.data_cls = MagicMock(return_value=s.data)
        s.image_cls = MagicMock(return_value=s.image)
        monkeypatch.setattr(main_module, "DataHandler", s.data_cls)
        monkeypatch.setattr(main_module, "ImageHandler", s.image_cls)
        monkeypatch.setattr(main_module, "run_claims_pipeline", s.claims)
        monkeypatch.setattr(main_module, "run_evaluation_report_pipeline", s.evaluation)
        monkeypatch.setattr(main_module, "show_banner", s.banner)
        s.run = main_module.run_main_pipeline
        return s

    def test_default_run(self, stubs):
        args = {"eda": False, "eval": False, "sample": False}
        stubs.run(args)

        stubs.data.load_data.assert_called_once_with(args)
        stubs.image_cls.assert_called_once_with(False)
        stubs.data.describe.assert_not_called()
        stubs.claims.assert_called_once_with(stubs.data, stubs.image)
        stubs.evaluation.assert_not_called()

    def test_eda_describes_data_and_enables_image_eda(self, stubs):
        stubs.run({"eda": True})
        stubs.image_cls.assert_called_once_with(True)
        stubs.data.describe.assert_called_once()

    def test_eval_runs_evaluation_after_claims(self, stubs):
        order = []
        stubs.claims.side_effect = lambda *a: order.append("claims")
        stubs.evaluation.side_effect = lambda *a: order.append("eval")
        stubs.run({"eval": True})
        assert order == ["claims", "eval"]
        stubs.evaluation.assert_called_once_with(stubs.data, stubs.image)

    def test_missing_keys_default_to_off(self, stubs):
        stubs.run({})
        stubs.image_cls.assert_called_once_with(False)
        stubs.evaluation.assert_not_called()

    def test_banner_lists_args_with_icons(self, stubs):
        stubs.run({"eda": True, "sample": False})
        title, subtitles = stubs.banner.call_args.args
        assert title == "HACKERRANK ORCHESTRATE"
        assert subtitles[0].startswith("The Insurance Agent")
        assert any("eda" in s and "✅" in s for s in subtitles)
        assert any("sample" in s and "❌" in s for s in subtitles)

    def test_pipeline_failure_propagates_and_skips_eval(self, stubs):
        stubs.claims.side_effect = RuntimeError("boom")
        with pytest.raises(RuntimeError):
            stubs.run({"eval": True})
        stubs.evaluation.assert_not_called()


class TestScriptEntryPoint:
    def test_main_block_runs_pipeline_and_prints_run_ids(
        self, legacy_modules, monkeypatch, capsys
    ):
        run_claims = MagicMock()
        fake_modules = {
            "pipelines.run_claims_pipeline": types.SimpleNamespace(
                run_claims_pipeline=run_claims
            ),
            "evaluation.main": types.SimpleNamespace(
                run_evaluation_report_pipeline=MagicMock()
            ),
            "src.data_handler": types.SimpleNamespace(
                DataHandler=MagicMock(return_value=MagicMock())
            ),
            "src.image_handler": types.SimpleNamespace(
                ImageHandler=MagicMock(return_value=MagicMock())
            ),
        }
        for name, mod in fake_modules.items():
            monkeypatch.setitem(sys.modules, name, mod)
        monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
        monkeypatch.setattr(sys, "argv", ["main.py", "--sample"])

        main_path = Path(__file__).resolve().parents[1] / "main.py"
        runpy.run_path(str(main_path), run_name="__main__")

        out = capsys.readouterr().out
        assert "START RUN ID:" in out and "END RUN ID:" in out
        assert "Run Time" in out
        run_claims.assert_called_once()
