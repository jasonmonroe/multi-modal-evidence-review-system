# code/tests/src/test_data_handler.py

import pandas as pd
import pytest

from src import data_handler as dh


@pytest.fixture
def csv_dir(tmp_path, monkeypatch):
    """Write tiny CSVs and point the module's path constants at them."""
    files = {
        "CLAIMS_FILE": ("claims.csv", "user_id,claim_object\nu1,car\nu2,laptop\n"),
        "SAMPLE_CLAIMS_FILE": ("sample.csv", "user_id,claim_object\ns1,package\n"),
        "EVIDENCE_REQ_FILE": ("evidence.csv", "requirement_id,claim_object\nR1,all\n"),
        "OUTPUT_FILE": ("output.csv", "user_id\nu1\n"),
        "USER_HISTORY_FILE": ("history.csv", "user_id,history_flags\nu1,none\n"),
    }
    for const, (name, body) in files.items():
        path = tmp_path / name
        path.write_text(body)
        monkeypatch.setattr(dh, const, str(path))
    return tmp_path


class TestInit:
    def test_starts_empty(self):
        handler = dh.DataHandler()
        for name in ("claims", "evidence_reqs", "output", "user_history"):
            frame = getattr(handler, name)
            assert isinstance(frame, pd.DataFrame)
            assert frame.empty
        assert handler._loaded_type is None


class TestLoadData:
    def test_production_loads_all_frames(self, csv_dir):
        handler = dh.DataHandler()
        handler.load_data({})

        assert list(handler.claims["user_id"]) == ["u1", "u2"]
        assert list(handler.evidence_reqs.columns) == ["requirement_id", "claim_object"]
        assert list(handler.user_history["history_flags"]) == ["none"]
        assert list(handler.output["user_id"]) == ["u1"]
        assert handler._loaded_type == "production"

    def test_sample_flag_loads_sample_file(self, csv_dir):
        handler = dh.DataHandler()
        handler.load_data({"sample": True})
        assert list(handler.claims["user_id"]) == ["s1"]
        assert handler._loaded_type == "sample"

    def test_falsy_sample_flag_is_production(self, csv_dir):
        handler = dh.DataHandler()
        handler.load_data({"sample": False})
        assert handler._loaded_type == "production"

    def test_repeat_load_same_type_skips_io(self, csv_dir, monkeypatch):
        handler = dh.DataHandler()
        handler.load_data({})

        calls = []
        monkeypatch.setattr(
            handler, "_safe_read_csv", lambda p: calls.append(p) or pd.DataFrame()
        )
        handler.load_data({})
        assert calls == []

    def test_switching_type_reloads(self, csv_dir):
        handler = dh.DataHandler()
        handler.load_data({})
        handler.load_data({"sample": True})
        assert list(handler.claims["user_id"]) == ["s1"]
        handler.load_data({})
        assert list(handler.claims["user_id"]) == ["u1", "u2"]

    def test_missing_claims_file_gives_empty_frame(self, csv_dir, monkeypatch, capsys):
        monkeypatch.setattr(dh, "CLAIMS_FILE", str(csv_dir / "nope.csv"))
        handler = dh.DataHandler()
        handler.load_data({})
        assert handler.claims.empty
        assert "not found" in capsys.readouterr().out

    def test_empty_claims_are_reloaded_next_time(self, csv_dir, monkeypatch):
        # guard clause requires non-empty claims, so a failed load is retried
        monkeypatch.setattr(dh, "CLAIMS_FILE", str(csv_dir / "nope.csv"))
        handler = dh.DataHandler()
        handler.load_data({})
        monkeypatch.setattr(dh, "CLAIMS_FILE", str(csv_dir / "claims.csv"))
        handler.load_data({})
        assert len(handler.claims) == 2


class TestSafeReadCsv:
    def test_reads_existing_file(self, csv_dir):
        df = dh.DataHandler()._safe_read_csv(str(csv_dir / "claims.csv"))
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (2, 2)

    def test_missing_file_returns_empty_frame_and_warns(self, tmp_path, capsys):
        df = dh.DataHandler()._safe_read_csv(str(tmp_path / "missing.csv"))
        assert isinstance(df, pd.DataFrame) and df.empty
        assert "Warning" in capsys.readouterr().out

    def test_individual_loaders_assign_attributes(self, csv_dir):
        handler = dh.DataHandler()
        handler._load_claims(str(csv_dir / "claims.csv"))
        handler._load_evidence_reqs(str(csv_dir / "evidence.csv"))
        handler._load_output(str(csv_dir / "output.csv"))
        handler._load_user_history(str(csv_dir / "history.csv"))
        assert len(handler.claims) == 2
        assert "requirement_id" in handler.evidence_reqs
        assert "user_id" in handler.output
        assert "history_flags" in handler.user_history


class TestDescribe:
    def test_empty_claims_message(self, capsys):
        dh.DataHandler().describe()
        assert "empty" in capsys.readouterr().out

    def test_profile_includes_shape_columns_and_distribution(self, csv_dir, capsys):
        handler = dh.DataHandler()
        handler.load_data({})
        handler.describe()
        out = capsys.readouterr().out
        assert "(2, 2)" in out
        assert "user_id" in out
        assert "Object Distribution" in out
        assert "car" in out

    def test_skips_distribution_without_claim_object(self, capsys):
        handler = dh.DataHandler()
        handler.claims = pd.DataFrame({"user_id": ["a", "b"]})
        handler.describe()
        assert "Object Distribution" not in capsys.readouterr().out
