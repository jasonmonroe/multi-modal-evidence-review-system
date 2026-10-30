# code/tests/src/test_claim_handler.py

import csv

import pytest

from src import claim_handler as ch
from src.claim_handler import ClaimHandler
from src.constants import CSV_HEADERS


def full_dataset(**over):
    data = {
        "user_id": "user_001",
        "image_paths": ["images/a.jpg", "images/b.jpg"],
        "user_claim": "Customer: hi\nAgent: hello",
        "claim_object": "car",
        "evidence_standard_met": True,
        "evidence_standard_met_reason": "  clear photo  ",
        "risk_flags": "none",
        "issue_type": "dent",
        "object_part": "door",
        "claim_status": "supported",
        "claim_status_justification": "  Dent visible.  ",
        "supporting_image_ids": ["img_1.jpg", "img_2.jpg"],
        "valid_image": True,
        "severity": "low",
    }
    data.update(over)
    return data


class TestInit:
    def test_all_attributes_start_none(self):
        assert all(v is None for v in ClaimHandler().__dict__.values())

    def test_attribute_order_matches_csv_headers(self):
        # export_csv_row emits __dict__ order, so it must equal the header row
        assert list(ClaimHandler().__dict__) == CSV_HEADERS


class TestUpdate:
    def test_sets_known_attributes(self):
        h = ClaimHandler()
        h.update({"user_id": "u9", "severity": "high"})
        assert h.user_id == "u9" and h.severity == "high"

    def test_ignores_unknown_keys(self):
        h = ClaimHandler()
        h.update({"usage": {"total_tokens": 5}, "csv_row": object(), "images": []})
        assert not hasattr(h, "usage") and not hasattr(h, "images")
        assert list(h.__dict__) == CSV_HEADERS

    def test_empty_update_is_noop(self):
        h = ClaimHandler()
        h.update({})
        assert all(v is None for v in h.__dict__.values())

    def test_overwrites_previous_values(self):
        h = ClaimHandler()
        h.update({"severity": "low"})
        h.update({"severity": "high"})
        assert h.severity == "high"


class TestFormatValuesForOutput:
    def format(self, **over):
        h = ClaimHandler()
        h.update(full_dataset(**over))
        h._format_values_for_output()
        return h

    def test_newlines_in_claim_become_pipes(self):
        assert self.format().user_claim == "Customer: hi | Agent: hello"

    def test_claim_without_newline_unchanged(self):
        assert self.format(user_claim="one line").user_claim == "one line"

    def test_image_paths_list_joined(self):
        assert self.format().image_paths == "images/a.jpg;images/b.jpg"

    def test_image_paths_string_untouched(self):
        assert self.format(image_paths="x.jpg;y.jpg").image_paths == "x.jpg;y.jpg"

    def test_supporting_ids_list_joined_and_stringified(self):
        h = self.format(supporting_image_ids=["img_1.jpg", 2])
        assert h.supporting_image_ids == "img_1.jpg;2"

    def test_supporting_ids_string_untouched(self):
        assert self.format(supporting_image_ids="none").supporting_image_ids == "none"

    def test_empty_supporting_ids_list(self):
        assert self.format(supporting_image_ids=[]).supporting_image_ids == ""

    @pytest.mark.parametrize("value,expected", [(True, "true"), (False, "false")])
    def test_booleans_lowercased(self, value, expected):
        h = self.format(evidence_standard_met=value, valid_image=value)
        assert h.evidence_standard_met == expected
        assert h.valid_image == expected
        assert isinstance(h.valid_image, str)

    def test_strings_stripped(self):
        h = self.format()
        assert h.evidence_standard_met_reason == "clear photo"
        assert h.claim_status_justification == "Dent visible."

    def test_non_bool_non_str_left_alone(self):
        h = self.format(evidence_standard_met=1, valid_image=None)
        assert h.evidence_standard_met == 1
        assert h.valid_image is None

    @pytest.mark.xfail(
        raises=TypeError, strict=True, reason="user_claim=None is not guarded"
    )
    def test_missing_user_claim_does_not_crash(self):
        ClaimHandler()._format_values_for_output()


class TestExportCsvRow:
    def export(self, **over):
        h = ClaimHandler()
        h.update(full_dataset(**over))
        return h.export_csv_row()

    def test_row_length_matches_headers(self):
        assert len(self.export()) == len(CSV_HEADERS)

    def test_row_values_in_header_order(self):
        row = dict(zip(CSV_HEADERS, self.export()))
        assert row["user_id"] == "user_001"
        assert row["claim_status"] == "supported"
        assert row["evidence_standard_met"] == "true"
        assert row["valid_image"] == "true"
        assert row["supporting_image_ids"] == "img_1.jpg;img_2.jpg"
        assert row["user_claim"] == "Customer: hi | Agent: hello"

    def test_apostrophes_removed(self):
        row = dict(
            zip(
                CSV_HEADERS,
                self.export(user_claim="It's the owner’s car", object_part="d'oor"),
            )
        )
        assert row["user_claim"] == "Its the owners car"
        assert row["object_part"] == "door"

    def test_none_becomes_empty_string(self):
        row = dict(zip(CSV_HEADERS, self.export(severity=None)))
        assert row["severity"] == ""

    def test_output_cells_are_strings_or_scalars(self):
        for cell in self.export():
            assert isinstance(cell, (str, int, float))
            assert cell is not None

    def test_does_not_emit_unexpected_keys(self):
        h = ClaimHandler()
        h.update(full_dataset(extra="zzz"))
        assert "zzz" not in h.export_csv_row()


class TestProgress:
    full, empty = "✅️", "☑️"

    def test_first_of_four(self, capsys):
        out = ClaimHandler().progress(0, 4)
        assert out.count(self.full) == 1
        assert out.count(self.empty) == 3
        assert out.endswith("\t25.0%")

    def test_last_row_is_complete(self, capsys):
        out = ClaimHandler().progress(3, 4)
        assert out.count(self.full) == 4
        assert self.empty not in out
        assert out.endswith("100.0%")

    def test_returns_str_and_prints_blank_line(self, capsys):
        out = ClaimHandler().progress(0, 1)
        assert isinstance(out, str)
        assert capsys.readouterr().out == "\n\n"

    def test_zero_total_raises(self):
        with pytest.raises(ZeroDivisionError):
            ClaimHandler().progress(0, 0)


class TestWriteOutput:
    def test_roundtrips_rows_with_commas_and_newlines(self, tmp_path, monkeypatch):
        target = tmp_path / "out.csv"
        monkeypatch.setattr(ch, "TEST_FILE", str(target))
        rows = [list(CSV_HEADERS), ["a,b", 'say "hi"', "line1\nline2"] + [""] * 11]

        ClaimHandler().write_output(rows)

        with open(target, newline="", encoding="utf-8") as f:
            assert list(csv.reader(f)) == rows

    def test_overwrites_existing_file(self, tmp_path, monkeypatch):
        target = tmp_path / "out.csv"
        target.write_text("stale,data\n")
        monkeypatch.setattr(ch, "TEST_FILE", str(target))
        ClaimHandler().write_output([["h"], ["v"]])
        assert target.read_text().splitlines() == ["h", "v"]

    def test_empty_rows_creates_empty_file(self, tmp_path, monkeypatch):
        target = tmp_path / "out.csv"
        monkeypatch.setattr(ch, "TEST_FILE", str(target))
        ClaimHandler().write_output([])
        assert target.read_text() == ""

    def test_announces_target(self, tmp_path, monkeypatch, capsys):
        target = tmp_path / "out.csv"
        monkeypatch.setattr(ch, "TEST_FILE", str(target))
        ClaimHandler().write_output([])
        assert str(target) in capsys.readouterr().out
