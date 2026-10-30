# code/tests/src/test_utils.py

import re

import pytest

from src import utils
from src.constants import PEP8_LINE_LEN


class TestGetRunId:
    def test_format(self):
        run_id = utils.get_run_id()
        assert isinstance(run_id, str)
        assert re.fullmatch(r"[0-9A-F]{5}", run_id)

    def test_ids_vary(self):
        assert len({utils.get_run_id() for _ in range(50)}) > 1


class TestTimer:
    def test_start_timer_returns_float(self):
        assert isinstance(utils.start_timer(), float)

    def test_start_timer_is_monotonic_enough(self):
        a = utils.start_timer()
        assert utils.start_timer() >= a

    def test_get_time_format(self):
        assert utils.get_time(0, 65.5) == "1m 5s 500ms"

    def test_get_time_zero(self):
        assert utils.get_time(10, 10) == "0m 0s 0ms"

    def test_get_time_is_order_independent(self):
        assert utils.get_time(65.5, 0) == utils.get_time(0, 65.5)

    def test_get_time_drops_whole_hours(self):
        # only minutes/seconds are reported
        assert utils.get_time(0, 3600 + 5) == "0m 5s 0ms"

    def test_get_time_defaults_end_to_now(self, monkeypatch):
        monkeypatch.setattr(utils.time, "time", lambda: 130.0)
        assert utils.get_time(100.0) == "0m 30s 0ms"

    def test_show_timer_prints(self, capsys, monkeypatch):
        monkeypatch.setattr(utils.time, "time", lambda: 61.0)
        utils.show_timer(0.0)
        assert "Run Time: 1m 1s 0ms" in capsys.readouterr().out


class TestBanners:
    def test_top_btm_line(self):
        line = utils._make_top_btm_line()
        assert len(line) == PEP8_LINE_LEN
        assert line[0] == line[-1] == "+"
        assert set(line[1:-1]) == {"-"}

    def test_title_banner_centered(self, capsys):
        utils._create_title_banner("HELLO")
        lines = [l for l in capsys.readouterr().out.splitlines() if l]
        assert len(lines) == 3
        assert lines[0] == lines[2] == utils._make_top_btm_line()
        assert len(lines[1]) == PEP8_LINE_LEN
        assert lines[1].startswith("| ") and lines[1].endswith(" |")
        assert lines[1].strip("| ") == "HELLO"
        left = len(lines[1][2:]) - len(lines[1][2:].lstrip())
        right = len(lines[1][:-2]) - len(lines[1][:-2].rstrip())
        assert abs(left - right) <= 1

    def test_title_banner_left_aligned(self, capsys):
        utils._create_title_banner("HELLO", center_text=False)
        title = [l for l in capsys.readouterr().out.splitlines() if l][1]
        assert title.startswith("| HELLO ")
        assert len(title) == PEP8_LINE_LEN

    def test_title_banner_truncates_long_text(self, capsys):
        utils._create_title_banner("X" * 500)
        title = [l for l in capsys.readouterr().out.splitlines() if l][1]
        assert len(title) == PEP8_LINE_LEN
        assert "X" * 72 not in title

    def test_title_banner_odd_padding_keeps_width(self, capsys):
        utils._create_title_banner("ABCD")
        title = [l for l in capsys.readouterr().out.splitlines() if l][1]
        assert len(title) == PEP8_LINE_LEN

    def test_get_wrapped_lines_string(self):
        lines = utils._get_wrapped_lines("word " * 40, 20)
        assert len(lines) > 1
        assert all(len(l) <= 20 for l in lines)

    def test_get_wrapped_lines_list_mixed_types(self):
        lines = utils._get_wrapped_lines(["short", 42, None], 20)
        assert lines == ["short", "42", "None"]

    def test_get_wrapped_lines_unsupported_type(self):
        assert utils._get_wrapped_lines(123, 20) == []

    @pytest.mark.parametrize("bad", ["", [], None, 5, {"a": 1}])
    def test_subtitle_ignores_invalid_or_empty(self, bad, capsys):
        utils._create_subtitle_banner(bad)
        assert capsys.readouterr().out == ""

    def test_subtitle_lines_fixed_width_and_closed(self, capsys):
        utils._create_subtitle_banner(["one", "two"])
        lines = capsys.readouterr().out.splitlines()
        assert len(lines) == 3  # two rows + closing rule
        assert all(len(l) == PEP8_LINE_LEN for l in lines)
        assert lines[-1] == utils._make_top_btm_line()

    def test_subtitle_centered(self, capsys):
        utils._create_subtitle_banner("mid", center_text=True)
        row = capsys.readouterr().out.splitlines()[0]
        assert len(row) == PEP8_LINE_LEN
        assert row.index("mid") > 2

    def test_subtitle_centered_odd_padding_keeps_width(self, capsys):
        utils._create_subtitle_banner("ab", center_text=True)
        assert len(capsys.readouterr().out.splitlines()[0]) == PEP8_LINE_LEN

    def test_subtitle_strips_embedded_newlines(self, capsys):
        utils._create_subtitle_banner(["a\nb"])
        row = capsys.readouterr().out.splitlines()[0]
        assert "a b" in row and len(row) == PEP8_LINE_LEN

    def test_show_banner_with_subtitle(self, capsys):
        utils.show_banner("TITLE", "subtitle text")
        out = capsys.readouterr().out
        assert "TITLE" in out and "subtitle text" in out

    @pytest.mark.parametrize("empty", ["", None, []])
    def test_show_banner_without_subtitle(self, empty, capsys):
        utils.show_banner("TITLE", empty)
        out = capsys.readouterr().out
        assert "TITLE" in out
        assert out.count("+--") == 2  # title box only


class TestLogChatTranscript:
    def test_appends_entries(self, tmp_path, monkeypatch):
        log = tmp_path / "log.txt"
        monkeypatch.setattr(utils, "CHAT_TRANSCRIPT_FILE", str(log))

        utils.log_chat_transcript("STAGE ONE", "first body")
        utils.log_chat_transcript("STAGE TWO", "second body")

        text = log.read_text(encoding="utf-8")
        assert "STAGE: STAGE ONE" in text and "first body" in text
        assert text.index("first body") < text.index("second body")
        assert "=" * PEP8_LINE_LEN in text

    def test_timestamp_format(self, tmp_path, monkeypatch):
        log = tmp_path / "log.txt"
        monkeypatch.setattr(utils, "CHAT_TRANSCRIPT_FILE", str(log))
        utils.log_chat_transcript("S", "c")
        assert re.search(r"\[\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\] - STAGE: S", log.read_text())

    def test_unicode_content(self, tmp_path, monkeypatch):
        log = tmp_path / "log.txt"
        monkeypatch.setattr(utils, "CHAT_TRANSCRIPT_FILE", str(log))
        utils.log_chat_transcript("S", "résumé ✅")
        assert "résumé ✅" in log.read_text(encoding="utf-8")

    def test_does_not_truncate_existing_log(self, tmp_path, monkeypatch):
        log = tmp_path / "log.txt"
        log.write_text("existing\n", encoding="utf-8")
        monkeypatch.setattr(utils, "CHAT_TRANSCRIPT_FILE", str(log))
        utils.log_chat_transcript("S", "c")
        assert log.read_text(encoding="utf-8").startswith("existing\n")
