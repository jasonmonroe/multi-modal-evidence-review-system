# code/src/utils.py

# +---------------------------------------------------------------------------+
# |                            UTILITY FUNCTIONS                              |
# +---------------------------------------------------------------------------+
# Utility Functions used throughout project.

# Python Libraries
import textwrap
import time
import uuid
from datetime import datetime

# Local Libraries
from src.constants import CHAT_TRANSCRIPT_FILE, MSEC, PEP8_LINE_LEN, SECS_IN_MIN


def get_run_id() -> str:
    """Generates a unique ID for the current run."""
    return uuid.uuid4().hex[:5].upper()


def start_timer() -> float:
    """
    Start a timer
    """
    return time.time()


def get_time(start_time_float: float, end_time_float: float | None = None) -> str:

    if end_time_float is None:
        end_time_float = time.time()

    diff = abs(end_time_float - start_time_float)
    _, remainder = divmod(diff, SECS_IN_MIN * SECS_IN_MIN)
    minutes, seconds = divmod(remainder, SECS_IN_MIN)
    fractional_seconds = seconds - int(seconds)

    ms = fractional_seconds * MSEC
    return f"{int(minutes)}m {int(seconds)}s {int(ms)}ms"


def show_timer(start_time_int: float) -> None:
    print(f"⏱ Run Time: {get_time(start_time_int)}")


def _make_top_btm_line() -> str:
    open_close_len = 2  # open close of char `+` or `|`
    max_line_len = PEP8_LINE_LEN - open_close_len

    return "+" + ("-" * max_line_len) + "+"


def _create_title_banner(text: str, center_text: bool = True) -> None:
    open_close_len = 4  # open close of char `+` or `|`
    max_line_len = PEP8_LINE_LEN - open_close_len

    # Trim off any chars after limit plus two spaces for blanks
    text = text[0 : max_line_len - open_close_len]
    text_len = len(text)
    padding_len = max_line_len - text_len

    if center_text:
        # If uneven padding add an extra length for the right side
        extra_len = 0 if padding_len % 2 == 0 else 1

        padding_len = padding_len // 2
        title_line = (
            "| " + (" " * padding_len) + text + (" " * (padding_len + extra_len)) + " |"
        )

    else:
        # Remove last two characters to account for open/close spacing
        title_line = "| " + text + (" " * padding_len) + " |"

    top_btm_line = _make_top_btm_line()

    # Print title banner
    print("\n")
    print(top_btm_line)
    print(title_line)
    print(top_btm_line)


def _create_subtitle_banner(text: str | list, center_text: bool = False) -> None:
    # Reconstructs the guard to safely catch wrong types OR empty values
    if not isinstance(text, (str, list)) or not text:
        return

    open_close_len = 4  # open close of char `+` or `|` plus space
    max_line_len = PEP8_LINE_LEN - open_close_len
    wrapped_lines = _get_wrapped_lines(text, max_line_len)

    # Now that the data is a list format it for display.
    for line in wrapped_lines:
        # Clean up any rogue newline markers so they don't break string length math
        line = line.replace("\n", " ").strip()
        line_len = len(line)
        padding_len = max_line_len - line_len

        if center_text:
            extra_len = 0 if padding_len % 2 == 0 else 1
            padding_len = padding_len // 2
            padded_line = (
                "| "
                + (" " * padding_len)
                + line
                + (" " * (padding_len + extra_len))
                + " |"
            )
        else:
            padded_line = "| " + line + (" " * padding_len) + " |"

        print(padded_line)

    # Close the subtitle
    if len(wrapped_lines) > 0:
        print(_make_top_btm_line())


def _get_wrapped_lines(text: str | list, max_line_len: int) -> list:
    wrapped_lines = []

    if isinstance(text, list):
        # Explicitly wrap each individual item inside the list
        for item in text:
            if isinstance(item, str):
                wrapped_lines.extend(textwrap.wrap(item, width=max_line_len))
            else:
                wrapped_lines.append(str(item))

    elif isinstance(text, str):
        wrapped_lines = textwrap.wrap(text, width=max_line_len)

    return wrapped_lines


def show_banner(
    title: str,
    subtitle: str | list | None = "",
    center_title_text: bool = True,
    center_subtitle_text: bool = False,
) -> None:
    _create_title_banner(title, center_title_text)

    if subtitle:
        _create_subtitle_banner(subtitle, center_subtitle_text)


def log_chat_transcript(stage: str, content: str) -> None:
    """
    Appends a formatted execution step or model interaction directly
    to the required evaluation log path.

    :param stage:
    :param content:
    :param filepath:
    :return:
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    divider = "=" * PEP8_LINE_LEN

    log_entry = f"\n{divider}\n[{timestamp}] - STAGE: {stage}\n{divider}\n{content}\n"

    # Ensure the file appends cleanly
    with open(CHAT_TRANSCRIPT_FILE, "a", encoding="utf-8") as log_file:
        log_file.write(log_entry)
