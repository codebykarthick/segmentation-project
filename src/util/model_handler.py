import curses
from .constants import CONSTANTS
from .logger import setup_logger
import os

log = setup_logger()


def list_model_weights(extension=".pth", sub_dir="", filters=[]):
    """ List all available model weights in the given directory. """
    actual_path = os.path.join(CONSTANTS["WEIGHTS_PATH"], sub_dir)
    if not os.path.exists(actual_path):
        log.error("No weights directory found.")
        return []

    return [
        f for f in sorted(os.listdir(actual_path))
        if f.endswith(extension) and (not filters or any(substr in f for substr in filters))
    ]


def select_model_weight(stdscr, weights):
    """ Interactive model selection using arrow keys. """
    curses.curs_set(0)  # Hide cursor
    idx = 0  # Start selection at first item

    while True:
        stdscr.clear()
        stdscr.addstr("Select a model weight file:\n", curses.A_BOLD)

        # Display weight files
        for i, weight in enumerate(weights):
            if i == idx:
                # Highlight selected
                stdscr.addstr(f"> {weight}\n", curses.A_REVERSE)
            else:
                stdscr.addstr(f"  {weight}\n")

        stdscr.refresh()

        key = stdscr.getch()

        if key == curses.KEY_UP and idx > 0:
            idx -= 1
        elif key == curses.KEY_DOWN and idx < len(weights) - 1:
            idx += 1
        elif key in [curses.KEY_ENTER, 10, 13]:  # Enter key pressed
            return weights[idx]


def load_selected_model(sub_dir: str = "", filters: list = []):
    """ Display model weights and let user select one. """
    weights = list_model_weights(sub_dir=sub_dir, filters=filters)

    if not weights:
        log.error("No model weights found!")
        return None

    # Launch interactive selection
    selected_weight = curses.wrapper(select_model_weight, weights)
    log.info(f"Selected model: {selected_weight}")
    selected_weight = os.path.join(
        os.getcwd(), CONSTANTS["WEIGHTS_PATH"], sub_dir, selected_weight)
    log.info(f"Full path of weight: {selected_weight}")

    return selected_weight
