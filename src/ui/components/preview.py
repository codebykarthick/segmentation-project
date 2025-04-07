import dearpygui.dearpygui as dpg
import numpy as np
from ui.components.config import get_config
from util.logger import setup_logger

config = get_config()
preview_width = None
preview_height = None
prompt_mode = False
added_pixels = []
log = setup_logger()


def create_preview():
    with dpg.window(label="Image Preview", width=config["width"]-250,
                    height=config["height"], pos=(250, 0),
                    tag="img_preview"):
        dpg.add_text("No image loaded yet! Please select an image to continue.",
                     tag="preview_placeholder_text")
    create_prompt_overlay()


def create_prompt_overlay():
    with dpg.window(label="Overlay", width=config["width"]-250,
                    height=config["height"], pos=(250, 0),
                    tag="overlay", no_move=True, no_resize=True,
                    no_background=True):
        pass


def clear_prompt(sender, app_data):
    global added_pixels
    added_pixels = []
    draw_overlay()


def generate_prompt():
    global added_pixels
    width, height = preview_width, preview_height
    np.set_printoptions(threshold=np.inf, linewidth=200, suppress=True)
    prompt = np.zeros(shape=(width, height))
    brush_radius = config["brush_size"]

    log.info(f"Creating mask of size: ({width}x{height})")

    for (x, y) in added_pixels:
        x, y = int(x), int(y)  # Ensure integer indexing
        for dx in range(-brush_radius, brush_radius + 1):
            for dy in range(-brush_radius, brush_radius + 1):
                if 0 <= x + dx < width and 0 <= y + dy < height:
                    # Mark the pixel and surrounding area as 1
                    prompt[y + dy, x + dx] = 1

    return prompt


def insert_image_into_preview(selected_file, is_prompt):
    global preview_width, preview_height, prompt_mode, added_pixels

    # Remove Old Image Before Adding a New One
    if dpg.does_item_exist("preview_image"):
        dpg.delete_item("preview_image")

    if dpg.does_item_exist("preview_texture"):
        dpg.delete_item("preview_texture")

    # Load Image into Dear PyGui
    with dpg.texture_registry():
        width, height, _, data = dpg.load_image(selected_file)
        dpg.add_static_texture(width, height, data, tag="preview_texture")
        preview_width = width
        preview_height = height
        prompt_mode = is_prompt

    dpg.add_image("preview_texture", parent="img_preview",
                  tag="preview_image", width=preview_width, height=preview_height)

    added_pixels = []

    if prompt_mode:
        # Add event listeners to record the pixels that
        # need to be included for the prompting.
        with dpg.handler_registry():
            # Fires when mouse is pressed
            dpg.add_mouse_drag_handler(callback=on_mouse_drag, threshold=5)
            dpg.add_mouse_release_handler(callback=on_mouse_release)


def on_mouse_drag(sender, app_data):
    """
    Custom event handler to record the pixels selected as prompts by the user.
    User must be holding down the shift key for the system to record the pixels
    """
    if dpg.is_key_down(dpg.mvKey_LShift) or dpg.is_key_down(dpg.mvKey_RShift):
        x, y = dpg.get_mouse_pos(local=True)
        # To account for the border offset and preview zooming.
        x, y = (x-10), (y-10)
        if ((x > 0 and x <= preview_width) and (y > 0 and y <= preview_height)):
            added_pixels.append((x, y))

        if len(added_pixels) < 1000:
            draw_overlay()


def draw_overlay():
    if dpg.does_item_exist("show_selection"):
        dpg.delete_item("show_selection")

    with dpg.drawlist(preview_width, preview_height, parent="overlay", pos=(0, 0), tag="show_selection"):
        for (x, y) in added_pixels:
            dpg.draw_circle(center=(x, y), radius=config["brush_size"],
                            color=(255, 255, 0, 150), fill=(255, 255, 0, 100))


def on_mouse_release(sender, app_data):
    """
    Custom event handler to flush the selected pixels to the store for the generated prompt
    """
    if len(added_pixels) > 1000:
        draw_overlay()
    print(f"Recorded {len(added_pixels)} pixel points")
