import dearpygui.dearpygui as dpg
from ui.components.config import get_config
from ui.components.preview import insert_image_into_preview, clear_prompt, generate_prompt
from util.logger import setup_logger

log = setup_logger()
config = get_config()
prompt_mode = False


def sidebar_callback(sender, app_data):
    """Handles model selection and opens file dialog"""
    global prompt_mode

    selected_model = dpg.get_value(sender)
    print(f"Selected Model: {selected_model}")

    if app_data == "Prompt-Based":
        prompt_mode = True
        dpg.set_value("instruction_text",
                      "Hold Shift & Drag to highlight areas")
        dpg.configure_item("clr_prompt_btn", show=True)
    else:
        prompt_mode = False
        dpg.configure_item("prompt_btn", show=False)
        dpg.configure_item("clr_prompt_btn", show=False)
        dpg.set_value("instruction_text", "")

    # Open file dialog for image selection
    with dpg.file_dialog(directory_selector=False, show=True, callback=file_selected_callback,
                         width=600, height=400, tag="file_selector",
                         default_path="./data/processed/Test/color/", default_filename=""):
        dpg.add_file_extension("JPG Image (*.jpg){.jpg}")
        dpg.add_file_extension("PNG Image (*.png){.png}")


def file_selected_callback(sender, app_data):
    """
    Handle file selection and load the image
    """
    global prompt_mode

    selected_file = app_data['file_path_name']
    print(f"Selected File: {selected_file}")
    dpg.set_value("selected_file_text", f"File Selected: {selected_file}")

    # Remove Placeholder Text
    if dpg.does_item_exist("preview_placeholder_text"):
        dpg.delete_item("preview_placeholder_text")

    insert_image_into_preview(selected_file, prompt_mode)


def segment_image_callback(sender, app_data):
    """
    Runs the segmentation model and replaces the image with the segmentation result
    """
    log.info("Capturing user highlighted prompt")

    # Check if this prompt is accurate for the provided image by dumping
    prompt = generate_prompt()


def create_sidebar():
    """
    Creates the sidebar UI
    """
    with dpg.window(label="Sidebar", width=250, height=config["height"], pos=(0, 0)):
        dpg.add_text("Select Model to use for segmentation task:", wrap=250)
        dpg.add_spacer(height=config["spacer"])
        with dpg.group():
            dpg.add_radio_button(["<none>", "UNet", "Autoencoder", "CLIP", "Prompt-Based"],
                                 default_value="<none>",
                                 callback=sidebar_callback, tag="model_selection")
            dpg.add_spacer(height=config["spacer"])
            dpg.add_text("", tag="instruction_text", wrap=250)
            dpg.add_spacer(height=config["spacer"])
            dpg.add_text("Selected File: ", tag="selected_file_text", wrap=250)
            dpg.add_spacer(height=config["spacer"])
            dpg.add_button(label="Clear Prompt", indent=60,
                           show=False, tag="clr_prompt_btn", callback=clear_prompt)
            dpg.add_spacer(height=config["spacer"])
            dpg.add_button(label="Run Segmentation\n  on Selection", indent=50,
                           callback=segment_image_callback)
