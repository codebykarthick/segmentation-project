import dearpygui.dearpygui as dpg
from ui.components.config import get_config
from ui.components.model_adapter import create_model_and_segment
from ui.components.preview import insert_image_into_preview, clear_prompt, generate_prompt
from util.logger import setup_logger

log = setup_logger()
config = get_config()
prompt_mode = False

model_to_weights = {
    "UNet": "unet",
    "Autoencoder": "autoencoder_segmentation",
    "CLIP": "clip_segmentation",
    "Prompt-Based": "prompt_segmentation"
}

model_path = None
model_type = None
image_path = None


def sidebar_callback(sender, app_data):
    """Handles model selection and opens file dialog"""
    global prompt_mode, model_type

    selected_model = dpg.get_value(sender)
    print(f"Selected Model: {selected_model}")

    if app_data == "Prompt-Based":
        prompt_mode = True
        dpg.set_value("instruction_text",
                      "Hold Shift & Drag to highlight areas")
        dpg.configure_item("clr_prompt_btn", show=True)
    else:
        prompt_mode = False
        dpg.configure_item("clr_prompt_btn", show=False)
        dpg.set_value("instruction_text", "")

    model_type = model_to_weights[selected_model]

    # Open file dialog for model path selection
    with dpg.file_dialog(directory_selector=False, show=True, callback=weights_selected_callback,
                         width=600, height=400, tag="weight_selector",
                         default_path=f"./weights/{model_type}/", default_filename=""):
        dpg.add_file_extension("PyTorch model weights (*.pth){.pth}")


def weights_selected_callback(sender, app_data):
    """
    Handle model weights selection and get the full path of the model.
    """
    global model_path
    model_path = app_data['file_path_name']
    print(f"Loading model weights from path: {model_path}")
    # Open file dialog for image selection
    with dpg.file_dialog(directory_selector=False, show=True, callback=image_selected_callback,
                         width=600, height=400, tag="file_selector",
                         default_path="./data/processed/Test/color/", default_filename=""):
        dpg.add_file_extension("JPG Image (*.jpg){.jpg}")
        dpg.add_file_extension("PNG Image (*.png){.png}")


def image_selected_callback(sender, app_data):
    """
    Handle file selection and load the image
    """
    global prompt_mode, image_path

    image_path = app_data['file_path_name']
    print(f"Selected File: {image_path}")
    dpg.set_value("selected_file_text", f"File Selected: {image_path}")

    # Remove Placeholder Text
    if dpg.does_item_exist("preview_placeholder_text"):
        dpg.delete_item("preview_placeholder_text")

    insert_image_into_preview(image_path, prompt_mode)


def segment_image_callback(sender, app_data):
    """
    Runs the segmentation model and replaces the image with the segmentation result
    """
    global prompt_mode, model_type, model_path, image_path
    log.info("Running segmentation")

    # Check if this prompt is accurate for the provided image by dumping
    if prompt_mode == True:
        prompt = generate_prompt()
        # Add the prompt to the input image as a 4th dimension
        # and then segment it
    else:
        prompt = None

    # We have the image ready segment it.
    final_mask_path = create_model_and_segment(
        image_path, model_type, model_path, prompt_mode, prompt)

    if prompt_mode == True:
        clear_prompt(sender, app_data)
    insert_image_into_preview(final_mask_path, prompt_mode)


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
