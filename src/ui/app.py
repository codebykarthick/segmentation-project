import dearpygui.dearpygui as dpg
from ui.components.config import get_config
from ui.components.preview import create_preview
from ui.components.sidebar import create_sidebar
from util.logger import setup_logger

config = get_config()
log = setup_logger()

log.info("Creating UI")

dpg.create_context()
dpg.create_viewport(title="CV Project UI",
                    width=config["width"], height=config["height"])
create_preview()
create_sidebar()

dpg.setup_dearpygui()
dpg.show_viewport()
dpg.start_dearpygui()
dpg.destroy_context()
