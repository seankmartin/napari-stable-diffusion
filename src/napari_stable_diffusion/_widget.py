from magicgui import magicgui
import napari
from .widget_functionality import create_images, load_model_func, display_data
from napari.utils.notifications import show_info


def diffusion_widget():
    @magicgui(
        call_button=True,
        persist=True,
        stop_button=dict(widget_type="PushButton", text="Stop running"),
        img_width=dict(
            widget_type="SpinBox", min=256, max=1920, step=8, value=512
        ),
        img_height=dict(
            widget_type="SpinBox", min=256, max=1080, step=8, value=512
        ),
    )
    def widget(
        viewer: napari.Viewer,
        stop_button,
        img_width,
        img_height,
        prompt: str,
        num_iters: int = 60,
        num_images: int = 9,
        guidance_scale: float = 7.5,
        nsfw_filter: bool = True,
    ):
        @widget.call_button.changed.connect
        def run():
            if widget.started:
                widget.worker.quit()
                widget.worker = create_images(widget)
                widget.worker.yielded.connect(display_data)
                widget.worker.start()
            else:
                widget.worker.start()
                widget.started = True

    @widget.stop_button.changed.connect
    def quit():
        show_info("Closing worker")
        widget.worker.quit()

    widget.started = False
    widget.call_button.text = "Loading..."
    widget.call_button.enabled = False
    load_model_func(widget)
    widget.worker = create_images(widget)
    widget.worker.yielded.connect(display_data)

    return widget
