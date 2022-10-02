from magicgui import magicgui
import napari
from .widget_functionality import create_images, load_model_func


def diffusion_widget():
    @magicgui(
        call_button=True,
        load_model_button=dict(widget_type="PushButton", text="Load Model"),
    )
    def widget(
        load_model_button,
        prompt: str,
        num_iters: int = 60,
        num_images: int = 9,
        nsfw_filter: bool = True,
    ):
        @widget.call_button.changed.connect
        def run():
            worker = create_images(widget)
            worker.returned.connect(display_data)
            worker.start()

    def display_data():
        metadata = widget.seeds
        # TODO don't get a new napari to launch
        napari.view_image(widget.data, name=widget.prompt.value)

    @widget.load_model_button.changed.connect
    def load_model(event=None):
        load_model_func(widget)

    return widget
