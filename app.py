import napari

viewer = napari.Viewer()
viewer.window.add_plugin_dock_widget("napari-stable-diffusion", "Diffusion")
