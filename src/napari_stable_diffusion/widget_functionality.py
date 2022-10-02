import random

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from torch import autocast
from tqdm import tqdm

YOUR_AUTH_CODE = ""


def gen_random(used_seeds):
    while True:
        seed = random.randint(0, 18446744073709551615)
        if seed not in used_seeds:
            break
    used_seeds.append(seed)
    return seed


def load_model_func(widget):
    if hasattr(widget, "pipe"):
        show_info("Already loaded the model")
        return
    make_model(widget)


def set_ready(widget):
    widget.call_button.text = "Run"
    widget.call_button.enabled = True


@thread_worker(connect={"returned": set_ready})
def make_model(widget):
    show_info("Making the model...")
    model_id = "CompVis/stable-diffusion-v1-4"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        use_auth_token=YOUR_AUTH_CODE,
        revision="fp16",
        torch_dtype=torch.float16,
        cache_dir="model",
        resume_download=True,
    )
    pipe = pipe.to("cuda")
    torch.backends.cudnn.benchmark = True
    show_info("Done making the model...")
    widget.pipe = pipe
    return widget


def infer(pipe, prompt, samples, steps, scale, seed, no_filter, width, height):
    def dummy(images, **kwargs):
        return images, [False]

    if no_filter:
        pipe.safety_checker = dummy

    generator = torch.Generator(device="cuda").manual_seed(seed)

    with autocast("cuda"):
        return pipe(
            [prompt] * samples,
            num_inference_steps=steps,
            guidance_scale=scale,
            generator=generator,
            height=height,
            width=width,
        )


def display_data(widget):
    try:
        widget.viewer.value.layers[widget.prompt.value].data = widget._data
    except KeyError:
        widget.viewer.value.add_image(widget._data, name=widget.prompt.value)


@thread_worker
def create_images(widget):
    pipe = widget.pipe
    num_images = widget.num_images.value
    num_iters = widget.num_iters.value
    prompt = widget.prompt.value
    no_filter = not widget.nsfw_filter.value
    width = widget.img_width.value
    height = widget.img_height.value

    max_tries = 6
    used_seeds = []
    images = []
    seeds_ld = []
    for _ in tqdm(range(num_images)):
        tries = 0
        seed = gen_random(used_seeds)
        img = infer(
            pipe, prompt, 1, num_iters, 7.5, seed, no_filter, width, height
        )
        while tries < max_tries and img.nsfw_content_detected[0]:
            show_info("Detected nsfw content - running again with new seed")
            seed = gen_random(used_seeds)
            img = infer(
                pipe, prompt, 1, num_iters, 7.5, seed, no_filter, width, height
            )
            tries += 1
        images.append(np.array(img.images[0]))
        widget._data = np.array(images)
        widget._seeds = seeds_ld
        seeds_ld.append({"seed": seed})
        yield widget
