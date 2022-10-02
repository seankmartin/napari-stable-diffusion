import random

import numpy as np
import torch
from diffusers import StableDiffusionPipeline
from napari.qt.threading import thread_worker
from napari.utils.notifications import show_info
from torch import autocast
from tqdm import tqdm

YOUR_AUTH_CODE = "Your code"


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
    worker = make_model(widget)
    worker.start()


@thread_worker
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


def infer(pipe, prompt, samples, steps, scale, seed, no_filter):
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
        )


@thread_worker
def create_images(widget):
    pipe = widget.pipe
    num_images = widget.num_images.value
    num_iters = widget.num_iters.value
    prompt = widget.prompt.value
    no_filter = not widget.nsfw_filter.value

    max_tries = 6
    used_seeds = []
    images = []
    seeds_ld = []
    for _ in tqdm(range(num_images)):
        tries = 0
        seed = gen_random(used_seeds)
        img = infer(pipe, prompt, 1, num_iters, 7.5, seed, no_filter)
        while tries < max_tries and img.nsfw_content_detected[0]:
            show_info("Detected nsfw content - running again with new seed")
            seed = gen_random(used_seeds)
            img = infer(pipe, prompt, 1, num_iters, 7.5, seed, no_filter)
            tries += 1
        images.append(np.array(img.images[0]))
        seeds_ld.append({"seed": seed})
    widget.data = images
    widget.seeds = seeds_ld
