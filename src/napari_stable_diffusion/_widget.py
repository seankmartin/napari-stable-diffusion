import torch
import random
from pathlib import Path
from torch import autocast
from diffusers import StableDiffusionPipeline
from skm_pyutils.plot import GridFig
from skm_pyutils.path import get_all_files_in_dir
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib as mpl
import typer
from tqdm import tqdm
from math import ceil
from magicgui import magic_factory

app = typer.Typer()

YOUR_AUTH_CODE = "Can't steal this"


def make_model():
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
    return pipe


def infer(pipe, prompt, samples, steps, scale, seed, no_filter):
    def dummy (images, **kwargs):
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


def gen_random(used_seeds):
    while True:
        seed = random.randint(0, 18446744073709551615)
        if seed not in used_seeds:
            break
    used_seeds.append(seed)
    return seed


def create_images(pipe, prompt, num_images, num_iters, output_dir, no_filter):
    output_dir.mkdir(exist_ok=True)
    image_paths = []
    max_tries = 6
    used_seeds = []
    for _ in tqdm(range(num_images)):
        tries = 0
        seed = gen_random(used_seeds)
        img = infer(pipe, prompt, 1, num_iters, 7.5, seed, no_filter)
        while tries < max_tries and img.nsfw_content_detected[0]:
            print("Detected nsfw content - running again with new seed")
            seed = gen_random(used_seeds)
            img = infer(pipe, prompt, 1, num_iters, 7.5, seed, no_filter)
            tries += 1
        img_path = output_dir / f"{seed}.png"
        img.images[0].save(img_path)
        image_paths.append(img_path)
    print(used_seeds)


def grid_images(dir_, num_images, dpi=200):
    image_paths = get_all_files_in_dir(dir_, ext=".png")[:num_images]
    mpl.rcParams["figure.subplot.left"] = 0.08
    mpl.rcParams["figure.subplot.right"] = 0.92
    mpl.rcParams["figure.subplot.bottom"] = 0.1
    mpl.rcParams["figure.subplot.top"] = 0.9

    if num_images == 4:
        rows = 2
        cols = 2
    else:
        rows = ceil(num_images / 3)
        cols = min(num_images, 3)

    gf = GridFig(
        rows=rows,
        cols=cols,
        size_multiplier_x=2,
        size_multiplier_y=2,
        wspace=0.12,
        hspace=0.12,
        tight_layout=True,
    )
    for p in image_paths:
        img = mpimg.imread(p)
        ax = gf.get_next()
        plt.axis("off")
        ax.imshow(img)
    gf.savefig(dir_ / f"{dir_.name}.png", dpi=dpi)
    plt.close(gf.get_fig())


def _on_init(self):
    self.pipe = make_model()

@magic_factory(
    widget_init=_on_init
)
def diffusion_widget(
    prompt: str,
    num_images: int = 9,
    num_iters: int = 60,
    overwrite: bool = False,
    dpi: int = 300,
    model: bool = True,
    nsfw_filter: bool = True,
) -> Image:
    output_dir = Path(prompt.replace(" ", "-").replace(":", "_"))
    if not output_dir.exists() or overwrite:
        create_images(pipe, prompt, num_images, num_iters, output_dir, no_filter=not nsfw_filter)
    grid_images(output_dir, num_images, dpi=dpi)
