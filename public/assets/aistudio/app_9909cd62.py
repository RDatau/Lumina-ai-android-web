import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import spaces
import random
import torch
import gradio as gr
from PIL import Image
from diffusers import QwenImage21Pipeline

MODEL_ID = os.environ.get("QWEN_IMAGE_MODEL", "Qwen/Qwen-Image-2.1")
LORA_REPO = os.environ.get(
    "UNCENSORED_LORA_REPO", "abenzerps/Qwen-Image-2.1-Uncensored-GGUF"
)
LORA_WEIGHT = os.environ.get(
    "UNCENSORED_LORA_WEIGHT", "qwen-image-2.1-uncensored-lora.safetensors"
)

SIZE_PRESETS = {
    "1024x1024 (1:1)": (1024, 1024),
    "1344x768 (16:9)": (1344, 768),
    "768x1344 (9:16)": (768, 1344),
    "1184x864 (4:3)": (1184, 864),
    "864x1184 (3:4)": (864, 1184),
}

pipe = QwenImage21Pipeline.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16)
pipe.load_lora_weights(LORA_REPO, weight_name=LORA_WEIGHT)
pipe.to("cuda")
pipe.vae.enable_tiling(
    tile_sample_min_height=1024,
    tile_sample_min_width=1024,
    tile_sample_stride_height=768,
    tile_sample_stride_width=768,
)


def _seed(seed: int, randomize_seed: bool) -> int:
    if randomize_seed or seed is None or int(seed) < 0:
        return random.randint(0, 2**32 - 1)
    return int(seed)


def _set_lora(scale: float) -> None:
    try:
        pipe.set_adapters(["default"], adapter_weights=[float(scale)])
    except Exception:
        pass


def _as_pil(img) -> Image.Image:
    if img is None:
        raise gr.Error("Upload an image.")
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    return Image.fromarray(img).convert("RGB")


@spaces.GPU(duration=120)
def generate(
    prompt: str,
    size: str,
    steps: int,
    seed: int,
    randomize_seed: bool,
    lora_scale: float,
) -> tuple:
    """Text-to-image with Qwen-Image-2.1 + uncensored LoRA."""
    if not prompt or not prompt.strip():
        raise gr.Error("Enter a prompt.")
    seed = _seed(seed, randomize_seed)
    width, height = SIZE_PRESETS.get(size, (1024, 1024))
    generator = torch.Generator("cuda").manual_seed(seed)
    _set_lora(lora_scale)
    image = pipe(
        prompt=prompt.strip(),
        width=width,
        height=height,
        num_inference_steps=int(steps),
        true_cfg_scale=1.0,
        generator=generator,
    ).images[0]
    return image, seed


@spaces.GPU(duration=120)
def edit(
    prompt: str,
    input_image,
    extra_images: list | None,
    steps: int,
    seed: int,
    randomize_seed: bool,
    lora_scale: float,
) -> tuple:
    """Edit 1–10 images with Qwen-Image-2.1 + uncensored LoRA."""
    if not prompt or not prompt.strip():
        raise gr.Error("Enter an edit instruction.")
    images = [_as_pil(input_image)]
    if extra_images:
        for item in extra_images:
            path = item[0] if isinstance(item, (list, tuple)) else item
            images.append(Image.open(path).convert("RGB"))
    if len(images) > 10:
        raise gr.Error("Max 10 reference images.")
    seed = _seed(seed, randomize_seed)
    generator = torch.Generator("cuda").manual_seed(seed)
    _set_lora(lora_scale)
    w, h = images[0].size
    # keep ~1K area so ZeroGPU large fits
    scale = (1024 * 1024 / max(w * h, 1)) ** 0.5
    width = max(64, int(w * scale) // 32 * 32)
    height = max(64, int(h * scale) // 32 * 32)
    out = pipe(
        prompt=prompt.strip(),
        image=images if len(images) > 1 else images[0],
        width=width,
        height=height,
        num_inference_steps=int(steps),
        true_cfg_scale=1.0,
        generator=generator,
    ).images[0]
    return out, seed


with gr.Blocks(title="Qwen-Image-2.1 Uncensored") as demo:
    gr.Markdown(
        """
# Qwen-Image-2.1 Uncensored — generate & edit

[abenzerps/Qwen-Image-2.1-Uncensored-GGUF](https://huggingface.co/abenzerps/Qwen-Image-2.1-Uncensored-GGUF)
via Diffusers + uncensored LoRA (not Comfy GGUF).

**Generate** = text-to-image. **Edit** = 1–10 refs (`image=` on the same pipeline).

**18+.** No safety filter.
"""
    )
    with gr.Tabs():
        with gr.Tab("Generate"):
            with gr.Row():
                with gr.Column():
                    g_prompt = gr.Textbox(label="Prompt", lines=4)
                    g_size = gr.Dropdown(
                        choices=list(SIZE_PRESETS.keys()),
                        value="1024x1024 (1:1)",
                        label="Size",
                    )
                    g_steps = gr.Slider(8, 50, value=28, step=1, label="Steps")
                    g_seed = gr.Number(value=0, label="Seed", precision=0)
                    g_rand = gr.Checkbox(value=True, label="Randomize seed")
                    g_lora = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="LoRA scale")
                    g_btn = gr.Button("Generate", variant="primary")
                with gr.Column():
                    g_out = gr.Image(type="pil", label="Image")
                    g_out_seed = gr.Number(label="Used seed", precision=0)
            gr.Examples(
                examples=[
                    ["A neon shop sign that reads \"QWEN IMAGE 2.1\", rainy night"],
                    ["Oil painting of a red fox in snow, soft morning light"],
                ],
                inputs=[g_prompt],
                cache_examples=False,
            )
            g_btn.click(
                generate,
                inputs=[g_prompt, g_size, g_steps, g_seed, g_rand, g_lora],
                outputs=[g_out, g_out_seed],
            )

        with gr.Tab("Edit"):
            with gr.Row():
                with gr.Column():
                    e_image = gr.Image(type="pil", label="Main image")
                    e_extra = gr.Gallery(
                        label="Extra refs (optional, up to 9)",
                        type="filepath",
                        columns=3,
                    )
                    e_prompt = gr.Textbox(
                        label="Edit instruction",
                        lines=3,
                        placeholder="Change the background to a sunset beach",
                    )
                    e_steps = gr.Slider(8, 50, value=28, step=1, label="Steps")
                    e_seed = gr.Number(value=0, label="Seed", precision=0)
                    e_rand = gr.Checkbox(value=True, label="Randomize seed")
                    e_lora = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="LoRA scale")
                    e_btn = gr.Button("Edit", variant="primary")
                with gr.Column():
                    e_out = gr.Image(type="pil", label="Edited")
                    e_out_seed = gr.Number(label="Used seed", precision=0)
            e_btn.click(
                edit,
                inputs=[e_prompt, e_image, e_extra, e_steps, e_seed, e_rand, e_lora],
                outputs=[e_out, e_out_seed],
            )

if __name__ == "__main__":
    demo.launch(mcp_server=True)
