
from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import Optional

import torch
from PIL import Image

from app.config import get_settings


# ============================================================
# ComicCraft - AI Image Generator
# ============================================================
#
# Uses:
#   Hugging Face Diffusers
#   Stable Diffusion v1.5
#
# Main functions:
#   generate_image(prompt)
#   generate_panel_image(prompt, panel_number)
#
# Output:
#   /static/panels/<filename>.png
#
# ============================================================


# ------------------------------------------------------------
# Project directories
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

STATIC_DIR = PROJECT_ROOT / "static"
PANELS_DIR = STATIC_DIR / "panels"

PANELS_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Default configuration
# ------------------------------------------------------------

# Current Hugging Face repository for Stable Diffusion v1.5.
#
# This replaces the old:
# runwayml/stable-diffusion-v1-5
#
# The model is documented by Hugging Face as:
# stable-diffusion-v1-5/stable-diffusion-v1-5
DEFAULT_MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"


IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512

# Keep this moderate so CPU systems do not take excessively long.
NUM_INFERENCE_STEPS = 20

GUIDANCE_SCALE = 7.5


# Cached pipeline.
_PIPELINE = None
_PIPELINE_MODEL_ID = None


# ------------------------------------------------------------
# Device detection
# ------------------------------------------------------------

def _get_device() -> str:
    """
    Select the best available device.

    NVIDIA CUDA GPU -> cuda
    Otherwise -> cpu
    """

    if torch.cuda.is_available():
        return "cuda"

    return "cpu"


# ------------------------------------------------------------
# Data type
# ------------------------------------------------------------

def _get_dtype(device: str):
    """
    Select a safe torch dtype.

    CUDA:
        float16

    CPU:
        float32
    """

    if device == "cuda":
        return torch.float16

    return torch.float32


# ------------------------------------------------------------
# Hugging Face token
# ------------------------------------------------------------

def _get_hf_token() -> Optional[str]:
    """
    Read the Hugging Face token.

    The project config uses:

        hf_api_key

    The function also checks common environment variable names.
    """

    # First use application settings.
    try:
        settings = get_settings()

        value = getattr(settings, "hf_api_key", None)

        if value:
            return str(value).strip()

    except Exception:
        pass

    # Fallback environment variables.
    for env_name in (
        "HF_API_KEY",
        "HUGGINGFACE_API_KEY",
        "HUGGINGFACEHUB_API_TOKEN",
        "HF_TOKEN",
    ):
        value = os.getenv(env_name)

        if value:
            return value.strip()

    return None


# ------------------------------------------------------------
# Model ID
# ------------------------------------------------------------

def _get_model_id() -> str:
    """
    Get the model from config.py / .env.

    If image_model is empty, use the current Stable Diffusion
    v1.5 repository.
    """

    try:
        settings = get_settings()

        model_id = getattr(settings, "image_model", None)

        if model_id:
            return str(model_id).strip()

    except Exception:
        pass

    return DEFAULT_MODEL_ID


# ------------------------------------------------------------
# Load Stable Diffusion
# ------------------------------------------------------------

def _load_pipeline():
    """
    Load the Stable Diffusion pipeline once.

    The model is cached in memory after the first successful load.
    """

    global _PIPELINE
    global _PIPELINE_MODEL_ID

    model_id = _get_model_id()

    # Return existing pipeline if it already uses the requested model.
    if _PIPELINE is not None and _PIPELINE_MODEL_ID == model_id:
        return _PIPELINE

    try:
        from diffusers import StableDiffusionPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Diffusers is not installed.\n\n"
            "Run:\n"
            "python -m pip install diffusers transformers accelerate torch"
        ) from exc

    device = _get_device()
    dtype = _get_dtype(device)
    hf_token = _get_hf_token()

    print()
    print("=" * 65)
    print("ComicCraft AI Image Generator")
    print("=" * 65)
    print(f"Model : {model_id}")
    print(f"Device: {device}")
    print(f"Token : {'configured' if hf_token else 'not configured'}")
    print("=" * 65)

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    try:
        load_kwargs = {
            "torch_dtype": dtype,
            "use_safetensors": True,
        }

        # Token is passed only when configured.
        if hf_token:
            load_kwargs["token"] = hf_token

        print("Loading Stable Diffusion model...")
        print("The first download can take several minutes.")
        print()

        try:
            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                **load_kwargs,
            )

        except TypeError:
            # Compatibility with older Diffusers versions.
            load_kwargs.pop("token", None)

            if hf_token:
                load_kwargs["use_auth_token"] = hf_token

            pipeline = StableDiffusionPipeline.from_pretrained(
                model_id,
                **load_kwargs,
            )

    except Exception as exc:

        error_text = str(exc)

        print()
        print("=" * 65)
        print("Stable Diffusion loading failed")
        print("=" * 65)
        print(error_text)
        print("=" * 65)

        raise RuntimeError(
            "Unable to load the Stable Diffusion model.\n\n"
            f"Model: {model_id}\n\n"
            "Check the following:\n"
            "1. Internet connection is working.\n"
            "2. Hugging Face token is correctly stored in .env.\n"
            "3. The model repository is accessible.\n"
            "4. Your computer has enough RAM/storage.\n"
            "5. The first model download has completed successfully.\n\n"
            f"Original error: {error_text}"
        ) from exc

    # --------------------------------------------------------
    # Move model to device
    # --------------------------------------------------------

    try:
        pipeline = pipeline.to(device)

    except Exception as exc:
        raise RuntimeError(
            f"Unable to move Stable Diffusion model to {device}: {exc}"
        ) from exc

    # --------------------------------------------------------
    # Memory optimizations
    # --------------------------------------------------------

    # Attention slicing works on CPU and GPU and reduces memory usage.
    try:
        pipeline.enable_attention_slicing()
    except Exception:
        pass

    # xFormers is optional.
    try:
        pipeline.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # Disable safety checker only if the pipeline allows it.
    # We do NOT force-disable it.

    _PIPELINE = pipeline
    _PIPELINE_MODEL_ID = model_id

    print()
    print("Stable Diffusion pipeline loaded successfully.")
    print()

    return _PIPELINE


# ------------------------------------------------------------
# Prompt cleaning
# ------------------------------------------------------------

def _clean_prompt(prompt: str) -> str:
    """
    Clean the incoming prompt.
    """

    if prompt is None:
        prompt = ""

    prompt = str(prompt).strip()

    # Remove excessive whitespace.
    prompt = re.sub(r"\s+", " ", prompt)

    # Avoid unnecessarily huge prompts.
    return prompt[:1800]


# ------------------------------------------------------------
# Comic prompt
# ------------------------------------------------------------

def _build_comic_prompt(prompt: str) -> str:
    """
    Convert the story scene into a comic-style image prompt.
    """

    prompt = _clean_prompt(prompt)

    if not prompt:
        prompt = (
            "a brave young character standing in an enchanted forest"
        )

    return (
        "professional comic book illustration, "
        "high quality digital comic art, "
        "detailed character design, "
        "expressive face, "
        "clear facial features, "
        "dynamic storytelling composition, "
        "strong foreground and background separation, "
        "cinematic lighting, "
        "vibrant colors, "
        "clean ink line art, "
        "polished digital painting, "
        "beautiful environment, "
        "professional illustration, "
        "single coherent scene, "
        "no text, no letters, no watermark, "
        f"{prompt}"
    )


# ------------------------------------------------------------
# Negative prompt
# ------------------------------------------------------------

def _negative_prompt() -> str:
    """
    Reduce common Stable Diffusion artifacts.
    """

    return (
        "blurry, low quality, low resolution, "
        "bad anatomy, deformed anatomy, distorted face, "
        "deformed hands, malformed hands, extra fingers, "
        "missing fingers, extra arms, extra legs, "
        "duplicate person, duplicate character, "
        "duplicate objects, disfigured, "
        "cropped, out of frame, "
        "text, letters, words, typography, logo, watermark, "
        "signature, jpeg artifacts, noisy, ugly, "
        "oversaturated, poorly drawn"
    )


# ------------------------------------------------------------
# Safe filename
# ------------------------------------------------------------

def _safe_filename(prefix: str = "comic_panel") -> str:
    """
    Create a unique safe PNG filename.
    """

    clean_prefix = re.sub(
        r"[^a-zA-Z0-9_-]+",
        "_",
        prefix,
    ).strip("_")

    if not clean_prefix:
        clean_prefix = "comic_panel"

    return f"{clean_prefix}_{uuid.uuid4().hex[:10]}.png"


# ------------------------------------------------------------
# Generate image
# ------------------------------------------------------------

def generate_image(
    prompt: str,
    filename: Optional[str] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Generate one comic panel.

    Parameters
    ----------
    prompt:
        Scene description.

    filename:
        Optional output filename.

    seed:
        Optional deterministic seed.

    Returns
    -------
    str
        Static URL:
        /static/panels/<filename>.png
    """

    pipeline = _load_pipeline()

    device = _get_device()

    final_prompt = _build_comic_prompt(prompt)
    negative_prompt = _negative_prompt()

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    if seed is None:
        seed = int.from_bytes(os.urandom(4), "big")

    if device == "cuda":
        generator = torch.Generator(
            device="cuda"
        ).manual_seed(seed)
    else:
        generator = torch.Generator(
            device="cpu"
        ).manual_seed(seed)

    print()
    print("-" * 65)
    print("Generating ComicCraft image...")
    print(f"Seed: {seed}")
    print("-" * 65)

    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    try:

        with torch.inference_mode():

            result = pipeline(
                prompt=final_prompt,
                negative_prompt=negative_prompt,
                height=IMAGE_HEIGHT,
                width=IMAGE_WIDTH,
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                generator=generator,
            )

        if not result.images:
            raise RuntimeError(
                "Stable Diffusion returned no images."
            )

        image = result.images[0]

    except Exception as exc:
        raise RuntimeError(
            f"ComicCraft image generation failed: {exc}"
        ) from exc

    # --------------------------------------------------------
    # Validate image
    # --------------------------------------------------------

    if not isinstance(image, Image.Image):
        raise RuntimeError(
            "Stable Diffusion returned an invalid image."
        )

    # --------------------------------------------------------
    # Filename
    # --------------------------------------------------------

    if filename:

        filename = Path(filename).name

        if not filename.lower().endswith(".png"):
            filename += ".png"

    else:

        filename = _safe_filename()

    output_path = PANELS_DIR / filename

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    try:

        image.save(
            output_path,
            format="PNG",
            optimize=True,
        )

    except Exception as exc:

        raise RuntimeError(
            f"Unable to save generated image: {exc}"
        ) from exc

    print(f"Image saved: {output_path}")
    print()

    return f"/static/panels/{output_path.name}"


# ------------------------------------------------------------
# Generate comic panel
# ------------------------------------------------------------

def generate_panel_image(
    image_prompt: str,
    panel_number: Optional[int] = None,
) -> str:
    """
    Generate a numbered comic panel.
    """

    if panel_number is not None:

        filename = (
            f"panel_{panel_number}_"
            f"{uuid.uuid4().hex[:8]}.png"
        )

    else:

        filename = None

    return generate_image(
        prompt=image_prompt,
        filename=filename,
    )


# ------------------------------------------------------------
# Clear generated images
# ------------------------------------------------------------

def clear_generated_images() -> None:
    """
    Delete generated panel images.
    """

    if not PANELS_DIR.exists():
        return

    for file in PANELS_DIR.iterdir():

        if not file.is_file():
            continue

        if file.suffix.lower() not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
        }:
            continue

        try:
            file.unlink()
        except OSError:
            pass


# ------------------------------------------------------------
# Status
# ------------------------------------------------------------

def image_generator_status() -> dict:
    """
    Return diagnostic information.
    """

    device = _get_device()
    model_id = _get_model_id()

    return {
        "model": model_id,
        "device": device,
        "pipeline_loaded": _PIPELINE is not None,
        "output_directory": str(PANELS_DIR),
        "image_size": f"{IMAGE_WIDTH}x{IMAGE_HEIGHT}",
        "inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "huggingface_token_configured": bool(
            _get_hf_token()
        ),
    }

