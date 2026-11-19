"""BLIP captions with the ORIGINAL Salesforce code and checkpoint (pinned, checksummed).

    python -m harness.caption_blip_orig --split dev

Writes ``$VTAS_DATA/captions/blip_orig_paper_<split>.json`` = {"config": ..., "captions":
{image_id: caption}}. Captions are data and are not committed.

Why not Hugging Face. ``Salesforce/blip-image-captioning-base`` (used by
``caption_blip.py`` in Chat 1) never emits WordPiece continuation tokens: on the dev split
it writes "a gi in a zoo" for giraffe, "frc" for frisbee, "fire hydra", and loops such as
"pew pew pew ...". It does the same under transformers 4.36.2 and 4.46.3. The original
BLIP code with the COCO caption checkpoint writes the full words on the same images
(Chat 2 diagnosis, PROGRESS.md). CHAIR results must use this module.

Model. BLIP ViT-B/16, checkpoint ``model_base_caption_capfilt_large.pth`` (BLIP paper,
COCO captioning, CapFilt-L), the checkpoint named in the repo's configs/caption_coco.yaml.
Source files come from github.com/salesforce/BLIP at a pinned commit, each SHA-256
checked. The weights are loaded with ``torch.load(weights_only=True)`` (no pickled code
runs). The key handling of BLIP's ``load_checkpoint`` (pos-embed interpolation, dropping
shape mismatches) is repeated here; strict loading then requires zero missing keys.

Decoding ("paper", configs/caption_coco.yaml): prompt "a picture of ", 3 beams,
max_length 20, min_length 5 (BLIP counts the prompt tokens in both), repetition_penalty
1.0; prompt stripped as in BLIP. Image transform as in BLIP's caption eval: 384x384
bicubic resize, CLIP mean/std normalisation.

One deliberate change to BLIP_Decoder.generate: BLIP repeats the image embeddings once
per beam because the transformers release it was written for (4.15) did not.
transformers >= 4.27 expands every model kwarg per beam inside generate(), so keeping
BLIP's repeat would expand twice (3 x 3 = 9 rows: a shape error). The repeat is dropped;
everything else in that function is unchanged.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path

from harness import data_dir
from harness.coco import image_path, load_split
from harness.download import fetch

BLIP_COMMIT = "056a169437371659074aa2732649d5de3bffb4a8"
BLIP_RAW = f"https://raw.githubusercontent.com/salesforce/BLIP/{BLIP_COMMIT}/" + "{}"
BLIP_FILES = {  # path in the repo -> sha256 of the raw (LF) file
    "models/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "models/blip.py": "118f1f96297af174d665fb75a62ce6f93c8451a8681458fff5a4240f444d27ea",
    "models/med.py": "faf785d868e0f758a27cc621c238a762e1c870328ac104e178d8c7f9db9b9159",
    "models/vit.py": "c16ef04bd471dc4d6ea1cae57982891670949bd9117ad209750bc88622539454",
    "configs/med_config.json": "d8b5029cc3c0124313d17009d043f0701bbe497f3721fa4ceb818720fe665326",
    "LICENSE.txt": "7d5df2d28b2bf297ecec0b51eb23f6d8d0c39338dae8d92f6d61d9441f77b856",
}
CHECKPOINT_URL = ("https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/"
                  "model_base_caption_capfilt_large.pth")
CHECKPOINT_SHA256 = "96ac8749bd0a568c274ebe302b3a3748ab9be614c737f3d8c529697139174086"
TOKENIZER_ID = "google-bert/bert-base-uncased"
TOKENIZER_REVISION = "86b5e0934494bd15c9632b12f734a8a67f723594"
IMAGE_SIZE = 384
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
VARIANTS = {
    "paper": {"prompt": "a picture of ",
              "generate": {"num_beams": 3, "max_length": 20, "min_length": 5,
                           "repetition_penalty": 1.0}},
}


def blip_dir() -> Path:
    return data_dir() / "external" / f"BLIP-{BLIP_COMMIT[:7]}"


def download() -> dict[str, Path]:
    """Fetch the pinned BLIP source files and the checkpoint (each SHA-256 verified)."""
    root = blip_dir()
    for rel, sha in BLIP_FILES.items():
        fetch(BLIP_RAW.format(rel), root / rel, sha)
    ckpt = fetch(CHECKPOINT_URL, data_dir() / "weights" / "model_base_caption_capfilt_large.pth",
                 CHECKPOINT_SHA256)
    return {"blip_dir": root, "checkpoint": ckpt}


def _import_blip(root: Path):
    """Import BLIP's ``models.blip`` from the pinned files."""
    try:
        importlib.import_module("fairscale")
    except ImportError:
        # vit.py imports fairscale only for activation checkpointing (training);
        # inference never calls it. A stub keeps fairscale out of the dependencies.
        for name in ("fairscale", "fairscale.nn", "fairscale.nn.checkpoint",
                     "fairscale.nn.checkpoint.checkpoint_activations"):
            sys.modules.setdefault(name, types.ModuleType(name))
        sys.modules["fairscale.nn.checkpoint.checkpoint_activations"].checkpoint_wrapper = \
            lambda module, **kw: module
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    blip = importlib.import_module("models.blip")

    def init_tokenizer():  # BLIP's init_tokenizer with the tokenizer revision pinned
        tok = blip.BertTokenizer.from_pretrained(TOKENIZER_ID, revision=TOKENIZER_REVISION)
        tok.add_special_tokens({"bos_token": "[DEC]"})
        tok.add_special_tokens({"additional_special_tokens": ["[ENC]"]})
        tok.enc_token_id = tok.additional_special_tokens_ids[0]
        return tok

    blip.init_tokenizer = init_tokenizer
    return blip


def load_model(paths: dict[str, Path], prompt: str, device: str):
    import torch

    blip = _import_blip(paths["blip_dir"])
    model = blip.BLIP_Decoder(image_size=IMAGE_SIZE, vit="base", prompt=prompt,
                              med_config=str(paths["blip_dir"] / "configs" / "med_config.json"))
    state = torch.load(paths["checkpoint"], map_location="cpu", weights_only=True)["model"]
    state["visual_encoder.pos_embed"] = blip.interpolate_pos_embed(
        state["visual_encoder.pos_embed"], model.visual_encoder)
    own = model.state_dict()
    dropped = [k for k in own if k in state and state[k].shape != own[k].shape]
    for k in dropped:
        del state[k]
    msg = model.load_state_dict(state, strict=False)
    if msg.missing_keys or dropped:
        raise RuntimeError(f"checkpoint mismatch: missing {msg.missing_keys[:5]}, "
                           f"shape-dropped {dropped[:5]}")
    return model.to(device).eval(), list(msg.unexpected_keys)


def beam_generate(model, image, num_beams: int, max_length: int, min_length: int,
                  repetition_penalty: float) -> list[str]:
    """BLIP_Decoder.generate, beam branch, without the second per-beam expansion."""
    import torch

    image_embeds = model.visual_encoder(image)
    image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
    model_kwargs = {"encoder_hidden_states": image_embeds, "encoder_attention_mask": image_atts}
    prompt = [model.prompt] * image.size(0)
    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(image.device)
    input_ids[:, 0] = model.tokenizer.bos_token_id
    input_ids = input_ids[:, :-1]
    outputs = model.text_decoder.generate(input_ids=input_ids, max_length=max_length,
                                          min_length=min_length, num_beams=num_beams,
                                          eos_token_id=model.tokenizer.sep_token_id,
                                          pad_token_id=model.tokenizer.pad_token_id,
                                          repetition_penalty=repetition_penalty,
                                          **model_kwargs)
    return [model.tokenizer.decode(o, skip_special_tokens=True)[len(model.prompt):]
            for o in outputs]


def generate(image_ids, variant: str = "paper", batch_size: int = 1,
             device: str | None = None) -> tuple[dict[int, str], dict]:
    import torch
    import transformers
    from PIL import Image
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    if not (4, 27) <= (major, minor) < (4, 50):
        # >= 4.27: per-beam expansion of model kwargs (see docstring). < 4.50: BLIP's
        # BertLMHeadModel gets generate() only through PreTrainedModel's GenerationMixin.
        raise RuntimeError(f"needs 4.27 <= transformers < 4.50, found {transformers.__version__}")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = VARIANTS[variant]
    paths = download()
    model, unexpected = load_model(paths, cfg["prompt"], device)
    tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    out: dict[int, str] = {}
    ids = list(image_ids)
    for start in range(0, len(ids), batch_size):
        batch = ids[start:start + batch_size]
        x = torch.stack([tf(Image.open(image_path(i)).convert("RGB")) for i in batch]).to(device)
        with torch.inference_mode():
            caps = beam_generate(model, x, **cfg["generate"])
        for i, c in zip(batch, caps):
            out[int(i)] = c.strip()
        print(f"\r{min(start + batch_size, len(ids))}/{len(ids)}", end="", flush=True)
    print()
    import timm

    config = {
        "implementation": "salesforce/BLIP (original code)", "blip_commit": BLIP_COMMIT,
        "checkpoint": CHECKPOINT_URL.rsplit("/", 1)[1], "checkpoint_sha256": CHECKPOINT_SHA256,
        "tokenizer": TOKENIZER_ID, "tokenizer_revision": TOKENIZER_REVISION,
        "variant": variant, **cfg, "image_size": IMAGE_SIZE, "batch_size": batch_size,
        "dtype": "float32", "device": device, "torch": torch.__version__,
        "transformers": transformers.__version__, "timm": timm.__version__,
        "unexpected_checkpoint_keys": len(unexpected),
        "deviation": "image_embeds not repeated per beam (transformers >= 4.27 does it)",
    }
    return out, config


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="python -m harness.caption_blip_orig")
    p.add_argument("--split", default="dev")
    p.add_argument("--variant", choices=list(VARIANTS), default="paper")
    p.add_argument("--batch-size", type=int, default=1,
                   help="1 (default) = exactly BLIP's per-image beam search")
    p.add_argument("--download-only", action="store_true")
    args = p.parse_args(argv)
    if args.download_only:
        print(download())
        return
    caps, config = generate(load_split(args.split), args.variant, args.batch_size)
    config["split"] = args.split
    dest = data_dir() / "captions" / f"blip_orig_{args.variant}_{args.split}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps({"config": config, "captions": caps}, indent=1), encoding="utf-8")
    print(f"wrote {len(caps)} captions to {dest}")


if __name__ == "__main__":
    main()
