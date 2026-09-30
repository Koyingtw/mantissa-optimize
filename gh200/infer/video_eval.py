"""Text-to-video quality under reduced attention precision (CogVideoX, diffusers).

For each prompt and a fixed seed, generates the video once per attention mode (the linear layers
stay in the model dtype) and reports vs the native-attention video of the same prompt/seed:
PSNR and SSIM (frame-averaged), plus CLIP text-frame similarity (CLIPSIM) and consecutive-frame
CLIP similarity (temporal consistency) for every mode.  Attention modes as ppl_eval.make_attention:
'' (native), 'fp8' (per-head scales), 'fp8t' (per-tensor, FlashAttention-3 style), 'bf16:<k>'.
Saves the middle frame of every video as PNG and a side-by-side strip per prompt.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ppl_eval  # noqa: E402

PROMPTS = [
    "A golden retriever running on a sunny beach, waves crashing in the background, cinematic.",
    "A busy city street at night with neon signs reflecting on wet pavement, cars passing by.",
    "A close-up of a hummingbird hovering near a red flower, slow motion, high detail.",
    "An astronaut riding a horse on the surface of Mars, dust clouds rising.",
]


def ssim(a, b):
    """Mean SSIM over frames; a, b: (T, H, W, C) float in [0, 1]; 11x11 Gaussian window on luma."""
    # float64: on the GPU an FP32 conv2d may run in TF32 (10-bit mantissa), and E[x^2] - mu^2 then
    # cancels badly on near-uniform frames (SSIM > 1)
    a, b = a.double(), b.double()
    w = torch.tensor([0.299, 0.587, 0.114], device=a.device, dtype=torch.float64)
    ya, yb = (x @ w for x in (a, b))
    g = torch.exp(-(torch.arange(11, device=a.device, dtype=torch.float64) - 5.0) ** 2 / (2 * 1.5 ** 2))
    g = (g / g.sum())
    k2 = (g[:, None] * g[None, :])[None, None]
    ya, yb = ya[:, None], yb[:, None]
    mu_a, mu_b = F.conv2d(ya, k2), F.conv2d(yb, k2)
    saa = F.conv2d(ya * ya, k2) - mu_a ** 2
    sbb = F.conv2d(yb * yb, k2) - mu_b ** 2
    sab = F.conv2d(ya * yb, k2) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (saa + sbb + c2))
    return m.mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/nfs/home/koying/models/CogVideoX-2b")
    ap.add_argument("--clip", default="/nfs/home/koying/models/clip-vit-large-patch14")
    ap.add_argument("--specs", nargs="+", default=["native", "fp8t", "fp8", "bf16:3", "bf16:1", "bf16:0"])
    ap.add_argument("--prompts", type=int, default=4)
    ap.add_argument("--first", type=int, default=0, help="index of the first prompt")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--frames", type=int, default=49)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--out", default="/nfs/home/koying/mantissa-optimize/gh200/logs/infer/video")
    ap.add_argument("--tag", default="")
    a, _ = ap.parse_known_args()

    from diffusers import CogVideoXPipeline
    from transformers import CLIPModel, CLIPProcessor

    os.makedirs(a.out, exist_ok=True)
    dt = torch.bfloat16 if a.dtype == "bf16" else torch.float16
    pipe = CogVideoXPipeline.from_pretrained(a.model, torch_dtype=dt).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    clip = CLIPModel.from_pretrained(a.clip, torch_dtype=torch.float16).cuda().eval()
    proc = CLIPProcessor.from_pretrained(a.clip)
    name = os.path.basename(a.model.rstrip("/"))

    def clip_scores(frames, prompt):
        with torch.no_grad():
            inp = proc(text=[prompt], images=list(frames), return_tensors="pt", padding=True).to("cuda")
            img = F.normalize(clip.get_image_features(pixel_values=inp.pixel_values.half()).float(), dim=-1)
            txt = F.normalize(clip.get_text_features(input_ids=inp.input_ids, attention_mask=inp.attention_mask).float(), dim=-1)
        return (img @ txt.t()).mean().item(), (img[1:] * img[:-1]).sum(-1).mean().item()

    for pi, prompt in list(enumerate(PROMPTS))[a.first:a.first + a.prompts]:
        ref = None
        mids = []
        for spec in a.specs:
            F.scaled_dot_product_attention = ppl_eval._SDPA if spec == "native" else ppl_eval.make_attention(spec)
            t0 = time.time()
            with torch.no_grad():
                video = pipe(prompt=prompt, num_frames=a.frames, num_inference_steps=a.steps, guidance_scale=6.0,
                             generator=torch.Generator("cuda").manual_seed(a.seed), output_type="np").frames[0]
            F.scaled_dot_product_attention = ppl_eval._SDPA
            secs = time.time() - t0
            v = torch.from_numpy(np.asarray(video)).float().cuda()  # (T, H, W, C) in [0, 1]
            frames_u8 = (np.asarray(video) * 255).round().astype(np.uint8)
            clipsim, clipt = clip_scores(frames_u8, prompt)
            res = dict(tag=a.tag, model=name, dtype=a.dtype, prompt=pi, spec=spec, seconds=round(secs, 1),
                       clipsim=clipsim, clip_t=clipt)
            if ref is None:
                ref = v
            else:
                mse = ((v - ref) ** 2).mean().item()
                res.update(psnr=10 * np.log10(1.0 / max(mse, 1e-12)), ssim=ssim(v, ref))
            mid = frames_u8[len(frames_u8) // 2]
            mids.append(mid)
            import cv2
            cv2.imwrite(f"{a.out}/{name}_p{pi}_{spec.replace(':', '')}.png", cv2.cvtColor(mid, cv2.COLOR_RGB2BGR))
            print("RESULT " + json.dumps(res), flush=True)
        import cv2
        cv2.imwrite(f"{a.out}/{name}_p{pi}_strip.png", cv2.cvtColor(np.concatenate(mids, axis=1), cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
