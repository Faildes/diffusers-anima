"""Precomputing the schedule must preserve the ancestral sampler trajectory."""

import torch

from diffusers_anima.pipelines.anima.sampling import sample_euler_ancestral_rf


class _Bar:
    def update(self, count=1):
        pass

    def close(self):
        pass


class _Pipeline:
    def progress_bar(self, **kwargs):
        return _Bar()


def _reference(noise, sigmas, seed):
    latents = noise.clone()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for i in range(len(sigmas) - 1):
        sigma, sigma_next = sigmas[i], sigmas[i + 1]
        denoised = latents - sigma * latents * 0.25
        if i == len(sigmas) - 2:
            return denoised
        eta = 1.0
        downstep_ratio = 1.0 + (sigma_next / sigma - 1.0) * eta
        sigma_down = sigma_next * downstep_ratio
        alpha_ip1 = 1.0 - sigma_next
        alpha_down = 1.0 - sigma_down
        renoise_sq = sigma_next**2 - sigma_down**2 * alpha_ip1**2 / (alpha_down**2)
        renoise_coeff = renoise_sq.clamp_min(0).sqrt()
        sigma_down_ratio = sigma_down / sigma
        latents = sigma_down_ratio * latents + (1.0 - sigma_down_ratio) * denoised
        latents = (alpha_ip1 / alpha_down) * latents + (
            torch.randn(latents.shape, generator=generator) * renoise_coeff
        )
    raise AssertionError("No denoising steps")


def test_ancestral_sampler_matches_scalar_reference_with_cpu_rng():
    noise = torch.full((1, 1), 0.5, dtype=torch.float32)
    sigmas = torch.tensor([0.9, 0.6, 0.3, 0.0], dtype=torch.float32)
    seed = 731

    class Transformer:
        def __call__(self, latents, timestep, *, encoder_hidden_states, return_dict):
            return (latents * 0.25,)

    actual = sample_euler_ancestral_rf(
        Transformer(), _Pipeline(), noise,
        sigmas=sigmas,
        pos_cond=torch.zeros((1, 1, 1)),
        neg_cond=None,
        guidance_scale=1.0,
        eta=1.0,
        s_noise=1.0,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        cfg_batch_mode="split",
        model_dtype=torch.float32,
        callback_on_step_end=None,
        callback_on_step_end_tensor_inputs=["latents"],
    )
    torch.testing.assert_close(actual, _reference(noise, sigmas, seed), rtol=0, atol=1e-6)
