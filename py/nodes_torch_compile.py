import torch

class TorchCompileModelAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": { "model": ("MODEL",),
                             "backend": (["inductor", "cudagraphs"],),
                             "compile_mode": (["reduce-overhead", "default", "max-autotune"],),
                             "enabled": ("BOOLEAN", {"default": False}),
                             }}
    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"

    CATEGORY = "utils"
    EXPERIMENTAL = True

    def patch(self, model, backend, compile_mode, enabled):
        if not enabled:
            return (model, )
        m = model.clone()
        m.add_object_patch("diffusion_model", torch.compile(model=m.get_model_object("diffusion_model"), mode=compile_mode, backend=backend))
        return (m, )

NODE_CLASS_MAPPINGS = {
    "TorchCompileModelAdvanced": TorchCompileModelAdvanced,
}
