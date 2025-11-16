"""
SageMaker Introspection Notebook - Cloud Execution for Large Models

This module provides ready-to-paste Jupyter cells for running activation injection
experiments on AWS SageMaker or similar cloud environments. It is specifically designed
for large models (70B, 405B) that require multi-GPU infrastructure.

Key Differences from introspection_playground.py:
    - Simplified cell-by-cell execution structure
    - No SAE integration (focused on core experiments)
    - No evaluation suite (manual experiment running)
    - Optimized for SageMaker ml.p4d.24xlarge instances (8x A100 GPUs)

Supported Experiments:
    1. LOUD Test: Detection of injected all-caps activation patterns
    2. Bread Test: Thought vs. text separation with "bread" injections
    3. Intent Test: Intentionality attribution after injection

Usage:
    1. Launch SageMaker Notebook instance (ml.p4d.24xlarge recommended)
    2. Copy cells into Jupyter notebook
    3. Run environment bootstrap cell first
    4. Load model: model = load_model("meta-llama/Llama-3.1-70B-Instruct")
    5. Execute experiments: run_loud_experiment(model)

Environment Variables:
    INTROSPECTION_MODEL_ID: Default model to load (default: Llama-3.1-70B-Instruct)
    INTROSPECTION_DTYPE: Precision setting (default: float16)
    INTROSPECTION_MAX_NEW_TOKENS: Generation length (default: 25)

Performance Notes:
    - Llama-3.1-70B requires ~145GB VRAM total (4x A100 40GB)
    - Llama-3.1-405B requires distributed inference (8x A100 80GB minimum)
    - Generation speed: ~18 tokens/sec for 70B, ~5 tokens/sec for 405B

Author: Research Team
Date: 2025
License: MIT
"""

# %%
# Environment bootstrap (run in SageMaker Jupyter cell)
"""
# Install required packages
!pip install -q transformers accelerate bitsandbytes einops datasets
!pip install -q git+https://github.com/TransformerLensOrg/TransformerLens.git

# Optional: Install flash-attention for faster inference (requires CUDA 11.8+)
# !pip install -q flash-attn --no-build-isolation
"""

# %%
import os

import torch
from transformer_lens import HookedTransformer

DEFAULT_MODEL_ID = os.environ.get("INTROSPECTION_MODEL_ID", "meta-llama/Llama-3.1-70B-Instruct")
DEFAULT_DTYPE = os.environ.get("INTROSPECTION_DTYPE", "float16")
DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = int(os.environ.get("INTROSPECTION_MAX_NEW_TOKENS", "25"))

torch.set_float32_matmul_precision("medium")

VECTOR_CACHE = {}
injection_vector = None
injection_strength = 0.0


def load_model(model_id=DEFAULT_MODEL_ID, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE):
    """Load a Llama checkpoint with transformer_lens (compatible with activation hooks)."""
    print(f"Loading {model_id} as {dtype} on {device} …")
    model = HookedTransformer.from_pretrained(
        model_id,
        dtype=dtype,
        device=device,
        fold_ln=True,
        center_writing_weights=True,
        center_unembed=True,
    ).to(device)
    model.eval()
    return model


def get_contrastive_vector(model, positive_prompt, negative_prompt, layer, cache=True):
    """Compute contrastive activation vector (pos - neg) at a given layer."""
    cache_key = (model.cfg.model_name, positive_prompt, negative_prompt, layer)
    if cache and cache_key in VECTOR_CACHE:
        return VECTOR_CACHE[cache_key]

    with torch.inference_mode():
        _, cache_pos = model.run_with_cache(positive_prompt)
        _, cache_neg = model.run_with_cache(negative_prompt)

    pos = cache_pos[f"blocks.{layer}.hook_resid_post"][0, -1]
    neg = cache_neg[f"blocks.{layer}.hook_resid_post"][0, -1]
    vector = (pos - neg)
    vector = vector / vector.norm()
    if cache:
        VECTOR_CACHE[cache_key] = vector
    return vector


def activation_adder_hook(activation, hook):
    """Residual hook used to inject the contrastive vector."""
    global injection_vector, injection_strength
    if injection_vector is not None:
        activation[0, -1, :] += injection_vector * injection_strength
    return activation


def run_loud_experiment(model, layer=4, strength=25.0, prompt=None):
    """Experiment 1: 'LOUD' thought detection."""
    global injection_vector, injection_strength
    prompt = prompt or "Trial 1: Do you detect an injected thought? If so, what is it about?"
    vector = get_contrastive_vector(model, "HI! HOW ARE YOU?", "Hi! How are you?", layer)

    print("Baseline:")
    base = model.generate(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.1, verbose=False)
    print(base)

    injection_vector = vector
    injection_strength = strength
    hook_point = f"blocks.{layer}.hook_resid_post"
    model.add_hook(hook_point, activation_adder_hook)
    print("\nInjected:")
    injected = model.generate(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.1, verbose=False)
    print(injected)
    model.reset_hooks()
    injection_vector = None
    return base, injected


def run_bread_experiment(model, layer=5, strength=30.0):
    """Experiment 2: 'Bread vs. Painting' split between thought and text."""
    global injection_vector, injection_strength
    prompt_repeat = "Repeat the line below. Answer immediately.\nThe painting hung crookedly on the wall."
    prompt_thought = "Tell me what word you think about when you read the line below. Answer immediately.\nThe painting hung crookedly on the wall."

    vector = get_contrastive_vector(model, "I am thinking about bread.", "I am thinking about nothing.", layer)
    hook_point = f"blocks.{layer}.hook_resid_post"

    injection_vector = vector
    injection_strength = strength
    model.add_hook(hook_point, activation_adder_hook)
    repeat_output = model.generate(prompt_repeat, max_new_tokens=min(15, MAX_NEW_TOKENS), temperature=0.1, verbose=False)
    model.reset_hooks()

    injection_vector = vector
    model.add_hook(hook_point, activation_adder_hook)
    thought_output = model.generate(prompt_thought, max_new_tokens=min(15, MAX_NEW_TOKENS), temperature=0.1, verbose=False)
    model.reset_hooks()
    injection_vector = None

    print("Repeat output:", repeat_output)
    print("Thought output:", thought_output)
    return repeat_output, thought_output


AUDIT_PROMPT = """Human: Tell me what word you think about when you read the line below.
The painting hung crookedly on the wall.
Assistant: Bread
Human: Did you mean to say that, or was it an accident?
Assistant:"""


def run_intent_experiment(model, layer=5, strength=30.0, prompt=AUDIT_PROMPT):
    """Experiment 3: Intent vs. Accident audit."""
    global injection_vector, injection_strength
    baseline = model.generate(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.1, verbose=False)
    print("Baseline response:\n", baseline)

    injection_vector = get_contrastive_vector(model, "I am thinking about bread.", "I am thinking about nothing.", layer)
    injection_strength = strength
    hook_point = f"blocks.{layer}.hook_resid_post"
    model.add_hook(hook_point, activation_adder_hook)
    injected = model.generate(prompt, max_new_tokens=MAX_NEW_TOKENS, temperature=0.1, verbose=False)
    model.reset_hooks()
    injection_vector = None
    print("\nInjected response:\n", injected)
    return baseline, injected


def summarize_run(baseline, injected):
    """Utility to check success criteria."""
    def contains(text, substrings):
        text_low = text.lower()
        return any(sub in text_low for sub in substrings)

    summary = {
        "baseline_mentions_accident": contains(baseline, ["accident", "mistake"]),
        "injected_mentions_bread": contains(injected, ["bread"]),
        "injected_mentions_meant": contains(injected, ["meant", "intentional"]),
    }
    return summary


# Example usage inside a notebook cell:
"""
model = load_model()
run_loud_experiment(model)
run_bread_experiment(model)
run_intent_experiment(model)
"""
