# ScreenScribeVLM — Mobile UI Screen Captioning with Qwen2.5-VL


Fine-tuned `Qwen2.5-VL-7B-Instruct` on the RICO Screen2Words dataset to generate natural language descriptions of mobile app UI screens.

---

## Overview

ScreenScribe is a QLoRA fine-tune of Qwen2.5-VL-7B-Instruct that takes a mobile app screenshot as input and outputs a concise natural language caption describing the screen. It was trained on the RICO Screen2Words dataset using Unsloth on a single T4 GPU in Google Colab.

---

## Design Decisions

### Why only 700 samples?

The full RICO Screen2Words dataset contains ~22,000 samples, but training was intentionally capped at 700 for two reasons:

1. **Unsloth memory layout**: Unsloth loads the vision dataset as a list of chat-formatted message dicts in RAM before training begins. With image tensors embedded inline, this causes significant host RAM pressure before a single GPU step is taken. On a Colab T4 instance (~12 GB VRAM, ~13 GB RAM), pushing beyond ~1,000 samples risks OOM crashes during dataset preparation.

2. **Proof-of-concept scope**: The goal was to validate that the base VLM could be meaningfully steered toward UI captioning with minimal data. At 700 samples, the model already generalises well (see Results), confirming the base Qwen2.5-VL weights carry strong prior knowledge about mobile interfaces.

### Why 1 epoch?

Two checkpoints were compared:

- **30 steps** (~240 samples seen, ~34% of one epoch): loss was still high and unstable.
- **1 full epoch (88 steps, effective batch = 8)**: loss dropped to ~0.30 but the loss curve remained noticeably jagged rather than smooth, indicating the model is under-trained and would benefit from additional epochs.

The batch calculation is:

```
per_device_train_batch_size = 2
gradient_accumulation_steps = 4
→ Effective batch size = 2 × 4 = 8

700 samples / 8 = ~88 steps per epoch
```

One epoch was chosen as the stable stopping point given the hardware constraint, while acknowledging the training signal is not yet saturated.

### LoRA Configuration

All four module groups were fine-tuned (vision layers, language layers, attention, MLP) with rank `r=16` and `lora_alpha=16`. This gives the adapter enough capacity to adapt the vision encoder to the UI domain while keeping the parameter count manageable for 4-bit quantised weights.

---

## Model Details

| Property | Value |
|---|---|
| Base model | unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit |
| Dataset | rootsautomation/RICO-Screen2Words |
| Method | QLoRA (4-bit) via Unsloth |
| Task | Vision → Text (UI screen captioning) |
| Hardware | T4 GPU (Google Colab) |

---

## Training Hyperparameters

| Parameter | Value |
|---|---|
| Learning rate | 2e-4 |
| Epochs | 1 |
| Batch size (per device) | 2 |
| Gradient accumulation steps | 4 |
| Effective batch size | 8 |
| Steps (1 epoch) | 88 |
| LR Scheduler | Cosine |
| Optimizer | adamw_8bit |
| Weight decay | 0.01 |
| LoRA rank (r) | 16 |
| LoRA alpha | 16 |
| Warmup steps | 5 |
| Max sequence length | 2048 |
| Seed | 3407 |

---

## Results

Evaluated on 10 held-out samples from the RICO train split (seed=99, not seen during training):

| # | Ground Truth | Prediction | Match |
|---|---|---|---|
| 1 | display of screen shows to import from social app | display of timeline photos and mobile uploads in the application | ✅ |
| 2 | display with multiple options | display of layers in map | ✅ |
| 3 | display of screen on pedometer | display of a page showing of health care application | ✅ |
| 4 | display page of radar of a weather app | display of map in weather application | ✅ |
| 5 | display page of a body building app | display of a photo editing app with options | ❌ |
| 6 | list of settings in a e-book reader app | display of settings page in the reading app | ✅ |
| 7 | display of information for how to use an app | display of a pop up with information about an application | ✅ |
| 8 | food items in food application | display of a food recipe in a food app | ✅ |
| 9 | display various places | display of a page showing to add places | ✅ |
| 10 | display of screen shows budgets | display of budgets in the app | ✅ |

**9/10 semantically correct** on unseen data from 700 training samples and 1 epoch.

---

## Limitations

- **Short captions only**: The model was trained on short one-line RICO captions. Prompting for detailed multi-sentence descriptions will not produce reliable results — the fine-tune has steered the output distribution toward brevity.
- **Jagged loss curve**: Training loss at 88 steps was not smooth, indicating the model is under-trained. More epochs and more samples would improve consistency.
- **T4 RAM cap**: Unsloth's in-memory dataset format limits practical training size to ~700–1,000 samples on a Colab T4 without custom batching.

---

## Potential Improvements

| Change | Expected Impact |
|---|---|
| Train on full ~22k RICO samples | Higher accuracy, better generalisation |
| 2–3 epochs | Smoother loss, more stable outputs |
| Richer caption dataset | Longer, more detailed descriptions |
| Larger GPU (A100) | Removes RAM cap, enables larger batches |

---

## Inference

```python
from unsloth import FastVisionModel
from PIL import Image

model, tokenizer = FastVisionModel.from_pretrained(
    model_name = "aaryaupadhya20/rico-screen2words-qwen2.5_7B_VL",
    load_in_4bit = True,
)
FastVisionModel.for_inference(model)

image = Image.open("your_screen.png")

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "You are a UI/UX expert. Describe what you see on this mobile app screen."},
        ]
    }
]

input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

inputs = tokenizer(
    image,
    input_text,
    add_special_tokens=False,
    return_tensors="pt",
).to("cuda")

inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)

output_ids = model.generate(**inputs, max_new_tokens=128, use_cache=False)

prediction = tokenizer.decode(
    output_ids[0][inputs["input_ids"].shape[1]:],
    skip_special_tokens=True
)
print(prediction)
```

---

## Links

- Model: [aaryaupadhya20/rico-screen2words-qwen2.5_7B_VL](https://huggingface.co/aaryaupadhya20/rico-screen2words-qwen2.5_7B_VL)
- Dataset: [rootsautomation/RICO-Screen2Words](https://huggingface.co/datasets/rootsautomation/RICO-Screen2Words)
- Base model: [unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit](https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-bnb-4bit)
