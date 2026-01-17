"""Test the fine-tuned adapter on Modal.

Usage:
    modal run modal_test_adapter.py
    modal run modal_test_adapter.py --question "كيفاش نجدد البطاقة الوطنية؟"
"""

import modal

app = modal.App("test-qwen-tone-adapter")

# Use the adapter volume
volume = modal.Volume.from_name("qwen-tone-adapter", create_if_missing=False)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0,<2.4.0",
        "transformers>=4.41.0,<4.45.0",
        "peft>=0.7.0,<0.12.0",
        "accelerate>=0.25.0,<0.34.0",
        "bitsandbytes>=0.41.0,<0.44.0",
        "scipy",
        "sentencepiece",
    )
    .env({"HF_HOME": "/cache/huggingface"})
)

hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

SYSTEM_PROMPT = """أنت مساعد إداري مغربي ودود. تساعد المواطنين في الإجراءات الإدارية بأسلوب بسيط ومباشر.
أجب بعربية بسيطة وسهلة الفهم. كن مختصراً ومفيداً.
إذا لم تعرف معلومة، قل ذلك بوضوح. لا تخترع معلومات."""


@app.function(
    gpu="A100",
    timeout=60 * 10,
    image=image,
    volumes={
        "/adapter": volume,
        "/cache": hf_cache,
    },
)
def test_adapter(questions: list[str], use_adapter: bool = True):
    """Test the adapter with sample questions.

    Args:
        questions: List of questions to test
        use_adapter: If True, use fine-tuned adapter. If False, use base model.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    base_model = "Qwen/Qwen2.5-7B-Instruct"
    adapter_path = "/adapter/qwen25_7b_tone_lora"

    print("=" * 60)
    print(f"Testing {'WITH ADAPTER' if use_adapter else 'BASE MODEL ONLY'}")
    print("=" * 60)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    # Load model in 4-bit
    print("Loading model in 4-bit...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )

    # Load adapter if requested
    if use_adapter:
        print(f"Loading adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()

    print("\n" + "=" * 60)
    print("GENERATING RESPONSES")
    print("=" * 60)

    results = []
    for i, question in enumerate(questions):
        print(f"\n--- Question {i + 1} ---")
        print(f"Q: {question}")

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]

        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(
            outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        print(f"A: {response}")
        results.append({"question": question, "response": response})

    return results


@app.local_entrypoint()
def main(question: str = "", no_adapter: bool = False):
    """Test the fine-tuned adapter.

    Args:
        question: Custom question to test (optional)
        no_adapter: Test base model without adapter for comparison
    """
    # Default test questions
    test_questions = [
        "كيفاش نجدد البطاقة الوطنية؟",
        "شنو الوثائق اللي خصني نجيب باش نخرج جواز السفر؟",
        "بغيت نعرف كيفاش ندير شهادة السكنى",
        "فين نمشي باش نصادق على الوثائق؟",
    ]

    if question:
        test_questions = [question]

    print("Testing fine-tuned Qwen2.5-7B adapter")
    print(f"Questions: {len(test_questions)}")
    print(f"Mode: {'Base model' if no_adapter else 'With adapter'}")
    print("-" * 40)

    results = test_adapter.remote(test_questions, use_adapter=not no_adapter)

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for r in results:
        print(f"\nQ: {r['question']}")
        print(f"A: {r['response']}")
