"""Modal inference endpoint for fine-tuned Qwen2.5-7B with LoRA adapter.

This endpoint serves the Idarati assistant model for Moroccan administrative procedures.

Usage:
    # Deploy the endpoint
    modal deploy modal_inference_endpoint.py

    # Test locally (runs the function directly)
    modal run modal_inference_endpoint.py

Endpoint:
    POST /generate
    {
        "message": "كيفاش نجدد البطاقة الوطنية؟",
        "context": "",           # Optional: RAG context from Supabase
        "mode": "chat",          # "chat" (conversational) or "rag" (JSON output)
        "max_tokens": 256,       # Optional, default 256
        "temperature": 0.7       # Optional, default 0.7
    }

    Returns:
    {
        "response": "...",
        "model": "qwen25-7b-tone-lora",
        "tokens_generated": 85,
        "mode": "chat"
    }
"""

import modal

# App configuration
app = modal.App("idarati-qwen-inference")

# Volumes for model weights
adapter_volume = modal.Volume.from_name("qwen-tone-adapter", create_if_missing=False)
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)

# Image with all dependencies
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
        "fastapi",
    )
    .env({"HF_HOME": "/cache/huggingface"})
)

# System prompts
CHAT_SYSTEM_PROMPT = """أنت مساعد إداري مغربي ودود. تساعد المواطنين في الإجراءات الإدارية بأسلوب بسيط ومباشر.
أجب بعربية بسيطة وسهلة الفهم. كن مختصراً ومفيداً.
إذا لم تعرف معلومة، قل ذلك بوضوح. لا تخترع معلومات."""

RAG_SYSTEM_PROMPT = """أنت "مساعد إدارتي" (Idarati Assistant). مهمتك هي تبسيط المساطر الإدارية المغربية.
سأزودك بسياق (Context) مستخرج من قاعدة بيانات idarati.ma.

يجب عليك:
1. استخراج المعلومات من السياق المقدم فقط.
2. حدّد المعرفين thematicId و procedureId للمسطرة الأكثر ارتباطًا بسؤال المستخدم، ويجب أن يكونا موجودين في السياق.
3. لا تبنِ روابط بنفسك، سنبنيها برمجيًا.

القواعد:
- لغة عربية سليمة (فصحى أو دارجة مهذبة).
- إذا لم تجد معلومة في السياق، اطلب من المستخدم توضيح طلبه.

أعد النتيجة في JSON فقط، بدون أي نص إضافي.
المفاتيح المطلوبة بالضبط:
- summary: فقرة قصيرة ترشد المستخدم بشكل عام بناءً على السياق.
- procedureId: معرّف المسطرة (من السياق فقط).
- thematicId: معرّف الموضوع (من السياق فقط).
مثال مختصر:
{"summary":"...","procedureId":"...","thematicId":"..."}
لا تضف مفاتيح أخرى ولا تغلف JSON بعلامات Markdown."""


@app.cls(
    gpu="A100",
    timeout=60 * 10,
    image=image,
    volumes={
        "/adapter": adapter_volume,
        "/cache": hf_cache,
    },
    scaledown_window=300,  # Keep container warm for 5 minutes
)
class QwenInference:
    """Inference class for fine-tuned Qwen model."""

    @modal.enter()
    def load_model(self):
        """Load the model and adapter once when container starts."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel

        base_model = "Qwen/Qwen2.5-7B-Instruct"
        adapter_path = "/adapter/qwen25_7b_tone_lora"

        print("=" * 60)
        print("Loading Qwen2.5-7B-Instruct with LoRA adapter...")
        print("=" * 60)

        # Load tokenizer
        print("\n[1/3] Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True
        )

        # Load model in 4-bit quantization
        print("[2/3] Loading model in 4-bit quantization...")
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

        # Load LoRA adapter
        print(f"[3/3] Loading LoRA adapter from {adapter_path}...")
        self.model = PeftModel.from_pretrained(model, adapter_path)
        self.model.eval()

        print("\n" + "=" * 60)
        print("Model loaded successfully!")
        print("=" * 60)

    @modal.fastapi_endpoint(method="POST", docs=True)
    def generate(self, request: dict) -> dict:
        """Generate a response for the given message.

        Args:
            request: JSON body with:
                - message (str): User's question
                - context (str, optional): RAG context from Supabase
                - mode (str, optional): "chat" or "rag" (default: "chat")
                - max_tokens (int, optional): Max tokens to generate (default: 256)
                - temperature (float, optional): Sampling temperature (default: 0.7)

        Returns:
            JSON response with:
                - response (str): Model's response
                - model (str): Model identifier
                - tokens_generated (int): Number of tokens generated
                - mode (str): Mode used
        """
        import torch

        # Parse request
        message = request.get("message", "")
        context = request.get("context", "")
        mode = request.get("mode", "chat").lower()
        max_tokens = request.get("max_tokens", 256)
        temperature = request.get("temperature", 0.7)

        if not message:
            return {"error": "message is required", "status": 400}

        # Select system prompt based on mode
        if mode == "rag":
            system_prompt = RAG_SYSTEM_PROMPT
            # Build user content with context
            if context:
                user_content = f"Context:\n{context}\n\nUser Question: {message}\n"
            else:
                user_content = f"User Question: {message}\n"
        else:
            system_prompt = CHAT_SYSTEM_PROMPT
            user_content = message

        # Build chat messages
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        # Tokenize
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        input_length = inputs.input_ids.shape[1]

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode response
        response = self.tokenizer.decode(
            outputs[0][input_length:], skip_special_tokens=True
        )
        tokens_generated = outputs[0].shape[0] - input_length

        return {
            "response": response,
            "model": "qwen25-7b-tone-lora",
            "tokens_generated": tokens_generated,
            "mode": mode,
        }

    @modal.fastapi_endpoint(method="GET", docs=True)
    def health(self) -> dict:
        """Health check endpoint."""
        return {
            "status": "ok",
            "model": "qwen25-7b-tone-lora",
            "base_model": "Qwen/Qwen2.5-7B-Instruct",
        }


@app.local_entrypoint()
def main():
    """Test the endpoint locally."""
    print("Testing Qwen inference endpoint...")

    # Create instance
    inference = QwenInference()

    # Test chat mode
    print("\n" + "=" * 60)
    print("TEST 1: Chat Mode")
    print("=" * 60)
    result = inference.generate.remote(
        {
            "message": "كيفاش نجدد البطاقة الوطنية؟",
            "mode": "chat",
        }
    )
    print(f"Response: {result['response']}")
    print(f"Tokens: {result['tokens_generated']}")

    # Test RAG mode with sample context
    print("\n" + "=" * 60)
    print("TEST 2: RAG Mode")
    print("=" * 60)
    sample_context = """Procedure: تجديد البطاقة الوطنية للتعريف الإلكترونية
ID: proc-cnie-renewal
ThematicID: thematic-identity
Content: يمكن لكل مواطن مغربي تجديد بطاقته الوطنية للتعريف الإلكترونية. الوثائق المطلوبة: البطاقة القديمة، صورتان شمسيتان، وصل الأداء. التوجه إلى المصلحة المختصة بمحل السكنى."""

    result = inference.generate.remote(
        {
            "message": "شنو الوثائق اللي خاصني لتجديد البطاقة؟",
            "context": sample_context,
            "mode": "rag",
        }
    )
    print(f"Response: {result['response']}")
    print(f"Tokens: {result['tokens_generated']}")

    print("\n" + "=" * 60)
    print("Tests completed!")
    print("=" * 60)
