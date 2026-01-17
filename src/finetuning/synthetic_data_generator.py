"""Gemini-based synthetic training data generator."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import requests
from google import genai

from finetuning.config import FineTuningConfig

DEFAULT_OPENROUTER_MODEL = "google/gemini-2.0-flash-exp:free"
LLAMA_SERVER_ALIASES = {"llama-server", "llama-cpp", "local"}


@dataclass
class ProcedureRecord:
    """Represents a procedure from the processed Idarati JSON format."""

    id: str
    title: str
    content: str
    administration: str
    price: str
    delay: str
    average_delay: int
    documents: List[str]
    legal_basis: List[str]
    additional_delays: str

    @classmethod
    def from_json(cls, data: Dict) -> "ProcedureRecord":
        metadata = data.get("metadata") or {}
        content = data.get("content", "")

        documents = []
        doc_section = re.search(
            r"(?:^|\\n)##\\s*الوثائق المطلوبة\\s*(.*?)(?=\\n##\\s*السند القانوني|$)",
            content,
            re.DOTALL,
        )
        if not doc_section:
            doc_section = re.search(
                r"الوثائق:(.*?)(?=السند القانوني:|$)", content, re.DOTALL
            )
        if doc_section:
            doc_lines = doc_section.group(1).strip().split("\n")
            documents = [
                line.strip("- ").strip()
                for line in doc_lines
                if line.strip().startswith("-")
            ]
        if not documents:
            documents = [doc for doc in metadata.get("documents", []) if doc]

        legal_basis = []
        legal_section = re.search(r"السند القانوني:(.*?)$", content, re.DOTALL)
        if legal_section:
            legal_lines = legal_section.group(1).strip().split("\n")
            legal_basis = [
                line.strip("- ").strip()
                for line in legal_lines
                if line.strip().startswith("-")
            ]

        return cls(
            id=data.get("id", ""),
            title=data.get("title", ""),
            content=content,
            administration=metadata.get("administration", ""),
            price=(metadata.get("price") or "").strip(),
            delay=metadata.get("delay", ""),
            average_delay=metadata.get("averageDelay") or 0,
            documents=documents,
            legal_basis=legal_basis,
            additional_delays=(metadata.get("duration") or [""])[0]
            if metadata.get("duration")
            else "",
        )


class SyntheticDataGenerator:
    """Generate synthetic Q&A training data using Gemini."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_keys: Optional[List[str]] = None,
        model_name: Optional[str] = None,
        config: Optional[FineTuningConfig] = None,
    ) -> None:
        self.config = config or FineTuningConfig()
        self.model_name = model_name or self.config.model_name
        self.use_llama_server = self.model_name in LLAMA_SERVER_ALIASES
        self.api_key = api_key
        self.api_keys = [key for key in (api_keys or []) if key]
        self.gemini_key_index = 0
        self.use_openrouter = "/" in self.model_name and not self.use_llama_server
        self.client: Optional[genai.Client] = None
        if not self.use_openrouter and not self.use_llama_server:
            gemini_key = self._current_gemini_key()
            if gemini_key:
                self.client = genai.Client(api_key=gemini_key)
        self.generated_examples: List[Dict] = []

    def _current_gemini_key(self) -> Optional[str]:
        if self.api_keys:
            return self.api_keys[self.gemini_key_index]
        return self.api_key

    def _rotate_gemini_key(self) -> bool:
        if not self.api_keys or len(self.api_keys) <= 1:
            return False
        self.gemini_key_index = (self.gemini_key_index + 1) % len(self.api_keys)
        self.client = genai.Client(api_key=self._current_gemini_key())
        return True

    def load_procedures(self, json_file: str | Path) -> List[ProcedureRecord]:
        with open(json_file, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        records = []
        if isinstance(data, list):
            for item in data:
                records.append(ProcedureRecord.from_json(item))
        else:
            records.append(ProcedureRecord.from_json(data))
        return records

    def generate_examples_for_procedure(
        self, procedure: ProcedureRecord, num_examples: int
    ) -> List[Dict]:
        if self.use_llama_server:
            return self._generate_with_llama_server(procedure, num_examples)
        if self.use_openrouter:
            return self._generate_with_openrouter(procedure, num_examples)
        if self.client:
            return self._generate_with_gemini(procedure, num_examples)
        return self._generate_rule_based(procedure, num_examples)

    def _generate_with_llama_server(
        self, procedure: ProcedureRecord, num_examples: int
    ) -> List[Dict]:
        prompt = self._build_prompt(procedure, num_examples)

        payload = {
            "model": self.config.llama_server_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
        }

        try:
            response = requests.post(
                f"{self.config.llama_server_url}/v1/chat/completions",
                json=payload,
                timeout=120,
            )
            response.raise_for_status()
            response_json = response.json()
            payload_text = response_json["choices"][0]["message"]["content"]
            examples = self._parse_json_response(payload_text)
        except Exception as exc:
            print(f"API Error for {procedure.id}: {exc}")
            return []

        for ex in examples:
            ex["procedure_id"] = procedure.id
            ex["procedure_title"] = procedure.title
            ex["context"] = procedure.content

        return examples[:num_examples]

    def _generate_with_openrouter(
        self, procedure: ProcedureRecord, num_examples: int
    ) -> List[Dict]:
        if not self.api_key:
            raise RuntimeError("OpenRouter API key is required")

        prompt = self._build_prompt(procedure, num_examples)
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_output_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost",
            "X-Title": "administrative-data-collection",
        }

        try:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=90,
            )
            response.raise_for_status()
            response_json = response.json()
            payload_text = response_json["choices"][0]["message"]["content"]
            examples = self._parse_json_response(payload_text)
        except Exception as exc:
            print(f"API Error for {procedure.id}: {exc}")
            return []

        for ex in examples:
            ex["procedure_id"] = procedure.id
            ex["procedure_title"] = procedure.title
            ex["context"] = procedure.content

        return examples[:num_examples]

    def _generate_with_gemini(
        self, procedure: ProcedureRecord, num_examples: int
    ) -> List[Dict]:
        prompt = self._build_prompt(procedure, num_examples)

        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "response_mime_type": "application/json",
                    "temperature": self.config.temperature,
                    "max_output_tokens": self.config.max_output_tokens,
                },
            )
            payload = getattr(response, "text", None)
            if not payload:
                payload = response.candidates[0].content.parts[0].text
            examples = self._parse_json_response(payload)
        except Exception as exc:
            print(f"API Error for {procedure.id}: {exc}")
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                if self._rotate_gemini_key():
                    try:
                        response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=prompt,
                            config={
                                "response_mime_type": "application/json",
                                "temperature": self.config.temperature,
                                "max_output_tokens": self.config.max_output_tokens,
                            },
                        )
                        payload = getattr(response, "text", None)
                        if not payload:
                            payload = response.candidates[0].content.parts[0].text
                        examples = self._parse_json_response(payload)
                    except Exception as retry_exc:
                        print(f"API Error for {procedure.id}: {retry_exc}")
                        return []
                else:
                    return []
            else:
                return []

        for ex in examples:
            ex["procedure_id"] = procedure.id
            ex["procedure_title"] = procedure.title
            ex["context"] = procedure.content

        return examples[:num_examples]

    def _build_prompt(self, procedure: ProcedureRecord, num_examples: int) -> str:
        documents_preview = "\n".join(f"- {doc}" for doc in procedure.documents[:6])
        legal_preview = "\n".join(f"- {law}" for law in procedure.legal_basis[:4])

        return f"""أنت خبير في توليد بيانات تدريب لنموذج ذكاء اصطناعي يساعد المواطنين المغاربة في الإجراءات الإدارية.

**الإجراء:**
العنوان: {procedure.title}
الإدارة المسؤولة: {procedure.administration}
التكلفة: {procedure.price}
المدة: {procedure.delay}
المتوسط بالأيام: {procedure.average_delay}

**الوثائق المطلوبة:**
{documents_preview or "غير مذكورة"}

**السند القانوني:**
{legal_preview or "غير مذكور"}

**مهمتك:** إنشاء {num_examples} أمثلة متنوعة من أزواج السؤال-الجواب.

**توزيع الأمثلة المطلوبة:**
1. سؤال عن التكلفة (بالدارجة المغربية)
2. سؤال عن المدة الزمنية (رسمي)
3. سؤالان عن الوثائق المطلوبة (واحد دارجة، واحد رسمي)
4. سؤال عن الإدارة المسؤولة
5. سؤال شامل يجمع عدة جوانب
6. سؤال بالدارجة المغربية (موضوع حر)

**قواعد مهمة:**
- استخدم تعبيرات دارجة حقيقية: "شحال", "شنو", "كيفاش", "واش", "خاصني", "بغيت", "غادي"
- الأجوبة يجب أن تكون بالعربية الفصحى دائما (حتى لو السؤال بالدارجة)
- لا تخترع معلومات - استخدم فقط البيانات المتوفرة
- الأجوبة بدون قوائم نقطية - استخدم فقرات طبيعية
- إذا معلومة غير متوفرة، قل: "المعلومات المتوفرة لا تشير إلى..."

**صيغة الإخراج (JSON فقط، بدون نص إضافي):**
[
  {{
    "question_type": "cost",
    "question": "السؤال هنا",
    "answer": "الجواب هنا بفقرات طبيعية بدون نقاط",
    "required_info": ["معلومة رئيسية 1"],
    "language_style": "darija"
  }}
]
"""

    def _parse_json_response(self, response: str) -> List[Dict]:
        json_match = re.search(r"```json\s*(.*?)\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            json_match = re.search(r"\[\s*\{.*?\}\s*\]", response, re.DOTALL)
            json_str = json_match.group(0) if json_match else response

        try:
            parsed = json.loads(json_str)
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
        return []

    def _generate_rule_based(
        self, procedure: ProcedureRecord, num_examples: int
    ) -> List[Dict]:
        examples: List[Dict] = []

        if procedure.price:
            examples.append(
                {
                    "question_type": "cost",
                    "question": "شحال كتكلف هاد الوثيقة؟"
                    if "مجان" not in procedure.price
                    else "واش هاد الإجراء مجاني؟",
                    "answer": self._generate_cost_answer(procedure),
                    "required_info": [procedure.price],
                    "language_style": "darija",
                    "procedure_id": procedure.id,
                    "procedure_title": procedure.title,
                    "context": procedure.content,
                }
            )

        if procedure.delay:
            examples.append(
                {
                    "question_type": "timeline",
                    "question": "ما هي المدة اللازمة لمعالجة هذا الطلب؟",
                    "answer": self._generate_timeline_answer(procedure),
                    "required_info": [procedure.delay, str(procedure.average_delay)],
                    "language_style": "formal_arabic",
                    "procedure_id": procedure.id,
                    "procedure_title": procedure.title,
                    "context": procedure.content,
                }
            )

        if procedure.documents:
            examples.append(
                {
                    "question_type": "documents",
                    "question": "شنو الوراق اللي خاصني نجيب؟",
                    "answer": self._generate_documents_answer(
                        procedure, style="natural"
                    ),
                    "required_info": procedure.documents[:3],
                    "language_style": "darija",
                    "procedure_id": procedure.id,
                    "procedure_title": procedure.title,
                    "context": procedure.content,
                }
            )
            examples.append(
                {
                    "question_type": "documents",
                    "question": "ما هي الوثائق المطلوبة لهذا الإجراء؟",
                    "answer": self._generate_documents_answer(
                        procedure, style="formal"
                    ),
                    "required_info": procedure.documents[:3],
                    "language_style": "formal_arabic",
                    "procedure_id": procedure.id,
                    "procedure_title": procedure.title,
                    "context": procedure.content,
                }
            )

        if procedure.administration:
            examples.append(
                {
                    "question_type": "administration",
                    "question": "شكون المسؤول على هاد الإجراء؟",
                    "answer": f"الإدارة المسؤولة عن معالجة هذا الإجراء هي: {procedure.administration}",
                    "required_info": [procedure.administration],
                    "language_style": "darija",
                    "procedure_id": procedure.id,
                    "procedure_title": procedure.title,
                    "context": procedure.content,
                }
            )

        examples.append(
            {
                "question_type": "comprehensive",
                "question": "كيفاش نقدر نحصل على هاد الوثيقة؟ عطيني معلومات كاملة",
                "answer": self._generate_comprehensive_answer(procedure),
                "required_info": [
                    procedure.price,
                    procedure.delay,
                    procedure.administration,
                ],
                "language_style": "mixed",
                "procedure_id": procedure.id,
                "procedure_title": procedure.title,
                "context": procedure.content,
            }
        )

        darija_questions = [
            "واش هاد الإجراء صعيب؟",
            "شحال غادي ياخد الوقت تقريبا؟",
            "فين نمشي باش نقدم الطلب؟",
        ]
        examples.append(
            {
                "question_type": "general",
                "question": darija_questions[len(examples) % len(darija_questions)],
                "answer": self._generate_general_answer(procedure),
                "required_info": [procedure.title],
                "language_style": "darija",
                "procedure_id": procedure.id,
                "procedure_title": procedure.title,
                "context": procedure.content,
            }
        )

        return examples[:num_examples]

    def _generate_cost_answer(self, proc: ProcedureRecord) -> str:
        if "مجان" in proc.price or "مجانا" in proc.price:
            return f"هذا الإجراء مجاني تماما ولا يتطلب أي رسوم. يمكنك الحصول على {proc.title} دون أي تكلفة."
        return f"تكلفة الحصول على {proc.title} هي {proc.price}. يرجى التأكد من تجهيز المبلغ المطلوب عند تقديم الطلب."

    def _generate_timeline_answer(self, proc: ProcedureRecord) -> str:
        answer = f"الآجل القانوني لمعالجة طلب {proc.title} هو {proc.delay}"

        if proc.average_delay > 0:
            answer += f"، مع متوسط تقديري للمعالجة يبلغ حوالي {proc.average_delay} يوم"

        if proc.additional_delays and "على الفور" not in proc.additional_delays:
            answer += (
                ". يجب الأخذ بعين الاعتبار أنه قد تكون هناك آجال إضافية حسب طبيعة الطلب"
            )

        answer += "."
        return answer

    def _generate_documents_answer(
        self, proc: ProcedureRecord, style: str = "natural"
    ) -> str:
        if not proc.documents:
            return (
                "المعلومات المتوفرة لا تحدد الوثائق المطلوبة بشكل واضح. "
                "يُنصح بالتواصل مع الإدارة المسؤولة للحصول على القائمة الكاملة."
            )

        intro = (
            "للحصول على هذا الإجراء، يجب تقديم الوثائق التالية: "
            if style == "formal"
            else "الوثائق اللي خاصك تجيب هي: "
        )

        docs_to_list = proc.documents[:5]
        if len(docs_to_list) <= 3:
            docs_text = (
                "، و".join(docs_to_list[:-1]) + f" و {docs_to_list[-1]}"
                if len(docs_to_list) > 1
                else docs_to_list[0]
            )
        else:
            docs_text = "، ".join(docs_to_list[:-1]) + f"، وأخيرا {docs_to_list[-1]}"

        answer = intro + docs_text

        if len(proc.documents) > 5:
            answer += ". بالإضافة إلى وثائق أخرى قد تكون مطلوبة حسب الحالة"

        answer += ". يُنصح بالتأكد من صحة واكتمال جميع الوثائق قبل التقديم."

        return answer

    def _generate_comprehensive_answer(self, proc: ProcedureRecord) -> str:
        answer = f"للحصول على {proc.title}، "

        if proc.administration:
            answer += f"يجب التوجه إلى {proc.administration}. "

        if proc.documents:
            answer += f"بالنسبة للوثائق المطلوبة، فهي تشمل أساسا: {', '.join(proc.documents[:3])}"
            answer += " وغيرها. " if len(proc.documents) > 3 else ". "

        if proc.price:
            answer += (
                "هذا الإجراء مجاني. "
                if "مجان" in proc.price
                else f"التكلفة الإجمالية هي {proc.price}. "
            )

        if proc.delay:
            answer += f"أما المدة فهي {proc.delay}"
            if proc.average_delay > 0:
                answer += f" (حوالي {proc.average_delay} يوم)"
            answer += ". "

        answer += "ننصحك بتحضير جميع الوثائق مسبقا لتجنب أي تأخير."

        return answer

    def _generate_general_answer(self, proc: ProcedureRecord) -> str:
        return (
            f"بخصوص {proc.title}، يمكنك التقدم بطلبك لدى {proc.administration}. "
            f"المدة المتوقعة للمعالجة هي {proc.delay} والتكلفة {proc.price}."
        )

    def process_batch(
        self,
        procedures: List[ProcedureRecord],
        examples_per_procedure: Optional[int] = None,
        delay_between_calls: Optional[float] = None,
        checkpoint_every: Optional[int] = None,
        resume_from_checkpoint: Optional[bool] = None,
    ) -> List[Dict]:
        all_examples: List[Dict] = []
        examples_per_procedure = (
            examples_per_procedure or self.config.examples_per_procedure
        )
        delay_between_calls = (
            delay_between_calls
            if delay_between_calls is not None
            else self.config.delay_between_calls
        )
        checkpoint_every = (
            checkpoint_every
            if checkpoint_every is not None
            else self.config.checkpoint_every
        )
        resume_from_checkpoint = (
            resume_from_checkpoint
            if resume_from_checkpoint is not None
            else self.config.resume_from_checkpoint
        )

        processed_count = 0
        if self._uses_generation_api() and resume_from_checkpoint:
            checkpoint_examples, processed_ids = self._load_latest_checkpoint()
            if checkpoint_examples:
                all_examples.extend(checkpoint_examples)
                procedures = [
                    proc for proc in procedures if proc.id not in processed_ids
                ]
                processed_count = len(processed_ids)
                print(
                    f"Resuming from checkpoint: {len(processed_ids)} procedures already processed. "
                    f"{len(procedures)} remaining."
                )
        total = len(procedures)

        for idx, procedure in enumerate(procedures):
            print(f"Processing {idx + 1}/{total}: {procedure.title[:50]}...")
            try:
                examples = self.generate_examples_for_procedure(
                    procedure, examples_per_procedure
                )
                all_examples.extend(examples)
                if self._uses_generation_api() and idx < total - 1:
                    time.sleep(delay_between_calls)
                procedures_seen = processed_count + idx + 1
                if (
                    self._uses_generation_api()
                    and checkpoint_every > 0
                    and procedures_seen % checkpoint_every == 0
                ):
                    self._save_checkpoint(all_examples, procedures_seen)
            except Exception as exc:
                print(f"  Error: {exc}")
                continue

        self.generated_examples = all_examples
        return all_examples

    def _save_checkpoint(self, examples: List[Dict], procedures_seen: int) -> None:
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"checkpoint_proc_{procedures_seen}.jsonl"
        self._write_training_jsonl(examples, checkpoint_path)
        raw_path = checkpoint_dir / f"checkpoint_proc_{procedures_seen}_examples.jsonl"
        self._write_examples_jsonl(examples, raw_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    def _uses_generation_api(self) -> bool:
        return self.client is not None or self.use_openrouter or self.use_llama_server

    def save_training_data(self, output_file: str | Path) -> None:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._write_training_jsonl(self.generated_examples, output_path)

        print(f"\nSaved {len(self.generated_examples)} examples to {output_path}")

    def _write_training_jsonl(self, examples: List[Dict], output_path: Path) -> None:
        with open(output_path, "w", encoding="utf-8") as handle:
            for example in examples:
                training_item = {
                    "messages": [
                        {
                            "role": "system",
                            "content": self.config.system_prompt,
                        },
                        {
                            "role": "user",
                            "content": (
                                f"السياق:\n"
                                f"id: {example['procedure_id']}\n"
                                f"{example['context']}\n\n"
                                f"السؤال: {example['question']}"
                            ),
                        },
                        {
                            "role": "assistant",
                            "content": example["answer"],
                        },
                    ]
                }
                handle.write(json.dumps(training_item, ensure_ascii=False) + "\n")

    def _write_examples_jsonl(self, examples: List[Dict], output_path: Path) -> None:
        with open(output_path, "w", encoding="utf-8") as handle:
            for example in examples:
                handle.write(json.dumps(example, ensure_ascii=False) + "\n")

    def _load_latest_checkpoint(self) -> tuple[List[Dict], set[str]]:
        checkpoint_dir = Path(self.config.checkpoint_dir)
        if not checkpoint_dir.exists():
            return [], set()

        raw_candidates = list(checkpoint_dir.glob("checkpoint_proc_*_examples.jsonl"))
        if raw_candidates:
            latest = self._select_latest_checkpoint(raw_candidates)
            examples = self._load_examples_jsonl(latest)
            processed_ids = {
                ex.get("procedure_id", "") for ex in examples if ex.get("procedure_id")
            }
            return examples, processed_ids

        training_candidates = list(checkpoint_dir.glob("checkpoint_proc_*.jsonl"))
        if not training_candidates:
            return [], set()

        latest = self._select_latest_checkpoint(training_candidates)
        examples = self._load_examples_from_training(latest)
        processed_ids = {
            ex.get("procedure_id", "") for ex in examples if ex.get("procedure_id")
        }
        return examples, processed_ids

    def _select_latest_checkpoint(self, candidates: List[Path]) -> Path:
        def extract_number(path: Path) -> int:
            match = re.search(r"checkpoint_proc_(\\d+)", path.name)
            return int(match.group(1)) if match else 0

        return max(candidates, key=extract_number)

    def _load_examples_jsonl(self, path: Path) -> List[Dict]:
        examples = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                examples.append(json.loads(line))
        return examples

    def _load_examples_from_training(self, path: Path) -> List[Dict]:
        examples = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                examples.append(self._example_from_training_item(item))
        return examples

    def _example_from_training_item(self, item: Dict) -> Dict:
        messages = item.get("messages", [])
        user_content = messages[1]["content"] if len(messages) > 1 else ""
        answer = messages[2]["content"] if len(messages) > 2 else ""

        question_match = re.search(r"السؤال:\\s*(.+)$", user_content, re.DOTALL)
        question = question_match.group(1).strip() if question_match else ""

        context_match = re.search(
            r"السياق:\\n(.*?)(?:\\n\\nالسؤال:|$)", user_content, re.DOTALL
        )
        context_block = context_match.group(1).strip() if context_match else ""

        procedure_id = ""
        context_lines = context_block.splitlines()
        cleaned_lines = []
        for line in context_lines:
            id_match = re.search(r"id:\\s*([a-f0-9\\-]+)", line)
            if id_match:
                procedure_id = id_match.group(1)
            else:
                cleaned_lines.append(line)

        context = "\n".join(cleaned_lines).strip()

        return {
            "question_type": "unknown",
            "question": question,
            "answer": answer,
            "required_info": [],
            "language_style": "unknown",
            "procedure_id": procedure_id,
            "procedure_title": "",
            "context": context,
        }

    def save_metadata(self, output_file: str | Path) -> Dict:
        from collections import Counter

        metadata = {
            "total_examples": len(self.generated_examples),
            "by_question_type": dict(
                Counter(ex["question_type"] for ex in self.generated_examples)
            ),
            "by_language_style": dict(
                Counter(
                    ex.get("language_style", "unknown")
                    for ex in self.generated_examples
                )
            ),
            "unique_procedures": len(
                {ex["procedure_id"] for ex in self.generated_examples}
            ),
        }

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)

        print(f"Saved metadata to {output_path}")
        return metadata
