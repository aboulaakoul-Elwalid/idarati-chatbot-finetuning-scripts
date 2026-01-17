"""Validation checks for generated training data."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from typing import Dict, List


class TrainingDataValidator:
    """Validate quality of generated training data."""

    def __init__(self, jsonl_file: str):
        self.examples = self._load_jsonl(jsonl_file)
        self.issues: Dict[str, List[str]] = defaultdict(list)

    def _load_jsonl(self, file: str) -> List[Dict]:
        examples = []
        with open(file, "r", encoding="utf-8") as handle:
            for line in handle:
                examples.append(json.loads(line))
        return examples

    def validate_all(self) -> Dict:
        print("Running quality checks...\n")

        checks = [
            ("Format", self.check_format),
            ("Answers", self.check_answers),
            ("Diversity", self.check_diversity),
            ("Hallucinations", self.check_hallucinations),
            ("Language Mix", self.check_language_mix),
        ]

        results = {}
        for name, check_func in checks:
            print(f"Checking {name}...")
            results[name] = check_func()

        return results

    def check_format(self) -> Dict:
        issues = []

        for i, ex in enumerate(self.examples):
            messages = ex.get("messages", [])

            if len(messages) != 3:
                issues.append(f"Example {i}: Wrong number of messages ({len(messages)})")

            roles = [m.get("role") for m in messages]
            if roles != ["system", "user", "assistant"]:
                issues.append(f"Example {i}: Wrong role sequence {roles}")

            for msg in messages:
                if not msg.get("content", "").strip():
                    issues.append(f"Example {i}: Empty message content")

        self.issues["format"] = issues
        total = len(self.examples)
        return {
            "total_examples": total,
            "issues_found": len(issues),
            "pass_rate": (total - len(issues)) / total if total else 0,
        }

    def check_answers(self) -> Dict:
        issues = []
        avg_lengths = []

        for i, ex in enumerate(self.examples):
            answer = ex["messages"][2]["content"]
            words = answer.split()
            avg_lengths.append(len(words))

            if len(words) < 10:
                issues.append(f"Example {i}: Answer too short ({len(words)} words)")

            if len(words) > 200:
                issues.append(f"Example {i}: Answer too long ({len(words)} words)")

            if re.search(r"^[\s]*[-•*]", answer, re.MULTILINE):
                issues.append(f"Example {i}: Uses bullet points instead of prose")

            if any(placeholder in answer.lower() for placeholder in ["...", "xxx", "todo", "["]):
                issues.append(f"Example {i}: Contains placeholder text")

        self.issues["answers"] = issues
        total = len(avg_lengths)
        return {
            "avg_answer_length": sum(avg_lengths) / total if total else 0,
            "min_length": min(avg_lengths) if avg_lengths else 0,
            "max_length": max(avg_lengths) if avg_lengths else 0,
            "issues_found": len(issues),
        }

    def check_diversity(self) -> Dict:
        questions = []

        for ex in self.examples:
            user_msg = ex["messages"][1]["content"]
            question = re.search(r"السؤال:\s*(.+)$", user_msg, re.DOTALL)
            if question:
                questions.append(question.group(1).strip())

        question_counts = Counter(questions)
        duplicates = {q: c for q, c in question_counts.items() if c > 1}
        question_starts = Counter([q.split()[0] for q in questions if q])

        return {
            "total_questions": len(questions),
            "unique_questions": len(question_counts),
            "duplicate_count": len(duplicates),
            "top_duplicates": dict(list(duplicates.items())[:5]),
            "question_starts": dict(question_starts.most_common(10)),
        }

    def check_hallucinations(self) -> Dict:
        issues = []

        for i, ex in enumerate(self.examples):
            user_msg = ex["messages"][1]["content"]
            answer = ex["messages"][2]["content"]

            context_match = re.search(r"السياق:(.*?)السؤال:", user_msg, re.DOTALL)
            if not context_match:
                continue

            context = context_match.group(1)

            answer_numbers = set(re.findall(r"\d+", answer))
            context_numbers = set(re.findall(r"\d+", context))

            hallucinated_numbers = answer_numbers - context_numbers
            if hallucinated_numbers:
                issues.append(f"Example {i}: Possible hallucinated numbers: {hallucinated_numbers}")

        self.issues["hallucinations"] = issues
        return {
            "total_checked": len(self.examples),
            "potential_hallucinations": len(issues),
            "issues": issues[:10],
        }

    def check_language_mix(self) -> Dict:
        darija_patterns = ["شحال", "شنو", "كيفاش", "واش", "خاصني", "بغيت", "غادي", "ديال"]

        darija_count = 0
        formal_count = 0
        mixed_count = 0

        for ex in self.examples:
            question = ex["messages"][1]["content"]
            darija_matches = sum(1 for pattern in darija_patterns if pattern in question)

            if darija_matches >= 2:
                darija_count += 1
            elif darija_matches == 1:
                mixed_count += 1
            else:
                formal_count += 1

        total = len(self.examples)
        return {
            "darija": {
                "count": darija_count,
                "percentage": f"{darija_count/total*100:.1f}%" if total else "0%",
            },
            "formal": {
                "count": formal_count,
                "percentage": f"{formal_count/total*100:.1f}%" if total else "0%",
            },
            "mixed": {
                "count": mixed_count,
                "percentage": f"{mixed_count/total*100:.1f}%" if total else "0%",
            },
        }

    def generate_report(self, output_file: str = "quality_report.json") -> Dict:
        results = self.validate_all()

        report = {
            "summary": {
                "total_examples": len(self.examples),
                "total_issues": sum(len(issues) for issues in self.issues.values()),
                "quality_score": self._calculate_quality_score(),
            },
            "validation_results": results,
            "issues_by_category": {k: len(v) for k, v in self.issues.items()},
            "detailed_issues": self.issues,
        }

        with open(output_file, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)

        print(f"\nQuality Report saved to: {output_file}")
        return report

    def _calculate_quality_score(self) -> float:
        total_issues = sum(len(issues) for issues in self.issues.values())
        max_possible_issues = len(self.examples) * 5

        score = max(0, 100 - (total_issues / max_possible_issues * 100)) if max_possible_issues else 0
        return round(score, 2)

    def print_summary(self) -> None:
        results = self.validate_all()

        print("\n" + "=" * 70)
        print("QUALITY VALIDATION SUMMARY")
        print("=" * 70)

        print(f"\nTotal Examples: {len(self.examples)}")
        print(f"Quality Score: {self._calculate_quality_score()}/100")

        print(f"\n{'-' * 70}")
        print("Issues by Category:")
        print(f"{'-' * 70}")
        for category, issues in self.issues.items():
            print(f"  {category}: {len(issues)} issues")

        print(f"\n{'-' * 70}")
        print("Language Distribution:")
        print(f"{'-' * 70}")
        lang_mix = results["Language Mix"]
        for lang, stats in lang_mix.items():
            print(f"  {lang}: {stats['count']} ({stats['percentage']})")

        print(f"\n{'-' * 70}")
        print("Answer Statistics:")
        print(f"{'-' * 70}")
        answer_stats = results["Answers"]
        print(f"  Average length: {answer_stats['avg_answer_length']:.1f} words")
        print(f"  Range: {answer_stats['min_length']} - {answer_stats['max_length']} words")
