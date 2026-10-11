import json
import unittest

from portraitaes.inference.pipeline import (
    DIMENSION_TASKS,
    TASKS,
    TYPE_WEIGHTS,
    format_pipeline_result,
    get_system_prompt,
    parse_json_object,
    resolve_type_weights,
    run_full_pipeline,
)


PROMPT_MARKERS = {
    "type_agent": "照片类型识别模型",
    "structure": "结构维度评分模型",
    "composition": "构图维度评分模型",
    "image_quality": "画质维度评分模型",
    "tone": "影调维度评分模型",
    "theme_style": "题材维度评分模型",
    "emotion": "情绪维度评分模型",
}

SCORES = {
    "structure": 5.0,
    "composition": 4.0,
    "image_quality": 4.0,
    "tone": 3.5,
    "theme_style": 4.2,
    "emotion": 4.4,
}


def task_outputs():
    outputs = {
        "type_agent": {
            "parsed": {
                "photo_type": {
                    "weight_category": "V",
                    "weight_category_name": "社交类",
                },
                "weights_used": {task: 1 / 6 for task in DIMENSION_TASKS},
            },
            "raw": "",
        }
    }
    for task, score in SCORES.items():
        outputs[task] = {
            "parsed": {f"{task}_score": score},
            "raw": "",
        }
    return outputs


class InferencePipelineTests(unittest.TestCase):
    def test_all_prompts_are_packaged(self):
        self.assertEqual(set(TASKS), set(PROMPT_MARKERS))
        for task, marker in PROMPT_MARKERS.items():
            self.assertIn(marker, get_system_prompt(task))

    def test_every_type_weight_table_sums_to_one(self):
        self.assertEqual(len(TYPE_WEIGHTS), 9)
        for weights in TYPE_WEIGHTS.values():
            self.assertEqual(set(weights), set(DIMENSION_TASKS))
            self.assertAlmostEqual(sum(weights.values()), 1.0)

    def test_type_table_takes_precedence_over_generated_weights(self):
        router = {
            "photo_type": {"weight_category": "V"},
            "weights_used": {task: 1 / 6 for task in DIMENSION_TASKS},
        }
        code, weights, source = resolve_type_weights(router)
        self.assertEqual(code, "V")
        self.assertEqual(weights, TYPE_WEIGHTS["V"])
        self.assertEqual(source, "type_table")

    def test_complete_result_uses_type_weighted_sum(self):
        result = format_pipeline_result(task_outputs())
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["weight_category"], "V")
        self.assertEqual(result["dimension_scores"], SCORES)
        self.assertEqual(result["overall_score"], 4.07)
        self.assertEqual(result["weight_coverage"], 1.0)

    def test_missing_dimension_is_renormalized(self):
        outputs = task_outputs()
        outputs["image_quality"] = {
            "parsed": None,
            "raw": "not json",
            "error": "Model response is not a JSON object",
        }
        result = format_pipeline_result(outputs)
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["missing_dimensions"], ["image_quality"])
        self.assertEqual(result["weight_coverage"], 0.85)
        self.assertEqual(result["overall_score"], 4.0824)

    def test_full_pipeline_runs_router_before_specialists(self):
        calls = []

        def generate(messages):
            system = messages[0]["content"]
            user_content = messages[1]["content"]
            for task, marker in PROMPT_MARKERS.items():
                if marker not in system:
                    continue
                calls.append((task, user_content))
                if task == "type_agent":
                    return json.dumps({
                        "photo_type": {
                            "weight_category": "V",
                            "weight_category_name": "社交类",
                        },
                        "weights_used": TYPE_WEIGHTS["V"],
                    }, ensure_ascii=False)
                return json.dumps({f"{task}_score": SCORES[task]})
            raise AssertionError("unknown prompt")

        result = run_full_pipeline(
            generate,
            [{"type": "image", "image": "/tmp/example.jpg"}],
        )
        self.assertEqual([task for task, _ in calls], list(TASKS))
        for _, user_content in calls:
            self.assertEqual(
                user_content,
                [{"type": "image", "image": "/tmp/example.jpg"}],
            )
        self.assertEqual(result["overall_score"], 4.07)

    def test_parse_json_from_code_fence(self):
        self.assertEqual(
            parse_json_object('result:\n```json\n{"structure_score": 4.5}\n```'),
            {"structure_score": 4.5},
        )


if __name__ == "__main__":
    unittest.main()
