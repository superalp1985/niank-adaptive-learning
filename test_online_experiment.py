"""CPU contracts for complete-answer online evaluation and SQL visibility."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from online_benchmark import Answer
from online_experiment import (
    Method, ObservedSqlite, make_stream, run_method, select_methods,
)


ROOT = Path(__file__).resolve().parent


class OnlineExperimentContracts(unittest.TestCase):
    def test_generated_question_accepts_multiple_observed_answers(self):
        task = make_stream(31, "mixed", stages=2, questions_per_stage=4, dim=12)
        for questions in task.questions:
            for query in questions:
                proofs = task.world.reference(query)
                self.assertGreaterEqual(len(proofs), 2)
                self.assertIn("any entity reachable", task.world.render_question(query))
                for proof in proofs[:2]:
                    score = task.world.score(
                        query, Answer(proof.terminal, proof.support_ids),
                        task.world.visible_facts(query.as_of),
                    )
                    self.assertTrue(score.full_correct)
                    self.assertEqual(len(proof.support_ids), 2)

    def test_sqlite_contains_only_current_observations(self):
        task = make_stream(31, "shift", stages=2, questions_per_stage=2, dim=12)
        with tempfile.TemporaryDirectory(dir=ROOT / "cache") as tmp:
            store = ObservedSqlite(Path(tmp) / "facts.sqlite")
            try:
                first, future = task.batches[0][0], task.batches[1][0]
                with self.assertRaises(KeyError):
                    store.read(first.id, first.observed_at)
                cost = store.observe(task.batches[0])
                self.assertEqual(store.n, 64)
                self.assertGreater(cost, 0)
                self.assertEqual(store.read(first.id, 63)[2], first.target)
                with self.assertRaises(KeyError):
                    store.read(future.id, 127)
            finally:
                store.close()

    def test_scoring_rejects_wrong_source_even_if_terminal_is_permitted(self):
        task = make_stream(31, "mixed", stages=1, questions_per_stage=1, dim=12)
        query = task.questions[0][0]
        proof = task.world.reference(query)[0]
        score = task.world.score(
            query, Answer(proof.terminal, tuple(reversed(proof.support_ids))),
            task.world.visible_facts(query.as_of),
        )
        self.assertFalse(score.evidence_valid)
        self.assertFalse(score.full_correct)

    def test_complete_cpu_run_logs_independent_answers_and_charges_sql(self):
        task = make_stream(41, "return", stages=4, questions_per_stage=2, dim=12)
        with tempfile.TemporaryDirectory(dir=ROOT / "cache") as tmp:
            output = Path(tmp)
            result = run_method(
                task, Method("periodic_2", "periodic", 2), output,
                device="cpu", allow_cpu_test=True,
            )
            self.assertEqual(result["questions"], 8)
            self.assertEqual(result["observed_rows"], 256)
            self.assertGreater(result["components"]["sql_write"], 0)
            self.assertGreater(result["components"]["sql_read"], 0)
            self.assertGreater(result["components"]["maintenance"], 0)
            self.assertIsNone(result["stages"][2]["maintenance"])
            self.assertIsNotNone(result["stages"][3]["maintenance"])
            rows = [json.loads(line) for line in
                    (output / "periodic_2.answers.jsonl").read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(rows), 8)
            self.assertTrue(all(len(row["valid_alternatives"]) == 2 for row in rows))
            self.assertTrue(all(row["score"]["reason"] for row in rows))
            for row in rows:
                for hop in row["costs"]["hops"]:
                    self.assertLess(hop["payload"]["observed"], row["observed_facts"])

    def test_symbolic_reference_is_strong_and_not_canonical_proof_matching(self):
        task = make_stream(41, "mixed", stages=2, questions_per_stage=3, dim=12)
        with tempfile.TemporaryDirectory(dir=ROOT / "cache") as tmp:
            result = run_method(
                task, Method("symbolic", "symbolic"), Path(tmp),
                device="cpu", allow_cpu_test=True,
            )
            self.assertEqual(result["accuracy"], 1.0)
            self.assertEqual(result["components"]["query"], 0)
            self.assertGreater(result["components"]["sql_write"], 0)

    def test_selection_uses_full_answer_quality_and_per_stream_constraint(self):
        base = {"accuracy": 1., "total_work": 100.,
                "streams": {"mixed": 1., "shift": 1., "return": 1.}}
        results = {"dense": base, "symbolic": base}
        for kind in ("stale", "periodic", "incremental"):
            results[f"{kind}_1"] = {
                "accuracy": .995, "total_work": 30.,
                "streams": {"mixed": 1., "shift": .95, "return": 1.},
            }
            results[f"{kind}_2"] = {**base, "total_work": 60.}
            results[f"{kind}_4"] = {**base, "total_work": 90.}
        selected = select_methods(results)
        self.assertEqual(
            selected["selected_names"],
            ["dense", "symbolic", "stale_2", "periodic_2", "incremental_2"],
        )
        self.assertTrue(all(selected["quality_gate"].values()))


if __name__ == "__main__":
    unittest.main()
