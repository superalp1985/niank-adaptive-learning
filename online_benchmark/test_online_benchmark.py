"""CPU contract tests for the online branching-proof benchmark."""

from __future__ import annotations

import unittest
from dataclasses import replace
from unittest.mock import patch

from online_benchmark import (
    Answer,
    BranchingWorld,
    EventKind,
    Fact,
    OnlineSession,
    ProtocolError,
    Query,
    make_branching_world,
    training_schedule,
)


class RecordingLearner:
    def __init__(self) -> None:
        self.feedback = []

    def update(self, feedback):
        self.feedback.append(feedback)
        return {"updates": len(self.feedback)}


def example_world():
    facts = (
        Fact(1037, 0, 0, 1, 0),
        Fact(2881, 1, 1, 3, 1),
        Fact(6007, 0, 0, 2, 2),
        Fact(3119, 0, 0, 4, 3),
        Fact(8831, 4, 1, 5, 4),
        Fact(4999, 2, 1, 3, 5),
        Fact(7309, 1, 0, 5, 6),
    )
    queries = (
        Query("late", 0, (0, 1), frozenset((3,)), 6),
        Query("early", 0, (0, 1), frozenset((3,)), 2),
    )
    return BranchingWorld(facts, queries)


class BranchingProofTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if make_branching_world is not None:
            cls.world = example_world()
            cls.query = cls.world.queries[0]
            cls.visible = cls.world.visible_facts(cls.query.as_of)

    def setUp(self):
        self.assertIsNotNone(make_branching_world, "branching proof API is not implemented")

    def test_distinct_valid_proofs_are_accepted(self):
        proofs = self.world.reference(self.query)
        self.assertGreaterEqual(len(proofs), 2)
        self.assertNotEqual(proofs[0].support_ids, proofs[1].support_ids)

        for proof in proofs:
            score = self.world.score(
                self.query,
                Answer(proof.terminal, proof.support_ids),
                self.visible,
            )
            self.assertTrue(score.terminal_correct)
            self.assertTrue(score.evidence_valid)
            self.assertTrue(score.full_correct)

    def test_score_does_not_require_a_canonical_path(self):
        proofs = self.world.reference(self.query)
        first, second = proofs[:2]
        self.assertNotEqual(first.support_ids, second.support_ids)
        with patch.object(self.world, "reference", side_effect=AssertionError("oracle")):
            score = self.world.score(
                self.query, Answer(second.terminal, second.support_ids), self.visible
            )
        self.assertTrue(score.full_correct)

    def test_wrong_relation_or_disconnected_edge_is_rejected(self):
        proofs = self.world.reference(self.query)
        wrong = list(proofs[0].support_ids)
        wrong[0], wrong[1] = wrong[1], wrong[0]
        score = self.world.score(
            self.query, Answer(self.query.goal, tuple(wrong)), self.visible
        )
        self.assertFalse(score.evidence_valid)
        self.assertFalse(score.full_correct)

    def test_future_fact_is_not_valid_for_an_earlier_snapshot(self):
        earlier = self.world.queries[1]
        visible = self.world.visible_facts(earlier.as_of)
        future_proof = next(
            proof
            for proof in self.world.reference(self.query)
            if proof.support_ids not in {
                proof2.support_ids for proof2 in self.world.reference(earlier)
            }
        )
        score = self.world.score(
            earlier,
            Answer(earlier.goal, future_proof.support_ids),
            visible,
        )
        self.assertFalse(score.evidence_valid)
        self.assertFalse(score.full_correct)

    def test_missing_and_extra_support_records_are_rejected(self):
        proof = self.world.reference(self.query)[0]
        missing = proof.support_ids[:-1]
        extra = proof.support_ids + (proof.support_ids[0],)
        self.assertFalse(
            self.world.score(
                self.query, Answer(self.query.goal, missing), self.visible
            ).full_correct
        )
        self.assertFalse(
            self.world.score(
                self.query, Answer(self.query.goal, extra), self.visible
            ).full_correct
        )

    def test_distinct_terminal_entities_satisfying_goal_are_valid(self):
        query = replace(self.query, goals=frozenset((3, 5)))
        proofs = self.world.reference(query)
        self.assertEqual({proof.terminal for proof in proofs}, {3, 5})
        for proof in proofs:
            self.assertTrue(
                self.world.score(
                    query, Answer(proof.terminal, proof.support_ids), self.visible
                ).full_correct
            )

    def test_fabricated_record_id_is_rejected(self):
        proof = self.world.reference(self.query)[0]
        answer = Answer(proof.terminal, (-1,) + proof.support_ids[1:])
        self.assertFalse(self.world.score(self.query, answer, self.visible).full_correct)

    def test_valid_evidence_for_a_wrong_goal_is_not_a_correct_answer(self):
        score = self.world.score(self.query, Answer(5, (3119, 8831)), self.visible)
        self.assertTrue(score.evidence_valid)
        self.assertFalse(score.terminal_correct)
        self.assertFalse(score.full_correct)
        self.assertEqual(score.reason, "wrong_goal")

    def test_simple_path_constraint_rejects_a_cycle(self):
        facts = (Fact(1501, 0, 0, 1, 0), Fact(1502, 1, 1, 0, 1))
        world = BranchingWorld(facts)
        query = Query("cycle", 0, (0, 1), frozenset((0,)), 1, simple_path=True)
        answer = Answer(0, (1501, 1502))
        self.assertFalse(world.score(query, answer, facts).full_correct)
        self.assertFalse(world.reference(query))
        unrestricted = replace(query, simple_path=False)
        self.assertTrue(world.score(unrestricted, answer, facts).full_correct)

    def test_default_goal_description_names_the_query_constraint(self):
        self.assertIn("Entity 0003", self.query.observable().goal_description)

    def test_generated_goal_accepts_every_reachable_terminal(self):
        world = make_branching_world(17, n_entities=24, query_count=4, dim=12)
        for query in world.queries:
            reachable = {query.start}
            for relation in query.relations:
                reachable = {
                    fact.target
                    for fact in world.visible_facts(query.as_of)
                    if fact.source in reachable and fact.relation == relation
                }
            self.assertEqual(query.goals, frozenset(reachable))
            self.assertEqual(
                query.observable().goal_description,
                "any terminal reachable by this relation chain",
            )

    def test_repeated_generation_is_identical_and_uses_opaque_ids(self):
        first = make_branching_world(17, n_entities=24, query_count=4, dim=12)
        another = make_branching_world(17, n_entities=24, query_count=4, dim=12)
        self.assertEqual(first.facts, another.facts)
        self.assertEqual(first.queries, another.queries)
        self.assertTrue(all(fact.id > 1000 for fact in first.facts))
        self.assertEqual(len({fact.id for fact in first.facts}), len(first.facts))
        self.assertTrue(all(len(first.reference(query)) >= 2 for query in first.queries))


class EventProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if make_branching_world is not None:
            cls.world = example_world()

    def setUp(self):
        self.assertIsNotNone(OnlineSession, "online event API is not implemented")

    def _session_with_visible_facts(self):
        session = OnlineSession(self.world.facts)
        for fact in sorted(self.world.facts, key=lambda item: item.observed_at):
            if fact.observed_at <= self.world.queries[0].as_of:
                session.observe_fact(fact)
        return session

    def test_feedback_and_update_are_post_answer_events(self):
        session = self._session_with_visible_facts()
        query = self.world.queries[0]
        view = session.ask(query)
        proof = self.world.reference(query)[0]
        session.commit_answer(
            query.id,
            Answer(proof.terminal, proof.support_ids),
            route_cost=12.5,
            database_cost=2.25,
        )
        learner = RecordingLearner()
        with self.assertRaises(ProtocolError):
            session.update(query.id, learner)
        with self.assertRaises(ProtocolError):
            session.feedback(query.id)

        score_event = session.score(query.id)
        self.assertTrue(score_event.score.full_correct)
        with self.assertRaises(ProtocolError):
            session.update(query.id, learner)

        feedback_event = session.feedback(query.id)
        self.assertTrue(feedback_event.feedback.full_correct)
        update_event = session.update(query.id, learner)
        self.assertEqual(update_event.result["updates"], 1)
        self.assertEqual(len(learner.feedback), 1)
        self.assertEqual(learner.feedback[0].route_cost, 12.5)
        self.assertEqual(learner.feedback[0].database_cost, 2.25)

        kinds = [event.kind for event in session.events]
        self.assertEqual(
            kinds[-5:],
            [
                EventKind.ASK,
                EventKind.COMMIT_ANSWER,
                EventKind.SCORE,
                EventKind.FEEDBACK,
                EventKind.UPDATE,
            ],
        )

    def test_feedback_alone_does_not_update_learner(self):
        session = self._session_with_visible_facts()
        query = self.world.queries[0]
        learner = RecordingLearner()
        session.ask(query)
        session.commit_answer(query.id, Answer(None, ()))
        session.score(query.id)
        self.assertEqual(learner.feedback, [])
        session.feedback(query.id)
        self.assertEqual(learner.feedback, [])
        session.update(query.id, learner)
        self.assertEqual(len(learner.feedback), 1)

    def test_asked_snapshot_cannot_gain_later_facts(self):
        session = OnlineSession(self.world.facts)
        query = self.world.queries[1]
        for fact in sorted(self.world.facts, key=lambda item: item.observed_at):
            if fact.observed_at <= query.as_of:
                session.observe_fact(fact)
        view = session.ask(query)
        self.assertTrue(
            all(fact.observed_at <= query.as_of for fact in view.visible_records)
        )

        later = [
            fact for fact in self.world.facts if fact.observed_at > query.as_of
        ]
        self.assertTrue(later)
        for fact in later:
            session.observe_fact(fact)

        future_proof = next(
            proof
            for proof in self.world.reference(self.world.queries[0])
            if any(
                self.world.fact(record).observed_at > query.as_of
                for record in proof.support_ids
            )
        )
        session.commit_answer(
            query.id,
            Answer(query.goal, future_proof.support_ids),
        )
        score = session.score(query.id).score
        self.assertFalse(score.evidence_valid)
        self.assertFalse(score.full_correct)

    def test_ask_view_exposes_facts_but_not_evaluation_reference(self):
        session = self._session_with_visible_facts()
        view = session.ask(self.world.queries[0])
        self.assertTrue(view.visible_records)
        self.assertFalse(hasattr(view, "reference"))
        self.assertFalse(hasattr(view, "score"))
        self.assertTrue(all(not hasattr(record, "target") for record in view.visible_records))
        with self.assertRaises(ProtocolError):
            session.ask(self.world.queries[0])

    def test_event_order_rejects_duplicate_or_skipped_transitions(self):
        session = self._session_with_visible_facts()
        query = self.world.queries[0]
        proof = self.world.reference(query)[0]
        with self.assertRaises(ProtocolError):
            session.score(query.id)
        session.ask(query)
        with self.assertRaises(ProtocolError):
            session.ask(query)
        session.commit_answer(query.id, Answer(proof.terminal, proof.support_ids))
        with self.assertRaises(ProtocolError):
            session.commit_answer(query.id, Answer(proof.terminal, proof.support_ids))

    def test_unique_questions_and_repetitions_are_counted_separately(self):
        queries = training_schedule(self.world.queries, unique_count=2, repeats=3)
        self.assertEqual(len(queries), 6)
        self.assertEqual(len({query.id for query in queries}), 6)
        self.assertEqual(len({query.problem_id for query in queries}), 2)
        session = OnlineSession(self.world.facts)
        learner = RecordingLearner()
        unobserved = sorted(self.world.facts, key=lambda fact: fact.observed_at)
        for query in queries:
            while unobserved and unobserved[0].observed_at <= query.as_of:
                session.observe_fact(unobserved.pop(0))
            proof = self.world.reference(query)[0]
            session.ask(query)
            session.commit_answer(query.id, Answer(proof.terminal, proof.support_ids))
            session.score(query.id)
            session.feedback(query.id)
            session.update(query.id, learner)
        self.assertEqual(session.training_counts.unique_questions, 2)
        self.assertEqual(session.training_counts.presentations, 6)
        self.assertEqual(session.training_counts.feedback_events, 6)
        self.assertEqual(session.training_counts.updates, 6)
        self.assertEqual(session.training_counts.presentations_per_problem, (3, 3))

    def test_training_schedule_preserves_causal_time_and_problem_repetitions(self):
        queries = training_schedule(self.world.queries, unique_count=2, repeats=3)
        self.assertEqual(
            [query.as_of for query in queries], sorted(query.as_of for query in queries)
        )
        self.assertEqual(
            [query.problem_id for query in queries[:3]],
            [self.world.queries[1].problem_id] * 3,
        )

    def test_feedback_does_not_expose_true_support_or_target_set(self):
        session = self._session_with_visible_facts()
        query = self.world.queries[0]
        session.ask(query)
        session.commit_answer(query.id, Answer(query.goal, ()))
        session.score(query.id)
        feedback = session.feedback(query.id).feedback
        self.assertFalse(feedback.full_correct)
        self.assertFalse(hasattr(feedback, "reference"))
        self.assertFalse(hasattr(feedback, "support_ids"))
        self.assertFalse(hasattr(feedback, "goals"))
        self.assertFalse(hasattr(feedback, "evidence_valid"))

    def test_training_schedule_rejects_same_problem_counted_twice(self):
        repeated = replace(self.world.queries[0], id="different-presentation")
        with self.assertRaises(ValueError):
            training_schedule(
                [self.world.queries[0], repeated], unique_count=2, repeats=1
            )

    def test_unobserved_fact_cannot_be_read_from_registry(self):
        session = OnlineSession(self.world.facts)
        fact = self.world.facts[0]
        with self.assertRaises(ProtocolError):
            session.read_fact(fact.id)
        session.observe_fact(fact)
        self.assertEqual(session.read_fact(fact.id), fact)

    def test_a_future_observation_cannot_precede_a_historical_question(self):
        session = self._session_with_visible_facts()
        with self.assertRaises(ProtocolError):
            session.ask(self.world.queries[1])

    def test_question_time_cannot_move_backwards(self):
        session = OnlineSession(self.world.facts)
        session.ask(self.world.queries[0])
        with self.assertRaises(ProtocolError):
            session.ask(self.world.queries[1])
        with self.assertRaises(ProtocolError):
            session.observe_fact(self.world.facts[0])

    def test_fabricated_or_duplicate_observation_is_rejected(self):
        session = OnlineSession(self.world.facts)
        fact = self.world.facts[0]
        with self.assertRaises(ProtocolError):
            session.observe_fact(replace(fact, target=9))
        session.observe_fact(fact)
        with self.assertRaises(ProtocolError):
            session.observe_fact(fact)

    def test_frozen_evaluation_has_no_feedback_or_updates(self):
        session = OnlineSession(self.world.facts, feedback_enabled=False)
        for fact in self.world.facts:
            session.observe_fact(fact)
        query = self.world.queries[0]
        session.ask(query)
        session.commit_answer(query.id, Answer(None, ()))
        session.score(query.id)
        with self.assertRaises(ProtocolError):
            session.feedback(query.id)
        with self.assertRaises(ProtocolError):
            session.update(query.id, RecordingLearner())
        self.assertEqual(session.training_counts.feedback_events, 0)
        self.assertEqual(session.training_counts.updates, 0)

    def test_feedback_and_update_can_each_be_consumed_only_once(self):
        session = self._session_with_visible_facts()
        query = self.world.queries[0]
        session.ask(query)
        session.commit_answer(query.id, Answer(None, ()))
        session.score(query.id)
        session.feedback(query.id)
        with self.assertRaises(ProtocolError):
            session.feedback(query.id)
        session.update(query.id, RecordingLearner())
        with self.assertRaises(ProtocolError):
            session.update(query.id, RecordingLearner())

    def test_database_reads_and_supplied_costs_are_counted_once(self):
        session = OnlineSession(self.world.facts)
        for fact in self.world.facts:
            session.observe_fact(fact, observation_cost=0.5)
        query = self.world.queries[0]
        session.ask(query)
        session.read_fact(1037, query.id)
        session.read_fact(2881, query.id)
        session.commit_answer(
            query.id, Answer(3, (1037, 2881)), route_cost=2,
            database_cost=0.75, maintenance_cost=1.5, inference_seconds=0.1,
        )
        session.score(query.id)
        feedback = session.feedback(query.id).feedback
        session.update(query.id, RecordingLearner(), update_cost=4)
        self.assertEqual(feedback.database_cost, 2.75)
        self.assertEqual(feedback.total_work, 6.25)
        self.assertEqual(
            dict(session.costs),
            {"observe": 3.5, "route": 2.0, "database": 2.75,
             "maintenance": 1.5, "update": 4.0},
        )
        self.assertEqual(session.total_work, 13.75)

    def test_failed_update_is_consumed_and_accounted_without_retry(self):
        class FailingLearner(RecordingLearner):
            def update(self, feedback):
                super().update(feedback)
                raise RuntimeError("failed after changing model state")

        session = self._session_with_visible_facts()
        query = self.world.queries[0]
        session.ask(query)
        session.commit_answer(query.id, Answer(None, ()))
        session.score(query.id)
        session.feedback(query.id)
        learner = FailingLearner()
        with self.assertRaises(RuntimeError):
            session.update(query.id, learner, update_cost=3)
        with self.assertRaises(ProtocolError):
            session.update(query.id, learner, update_cost=3)
        self.assertEqual(len(learner.feedback), 1)
        self.assertEqual(session.costs["update"], 3.0)
        self.assertEqual(session.training_counts.updates, 1)
        self.assertEqual(session.events[-1].kind, EventKind.UPDATE)
        self.assertEqual(session.events[-1].result, {"error_type": "RuntimeError"})


if __name__ == "__main__":
    unittest.main()
