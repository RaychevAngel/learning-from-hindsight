"""Scientific reporting invariants, independent of GPU/model availability."""

import unittest

from hindsight.analysis import FIELDS, paired_interval, summarize


def row(episode, task, condition, success, candidate=None, role="Q", partition="development"):
    return dict(episode_id=episode, partition=partition, family="translation",
                condition=condition, candidate=candidate, task_id=task, role=role,
                success=success, format_ok=True, cap_hit=False)


def plan(records):
    return [{field: record[field] for field in FIELDS} for record in records]


class AnalysisTests(unittest.TestCase):
    def test_source_clusters_have_equal_weight_despite_different_task_counts(self):
        records = [row("a", "q", "baseline", False),
                   row("a", "q", "adapted_bidirectional", True)]
        for task in range(3):
            records += [row("b", str(task), "baseline", False),
                        row("b", str(task), "adapted_bidirectional", False)]
        result = summarize(records, expected=plan(records))
        comparison = result["paired_vs_baseline"][0]
        self.assertEqual(result["status"], "complete")
        self.assertEqual(comparison["episodes"], 2)
        self.assertEqual(comparison["mean_difference"], 0.5)
        self.assertEqual(comparison["ci95"], [0.0, 1.0])
        self.assertEqual(comparison["by_family"]["translation"]["episodes"], 2)

    def test_missing_planned_trial_excludes_whole_episode_not_only_missing_task(self):
        records = [row("a", task, condition, condition != "baseline")
                   for task in ("q1", "q2") for condition in ("baseline", "adapted_pi")]
        result = summarize(records[:-1], expected=plan(records))
        self.assertEqual(result["status"], "incomplete")
        self.assertEqual(len(result["missing_records"]), 1)
        self.assertEqual(result["paired_vs_baseline"][0]["episodes"], 0)
        self.assertIsNone(result["paired_vs_baseline"][0]["mean_difference"])

    def test_no_plan_cannot_claim_complete(self):
        result = summarize([row("a", "q", "baseline", True)])
        self.assertEqual(result["status"], "plan_unverified")
        self.assertIsNone(result["planned_records"])

    def test_duplicate_success_records_do_not_create_pseudoreplication(self):
        record = row("a", "q", "baseline", True)
        with self.assertRaisesRegex(ValueError, "Duplicate observation"):
            summarize([record, record])

    def test_unpaired_tasks_are_rejected_even_when_counts_match(self):
        records = [row("a", "q1", "baseline", False),
                   row("a", "q2", "adapted_bidirectional", True)]
        with self.assertRaisesRegex(ValueError, "Unpaired task sets"):
            summarize(records)

    def test_final_known_and_unseen_are_separate_and_require_freezing(self):
        records = []
        for partition in ("final_known", "final_unseen"):
            records += [row(partition, "q", "baseline", False, partition=partition),
                        row(partition, "q", "adapted_bidirectional", True, partition=partition)]
        with self.assertRaisesRegex(ValueError, "frozen protocol"):
            summarize(records)
        result = summarize(records, protocol_frozen=True)
        comparisons = result["paired_vs_baseline"]
        self.assertEqual({item["partition"] for item in comparisons},
                         {"final_known", "final_unseen"})
        self.assertTrue(all(item["episodes"] == 1 and item["ci95"] is None
                            for item in comparisons))

    def test_v_ordering_handles_ties_and_never_uses_q(self):
        records = []
        for role in ("V", "Q"):
            for task in ("1", "2"):
                task_id = role + task
                records.append(row("a", task_id, "baseline", False, role=role))
                for candidate in range(3):
                    for condition in ("direct_pi", "adapted_pi"):
                        score = candidate == 2
                        if condition == "adapted_pi" and candidate == 1:
                            score = task == "1"
                        records.append(row("a", task_id, condition,
                                           score if role == "V" else not score,
                                           candidate=candidate, role=role))
        item = summarize(records)["within_source_V_ordering"][0]
        self.assertEqual(item["concordant_pairs"], 2)
        self.assertEqual(item["direct_only_ties"], 1)
        self.assertEqual(item["discordant_pairs"], 0)
        self.assertAlmostEqual(item["kendall_tau_b"], 2 / (6 ** 0.5))

    def test_supplied_selection_is_not_reselected_using_query_outcomes(self):
        records = [row("a", "q", "baseline", False),
                   row("a", "q", "adapted_pi", False, candidate=0),
                   row("a", "q", "adapted_pi", True, candidate=1)]
        selection = dict(episode_id="a", condition="adapted_pi",
                         selected_candidate=0, derived_from="V")
        result = summarize(records, selections=[selection])
        self.assertEqual(result["frozen_V_selection_Q"][0]["mean_difference"], 0.0)
        selection["selected_candidate"] = None
        result = summarize(records, selections=[selection])
        self.assertEqual(result["frozen_V_selection_Q"][0]["mean_difference"], 0.0)
        selection["derived_from"] = "Q"
        with self.assertRaisesRegex(ValueError, "never Q"):
            summarize(records, selections=[selection])

    def test_false_is_valid_and_missing_outcomes_are_not_false(self):
        record = row("a", "q", "baseline", False)
        record["format_ok"] = False
        record["cap_hit"] = True
        result = summarize([record])
        self.assertEqual(result["conditions"][0]["source_mean_success"], 0.0)
        self.assertEqual(result["conditions"][0]["format_failures"], 1)
        self.assertEqual(result["conditions"][0]["cap_hits"], 1)
        del record["success"]
        with self.assertRaisesRegex(ValueError, "explicit boolean"):
            summarize([record])

    def test_meta_selection_does_not_invent_unplanned_query_work(self):
        records = [row("a", "v", "baseline", False, role="V", partition="meta_train")]
        event = dict(episode_id="a", condition="adapted_pi",
                     selected_candidate=None, derived_from="V")
        result = summarize(records, selections=[event], expected=plan(records))
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["frozen_V_selection_Q"], [])
        self.assertEqual(len(result["selections_without_query_plan"]), 1)

    def test_metadata_overlap_and_unplanned_work_are_rejected(self):
        v = row("a", "task", "baseline", True, role="V")
        q = row("a", "task", "baseline", True, role="Q")
        with self.assertRaisesRegex(ValueError, "both V and Q"):
            summarize([v, q])
        with self.assertRaisesRegex(ValueError, "outside the supplied plan"):
            summarize([v], expected=[])

    def test_bootstrap_is_seeded_and_has_no_single_cluster_precision_claim(self):
        self.assertEqual(paired_interval([0.3, -0.2, 1.0], seed=7),
                         paired_interval([0.3, -0.2, 1.0], seed=7))
        self.assertIsNone(paired_interval([1.0])["ci95"])
        with self.assertRaises(ValueError):
            paired_interval([float("nan")])


if __name__ == "__main__":
    unittest.main()
