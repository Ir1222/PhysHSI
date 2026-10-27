"""Native terminal panel, collection cadence and detailed logging regression."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_mixed_handoff_training import (
    MixedHandoffOnPolicyRunner, TensorEnvironment, actor_critic, configuration,
)


class TerminalLoggerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.root = Path(self.folder.name)
        self.teacher = self.root / "teacher.pt"
        self.output = io.StringIO()
        self.capture = contextlib.redirect_stdout(self.output)
        self.capture.__enter__()
        torch.save(dict(model_state_dict=actor_critic().state_dict()), self.teacher)

    def tearDown(self):
        self.capture.__exit__(None, None, None)
        self.folder.cleanup()

    def runner(self, interval=None, save_interval=500, steps=1, log=True, detailed=False):
        cfg = configuration(amp=False, steps=steps)
        cfg["runner"]["save_interval"] = save_interval
        cfg["runner"]["detailed_handoff_events"] = detailed
        if interval is not None:
            cfg["runner"]["terminal_log_interval"] = interval
        with patch("torch.utils.tensorboard.SummaryWriter"):
            runner = MixedHandoffOnPolicyRunner(TensorEnvironment(), cfg, self.teacher,
                                                log_dir=self.root / "logs" if log else None)
        self.output.seek(0)
        self.output.truncate(0)
        return runner

    def metrics(self):
        return [json.loads(line) for line in (self.root / "logs/training_metrics.jsonl").read_text().splitlines()]

    def test_panel_format_is_native_width_with_selected_metrics_only(self):
        runner = self.runner()
        runner.collection_iteration = 7
        runner.handoff_successes, runner.handoff_failures = 3, 1
        runner.teacher_steps, runner.student_steps = 100, 300
        runner.alg.optimizer_updates = 2
        metrics = dict(value=.125, surrogate=-.03, learning_rate=2e-4, student_valid_count=30,
                       student_valid_ratio=.75, amp_valid_fraction=.5, handoff_success_rate=.75,
                       **{"Perf/total_fps": 200., "Perf/collection_time": .15, "Perf/learning_time": .05,
                          "Policy/mean_noise_std": .31, "performance/task_reward": .12,
                          "performance/amp_reward_valid": .4, "performance/student/completed_survival_s": 1.5,
                          "performance/student_reward_weighted/carry_box_tilt": -99.,
                          "performance/task_reward_std": 555.})
        before = copy.deepcopy(metrics)
        runner.log(metrics, 20, 6., 9., final=True)
        text = self.output.getvalue()
        self.assertEqual(text.splitlines()[0], "#" * 80)
        self.assertIn("-" * 80, text)
        self.assertTrue(all(len(line) <= 80 for line in text.splitlines()))
        for label in ("Learning/Collection iteration 7/20 (final)", "Computation:", "Collection time:", "Learning time:",
                      "Value function loss:", "Surrogate loss:", "Mean action noise std:", "Learning rate:",
                      "Mean Student task reward:", "Mean AMP reward:", "Student-valid ratio:", "AMP-valid fraction:",
                      "Handoff success rate:", "Mean completed Student survival:", "Actual PPO optimizer updates:",
                      "Total timesteps:", "Total time (this run):", "ETA (collection limit):"):
            self.assertIn(label, text)
        for value in ("200 steps/s", "0.150s", "0.050s", "0.1250", "-0.0300", "2.00e-04", "75.00%", "1.500s"):
            self.assertIn(value, text)
        self.assertNotIn("carry_box_tilt", text)
        self.assertNotIn("task_reward_std", text)
        self.assertNotIn("555", text)
        self.assertEqual(metrics, before)

    def test_missing_samples_and_nonfinite_values_display_na(self):
        runner = self.runner()
        runner.log({"value": float("nan"), "surrogate": float("inf"), "student_valid_count": 0,
                    "amp_valid_fraction": 0., "handoff_success_rate": 0.}, 1, 1., 0.)
        text = self.output.getvalue()
        for label in ("Value function loss", "Surrogate loss", "Mean Student task reward", "Mean AMP reward",
                      "AMP-valid fraction", "Handoff success rate", "Mean completed Student survival"):
            line = next(line for line in text.splitlines() if label + ":" in line)
            self.assertTrue(line.endswith("N/A"), line)

    def test_default_prints_only_at_100_200_and_final_205(self):
        runner = self.runner(log=False)
        with patch.object(runner, "log", wraps=runner.log) as log:
            runner.learn(205, optimize=False)
        self.assertEqual([c.args[0]["collection_iterations"] for c in log.call_args_list], [100, 200, 205])
        self.assertEqual([c.kwargs["final"] for c in log.call_args_list], [False, False, True])
        self.assertEqual(runner.alg.optimizer_updates, 0)
        self.assertFalse(any(line.startswith("{") for line in self.output.getvalue().splitlines()))

    def test_terminal_save_and_tensorboard_cadences_are_independent(self):
        runner = self.runner(interval=3, save_interval=2)
        with patch.object(runner, "log", wraps=runner.log) as log, patch.object(runner, "save") as save:
            runner.learn(7, optimize=False)
        self.assertEqual([c.args[0]["collection_iterations"] for c in log.call_args_list], [3, 6, 7])
        self.assertEqual([c.args[0].name for c in save.call_args_list], ["model_2.pt", "model_4.pt", "model_6.pt", "model_7.pt"])
        self.assertEqual([m["collection_iterations"] for m in self.metrics()], list(range(1, 8)))
        scalars = runner.writer.add_scalar.call_args_list
        self.assertEqual([c.args[2] for c in scalars if c.args[0] == "collection_iterations"], list(range(1, 8)))
        self.assertTrue(any(c.args[0] == "performance/student_reward_weighted/carry_box_tilt" for c in scalars))
        runner.writer.add_histogram.assert_not_called()
        self.assertFalse((self.root / "logs/handoff_events.jsonl").exists())
        self.assertEqual(runner.writer.flush.call_count, 7)
        runner.writer.close.assert_called_once()

    def test_episode_rollout_cumulative_ratios_and_real_tensorboard_json(self):
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        cfg = configuration(amp=False, steps=10)
        directory = self.root / "metrics"
        runner = MixedHandoffOnPolicyRunner(TensorEnvironment(), cfg, self.teacher, log_dir=directory)
        runner.learn(2, max_optimizer_updates=None)
        rows = [json.loads(line) for line in (directory / "training_metrics.jsonl").read_text().splitlines()]
        first, second = rows
        self.assertAlmostEqual(first["student_valid_ratio"], 14 / 20)
        self.assertAlmostEqual(first["student_step_fraction"], 14 / 20)
        self.assertEqual(first["episode_student_fraction_count"], 0)
        self.assertEqual(first["student/completed_survival_s/count"], 0)
        self.assertEqual(first["student/censored_timeout_s/count"], 0)
        self.assertEqual(first["student/segment_duration_s/count"], 0)
        self.assertAlmostEqual(second["student_valid_ratio"], 16 / 20)
        self.assertAlmostEqual(second["student_step_fraction"], 30 / 40)
        self.assertAlmostEqual(second["episode_student_fraction"], (14 / 16 + 14 / 18) / 2)
        self.assertEqual(second["episode_student_fraction_count"], 2)
        self.assertAlmostEqual(second["amp_valid_fraction"], 10 / 16)
        self.assertAlmostEqual(second["cumulative_amp_valid_fraction"], 10 / 30)
        for key, count in (("student/segment_duration_s", 2),
                           ("student/completed_survival_s", 1), ("student/censored_timeout_s", 1)):
            self.assertAlmostEqual(second[key], .28)
            self.assertEqual(second[key + "/count"], count)
            self.assertAlmostEqual(second["distribution/" + key + "/median"], .28)
            self.assertAlmostEqual(second["distribution/" + key + "/p90"], .28)
        events = EventAccumulator(str(directory)).Reload()
        for key in ("student_valid_ratio", "student_step_fraction", "episode_student_fraction",
                    "episode_student_fraction_count", "amp_valid_fraction", "cumulative_amp_valid_fraction",
                    "student/segment_duration_s", "student/completed_survival_s", "student/censored_timeout_s",
                    "kl", "learning_rate"):
            scalars = events.Scalars(key)
            self.assertEqual(scalars[-1].step, 2)
            self.assertAlmostEqual(scalars[-1].value, second[key], places=6)
        # Resumed cumulative counters continue, while physical episodes restart.
        resumed = MixedHandoffOnPolicyRunner(TensorEnvironment(), cfg, self.teacher, initialize=False)
        resumed.load(directory / "model_2.pt")
        with patch.object(resumed, "log") as log:
            resumed.learn(1, optimize=False)
        self.assertEqual(resumed.teacher_steps + resumed.student_steps, 60)
        self.assertEqual(resumed.student_steps, 44)
        self.assertAlmostEqual(log.call_args.args[0]["student_step_fraction"], 44 / 60)
        self.assertEqual(log.call_args.args[0]["episode_student_fraction_count"], 0)

    def test_completed_teacher_only_episodes_contribute_zero_fraction(self):
        class TeacherOnlySecond(TensorEnvironment):
            def step(self, actions):
                result = super().step(actions)
                self.handoff.phase[1] = 0
                result[4]["handoff"]["handoff_trigger"][1] = False
                return result

        runner = MixedHandoffOnPolicyRunner(TeacherOnlySecond(), configuration(False, 10), self.teacher)
        with patch.object(runner, "log") as log:
            runner.learn(2, optimize=False)
        metrics = log.call_args.args[0]
        self.assertEqual(metrics["episode_student_fraction_count"], 2)
        self.assertAlmostEqual(metrics["episode_student_fraction"], (14 / 16 + 0) / 2)
        self.assertAlmostEqual(metrics["performance/episode/zero_student_fraction"], .5)
        self.assertEqual(metrics["student/segment_duration_s/count"], 1)

    def test_debug_events_and_histograms_are_explicitly_enabled(self):
        runner = self.runner(steps=18, detailed=True)
        runner.learn(1, optimize=False)
        self.assertTrue((self.root / "logs/handoff_events.jsonl").read_text().strip())
        self.assertTrue(runner.writer.add_histogram.called)

    def test_exact_interval_completion_prints_one_final_panel(self):
        runner = self.runner(interval=2, log=False)
        with patch.object(runner, "log", wraps=runner.log) as log:
            runner.learn(2, optimize=False)
        log.assert_called_once()
        self.assertTrue(log.call_args.kwargs["final"])

    def test_update_limit_prints_final_even_before_first_log_interval(self):
        runner = self.runner(steps=14)
        with patch.object(runner, "log", wraps=runner.log) as log:
            runner.learn(20, max_optimizer_updates=1)
        log.assert_called_once()
        self.assertTrue(log.call_args.kwargs["final"])
        self.assertEqual(log.call_args.args[3], 0.)  # ETA on early completion.
        self.assertEqual(runner.collection_iteration, 1)
        self.assertEqual(runner.alg.optimizer_updates, 1)
        self.assertIn("Mean AMP reward: N/A", self.output.getvalue())
        runner.assert_teacher_frozen()

    def test_timing_fps_and_elapsed_use_measured_times(self):
        runner = self.runner(steps=2)
        runner.reset()
        with patch("rsl_rl.runners.mixed_handoff_on_policy_runner.time.perf_counter", side_effect=[10., 11., 13., 16., 20.]), \
                patch.object(runner, "log", wraps=runner.log) as log:
            runner.learn(1, optimize=False)
        metrics = self.metrics()[0]
        self.assertEqual(metrics["Perf/collection_time"], 2.)
        self.assertEqual(metrics["Perf/learning_time"], 3.)
        self.assertEqual(metrics["Perf/total_fps"], 4./5.)
        self.assertEqual(log.call_args.args[2], 10.)

    def test_cuda_timing_synchronizes_each_distinct_device_and_cpu_does_not(self):
        runner = self.runner(log=False)
        with patch("torch.cuda.synchronize") as sync:
            runner._synchronize_timing()
            sync.assert_not_called()
            runner.device = torch.device("cuda:0")
            runner.env.device = "cuda:1"
            runner._synchronize_timing()
            self.assertEqual({c.args[0] for c in sync.call_args_list}, {torch.device("cuda:0"), torch.device("cuda:1")})

    def test_resume_allows_old_config_and_new_terminal_interval(self):
        runner = self.runner(log=False)
        runner.learn(1, optimize=False)
        saved = self.root / "student.pt"
        runner.save(saved)  # Legacy runner config has no terminal_log_interval.
        resumed = self.runner(interval=1, save_interval=7, log=False)
        resumed.load(saved)
        self.assertEqual(resumed.terminal_log_interval, 1)
        with patch.object(resumed, "log", wraps=resumed.log) as log:
            resumed.learn(2, optimize=False)
        self.assertEqual([c.args[0]["collection_iterations"] for c in log.call_args_list], [2, 3])
        self.assertEqual([c.args[1] for c in log.call_args_list], [3, 3])

    def test_invalid_terminal_intervals_are_rejected(self):
        for interval in (0, -1, 1.5, True):
            with self.subTest(interval=interval), self.assertRaisesRegex(ValueError, "terminal_log_interval"):
                self.runner(interval=interval, log=False)

    def test_log_interval_does_not_change_training_weights_or_update_counts(self):
        states = []
        for interval in (1, 100):
            torch.manual_seed(123)
            runner = self.runner(interval=interval, steps=14, log=False)
            runner.learn(3)
            runner.assert_teacher_frozen()
            states.append((copy.deepcopy(runner.alg.actor_critic.state_dict()), runner.alg.optimizer_updates,
                           runner.alg.optimizer_steps, runner.teacher_steps, runner.student_steps))
        for key, value in states[0][0].items():
            torch.testing.assert_close(value, states[1][0][key], atol=0, rtol=0)
        self.assertEqual(states[0][1:], states[1][1:])

    def test_resumed_eta_uses_iterations_in_this_invocation(self):
        runner = self.runner(interval=1, log=False)
        runner.collection_iteration = 40
        with patch("rsl_rl.runners.mixed_handoff_on_policy_runner.time.perf_counter",
                   side_effect=[0., 1., 2., 3., 5., 6., 7., 8., 10.]), \
                patch.object(runner, "log", wraps=runner.log) as log:
            runner.learn(2, optimize=False)
        self.assertEqual([c.args[0]["collection_iterations"] for c in log.call_args_list], [41, 42])
        self.assertEqual([c.args[1] for c in log.call_args_list], [42, 42])
        self.assertEqual([c.args[2] for c in log.call_args_list], [5., 10.])
        self.assertEqual([c.args[3] for c in log.call_args_list], [5., 0.])


if __name__ == "__main__":
    unittest.main()
