# BANA 690A: Federated Learning I, Foundations

Three hospitals want one readmission model and cannot pool their patients.
Tonight you are the coordinator. Eight short labs, each a question the
coordinator must answer: how to average, what the knobs cost, who pays when
hospitals differ, what vertical federation changes, what a bad hospital can do,
what the defenses cost, and what a federation charter must say.

Instructor repo: `github.com/pouyan1361/governed-fl-foundations`
Everything runs in a Codespace in seconds. numpy and scikit-learn only. No framework.

## Step 0: fork, then Codespace (do this first, every student, on your own fork)

1. On GitHub, open `pouyan1361/governed-fl-foundations` and click **Fork**.
   Keep the name. Create the fork under your own account.
   You work on your fork, never on the instructor's repo.
2. On YOUR fork, click the green **Code** button, the **Codespaces** tab,
   **Create codespace on main**.
3. **Wait for the line `Setup complete` in the terminal** (about two minutes).
   The packages install in the background after the editor opens. Do not type
   before that line appears.
4. In the terminal, run:

       python labs/lab0_world.py

   You should see the three hospitals and one patient in human units.
5. **If you see `ModuleNotFoundError: No module named 'numpy'`** (or pandas,
   sklearn), the setup did not finish. Run this, wait a minute, run Lab 0 again:

       python -m pip install -r requirements.txt

6. A yellow underline under `import numpy` in the editor is cosmetic. If the
   terminal runs the script, you are fine. To clear it: Ctrl + Shift + P,
   **Python: Select Interpreter**, choose `/usr/local/bin/python`.
7. A Codespace stuck on "Opening Remote" for more than three minutes: delete it
   from the Codespaces tab on your fork and create a new one.

## The labs

| Lab | Command | The question |
|---|---|---|
| 0 | `python labs/lab0_world.py` | Who holds what? |
| 1 | `python labs/lab1_fedavg.py` then `python -m pytest -q tests/test_lab1.py` | Implement the weighted average. Who gains from federating? |
| 2 | `python labs/lab2_knobs.py --grid` | Which knob cuts communication rounds, and what does it cost? |
| 3 | `python labs/lab3_skew.py --skew label --finetune` (also quantity, feature, concept) | When hospitals differ, who pays, and what fixes it? |
| 4 | `python labs/lab4_vertical.py` | Same patients, different columns: what crosses the wire, what leaks? |
| 5 | `python labs/lab5_poison.py` and `python labs/lab5_leak.py` | What can a bad hospital do? What can a curious coordinator read? |
| 6 | `python labs/lab6_defend.py`, `python labs/lab6_secagg.py`, `python labs/lab6_dp.py` | What does each defense stop, and what does it cost? |
| 7 | `python run_federation.py` and `python run_federation.py --attack boost` | The charter as code: policy, audit log, model card, gate |
| A3 | `python labs/a3_harness.py` | The experiment grid for Assignment 3 |

Lab 1 is the only one where you write code: one line, the weighted average,
in `labs/lab1_fedavg.py`. The test tells you when it is right. In Lab 7 you
edit `governance/federation_policy.json` and watch the coordinator enforce it.

## Turning in your notes (in-class credit)

Open `LAB_NOTES.md`, type your name, and answer each prompt as you finish
each lab. By the end of class:

1. Source Control (third icon on the left bar). Stage `LAB_NOTES.md` (click the +)
   and, if you changed it, `governance/federation_policy.json`.
2. Message: `Lab notes: Firstname Lastname`. Click **Commit**.
3. Click **Sync Changes** (or **Publish Branch**). This pushes to YOUR fork.
4. On github.com, open your fork, click **Contribute**, **Open pull request**, title
   `Lab notes: Firstname Lastname`, **Create pull request**. There is no
   workflow to wait for; the pull request is just how I find your notes.

## The files

    fl/world.py          the three hospitals, the insurer, the neutral test set, the skews
    fl/model.py          logistic regression as one vector; gradient written out
    fl/federation.py     the configurable FedAvg loop every lab uses from Lab 2 on
    fl/aggregators.py    weighted mean, median, trimmed mean, Krum, clipping
    fl/attacks.py        label flipping, boosting, backdoor
    fl/privacy.py        secure aggregation, Gaussian DP, gradient reconstruction, membership inference
    fl/audit.py          the round log and the federated model card
    governance/federation_policy.json   the charter the coordinator enforces (Lab 7)
    run_federation.py    the governed federation: charter pre-check, rounds, gate, evidence
    labs/                one script per lab; read the docstring at the top of each
    tests/               Lab 1 check, charter consistency, simulator sanity

## Rules

- Fork first. Work in a Codespace on your fork. Never clone the instructor repo to work in.
- Run every lab yourself. The numbers in your notes must come from your own terminal.
- Everything is synthetic. Nobody in these files is a real patient.
