# Review and ownership rules (MyoSuite)

Antigravity and Gemini: follow `AGENTS.md` and the review rules in `REVIEW.md` at the repository root, for writing code as
well as reviewing it. In short:

- Decide the owner before adding code: reusable model construction and anatomical/physical transforms belong in `myo_sim`;
  task, reward, training, experiment and analysis workflows belong in MyoSuite. Search both repositories before adding a
  helper, class or solver, and reuse the existing primitive.
- Make the smallest compatible change and remove what it supersedes.
- A person opens the pull request under their own account. Never add an AI co-author or attribution trailer to a commit (CLA).
- Before pushing, run `scripts/review/local_review.sh` (`MYO_REVIEW_AGENT=gemini` to review with Gemini CLI); it records the
  checks in the PR description.
