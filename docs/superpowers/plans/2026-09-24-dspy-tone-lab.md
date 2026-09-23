# DSPy tone lab implementation plan

1. Add a versioned prompt program and make the Worker render feedback from it.
2. Add attempt-sensitive escalation and contract tests.
3. Add fixed train/eval examples and Jev metric helpers.
4. Add a reproducible DSPy MIPROv2 compiler using Cloudflare's OpenAI-compatible
   endpoint for Granite and universal endpoint for Jev.
5. Run the lab, select the better prompt, and check in its result artifact.
6. Add a public `/lab` page and link it from the tool.
7. Update documentation, run all checks, deploy, verify live, commit, and push.
