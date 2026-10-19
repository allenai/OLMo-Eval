# SWE-bench Verified

`swe_bench_verified` runs an agent with a bash tool on each instance's prebuilt
SWE-bench image, collects its changes as a patch, and grades the patch in a fresh
container with the upstream `swebench` log parsers. The primary metric is
`resolve_rate`. Set `oracle=true` to grade the reference patches instead, which
checks the grading setup without a model.

Instance images come from Epoch AI's mirror on the GitHub Container Registry
(`ghcr.io/epoch-research/swe-bench.eval.x86_64.<instance_id>`), which needs no
login. Docker Hub, where the dataset's own image names point, rate-limits
anonymous pulls, so a full run from there stalls after about 100 images; set
`image_source=dataset` to use it anyway. Each instance's images are deleted
after it is graded, which keeps disk use flat over a full run; set
`cleanup_images=false` to keep them.

```bash
# A few instances on Beaker
uv run olmo-eval beaker launch -E swe_bench_verified -m qwen3-coder-30b -c ai2/jupiter \
    -A instance_ids=astropy__astropy-14309,django__django-10097 -A max_concurrency=4

# The full 500-instance set
uv run olmo-eval beaker launch -E swe_bench_verified -m qwen3-coder-30b -c ai2/jupiter \
    -A max_concurrency=8
```

The default scaffold is `vanillux`, the agent used for Terminal-Bench and for
tmax's SWE-bench Verified numbers. It gets the issue text as is, one `bash` tool,
64 steps, and samples at temperature 0.7, top_p 0.95 and 16384 max tokens; it
leaves thinking to the model's chat template. Set `scaffold=openai_agents` for
the SWE-agent-style prompt with `execute_bash_session` and `submit` tools, 100
turns, thinking off and the server's sampling defaults. Its run ends when the
model calls `submit()`; set `stop_on_submit=false` to let it keep going until
`max_turns`.

Each finished instance is appended to `swe_bench_verified_progress.jsonl` in the
output directory, so a job that is killed or preempted keeps its finished work.
Pass that file as `resume_from` to a new run to skip the instances it already
covers.

Other arguments include `limit`, `repos`, `max_turns`, `command_timeout`,
`agent_timeout`, `image_timeout`, `eval_timeout`, `include_hints`, and the
sampling arguments `temperature`, `top_p` and `max_tokens`;
`olmo-eval external-evals` lists them all with their defaults.

Qwen3.5 models need vLLM's Triton kernels on hosts without a CUDA compiler and
their XML tool parser: `-K gdn_prefill_backend=triton -K tool_call_parser=qwen3_xml`.
On B300 nodes, also pass `-K attention_backend=TRITON_ATTN -K language_model_only=true
-K startup_timeout=900 -e VLLM_USE_FLASHINFER_SAMPLER=0`.

With the default settings, three full runs of Qwen3.5-9B resolved 51.7% ± 0.7
(mean ± standard error).

Scores from a single run carry a few points of noise: in one repeated run of 97
sphinx instances, 25 changed outcome even though the total resolved count moved
by one.

