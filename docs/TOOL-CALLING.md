# Tool calling / agent harnesses

DeepSeek-V4-Flash uses the DSML tool-call format. Out of the box on this
stack, tool calling works for short direct prompts but degrades badly under
long agentic prompts (17k-token system prompt with many tools): the model
emits malformed DSML openers (`<|DSML|>` with no name/args) or drifts into
fabricated plain-text transcripts. Patches 11/11b/11c/12 fix this
server-side: 0/8 -> 8/8 structured tool calls on the same 17k-token payload,
with no client-side stop sequences needed.

## What the patches do

The insight: vLLM's structural-tag grammar machinery for DSML existed but was
dead code on the chat path, and its trigger was wrong for long prompts.

1. **Patch 11** — the chat endpoint never called `parser.adjust_request`, so
   the structural-tag grammar was never attached. Wire it in; requests with
   tools but no `tool_choice` default to `auto`.
2. **Patch 11b** — the registry skipped grammar construction for non-strict
   tools; many clients send `strict: None`. Build it anyway.
3. **Patch 11c** — the grammar trigger was the full `<|DSML|tool_calls>`
   opener, which the model rarely types unaided under long prompts. Trigger
   on the bare `<|DSML|>` token instead, so *any* DSML attempt is
   grammar-railroaded into a valid call.
4. **Patch 12** — the tokenizer appends a tool-format reminder as the *last*
   system message when tools are present (recency position), countering
   opener drift on long prompts.

## Server flags

```
--served-model-name dsv4-flash
--enable-auto-tool-choice
--tool-call-parser deepseek_v4
--tokenizer-mode deepseek_v4
```

(Already in `scripts/serve_256k_marlin.sh`.)

## Client configuration

OpenAI-compatible; example provider block (any harness):

```yaml
api: openai-completions
baseUrl: http://<server>:8000/v1
model: dsv4-flash
contextWindow: 262144
maxTokens: 16384
supportsTools: true
extraBody:
  temperature: 0.3
  top_p: 0.95
```

Notes:

- vLLM rejects `logit_bias` and `min_p` when speculative decode is on — don't
  send them.
- Client-side stop sequences like `"\n<tool"` can truncate plain-text
  tool-call impostors as a belt-and-suspenders measure, but after patches
  11-12 they were not needed (8/8 without them).
- If your harness's tool descriptions embed *other* models' calling
  conventions as examples (e.g. XML `<invoke>` blocks), they actively teach
  DSV4 the wrong grammar and increase drift. Strip them if you can.

## Trust boundary

Treat this model's output as untrusted, as you would any open-weights model
serving agentic workloads: under long prompts it can fabricate
instruction-shaped text (including text that looks like injected commands).
Don't pipe its raw output into another agent's context unfiltered, and gate
any tool execution on your harness's own permission model, not on the
model's claims.
