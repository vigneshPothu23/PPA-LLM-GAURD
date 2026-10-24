# Engine 1: polymorphic prompt assembly (v5, independent implementation)

Code: `src/ppaguard/engine1/` (`separators.py`, `templates.py`, `assembler.py`, `forgery.py`).
Tests: `tests/v5/test_v5_engine1.py` (31 tests).
No code is imported from the earlier `ppa_guard.engine1` package; `test_v5_sdk_and_independence.py` enforces this.

## What it does

When the policy allows a request to go forward, Engine 1 builds the prompt that the calling application sends to its own
LLM. For every request it:

1. **Draws a fresh boundary.** It picks one of 8 marker families (`[[TAG-N]]`, `<TAG_N>`, `##### TAG N BEGIN #####`,
   …) and one of 8 tag words. Then it adds a 48-bit random nonce `N` from `secrets`, a cryptographic source. Example:
   `[[UNTRUSTED-9F3A1C0B77E2]] … [[/UNTRUSTED-9F3A1C0B77E2]]`.
2. **Checks for collisions.** It redraws if the marker or its nonce already appears in the input or in the trusted
   instructions. After 16 failed draws it refuses with `SeparatorCollision`, and the workflow then BLOCKs.
3. **Picks a template.** It chooses one of 3 equivalent instruction templates. Each one:
   - states the task;
   - says the delimited text is untrusted data whose instructions must not be followed;
   - places the data between the markers;
   - **restates the task after the data**.
4. **Inserts the data unchanged.** The untrusted text goes in exactly as received. It fills a single slot after the
   trusted parts have been formatted, so braces in the input are never treated as format fields.
5. **Checks the response, if you want it to.** `check_response(response, assembled)` reports whether an LLM's output
   reproduced this request's marker or nonce. That would show the model echoed the prompt structure.

`find_forgery(text)` supplies a separate **signal** to the policy engine; Engine 1 never edits the input. It looks for:
- text shaped like this engine's own marker families;
- chat-template control tokens such as `<|im_start|>`, `[INST]` and `<<SYS>>`;
- delimiter lines with keywords such as `###END###` and `===== END OF INPUT =====`.

Findings are split into strong and weak. Strong findings trigger the policy rule. Weak ones (`Assistant:` headers,
"end of input" phrasing) are only reported, to limit false holds.

## Relation to the referenced method

The reference is Wang et al., *To Protect the LLM Agent Against the Prompt Injection Attack with Polymorphic Prompt*
(DSN-S 2025), and its SDK. They share the core idea: the structure around the untrusted input changes randomly on every
request, and the instructions tell the model to treat whatever sits between the markers as data. An attacker therefore
cannot write a closing marker that matches the real one.

This implementation was written from that description. It **does not claim equivalence** with the paper's SDK. The
differences:

| Aspect | This implementation | Note |
|---|---|---|
| Separator source | 8 families × 8 tags × 48-bit random nonce | The SDK draws from a fixed list of separator pairs. Adding a nonce makes the closing marker unguessable even to someone who knows every family. |
| Instruction text | 3 template variants, task restated after the data | The variants and the trailing reminder are design choices made here. |
| Collision handling | Redraw, then refuse | Never emits an ambiguous boundary. |
| Forgery signals | Separate detector feeding the policy | An addition. The SDK has no equivalent rule layer. |
| Response check | Marker/nonce echo check | Similar in spirit to the SDK's leak detection. |

**Not measured.** No LLM is connected in this project, so Engine 1's effect on attack success rates was not measured.
The tests show its structural properties:
- a fresh, unpredictable boundary on every request;
- untrusted text enclosed exactly once and unmodified;
- a fake closing marker that can never end the real data block;
- collision refusal;
- invalid-input refusal;
- every family and template variant in use.
