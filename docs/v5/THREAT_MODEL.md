# PPA Guard — Threat model for indirect prompt injection (context guard)

This document states what the experimental context guard (`src/ppaguard/experimental/context_guard/`) is
and is not designed to defend against. It is scoped to **indirect prompt injection**: untrusted content
that an LLM application ingests as *data* (an e-mail to summarise, a document to analyse, a table or code
file to process) which contains text that tries to steer the model as if it were an *instruction*.

## 1. System and trust boundaries

```
 user (trusted) ──► application ──► PPA Guard ──► application's own LLM
                        ▲
 untrusted content ─────┘   (e-mails, documents, tables, code, web snippets, tool output)
```

- **Trusted channel:** the application's own system instructions and the end user's request. PPA Guard
  never treats these as hostile.
- **Untrusted channel (in scope):** any third-party content the application places in front of the model.
  This is the "data channel". The context guard scores this content.
- The guard sits before the application's LLM call. Engine 1 (polymorphic assembly) still isolates the
  data structurally; the context guard is an additional *detection* signal, not a replacement for Engine 1.

## 2. Attacker capabilities (assumed)

The attacker can place arbitrary text into the untrusted channel — they wrote the e-mail, edited the
shared document, control the web page that gets retrieved, or seeded the record in a database. They **cannot**
see PPA Guard's per-request random boundaries (Engine 1 nonces), modify the application's system prompt, or
observe the guard's internal scores.

## 3. Attacks in scope

| Class | Example (sourced from published datasets, not authored here) |
|---|---|
| Goal hijack / instruction override | "Ignore the previous instructions and instead …" embedded in a document |
| Topic-shift directives | TaskTracker-style "A new directive is about to land …" inserted into a paragraph |
| Tool / API abuse via content | LLMail-Inject e-mail bodies that try to make an agent call a send-email API |
| Delimiter / structure forgery | Fake `<<end_context>>`, `[/INST]`, `<|assistant|>` markers that try to end the data region |
| Data exfiltration requests | Content asking the model to append its context or post data to a URL |
| Task-irrelevant injected instructions | A harmless but off-task request ("write an essay about …") planted in a document |

## 4. Explicitly out of scope

- **Direct prompt injection** typed by the end user themselves (that is the user's own channel; the frozen
  dev set's deepset rows are of this kind and are used only to measure over-blocking of the user channel).
- **Jailbreaks of the application's own model** that do not arrive through untrusted content.
- **Multi-turn/agentic state attacks**, memory poisoning across sessions, and attacks that depend on tool
  outputs the guard never sees.
- **Semantic attacks with no lexical, structural, or topical signal** — e.g. a single on-topic sentence
  that is a legitimate-looking instruction the user would also plausibly want. These are fundamentally
  ambiguous without knowing the user's intent and are a known gap (see §6).
- **Non-text payloads** (images, audio, steganography) and **encoding tricks** beyond simple delimiter
  forgery.
- **Adversarial adaptation to the guard itself.** The guard's features are public; an attacker who reads
  this repository can craft content that is on-topic, non-directive and free of delimiters. The guard
  raises the cost of the common, templated attacks in the datasets; it is not a robust adversarial defence.

## 5. What "protection" means here

The guard produces a calibrated document-level `P(injection)` that the existing policy maps to
ALLOW / WARN / BLOCK. The design goal is **not** perfect detection. It is to raise recall on *embedded*
injections while **lowering the false-positive rate on benign content that merely contains trigger words**,
compared with scoring the whole document with a single classifier. Whether it achieves this is an empirical
question answered in `CONTEXT_GUARD.md`, not asserted here.

## 6. Known weaknesses (by construction)

1. **Standalone malicious content.** The off-topic feature compares a segment to the rest of the document.
   An e-mail whose entire body is the attack has no benign "rest" to contrast against, so the guard falls
   back to the detector and lexical cues alone. Expect weaker performance on fully-malicious short e-mails
   than on injections embedded in larger benign content.
2. **On-topic injections.** An instruction that is topically consistent with the document defeats the
   off-topic feature.
3. **Paraphrase / novel phrasing.** Lexical cues are pattern-based; an attacker who avoids the known
   vocabulary reduces their contribution, leaving only the detector and off-topic signals.
4. **Localised detector inherits the base model's blind spots.** The guard runs the same ONNX detector on
   segments; where that model is weak (e.g. non-English, code), the localised score is weak too.
5. **Not validated for production.** See `CONTEXT_GUARD.md` for measured limits. The component is clearly
   labelled experimental and is not wired into the default workflow.
