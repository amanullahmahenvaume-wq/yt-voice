"""Runs once per matrix entry (one shard). Synthesizes this shard's slice of
text against the shared reference voice, with its own retry/backoff -- a
transient hiccup on one shard shouldn't force recomputing the other 39.
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


import difflib
import re

# সংখ্যা ও সংখ্যা-শব্দ দুই দিক থেকেই বাদ: TTS-ইনপুটে "$700", "2,000‑year‑old", "2nd‑century" আছে (আসল ডেটায় দেখা),
# আর ASR সেটা লেখে "seven hundred dollars", "two thousand year old", "second century" — এতে ভুয়া mismatch হতো।
_NUMWORDS = set("""zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen
seventeen eighteen nineteen twenty thirty forty fifty sixty seventy eighty ninety hundred thousand million billion trillion
point dollar dollars cent cents percent first second third fourth fifth sixth seventh eighth ninth tenth""".split())


def _tok(s: str) -> list[str]:
    return [t for t in re.findall(r"[a-z0-9']+", s.lower()) if not re.search(r"\d", t) and t not in _NUMWORDS]


def shard_check(expected: str, heard: str, min_ratio: float = 0.88, max_drop_run: int = 2,
                max_insert_run: int = 3) -> tuple[bool, float, int, int]:
    """প্রতিটি TTS shard-এর পরে চালান। (ok, ratio, লম্বা 'বাদ-পড়া' সারি, লম্বা 'বাড়তি/পুনরাবৃত্ত' সারি)।
    ~২৫ শব্দের shard-এ ০.৮৮ ≈ সর্বোচ্চ ৩টি ভুল শব্দ সহনীয়; TTS-এর আসল ব্যর্থতা (শব্দ গিলে ফেলা / বাবল / তোতলানো)
    ধরতে 'টানা বাদ' ও 'টানা বাড়তি' সারিও দেখা হয়। ASR নিজেও নাম ও সংখ্যায় ভুল করে, তাই ০.৯৩+ রাখলে ভালো shard-ও বাতিল হবে।"""
    a, b = _tok(expected), _tok(heard)
    if not a:
        return True, 1.0, 0, 0
    sm = difflib.SequenceMatcher(None, a, b)
    ops = sm.get_opcodes()
    longest_drop = max((i2 - i1 for tag, i1, i2, _j1, _j2 in ops if tag == "delete"), default=0)
    longest_insert = max((j2 - j1 for tag, _i1, _i2, j1, j2 in ops if tag == "insert"), default=0)
    ratio = sm.ratio()
    ok = ratio >= min_ratio and longest_drop <= max_drop_run and longest_insert <= max_insert_run
    return ok, ratio, longest_drop, longest_insert

_WHISPER = None


def transcribe(wav_path: Path) -> str:
    """faster-whisper, CPU int8; the model is loaded once per process (add `faster-whisper` to requirements.txt)."""
    global _WHISPER
    if _WHISPER is None:
        from faster_whisper import WhisperModel
        _WHISPER = WhisperModel("base.en", device="cpu", compute_type="int8")
    segments, _info = _WHISPER.transcribe(str(wav_path), language="en", beam_size=1)
    return " ".join(seg.text.strip() for seg in segments)


def asr_verdict(expected: str, wav_path: Path) -> dict | None:
    """None = the checker itself could not run (never fail a shard because of the checker)."""
    try:
        ok, ratio, drop, ins = shard_check(expected, transcribe(wav_path))
    except Exception as exc:  # noqa: BLE001
        print(f"::warning::ASR check skipped: {exc}")
        return None
    return {"ok": ok, "ratio": ratio, "drop": drop, "ins": ins}


def synthesize(ref_audio: Path, ref_text: str, gen_text: str, out_dir: Path, out_name: str) -> None:
    """Shells out to the documented `f5-tts_infer-cli` entry point rather
    than importing f5_tts's internal Python classes directly -- the
    internal API (the F5TTS class's constructor signature, module layout)
    has changed across versions, while these CLI flags
    (-r/-s/-t/-f/-o/-w, confirmed against the upstream infer_cli.py source)
    have stayed the stable, documented surface.

    NOTE: re-verify with `f5-tts_infer-cli --help` after any version bump
    pinned in requirements.txt -- this was written and confirmed against
    the plain flag layout in SWivid/F5-TTS's infer_cli.py, but this
    workflow can't be end-to-end tested without a GPU/model-weights
    environment, so treat the FIRST canary run as the real test of this
    exact command line.
    """
    gen_text_file = out_dir / f"{out_name}.gen.txt"
    gen_text_file.write_text(gen_text, encoding="utf-8")
    cmd = [
        "f5-tts_infer-cli",
        "-r", str(ref_audio),
        "-s", ref_text,
        "-f", str(gen_text_file),
        "-o", str(out_dir),
        "-w", f"{out_name}.wav",
    ]
    # 25 minutes -- this is CPU inference on a standard GitHub-hosted
    # runner with no GPU, plus (on a cache miss) downloading the ~1.5GB
    # model checkpoint from HuggingFace first. The actions/cache step
    # added to generate.yml's shard job should make every run AFTER the
    # first a lot faster, but this still needs to be generous enough for
    # a cold cache.
    subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1500)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--profile-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    shard = next(s for s in manifest["shards"] if s["index"] == args.shard_index)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not shard["text"].strip():
        # split_balanced() can hand back an empty pad chunk if the script
        # had fewer sentences than requested shards -- nothing to
        # synthesize; write a marker so collect_and_stitch.py can skip it
        # without treating it as a failure.
        (out_dir / "EMPTY").write_text("", encoding="utf-8")
        return

    ref_audio = Path(args.profile_dir) / "reference.wav"
    ref_meta = json.loads((Path(args.profile_dir) / "reference.json").read_text(encoding="utf-8"))
    ref_text = ref_meta.get("transcript", "")
    out_name = f"shard_{args.shard_index:03d}"

    last_exc: Exception | None = None
    for attempt in range(1, args.max_attempts + 1):
        try:
            synthesize(ref_audio, ref_text, shard["text"], out_dir, out_name)
            if not (out_dir / f"{out_name}.wav").exists():
                raise RuntimeError("f5-tts_infer-cli exited 0 but produced no output file")
            verdict = asr_verdict(shard["text"], out_dir / f"{out_name}.wav")
            if verdict is not None and not verdict["ok"]:
                msg = (f"ASR check failed (match {verdict['ratio']:.2f}, dropped-run {verdict['drop']}, extra-run {verdict['ins']})")
                if attempt < args.max_attempts:
                    raise RuntimeError(msg)          # handled by the generic retry branch below
                print(f"::warning::Shard {args.shard_index}: {msg} on the last attempt; keeping it (final voice_qa still applies)")
            print(f"Shard {args.shard_index}: done on attempt {attempt}.")
            return
        except subprocess.TimeoutExpired as exc:
            # Same visibility gap as CalledProcessError had: capture_output
            # buffers everything, so a timeout was showing "timed out
            # after Ns" with nothing else -- no way to tell whether it was
            # still downloading the model, still loading it, or actually
            # mid-synthesis when it got killed. exc.stdout/exc.stderr hold
            # whatever was captured up to the kill, which is exactly what
            # we need to tell those apart.
            last_exc = exc
            print(f"Shard {args.shard_index} attempt {attempt}/{args.max_attempts}: timed out after {exc.timeout}s")
            print(f"--- stdout so far ---\n{exc.stdout}")
            print(f"--- stderr so far ---\n{exc.stderr}")
            if attempt < args.max_attempts:
                time.sleep(min(60, 5 * 2 ** attempt))
        except subprocess.CalledProcessError as exc:
            # This is the one that was going silent before: check=True +
            # capture_output=True means the actual error text was captured
            # but never printed, so every failure just showed "exit status
            # 1" with no way to tell why. Surfacing stdout/stderr here is
            # the whole reason this except branch is split out from the
            # generic one below.
            last_exc = exc
            print(f"Shard {args.shard_index} attempt {attempt}/{args.max_attempts}: "
                  f"f5-tts_infer-cli exited {exc.returncode}")
            print(f"--- stdout ---\n{exc.stdout}")
            print(f"--- stderr ---\n{exc.stderr}")
            if attempt < args.max_attempts:
                time.sleep(min(60, 5 * 2 ** attempt))
        except Exception as exc:  # noqa: BLE001 -- any other failure should retry too, not just CalledProcessError
            last_exc = exc
            print(f"Shard {args.shard_index} attempt {attempt}/{args.max_attempts} failed: {exc}")
            if attempt < args.max_attempts:
                time.sleep(min(60, 5 * 2 ** attempt))
    print(f"::error::Shard {args.shard_index} failed after {args.max_attempts} attempts: {last_exc}")
    sys.exit(1)


if __name__ == "__main__":
    main()
