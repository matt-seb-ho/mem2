"""Phase 3 driver — on-policy corpus-global concept induction.

Runs the induction stages on dsv4f via the OFFICIAL DeepSeek API, with per-stage
token accounting (tokens only, decision D0). See docs/onpolicy_concept_induction_plan.md.

Stages implemented here:
  a  pseudocode + summary  (per solve)   -> stageA_pseudocode.json
  b  free-form tags        (per solve)   -> stageB_tags.json
  (c global unification + d typed synthesis land in follow-up commits)

Usage:
    python scripts/induce_library.py --solves <run_dir>/induction/solved_seeds.json --stage a
    python scripts/induce_library.py --solves <run_dir>/induction/solved_seeds.json --stage b
    python scripts/induce_library.py --solves ... --stage a --limit 3   # cheap smoke
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED = REPO_ROOT / "third_party" / "llm_wrapper"
if str(VENDORED) not in sys.path:
    sys.path.insert(0, str(VENDORED))

from llmplus import GenerationConfig, LLMClient, RetryConfig  # noqa: E402
from llmplus.model_registry import Provider  # noqa: E402

from mem2.concepts.induction import (  # noqa: E402
    StageUsage,
    aggregate_tags,
    build_stage_a_prompt,
    build_stage_b_prompt,
    build_stage_c_critique_prompt,
    build_stage_c_map_prompt,
    build_stage_c_reduce_prompt,
    build_stage_d_prompt,
    parse_stage_a,
    parse_stage_b,
    parse_stage_c_critique,
    parse_stage_c_groups,
    parse_stage_d,
)
from mem2.concepts.memory import ConceptMemory  # noqa: E402

MODEL = "deepseek-v4-flash"
DOTENV = str(REPO_ROOT / ".env")


def make_client(batch_size: int) -> LLMClient:
    return LLMClient(
        provider=Provider.DEEPSEEK,
        cache_dir=str(REPO_ROOT / ".llm_cache"),
        default_max_concurrency=batch_size,
        retry_cfg=RetryConfig(attempts=5, wait_min=1, wait_max=120),
        dotenv_path=DOTENV,
    )


async def _batch(client: LLMClient, prompts: list[str], batch_size: int,
                 max_tokens: int, temperature: float, progress_file: Path) -> list[str]:
    gen = GenerationConfig(n=1, temperature=temperature, max_tokens=max_tokens,
                           batch_size=batch_size, ignore_cache=False)
    results = await client.async_batch_generate(
        prompts=prompts, model=MODEL, gen_cfg=gen,
        progress_file=str(progress_file), show_progress=True,
    )
    return [(r[0] if r and r[0] else "") for r in results]


def _record_usage(client: LLMClient, stage: str, before: dict, out_dir: Path) -> None:
    after = client.get_token_usage_dict()
    usage = StageUsage.from_snapshot(stage, before, after, MODEL).to_dict()
    path = out_dir / "usage" / f"stage_{stage}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(usage, indent=2))
    print(f"[usage:{stage}] {usage}")


async def stage_a(client: LLMClient, solves: dict, out_dir: Path, args) -> None:
    uids = [u for u, v in solves.items() if v.get("code")]
    prompts = [build_stage_a_prompt(solves[u]["code"]) for u in uids]
    before = client.get_token_usage_dict()
    comps = await _batch(client, prompts, args.batch_size, args.max_tokens, args.temperature,
                         out_dir / "_progress_a.json")
    result = {}
    for u, c in zip(uids, comps):
        parsed = parse_stage_a(c)
        parsed["raw"] = c
        result[u] = parsed
    ok = sum(1 for v in result.values() if v["pseudocode"])
    (out_dir / "stageA_pseudocode.json").write_text(json.dumps(result, indent=2))
    _record_usage(client, "a", before, out_dir)
    print(f"[stage A] {ok}/{len(uids)} pseudocode parsed -> {out_dir/'stageA_pseudocode.json'}")


async def stage_b(client: LLMClient, solves: dict, out_dir: Path, args) -> None:
    stage_a_path = out_dir / "stageA_pseudocode.json"
    if not stage_a_path.exists():
        raise SystemExit("Stage A output missing — run --stage a first.")
    pseudo = json.loads(stage_a_path.read_text())
    uids = [u for u in pseudo if pseudo[u].get("pseudocode")]
    prompts = [build_stage_b_prompt(pseudo[u]["pseudocode"], pseudo[u]["summary"]) for u in uids]
    before = client.get_token_usage_dict()
    comps = await _batch(client, prompts, args.batch_size, args.max_tokens, args.temperature,
                         out_dir / "_progress_b.json")
    result, n_tags = {}, 0
    for u, c in zip(uids, comps):
        tags = parse_stage_b(c)
        result[u] = {"tags": tags, "raw": c}
        n_tags += len(tags)
    (out_dir / "stageB_tags.json").write_text(json.dumps(result, indent=2))
    _record_usage(client, "b", before, out_dir)
    print(f"[stage B] {n_tags} tags across {len(uids)} puzzles "
          f"({n_tags/max(1,len(uids)):.1f}/puzzle) -> {out_dir/'stageB_tags.json'}")


def _chunks(seq: list, size: int) -> list[list]:
    return [seq[i:i + size] for i in range(0, len(seq), size)]


async def stage_c(client: LLMClient, out_dir: Path, args) -> None:
    """LLM-driven vocabulary unification: map -> reduce -> mechanical freq -> critique."""
    stage_b = json.loads((out_dir / "stageB_tags.json").read_text())
    records = aggregate_tags(stage_b)
    tag_to_uids = {r["tag"]: r["uids"] for r in records}
    all_tags = {r["tag"] for r in records}
    before = client.get_token_usage_dict()

    # --- MAP: group each shard of tags into chunk-canonical concepts ---
    chunks = _chunks(records, args.chunk_size)
    map_prompts = [build_stage_c_map_prompt(c) for c in chunks]
    map_comps = await _batch(client, map_prompts, args.batch_size, 8192, args.temperature,
                             out_dir / "_progress_c_map.json")
    chunk_groups: list[dict] = []
    claimed: set[str] = set()
    for comp in map_comps:
        for g in parse_stage_c_groups(comp):
            members = [m for m in g["members"] if m in all_tags]
            if not members:
                continue
            g["members"] = members
            chunk_groups.append(g)
            claimed.update(members)
    # tags the model dropped -> own singleton group
    for t in all_tags - claimed:
        r = next(x for x in records if x["tag"] == t)
        chunk_groups.append({"canonical": t, "kind": r["kind"], "gloss":
                             (r["descriptions"][0] if r["descriptions"] else ""), "members": [t]})
    print(f"[stage C] map: {len(records)} tags -> {len(chunk_groups)} chunk-canonical groups")

    # --- REDUCE: iterative agglomeration in bounded chunks ---
    # Each node carries the original tags it covers (for freq + Stage-D evidence).
    # de-dup chunk-canonical names (map may repeat a name across chunks)
    by_name: dict[str, dict] = {}
    for g in chunk_groups:
        cur = by_name.setdefault(g["canonical"], {"canonical": g["canonical"], "kind": g["kind"],
                                                  "gloss": g.get("gloss", ""), "tags": []})
        cur["tags"] = list(dict.fromkeys(cur["tags"] + g["members"]))
        cur["kind"] = g["kind"]
    nodes = list(by_name.values())  # each: {canonical, kind, gloss, tags}

    rnd = 0
    while rnd < args.reduce_rounds:
        rnd += 1
        # alternate ordering so different synonyms co-occur across rounds
        nodes.sort(key=(lambda n: n["canonical"]) if rnd % 2 else (lambda n: -len(n["tags"])))
        rchunks = _chunks(nodes, args.reduce_chunk)
        rprompts = [build_stage_c_reduce_prompt(c) for c in rchunks]
        rcomps = await _batch(client, rprompts, args.batch_size, 8192, args.temperature,
                              out_dir / f"_progress_c_reduce{rnd}.json")
        new_nodes: list[dict] = []
        for chunk, comp in zip(rchunks, rcomps):
            cmap = {n["canonical"]: n for n in chunk}
            claimed_local: set[str] = set()
            for g in parse_stage_c_groups(comp):
                mem_names = [m for m in g["members"] if m in cmap and m not in claimed_local]
                if not mem_names:
                    continue
                claimed_local.update(mem_names)
                tags = list(dict.fromkeys(t for m in mem_names for t in cmap[m]["tags"]))
                new_nodes.append({"canonical": g["canonical"], "kind": g["kind"],
                                  "gloss": g["gloss"], "tags": tags})
            for nm, node in cmap.items():  # carry unclaimed through
                if nm not in claimed_local:
                    new_nodes.append(node)
        # merge identical names that emerged
        merged: dict[str, dict] = {}
        for n in new_nodes:
            cur = merged.setdefault(n["canonical"], {**n, "tags": []})
            cur["tags"] = list(dict.fromkeys(cur["tags"] + n["tags"]))
        new_nodes = list(merged.values())
        print(f"[stage C] reduce round {rnd}: {len(nodes)} -> {len(new_nodes)}")
        if len(new_nodes) >= len(nodes):
            nodes = new_nodes
            break
        nodes = new_nodes

    vocab: list[dict] = []
    for n in nodes:
        uids = sorted({u for t in n["tags"] for u in tag_to_uids.get(t, [])})
        vocab.append({"canonical": n["canonical"], "kind": n["kind"], "gloss": n["gloss"],
                      "member_tags": sorted(set(n["tags"])), "member_uids": uids,
                      "frequency": len(uids)})
    print(f"[stage C] reduce done: {len(vocab)} canonical concepts")

    vocab.sort(key=lambda c: -c["frequency"])
    primary = [c for c in vocab if c["frequency"] >= args.min_freq]
    appendix = [c for c in vocab if c["frequency"] < args.min_freq]

    # --- CRITIQUE (bounded loop on PRIMARY pool only): forceful merge/rename ---
    # The primary set (~50 concepts) is what becomes the library and is small
    # enough for the model to dedup well in one focused call. Iterate to stability.
    if not args.no_critique:
        def _apply(vmap, edits):
            for grp in edits["merges"]:
                keep = grp[0]
                if keep not in vmap:
                    continue
                for other in grp[1:]:
                    if other in vmap and other != keep:
                        vmap[keep]["member_tags"] = sorted(set(vmap[keep]["member_tags"]) | set(vmap[other]["member_tags"]))
                        vmap[keep]["member_uids"] = sorted(set(vmap[keep]["member_uids"]) | set(vmap[other]["member_uids"]))
                        vmap[keep]["frequency"] = len(vmap[keep]["member_uids"])
                        del vmap[other]
            for old, new in edits["renames"].items():
                if old in vmap and new not in vmap:
                    vmap[old]["canonical"] = new
                    vmap[new] = vmap.pop(old)
            return vmap

        for crit_round in range(args.critique_rounds):
            primary.sort(key=lambda c: -c["frequency"])
            crit = await _batch(client, [build_stage_c_critique_prompt(primary)],
                                args.batch_size, 4096, args.temperature,
                                out_dir / f"_progress_c_crit{crit_round}.json")
            edits = parse_stage_c_critique(crit[0])
            n_before = len(primary)
            primary = list(_apply({c["canonical"]: c for c in primary}, edits).values())
            print(f"[stage C] critique round {crit_round+1}: {len(edits['merges'])} merges, "
                  f"{len(edits['renames'])} renames -> {len(primary)} primary concepts")
            if len(primary) == n_before and not edits["renames"]:
                break
    (out_dir / "stageC_vocab.json").write_text(json.dumps(
        {"primary": primary, "appendix": appendix}, indent=2))
    _record_usage(client, "c", before, out_dir)
    print(f"[stage C] DONE: {len(primary)} primary (freq>={args.min_freq}), "
          f"{len(appendix)} appendix -> {out_dir/'stageC_vocab.json'}")
    print("[stage C] top concepts:", [(c["canonical"], c["frequency"]) for c in primary[:15]])


def _build_evidence(uids: list[str], pseudo: dict, stage_b: dict, member_tags: set[str],
                    max_puzzles: int, max_chars: int) -> str:
    blocks = []
    for uid in uids[:max_puzzles]:
        descs = [t["description"] for t in stage_b.get(uid, {}).get("tags", [])
                 if t["tag"] in member_tags and t.get("description")]
        ps = (pseudo.get(uid, {}).get("pseudocode") or "")[:max_chars]
        blocks.append(f"## puzzle {uid}\nrole: {'; '.join(descs[:2])}\npseudocode:\n{ps}")
    return "\n\n".join(blocks)


async def stage_d(client: LLMClient, out_dir: Path, args) -> None:
    vocab = json.loads((out_dir / "stageC_vocab.json").read_text())
    pseudo = json.loads((out_dir / "stageA_pseudocode.json").read_text())
    stage_b = json.loads((out_dir / "stageB_tags.json").read_text())
    pool = vocab["primary"] + (vocab["appendix"] if args.include_appendix else [])
    before = client.get_token_usage_dict()

    prompts = [build_stage_d_prompt(
        c["canonical"], c["kind"], c.get("gloss", ""),
        _build_evidence(c["member_uids"], pseudo, stage_b, set(c["member_tags"]),
                        args.max_evidence, args.evidence_chars)) for c in pool]
    comps = await _batch(client, prompts, args.batch_size, args.max_tokens, args.temperature,
                         out_dir / "_progress_d.json")

    mem = ConceptMemory()
    n_ok = 0
    for c, comp in zip(pool, comps):
        ann = parse_stage_d(comp, c["canonical"], c["kind"])
        if not ann:
            continue
        uids = c["member_uids"] or ["_induced"]
        mem.write_concept(uids[0], ann)
        name = ann["concept"]
        if name in mem.concepts:
            for u in uids:
                if u not in mem.concepts[name].used_in:
                    mem.concepts[name].used_in.append(u)
            n_ok += 1
    out_path = out_dir.parent.parent.parent.parent / "data" / "arc_agi" / "concept_memory" / args.library_name \
        if False else (Path(args.library_out) if args.library_out
                       else REPO_ROOT / "data" / "arc_agi" / "concept_memory" / "induced_concepts_v1.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mem.save_to_file(out_path)
    _record_usage(client, "d", before, out_dir)
    # also drop a copy next to the induction artifacts
    (out_dir / "induced_concepts_v1.json").write_text(out_path.read_text())
    print(f"[stage D] synthesized {n_ok}/{len(pool)} concepts "
          f"(structures={sum(1 for x in mem.concepts.values() if x.kind=='structure')}, "
          f"routines={sum(1 for x in mem.concepts.values() if x.kind=='routine')}) "
          f"custom_types={len(mem.custom_types)} -> {out_path}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--solves", default=None, help="path to solved_seeds.json (stages a/b)")
    ap.add_argument("--stage", required=True, choices=["a", "b", "c", "d"])
    ap.add_argument("--out-dir", default=None, help="induction dir; defaults to solves' parent")
    ap.add_argument("--train-only", action="store_true", default=True,
                    help="restrict to train-correct solves (D1 primary pool); on by default")
    ap.add_argument("--limit", type=int, default=0, help="cap #puzzles (smoke test)")
    ap.add_argument("--batch-size", type=int, default=48)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.3)
    # Stage C
    ap.add_argument("--chunk-size", type=int, default=70, help="tags per map shard (stage c)")
    ap.add_argument("--reduce-chunk", type=int, default=60, help="concepts per reduce-round chunk")
    ap.add_argument("--reduce-rounds", type=int, default=4, help="max agglomeration rounds")
    ap.add_argument("--min-freq", type=int, default=2, help="primary-pool frequency cutoff (D2)")
    ap.add_argument("--no-critique", action="store_true", help="skip stage-c critique loop")
    ap.add_argument("--critique-rounds", type=int, default=3, help="max primary-pool merge rounds")
    # Stage D
    ap.add_argument("--include-appendix", action="store_true",
                    help="also synthesize freq<min_freq concepts (stage d)")
    ap.add_argument("--max-evidence", type=int, default=6, help="member puzzles per concept (stage d)")
    ap.add_argument("--evidence-chars", type=int, default=700, help="pseudocode chars per puzzle (stage d)")
    ap.add_argument("--library-out", default=None,
                    help="output path for induced library (stage d)")
    args = ap.parse_args()

    if args.out_dir:
        out_dir = Path(args.out_dir)
    elif args.solves:
        out_dir = Path(args.solves).parent
    else:
        raise SystemExit("Provide --out-dir (stages c/d) or --solves (stages a/b).")
    out_dir.mkdir(parents=True, exist_ok=True)

    client = make_client(args.batch_size)
    if args.stage in ("a", "b"):
        if not args.solves:
            raise SystemExit("--solves required for stages a/b")
        solves = json.loads(Path(args.solves).read_text())
        if args.train_only:
            solves = {u: v for u, v in solves.items() if v.get("train_ok")}
        if args.limit:
            solves = dict(list(solves.items())[: args.limit])
        print(f"[induce] stage={args.stage} pool={len(solves)} out={out_dir}")
        if args.stage == "a":
            await stage_a(client, solves, out_dir, args)
        else:
            await stage_b(client, solves, out_dir, args)
    elif args.stage == "c":
        print(f"[induce] stage=c out={out_dir}")
        await stage_c(client, out_dir, args)
    elif args.stage == "d":
        print(f"[induce] stage=d out={out_dir}")
        await stage_d(client, out_dir, args)


if __name__ == "__main__":
    asyncio.run(main())
