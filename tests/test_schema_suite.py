"""M5-⑩：JSON Schema Test Suite（draft-07）子集回归（issue #24）。

对 tests/json_schema_test_suite/subset/ 的每个用例断言：
编译成功 + 数据字节走自动机结果与套件 valid 标志一致。
纳入/排除依据见同目录 SUBSET.md / SUBSET_EXCLUDED.md。
"""

import json
import pathlib
import sys

import pytest
import zealot_engine as ze

SUBSET = pathlib.Path(__file__).parent / "json_schema_test_suite" / "subset"


@pytest.fixture(scope="module")
def byte_vocab_grammar():
    """256 单字节字符词表：allowed_tokens/advance 即字节级走边（与生成器一致）。"""
    vocab = {chr(i): i for i in range(256)}
    tok_path = SUBSET / "_byte_vocab.json"
    tok_path.write_text(json.dumps({"model": {"type": "BPE", "vocab": vocab}}))
    return str(tok_path)


def _walk(gs, data):
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    st = 0
    for b in text.encode("utf-8"):
        if b not in gs.allowed_tokens(st):
            return False
        st = gs.advance(st, b)
    return gs.is_valid_final(st)


def _cases():
    for path in sorted(SUBSET.glob("*.json")):
        if path.name.startswith("_"):
            continue
        for case in json.loads(path.read_text()):
            for t in case["tests"]:
                yield path.name, case["description"], t["description"], case["schema"], t["data"], t["valid"]


ALL_CASES = list(_cases())


@pytest.mark.parametrize(
    "file,case,tcase,schema,data,valid",
    ALL_CASES,
    ids=[f"{f}:{c} / {t}" for f, c, t, _, _, _ in ALL_CASES],
)
def test_suite_case(file, case, tcase, schema, data, valid, byte_vocab_grammar):
    compiler = ze.JsonSchemaCompiler(64, 10000, 5000)
    gs = compiler.compile(json.dumps(schema))
    gs.load_tokenizer(byte_vocab_grammar)
    got = _walk(gs, data)
    assert got == valid, f"{file} [{case}] {tcase}: 期望 {valid} 实际 {got}"


def test_subset_manifest_consistent():
    """subset 文件均被 runner 收集（防止新文件被静默漏跑）。"""
    for path in sorted(SUBSET.glob("*.json")):
        if path.name.startswith("_"):
            continue
        if not json.loads(path.read_text()):
            continue  # 整文件排除（见 SUBSET_EXCLUDED.md），允许空子集文件
        assert any(f == path.name for f, _, _, _, _, _ in ALL_CASES), f"{path.name} 无用例"
    assert len(ALL_CASES) >= 140, f"子集异常缩水: {len(ALL_CASES)}"
