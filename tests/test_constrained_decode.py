"""M4 §9: JsonSchemaCompiler / ConstrainedGrammar PyO3 接口 + 对抗性 schema 拒绝。

§12-6 覆盖：构造 >10000 states 的 schema → SchemaTooComplex（RuntimeError）。
运行：`pytest tests/`（前置：`.venv/bin/maturin develop` 安装扩展）。
"""

import json

import pytest

import zealot_engine as ze

SIMPLE = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "active": {"type": "boolean"},
    },
    "required": ["name"],
}


def make_compiler(max_states=10_000, max_compile_ms=1000, cache_size=128):
    return ze.JsonSchemaCompiler(cache_size, max_states, max_compile_ms)


def test_compile_simple_object():
    c = make_compiler()
    g = c.compile(json.dumps(SIMPLE))
    assert isinstance(g, ze.ConstrainedGrammar)
    # 1 (object) + 3 properties
    assert g.state_count == 4
    tokens = g.allowed_tokens(0)
    assert isinstance(tokens, list) and all(isinstance(t, int) for t in tokens)
    assert g.is_valid_final(0) in (True, False)


def test_advance_cycles_through_states():
    c = make_compiler()
    g = c.compile(json.dumps(SIMPLE))
    s = g.advance(0, 123)
    assert isinstance(s, int) and 0 <= s < g.state_count


def test_compile_cache_hit_and_bypass():
    c = make_compiler()
    g1 = c.compile(json.dumps(SIMPLE))
    g2 = c.compile(json.dumps(SIMPLE))  # 命中缓存
    assert g2.state_count == g1.state_count
    g3 = c.compile(json.dumps(SIMPLE), bypass_cache=True)  # 绕过缓存
    assert g3.state_count == g1.state_count


def test_compile_enum_schema():
    c = make_compiler()
    g = c.compile(json.dumps({"enum": ["red", "green", "blue"]}))
    assert g.state_count >= 1


def test_compile_union_schema():
    c = make_compiler()
    g = c.compile(json.dumps({"anyOf": [{"type": "string"}, {"type": "integer"}]}))
    assert g.state_count == 2


def test_invalid_json_rejected():
    c = make_compiler()
    with pytest.raises(RuntimeError, match="Invalid JSON schema"):
        c.compile("this is not json")


def test_invalid_state_out_of_range():
    c = make_compiler()
    g = c.compile(json.dumps(SIMPLE))
    with pytest.raises(RuntimeError, match="out of range"):
        g.allowed_tokens(99)
    with pytest.raises(RuntimeError, match="out of range"):
        g.advance(99, 1)
    with pytest.raises(RuntimeError, match="out of range"):
        g.is_valid_final(99)


def test_excessive_nesting_rejected():
    """嵌套深度超过 MAX_NESTING_DEPTH=32 必须拒绝。"""
    c = make_compiler()
    schema = {"type": "string"}
    for _ in range(40):
        schema = {"type": "array", "items": schema}
    with pytest.raises(RuntimeError, match="nesting depth"):
        c.compile(json.dumps(schema))


def test_adversarial_schema_rejected_over_max_states():
    """§12-6：>10000 states 的对抗性 schema → SchemaTooComplex。"""
    c = make_compiler(max_states=10_000)
    # object 每属性计 1 state：10001 属性 → 10002 states > 10000
    props = {f"p{i}": {"type": "string"} for i in range(10_001)}
    schema = {"type": "object", "properties": props}
    with pytest.raises(RuntimeError, match="Schema too complex"):
        c.compile(json.dumps(schema))


def test_boundary_schema_accepted_at_max_states():
    """恰好等于 max_states 的 schema 必须被接受（边界语义）。"""
    c = make_compiler(max_states=10_000)
    props = {f"p{i}": {"type": "string"} for i in range(9_999)}  # 10000 states
    schema = {"type": "object", "properties": props}
    g = c.compile(json.dumps(schema))
    assert g.state_count == 10_000


def test_compile_timeout_guard():
    """max_compile_ms=0：耗时编译（边界内但量大）必须触发 SchemaCompileTimeout。
    注意：超限 schema 先触发 SchemaTooComplex（编译中途报错），
    到不了超时检查——因此用边界内的 9999 属性 schema。"""
    c = make_compiler(max_states=10_000, max_compile_ms=0)
    props = {f"p{i}": {"type": "string"} for i in range(9_999)}
    schema = {"type": "object", "properties": props}
    with pytest.raises(RuntimeError, match="timed out"):
        c.compile(json.dumps(schema))
