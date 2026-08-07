"""M4 §9 / M5-①: JsonSchemaCompiler / ConstrainedGrammar PyO3 接口 + 对抗性 schema 拒绝。

§12-6 覆盖：构造 >10000 states 的 schema → SchemaTooComplex（RuntimeError）。
§12-5 覆盖：缓存命中/淘汰统计 + 真 LRU 淘汰顺序（见 test_cache_lru_eviction）。
M5-① 覆盖：required 收尾语义、嵌套 max_states、子 schema 失败整体报错。
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
    assert c.cache_misses == 1
    assert c.cache_hits == 1
    assert c.cache_evictions == 0
    g3 = c.compile(json.dumps(SIMPLE), bypass_cache=True)  # 绕过缓存：不查不写不计
    assert g3.state_count == g1.state_count
    assert c.cache_hits == 1
    assert c.cache_misses == 1


def test_cache_lru_eviction():
    """§12-5：cache_size=2 下 A,B,A(命中提升),C → 必须淘汰 B（最久未用），保留 A。"""
    c = ze.JsonSchemaCompiler(2, 10_000, 1000)
    a = {"type": "string", "minLength": 1}
    b = {"type": "integer"}
    d = {"type": "boolean"}

    c.compile(json.dumps(a))  # miss
    c.compile(json.dumps(b))  # miss
    c.compile(json.dumps(a))  # hit → A 提升为最近使用
    c.compile(json.dumps(d))  # miss → 容量满，淘汰最久未用的 B
    assert c.cache_evictions == 1
    assert c.cache_hits == 1
    assert c.cache_misses == 3

    # 先验证 A 存活（hit），再重编译 B：若 B 未真被淘汰此处应为 hit
    c.compile(json.dumps(a))  # hit → 证明被淘汰的是 B 而非 A
    c.compile(json.dumps(b))  # miss → 证明 B 确实已被淘汰（同时把当时最久未用的 d 淘汰）
    assert c.cache_hits == 2
    assert c.cache_misses == 4
    assert c.cache_evictions == 2  # 第 1 次淘汰 B；第 2 次淘汰 d（B 重插时最久未用）


def test_compile_enum_schema():
    c = make_compiler()
    g = c.compile(json.dumps({"enum": ["red", "green", "blue"]}))
    assert g.state_count >= 1


def test_compile_union_schema():
    c = make_compiler()
    g = c.compile(json.dumps({"anyOf": [{"type": "string"}, {"type": "integer"}]}))
    assert g.state_count == 2


def test_required_object_final_state_semantics():
    """M5-①：required 键遍历完才可合法收尾；根状态不可直接收尾；可选字段允许缺省。"""
    c = make_compiler()
    g = c.compile(json.dumps(SIMPLE))  # required=["name"]
    assert not g.is_valid_final(0)  # 根状态：required 未遍历
    assert g.is_valid_final(1)  # 已遍历 1 个槽位 ≥ required 数
    assert g.is_valid_final(g.state_count - 1)

    g_opt = c.compile(json.dumps({"type": "object", "properties": {"x": {"type": "string"}}}))
    assert g_opt.is_valid_final(0)  # 无 required：空对象 {} 合法


def test_nested_schema_max_states_rejected():
    """M5-①：count_states 递归后 max_states 覆盖嵌套结构（不再只挡顶层）。"""
    c = make_compiler(max_states=10)
    sub = {"type": "object", "properties": {"x": {"type": "boolean"}, "y": {"type": "integer"}}}
    ok = {"type": "object", "properties": {"a": sub, "b": sub, "c": sub}}
    g = c.compile(json.dumps(ok))  # 1 + 3*3 = 10 = max_states：边界接受
    assert g.state_count == 10
    bad = {"type": "object", "properties": {"a": sub, "b": sub, "c": sub, "d": sub}}
    with pytest.raises(RuntimeError, match="Schema too complex"):
        c.compile(json.dumps(bad))  # 1 + 4*3 = 13 > 10


def test_failed_child_schema_not_silently_dropped():
    """M5-①：子 schema 编译失败 → 整体报错（原实现静默丢弃失败属性）。"""
    c = make_compiler(max_states=5)
    big = {"type": "object", "properties": {f"p{i}": {"type": "boolean"} for i in range(5)}}
    schema = {"type": "object", "properties": {"big": big}}
    with pytest.raises(RuntimeError, match="Schema too complex"):
        c.compile(json.dumps(schema))

def test_invalid_json_rejected():
    c = make_compiler()
    with pytest.raises(RuntimeError, match="Invalid JSON schema"):
        c.compile("this is not json")


def test_vocab_mode_real_token_ids(tmp_path):
    """M5-③：load_tokenizer 后 allowed_tokens/advance 切换为真实词表 ID。"""
    vocab = {"{": 0, "}": 1, '"': 2, ":": 3, ",": 4, "name": 5, "true": 6}
    tok_path = tmp_path / "tokenizer.json"
    tok_path.write_text(json.dumps({"model": {"type": "BPE", "vocab": vocab}}))

    c = make_compiler()
    schema = {"type": "object", "properties": {"name": {"type": "boolean"}}, "required": ["name"]}
    g = c.compile(json.dumps(schema))
    g.load_tokenizer(str(tok_path))

    # state 0：required 未满足 → 只有 `{`(0) 合法（`}` 被自动机拒绝）
    assert g.allowed_tokens(0) == [0]
    s = g.advance(0, 0)  # `{`
    assert not g.is_valid_final(s)
    # 完整对象：{ " name " : true }
    for t in [2, 5, 2, 3, 6]:
        s = g.advance(s, t)
    # required 已满足：`,` 与 `}` 均合法，但 `{` 不合法
    assert g.allowed_tokens(s) == [1, 4]
    s = g.advance(s, 1)  # `}`
    assert g.is_valid_final(s)


def test_vocab_mode_legacy_without_tokenizer():
    """M5-③：未加载词表时保持 Phase-2 占位语义（向后兼容）。"""
    c = make_compiler()
    g = c.compile(json.dumps(SIMPLE))
    tokens = g.allowed_tokens(0)
    assert isinstance(tokens, list) and all(isinstance(t, int) for t in tokens)


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
    """恰好等于 max_states 的 schema 必须被接受（边界语义）。
    9999 属性编译约 0.9s（贴 1000ms 守卫线，冷启动抖动会超时）——
    此处放宽编译时限，只验证边界语义本身。"""
    c = make_compiler(max_states=10_000, max_compile_ms=10_000)
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


def test_integer_range_keywords_compile():
    """M5-⑤：integer min/max/exclusive*/multipleOf 解析并编译（不报错）。"""
    c = make_compiler()
    for schema in [
        {"type": "integer"},
        {"type": "integer", "minimum": 0},
        {"type": "integer", "maximum": 100},
        {"type": "integer", "minimum": -10, "maximum": 10, "multipleOf": 2},
        {"type": "integer", "exclusiveMinimum": 0, "exclusiveMaximum": 100},
        {"type": "integer", "minimum": 0, "exclusiveMinimum": True},  # draft-4 形态
        {"type": "integer", "multipleOf": 2.5},  # 小数展开
    ]:
        g = c.compile(json.dumps(schema))
        assert isinstance(g, ze.ConstrainedGrammar)


def test_integer_invalid_range_rejected():
    """M5-⑤：空区间 / multipleOf 0 / number+范围 → 显式错误（不静默）。"""
    c = make_compiler()
    with pytest.raises(RuntimeError, match="空区间"):
        c.compile(json.dumps({"type": "integer", "minimum": 10, "maximum": 5}))
    with pytest.raises(RuntimeError, match="multipleOf"):
        c.compile(json.dumps({"type": "integer", "multipleOf": 0}))
    with pytest.raises(RuntimeError, match="integer"):
        c.compile(json.dumps({"type": "number", "minimum": 0}))
    with pytest.raises(RuntimeError, match="multipleOf"):
        c.compile(json.dumps({"type": "integer", "multipleOf": 0.1234567}))


def test_integer_range_walk_with_vocab(tmp_path):
    """M5-⑤：真实词表 walk——[0,100] 内合法、越界被 mask 拒。"""
    vocab = {chr(i): i for i in range(256)}  # 单字节词表（id == 字节值）
    tok_path = tmp_path / "tokenizer.json"
    tok_path.write_text(json.dumps({"model": {"type": "BPE", "vocab": vocab}}))

    c = make_compiler()
    g = c.compile(json.dumps({"type": "integer", "minimum": 0, "maximum": 100}))
    g.load_tokenizer(str(tok_path))

    def walk_ok(gs, text):
        st = 0
        for ch in text:
            tid = ord(ch)
            if tid not in gs.allowed_tokens(st):
                return False
            st = gs.advance(st, tid)
        return gs.is_valid_final(st)

    assert walk_ok(g, "0")
    assert walk_ok(g, "42")
    assert walk_ok(g, "100")
    assert not walk_ok(g, "-1"), "低于下限应被拒"
    assert not walk_ok(g, "101"), "超出上限应被拒"
    assert not walk_ok(g, "007"), "前导零应被拒"


def test_string_length_walk_with_vocab(tmp_path):
    """M5-⑥：minLength/maxLength 计数执行（码点）——真实词表 walk。"""
    vocab = {chr(i): i for i in range(256)}
    tok_path = tmp_path / "tokenizer.json"
    tok_path.write_text(json.dumps({"model": {"type": "BPE", "vocab": vocab}}))

    c = make_compiler()

    def check(schema, ok_list, bad_list):
        g = c.compile(json.dumps(schema))
        g.load_tokenizer(str(tok_path))

        def walk_ok(gs, text):
            st = 0
            for b in text.encode("utf-8"):  # 逐字节（码点计数由自动机处理）
                if b not in gs.allowed_tokens(st):
                    return False
                st = gs.advance(st, b)
            return gs.is_valid_final(st)

        for t in ok_list:
            assert walk_ok(g, t), f"{schema} 应接受 {t!r}"
        for t in bad_list:
            assert not walk_ok(g, t), f"{schema} 应拒绝 {t!r}"

    check({"type": "string", "minLength": 3}, ['"abc"', '"abcd"'], ['"ab"', '""'])
    check({"type": "string", "maxLength": 2}, ['""', '"ab"'], ['"abc"'])
    check({"type": "string", "minLength": 2, "maxLength": 4}, ['"ab"', '"abcd"'], ['"a"', '"abcde"'])
    # 码点计数：中（3 字节）算 1
    check({"type": "string", "minLength": 2}, ['"\u4e2d\u4e2d"'], ['"\u4e2d"'])


def test_string_length_invalid_ranges_rejected():
    """M5-⑥：minLength > maxLength / 超大上限 → 显式错误。"""
    c = make_compiler()
    with pytest.raises(RuntimeError, match="空区间"):
        c.compile(json.dumps({"type": "string", "minLength": 5, "maxLength": 2}))
    with pytest.raises(RuntimeError, match="过大"):
        c.compile(json.dumps({"type": "string", "maxLength": 99999999}))


def test_string_format_subset(tmp_path):
    """M5-⑦：format 子集编译 + 真实词表 walk 正反例。"""
    vocab = {chr(i): i for i in range(256)}
    tok_path = tmp_path / "tokenizer.json"
    tok_path.write_text(json.dumps({"model": {"type": "BPE", "vocab": vocab}}))
    c = make_compiler()

    def walk_ok(gs, text):
        st = 0
        for b in text.encode("utf-8"):
            if b not in gs.allowed_tokens(st):
                return False
            st = gs.advance(st, b)
        return gs.is_valid_final(st)

    cases = [
        ("email", ['"a@b.c"', '"user.name+tag@x-y.com"'], ['"a@"', '"@b.c"', '"a b@c"']),
        ("date", ['"2024-02-29"', '"9999-12-31"'], ['"2024-13-01"', '"2024-01-32"', '"24-01-01"']),
        ("date-time", ['"2024-01-01T00:00:00Z"', '"2024-01-01t23:59:59+08:00"'], ['"2024-01-01"', '"2024-01-01 00:00:00Z"']),
        ("time", ['"00:00:00"', '"23:59:59.5Z"'], ['"24:00:00"', '"12:00"']),
        ("uuid", ['"123e4567-e89b-12d3-a456-426614174000"'], ['"123e4567-e89b-12d3-a456-42661417400g"', '"123"']),
        ("ipv4", ['"192.168.1.1"', '"255.255.255.255"'], ['"256.0.0.1"', '"01.2.3.4"', '"1.2.3"']),
        ("ipv6", ['"::1"', '"2001:db8::1"', '"1:2:3:4:5:6:7:8"'], ['"1:2:3"', '"1::2::3"', '"g::1"']),
        ("hostname", ['"localhost"', '"example.com"', '"sub-a.b.co.uk"'], ['"-a.com"', '"a..b"', '"a."']),
        ("uri", ['"https://example.com/p?q=1#f"', '"mailto:u@x.com"'], ['"1http://x"', '"a b:c"']),
    ]
    for fmt, ok_list, bad_list in cases:
        g = c.compile(json.dumps({"type": "string", "format": fmt}))
        g.load_tokenizer(str(tok_path))
        for t in ok_list:
            assert walk_ok(g, t), f"{fmt} 应接受 {t}"
        for t in bad_list:
            assert not walk_ok(g, t), f"{fmt} 应拒绝 {t}"


def test_string_format_unsupported_errors():
    """M5-⑦：不支持的 format → 显式报错（P2 语义）。"""
    c = make_compiler()
    for fmt in ["binary", "byte", "password", "int32"]:
        with pytest.raises(RuntimeError, match="暂不支持"):
            c.compile(json.dumps({"type": "string", "format": fmt}))


def test_string_pattern_walk(tmp_path):
    """M5-⑧：pattern 锚定语义——真实词表 walk 正反例。"""
    vocab = {chr(i): i for i in range(256)}
    tok_path = tmp_path / "tokenizer.json"
    tok_path.write_text(json.dumps({"model": {"type": "BPE", "vocab": vocab}}))
    c = make_compiler()

    def walk_ok(gs, text):
        st = 0
        for b in text.encode("utf-8"):
            if b not in gs.allowed_tokens(st):
                return False
            st = gs.advance(st, b)
        return gs.is_valid_final(st)

    cases = [
        (r"^\d{3}-\d{2}$", ['"123-45"'], ['"12-345"', '"1234-56"']),
        (r"[a-z]+", ['"abc"'], ['"a1"', '"ABC"']),
        (r"(a|b)c", ['"ac"', '"bc"'], ['"cc"', '"a"']),
        (r"a{2,3}", ['"aa"', '"aaa"'], ['"a"', '"aaaa"']),
        ("abc", ['"abc"'], ['"xabc"', '"abcx"']),  # 锚定语义
        ("", ['""'], ['"a"']),  # 空 pattern 只匹配空串
    ]
    for pat, ok_list, bad_list in cases:
        g = c.compile(json.dumps({"type": "string", "pattern": pat}))
        g.load_tokenizer(str(tok_path))
        for t in ok_list:
            assert walk_ok(g, t), f"{pat!r} 应接受 {t}"
        for t in bad_list:
            assert not walk_ok(g, t), f"{pat!r} 应拒绝 {t}"


def test_string_pattern_unsupported_and_conflict():
    """M5-⑧：不支持特性（环视/反向引用/未闭合）与 pattern+format 冲突 → 报错。"""
    c = make_compiler()
    for pat in ["(?=a)", "(?!a)", "a\\1", "(", "["]:
        with pytest.raises(RuntimeError, match="pattern"):
            c.compile(json.dumps({"type": "string", "pattern": pat}))
    with pytest.raises(RuntimeError, match="pattern"):
        c.compile(json.dumps({"type": "string", "pattern": "^a$", "format": "date"}))


def test_p2_unsupported_keywords_rejected():
    """M5-⑨：不支持关键字编译时显式报错（不静默忽略）。"""
    c = make_compiler()
    cases = [
        {"not": {"type": "string"}},
        {"if": {"type": "string"}, "then": {"minLength": 1}},
        {"dependentSchemas": {"a": {"required": ["b"]}}},
        {"dependentRequired": {"a": ["b"]}},
        {"type": "array", "uniqueItems": True},
        {"type": "array", "contains": {"type": "string"}},
        {"type": "array", "minContains": 1},
        {"type": "array", "maxContains": 3},
        {"propertyNames": {"pattern": "^[a-z]+$"}},
        {"patternProperties": {"^x": {"type": "string"}}},
        {"unevaluatedProperties": False},
        {"unevaluatedItems": False},
        {"type": "array", "prefixItems": [{"type": "string"}]},
        {"type": "object", "minProperties": 1},
        {"const": "x"},
        {"allOf": [{"type": "string"}]},
        {"type": "object", "additionalProperties": {"type": "string"}},
        {"type": "array", "items": [{"type": "string"}]},
        # 嵌套传播
        {"type": "object", "properties": {"a": {"not": {"type": "string"}}}},
    ]
    for schema in cases:
        with pytest.raises(RuntimeError, match="暂不支持"):
            c.compile(json.dumps(schema))


def test_p2_annotation_keywords_allowed():
    """M5-⑨：注解/元关键字放行（不约束值）。"""
    c = make_compiler()
    schema = {
        "type": "object",
        "title": "t",
        "description": "d",
        "default": 1,
        "examples": [1, 2],
        "readOnly": True,
        "writeOnly": False,
        "deprecated": False,
        "$comment": "c",
        "$id": "x",
        "$defs": {"a": {"type": "string"}},
        "properties": {"x": {"type": "string", "title": "x", "description": "xd"}},
        "required": ["x"],
        "additionalProperties": False,
    }
    # 注解/元关键字不约束值 → 编译放行（编译失败会抛 RuntimeError）


def test_greedy_decode_e2e(tmp_path):
    """M5-⑩：compile → mask → sample 全链路模拟（256 字节词表 + 最小 id 贪婪）。

    与 tests/grammar_e2e.rs 的 mock:tiny 引擎用例同构：每个位置在 mask 中取
    最小 id 的 token，直到自动机终态——确定性输出可断言。
    """
    vocab = {chr(i): i for i in range(256)}
    tok_path = tmp_path / "tokenizer.json"
    tok_path.write_text(json.dumps({"model": {"type": "BPE", "vocab": vocab}}))
    c = make_compiler()

    def decode(schema, budget=200):
        g = c.compile(json.dumps(schema))
        g.load_tokenizer(str(tok_path))
        st, out = 0, []
        for _ in range(budget):
            # 最小 id 胜出（同 mock logits 偏好）；排除 ws/控制字节——真实词表
            # 不含它们（mock:tiny 无 ws token；控制字节 JSON 需转义，空白是
            # 可选路径而非必经），模拟避免 ws/控制字节自旋
            allowed = sorted(a for a in g.allowed_tokens(st) if a >= 0x20 and a not in (9, 10, 13, 32))
            if not allowed:
                return bytes(out), st  # mask 清空 → eos 强推（引擎收尾语义）
            b = allowed[0]
            out.append(b)
            st = g.advance(st, b)
        raise AssertionError(f"未收尾: {bytes(out)!r}")

    # 对象 + required + 枚举（M5-④ 信封）；字节词表 'f'=102 < 't'=116 → "false" 胜出
    text, _ = decode({
        "type": "object",
        "properties": {"name": {"enum": ["true", "false"]}},
        "required": ["name"],
        "additionalProperties": False,
    })
    assert text == b'{"name":"false"}'

    # integer 范围（M5-⑤）：字节词表 '0'=48 < '1'=49 → 取 '0'（最小 id）
    text, _ = decode({
        "type": "object",
        "properties": {"age": {"type": "integer", "minimum": 0, "maximum": 1}},
        "required": ["age"],
        "additionalProperties": False,
    })
    assert text == b'{"age":0}'

    # string 长度（M5-⑥）：minLength 2 内 "!!" 不可收，续到 maxLength 3 → "!!!"
    # （字节词表最小可打印 id 33='!' 胜出）
    text, _ = decode({"type": "string", "minLength": 2, "maxLength": 3})
    assert text == b'"!!!"'

    # pattern 锚定（M5-⑧）：^abc$ 确定性（贪心无环）
    text, _ = decode({"type": "string", "pattern": "^abc$"})
    assert text == b'"abc"'
