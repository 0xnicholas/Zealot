//! M5-⑩（issue #24）：compile → mask → sample 端到端用例。
//!
//! 全链路：JSON Schema 编译（JsonSchemaCompiler）→ 字节自动机 mask 钩子 →
//! mock:tiny 常量 logits（偏好最小 id）采样 → 确定性 token 输出断言。
//! 覆盖 M5-④（信封/required/additionalProperties）、⑤（integer 范围）、
//! ⑥（string min/maxLength）、⑧（pattern 锚定）的端到端行为。
//! （M5-⑦ format 无引擎级用例：mock:tiny 词表缺 `@`/`-` 等格式字符，
//! 由 pytest 真实字节词表 walk 覆盖。）

mod common;

use zealot_engine::constrained_decode::JsonSchemaCompiler;
use zealot_engine::engine::Engine;
use zealot_engine::mock_runner::MockModelRunner;
use zealot_engine::sampling::SamplingParams;
use zealot_engine::scheduler::{FinishReason, Priority, Scheduler, SchedulerConfig};

/// 编译 schema → mock:tiny 引擎采样到收尾，返回 token 序列与终止原因。
fn run_grammar(schema: &str) -> (Vec<i64>, Vec<FinishReason>) {
    let grammar = JsonSchemaCompiler::new(16, 10000, 500)
        .compile(schema, None)
        .expect("schema 应可编译");
    let runner = MockModelRunner::load("mock:tiny").unwrap();
    let tok = runner.tokenizer();
    let sched = Scheduler::new(SchedulerConfig::default()).unwrap();
    let mut engine = Engine::new(sched, Box::new(runner)).with_tokenizer(tok);
    let seq = engine
        .scheduler_mut()
        .make_sequence(
            "e2e".into(),
            vec![1, 2],
            32,
            Priority::Medium,
            Some(15), // eos = 空字节 token
            SamplingParams {
                temperature: 0.0,
                ..Default::default()
            },
        )
        .unwrap()
        .with_grammar(grammar);
    engine.scheduler_mut().add(seq);

    let mut tokens = Vec::new();
    let mut finished = Vec::new();
    while !engine.is_idle() {
        let res = engine.step().unwrap();
        tokens.extend(res.tokens.iter().map(|t| t.token));
        finished.extend(res.finished.iter().map(|f| f.reason));
    }
    (tokens, finished)
}

fn obj_schema(props: &[(&str, &str)], required: &[&str], additional: bool) -> String {
    let mut s = String::from(r#"{"type": "object", "properties": {"#);
    for (i, (k, v)) in props.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("\"{k}\": {v}"));
    }
    s.push_str("}, \"required\": [");
    for (i, k) in required.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push_str(&format!("\"{k}\""));
    }
    s.push_str(&format!("], \"additionalProperties\": {additional}}}"));
    s
}

#[test]
fn e2e_object_required_additional_false() {
    // M5-④：信封 + required + additionalProperties=false。
    // 值 true(3) < false(4)；收尾 `}`(10) < `,`(13)。
    let schema = obj_schema(
        &[
            ("name", r#"{"type": "boolean"}"#),
            ("age", r#"{"type": "integer"}"#),
        ],
        &["name"],
        false,
    );
    let (tokens, finished) = run_grammar(&schema);
    assert_eq!(tokens, vec![9, 11, 1, 11, 12, 3, 10, 15]);
    assert_eq!(finished, vec![FinishReason::Stop]);
}

#[test]
fn e2e_object_extra_key_rejected() {
    // M5-⑩ 修复回归：additionalProperties=false 时额外键不可达（xxx 键被屏蔽）。
    let schema = obj_schema(
        &[
            ("name", r#"{"type": "boolean"}"#),
            ("age", r#"{"type": "integer"}"#),
        ],
        &["name"],
        false,
    );
    let (tokens, _) = run_grammar(&schema);
    assert!(!tokens.contains(&8), "xxx 键被屏蔽: {tokens:?}");
}

#[test]
fn e2e_integer_range_zero_one() {
    // M5-⑤：integer [0,1] —— 贪心取最小 id：`1`(6) < `0`(7)。
    let schema = obj_schema(
        &[("age", r#"{"type": "integer", "minimum": 0, "maximum": 1}"#)],
        &["age"],
        false,
    );
    let (tokens, finished) = run_grammar(&schema);
    assert_eq!(tokens, vec![9, 11, 5, 11, 12, 6, 10, 15]);
    assert_eq!(finished, vec![FinishReason::Stop]);
}

#[test]
fn e2e_string_min_max_length() {
    // M5-⑥：minLength 5 / maxLength 7 —— `"name"`(4) 后可续不可收，
    // `"age"`(3) 续到 7 = max 只可收尾 → "nameage"。
    let schema = obj_schema(
        &[(
            "name",
            r#"{"type": "string", "minLength": 5, "maxLength": 7}"#,
        )],
        &["name"],
        false,
    );
    let (tokens, finished) = run_grammar(&schema);
    assert_eq!(tokens, vec![9, 11, 1, 11, 12, 11, 1, 5, 11, 10, 15]);
    assert_eq!(finished, vec![FinishReason::Stop]);
}

#[test]
fn e2e_pattern_anchored() {
    // M5-⑧：pattern 锚定整串 —— 只有 get_weather(0) 命中 `^get_weather$`。
    let schema = obj_schema(
        &[("name", r#"{"type": "string", "pattern": "^get_weather$"}"#)],
        &["name"],
        false,
    );
    let (tokens, finished) = run_grammar(&schema);
    assert_eq!(tokens, vec![9, 11, 1, 11, 12, 11, 0, 11, 10, 15]);
    assert_eq!(finished, vec![FinishReason::Stop]);
}

#[test]
fn e2e_compile_error_surfaces_400() {
    // M5-⑨/⑩：不支持关键字编译失败 → 显式错误（可映射 400），不静默降级。
    let err = JsonSchemaCompiler::new(16, 10000, 500)
        .compile(r#"{"type": "object", "uniqueItems": true}"#, None)
        .unwrap_err();
    assert!(err.to_string().contains("暂不支持"));
}
