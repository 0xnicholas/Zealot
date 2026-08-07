//! M4 §8.1 FFI 微基准：Constrained Decode schema 编译（issue #8）。
//!
//! 运行：`cargo bench`
//!
//! 验收阈值（docs/architecture.md §8.1）：
//! - decode_schema_simple  < 5μs  简单 object schema 编译
//! - decode_schema_reject  < 1ms  超 max_states 的对抗性 schema 拒绝

use criterion::{criterion_group, criterion_main, Criterion};
use zealot_engine::constrained_decode::schema::JsonSchemaCompiler;

const CACHE_SIZE: usize = 128;
const MAX_STATES: usize = 10_000;
const MAX_COMPILE_MS: u64 = 1000;

/// 简单 object schema 编译（bypass cache → 测真实编译路径，非缓存命中）。
fn bench_decode_schema_simple(c: &mut Criterion) {
    let compiler = JsonSchemaCompiler::new(CACHE_SIZE, MAX_STATES, MAX_COMPILE_MS);
    let schema = r#"{
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "active": {"type": "boolean"},
            "tags": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["name", "age"]
    }"#;

    c.bench_function("decode_schema_simple", |b| {
        b.iter(|| {
            let g = compiler.compile(schema, Some(true)).unwrap();
            std::hint::black_box(g.state_count());
        });
    });
}

/// 对抗性 schema（10001 个属性的 object，超过 max_states=10000）
/// 必须走 SchemaTooComplex 拒绝路径。
fn bench_decode_schema_reject(c: &mut Criterion) {
    let compiler = JsonSchemaCompiler::new(CACHE_SIZE, MAX_STATES, MAX_COMPILE_MS);
    let props: String = (0..MAX_STATES + 1)
        .map(|i| format!("\"p{i}\": {{\"type\": \"string\"}}"))
        .collect::<Vec<_>>()
        .join(",");
    let schema = format!("{{\"type\": \"object\", \"properties\": {{{props}}}}}");

    c.bench_function("decode_schema_reject", |b| {
        b.iter(|| {
            let r = compiler.compile(&schema, Some(true));
            debug_assert!(r.is_err(), "expected SchemaTooComplex rejection");
            std::hint::black_box(r.is_err());
        });
    });
}

criterion_group!(benches, bench_decode_schema_simple, bench_decode_schema_reject);
criterion_main!(benches);
