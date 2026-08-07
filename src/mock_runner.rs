//! dev 模式确定性 runner（M5-④）。
//!
//! `model_id = "mock:tiny"` 时不加载真实模型，返回固定 byte 词表 + 常量
//! logits（`vocab_len - id`，即采样偏好最小 id）。配合约束解码的 mask 钩子，
//! 在无模型环境（CI）下也能跑通 HTTP 端到端：grammar 屏蔽非法 token 后，
//! 每个位置选最小合法 id，输出确定可断言。

use crate::engine::{ModelRunner, StepOut};
use crate::error::ZealotError;
use crate::scheduler::Sequence;
use crate::tokenizer::Tokenizer;

/// mock:tiny 固定 byte 词表（id 序 = 采样偏好序，logits = len - id ⇒ 最小 id 胜出）：
///
/// - 内容位 token 置低 id（字符串内容确定输出 `get_weather`）；
/// - `name`(1) < `arguments`(2)：信封 required 序 [name, arguments] 下先发 name
///   （k 0→1），排除后 arguments（k 1→2），收尾 `}` 可达、无重复键死循环；
/// - 标点置高 id：`}`(10) < `,`(13)，值后优先收尾；
/// - 空字节 eos(15)：vocab 过滤排除（allowed_tokens_vocab 跳过空字节），
///   收尾状态由引擎侧 eos 强制放行兜底，不污染输出文本。
pub const MOCK_TINY_VOCAB: &[&str] = &[
    "get_weather",
    "name",
    "arguments",
    "true",
    "false",
    "age",
    "1",
    "0",
    "xxx",
    "{",
    "}",
    "\"",
    ":",
    ",",
    "\n",
    "",
];

#[derive(Debug)]
pub struct MockModelRunner {
    vocab: Vec<Option<String>>,
    eos: i64,
}

impl MockModelRunner {
    /// 按 model_id 解析 mock 模型；未知名称报错（与真实加载失败同路径）。
    pub fn load(model_id: &str) -> Result<MockModelRunner, ZealotError> {
        match model_id {
            "mock:tiny" => Ok(Self {
                vocab: MOCK_TINY_VOCAB
                    .iter()
                    .map(|s| Some(s.to_string()))
                    .collect(),
                eos: (MOCK_TINY_VOCAB.len() - 1) as i64,
            }),
            other => Err(ZealotError::Internal(format!(
                "unknown mock model '{other}' (supported: mock:tiny)"
            ))),
        }
    }

    /// 对应内置词表的 tokenizer（vocab-mode 真实 token 校验用）。
    pub fn tokenizer(&self) -> Tokenizer {
        Tokenizer::from_vocab(self.vocab.clone(), true)
    }
}

impl ModelRunner for MockModelRunner {
    fn tokenize_chat(&self, _messages: &[(String, String)]) -> Result<Vec<i64>, ZealotError> {
        Ok(vec![1, 2])
    }

    fn eos_token_id(&self) -> Option<i64> {
        Some(self.eos)
    }

    fn step(&mut self, batch: &mut [&mut Sequence]) -> Result<Vec<StepOut>, ZealotError> {
        let n = self.vocab.len() as f32;
        Ok(batch
            .iter()
            .map(|s| StepOut {
                request_id: s.request_id.clone(),
                token: None,
                logits: Some((0..self.vocab.len()).map(|i| n - i as f32).collect()),
                text: None,
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_load_rejects_unknown() {
        let err = MockModelRunner::load("mock:nope").unwrap_err();
        assert!(err.to_string().contains("unknown mock model"));
    }

    #[test]
    fn mock_tiny_eos_is_last_id() {
        let runner = MockModelRunner::load("mock:tiny").unwrap();
        assert_eq!(runner.eos_token_id(), Some(15));
        let tok = MockModelRunner::load("mock:tiny").unwrap().tokenizer();
        // 空字节 token：decode 为空串，不污染输出文本
        assert_eq!(tok.decode_single(15), Some(String::new()));
    }
}
