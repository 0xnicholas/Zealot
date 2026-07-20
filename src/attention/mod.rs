use crate::error::ZealotError;

pub trait AttentionBackend: Send {
    fn forward(
        &mut self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        batch: &AttentionBatch,
    ) -> Result<Vec<f32>, ZealotError>;
}

pub struct AttentionBatch {
    pub num_seqs: usize,
    pub num_heads: usize,
    pub head_dim: usize,
    pub max_seq_len: usize,
}

pub struct CpuAttention;

impl AttentionBackend for CpuAttention {
    fn forward(
        &mut self,
        query: &[f32],
        key: &[f32],
        value: &[f32],
        batch: &AttentionBatch,
    ) -> Result<Vec<f32>, ZealotError> {
        let expected = batch.num_seqs * batch.num_heads * batch.max_seq_len * batch.head_dim;
        if query.len() != expected || key.len() != expected || value.len() != expected {
            return Err(ZealotError::Internal(format!(
                "dimension mismatch: expected {}, got q={} k={} v={}",
                expected,
                query.len(),
                key.len(),
                value.len()
            )));
        }

        let n_heads = batch.num_heads;
        let d_head = batch.head_dim;
        let n_pos = batch.max_seq_len;
        let scale = 1.0_f32 / (d_head as f32).sqrt();
        let mut output = vec![0.0_f32; expected];

        for s in 0..batch.num_seqs {
            for h in 0..n_heads {
                let base = ((s * n_heads + h) * n_pos) * d_head;
                let q = &query[base..base + n_pos * d_head];
                let k = &key[base..base + n_pos * d_head];
                let v = &value[base..base + n_pos * d_head];
                let out = &mut output[base..base + n_pos * d_head];

                let mut scores = vec![0.0_f32; n_pos * n_pos];
                for i in 0..n_pos {
                    for j in 0..n_pos {
                        let mut dot = 0.0_f32;
                        for d in 0..d_head {
                            dot += q[i * d_head + d] * k[j * d_head + d];
                        }
                        scores[i * n_pos + j] = dot * scale;
                    }
                }

                // softmax per row
                for i in 0..n_pos {
                    let mut max = f32::NEG_INFINITY;
                    for j in 0..n_pos {
                        max = max.max(scores[i * n_pos + j]);
                    }
                    let mut sum = 0.0_f32;
                    for j in 0..n_pos {
                        sum += (scores[i * n_pos + j] - max).exp();
                    }
                    for j in 0..n_pos {
                        scores[i * n_pos + j] = (scores[i * n_pos + j] - max).exp() / sum;
                    }
                }

                for i in 0..n_pos {
                    for d in 0..d_head {
                        let mut acc = 0.0_f32;
                        for j in 0..n_pos {
                            acc += scores[i * n_pos + j] * v[j * d_head + d];
                        }
                        out[i * d_head + d] = acc;
                    }
                }
            }
        }
        Ok(output)
    }
}

pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            for inner in 0..k {
                c[i * n + j] += a[i * k + inner] * b[inner * n + j];
            }
        }
    }
    c
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_attention_single_seq_output_shape() {
        let mut attn = CpuAttention;
        let batch = AttentionBatch {
            num_seqs: 1,
            num_heads: 2,
            head_dim: 4,
            max_seq_len: 3,
        };
        let q = vec![1.0_f32; 1 * 2 * 3 * 4];
        let k = vec![1.0_f32; 1 * 2 * 3 * 4];
        let v = vec![1.0_f32; 1 * 2 * 3 * 4];
        let out = attn.forward(&q, &k, &v, &batch).unwrap();
        assert_eq!(out.len(), 1 * 2 * 3 * 4);
    }

    #[test]
    fn cpu_attention_returns_error_on_dimension_mismatch() {
        let mut attn = CpuAttention;
        let batch = AttentionBatch {
            num_seqs: 1,
            num_heads: 2,
            head_dim: 4,
            max_seq_len: 3,
        };
        let q = vec![1.0_f32; 10]; // wrong size
        let k = vec![1.0_f32; 1 * 2 * 3 * 4];
        let v = vec![1.0_f32; 1 * 2 * 3 * 4];
        let err = attn.forward(&q, &k, &v, &batch).unwrap_err();
        assert!(matches!(err, ZealotError::Internal(_)));
    }

    #[test]
    fn cpu_attention_multi_seq_separate_attention() {
        let mut attn = CpuAttention;
        let batch = AttentionBatch {
            num_seqs: 2,
            num_heads: 1,
            head_dim: 4,
            max_seq_len: 2,
        };
        // seq0 q all 1, seq1 q all 100
        let mut q = vec![1.0_f32; 2 * 1 * 2 * 4];
        let base1 = 1 * 1 * 2 * 4;
        for i in base1..q.len() {
            q[i] = 100.0;
        }
        // position-dependent k/v: pos0 gets half the dot-product sum of pos1,
        // so different query magnitudes produce different softmax distributions
        let k = vec![
            1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
        ];
        let v = k.clone();
        let out = attn.forward(&q, &k, &v, &batch).unwrap();
        let o0 = &out[0..8];
        let o1 = &out[8..16];
        assert_ne!(
            o0, o1,
            "different inputs should yield different attention outputs"
        );
    }

    #[test]
    fn matmul_dimensions() {
        let a = vec![1.0_f32, 2.0, 3.0, 4.0]; // 2x2
        let b = vec![5.0_f32, 6.0, 7.0, 8.0]; // 2x2
        let c = matmul(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }
}
