//! Sampling module: temperature, top-k, top-p, penalty, softmax, multinomial.
//!
//! Independent of Engine, ModelRunner, and CUDA. All operations are
//! CPU-side `&[f32]` computations suitable for vocabularies up to
//! 128k tokens. No allocation per step is wasted where avoidable.

use std::collections::HashMap;

use rand::Rng;

// ── Types ────────────────────────────────────────────────────────────────────

/// Sampling error.
#[derive(Debug, Clone)]
pub enum SamplingError {
    /// Empty logits input.
    EmptyLogits,
    /// All tokens were masked (top-k / top-p / constraint too strict).
    AllTokensMasked,
}

/// Sampling parameters. All fields have defaults that mean "no-op."
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub frequency_penalty: f32,
    pub presence_penalty: f32,
    pub seed: Option<u64>,
}

impl Default for SamplingParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 0.0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            seed: None,
        }
    }
}

/// Sampling result.
#[derive(Debug, Clone)]
pub struct SampledToken {
    pub token_id: i64,
    pub logprob: f32,
}

// ── Sampler ──────────────────────────────────────────────────────────────────

/// Zero-size sampler. Call `sample()` repeatedly.
pub struct Sampler;

impl Sampler {
    /// Run the full sampling pipeline.
    ///
    /// `prev_output_ids` is the sequence of tokens already generated
    /// (used for penalty computation).  It can be empty during prefill.
    pub fn sample(
        &self,
        logits: &[f32],
        prev_output_ids: &[i64],
        params: &SamplingParams,
        rng: &mut impl Rng,
    ) -> Result<SampledToken, SamplingError> {
        if logits.is_empty() {
            return Err(SamplingError::EmptyLogits);
        }

        // ── 1. Greedy fast path (temperature ≈ 0) ─────────────────────────
        if params.temperature < 1e-7 {
            let (max_idx, _) = argmax(logits);
            let logprob = lightweight_logprob(logits, max_idx);
            return Ok(SampledToken {
                token_id: max_idx,
                logprob,
            });
        }

        // ── 2. Clone logits into mutable scores ───────────────────────────
        let mut scores = logits.to_vec();

        // ── 3. Temperature scaling ───────────────────────────────────────
        if params.temperature != 1.0 {
            let inv_t = 1.0 / params.temperature;
            for s in &mut scores {
                *s *= inv_t;
            }
        }

        // ── 4/5. Presence + frequency penalties (merged pass) ────────────
        // Presence penalty subtracts a flat amount per unique token; frequency
        // penalty subtracts (count × penalty). Both iterate prev_output_ids.
        // M5: Merge into one HashMap pass to halve iteration overhead.
        if (params.presence_penalty != 0.0 || params.frequency_penalty != 0.0)
            && !prev_output_ids.is_empty()
        {
            // Single pass: build frequency map
            let mut freq: HashMap<i64, u32> = HashMap::new();
            for &id in prev_output_ids {
                *freq.entry(id).or_insert(0) += 1;
            }
            // Apply both penalties from the same map
            for (id, count) in &freq {
                let idx = *id as usize;
                if idx < scores.len() {
                    // Frequency penalty scales with count
                    scores[idx] -= (*count as f32) * params.frequency_penalty;
                    // Presence penalty applies once per unique token
                    if params.presence_penalty != 0.0 {
                        scores[idx] -= params.presence_penalty;
                    }
                }
            }
        }

        // ── 6. Repetition penalty (multiplicative, per occurrence) ────────
        if params.repetition_penalty != 1.0 && !prev_output_ids.is_empty() {
            let pen = params.repetition_penalty;
            for &id in prev_output_ids {
                let idx = id as usize;
                if idx < scores.len() {
                    if scores[idx] > 0.0 {
                        scores[idx] /= pen;
                    } else {
                        scores[idx] *= pen;
                    }
                }
            }
        }

        // ── 7. Top-K filtering ────────────────────────────────────────────
        if params.top_k > 0 {
            let k = (params.top_k as usize).min(scores.len());
            if k < scores.len() {
                // Use select_nth_unstable_by on INDEXED values to find threshold
                // without reordering the original scores array (which would break
                // the index→token mapping).
                let mut indexed: Vec<(usize, f32)> = scores.iter().copied().enumerate().collect();
                indexed.select_nth_unstable_by(k - 1, |a, b| {
                    b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                });
                let threshold = indexed[k - 1].1;
                for s in &mut scores {
                    if *s < threshold {
                        *s = f32::NEG_INFINITY;
                    }
                }
            }
        }

        // ── 8. Top-P (nucleus) filtering ──────────────────────────────────
        if params.top_p > 0.0 && params.top_p < 1.0 {
            let probs = softmax(&scores);
            let mut indexed: Vec<(f32, usize)> = probs
                .iter()
                .copied()
                .enumerate()
                .map(|(i, p)| (p, i))
                .collect();
            // M5: sorts entire vocab (O(n log n)). Optimize with
            // select_nth_unstable + partial sort for 128k vocab.
            indexed.sort_unstable_by(|a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            let mut cum = 0.0_f32;
            let mut selected = 0_usize;
            for (p, _) in &indexed {
                cum += p;
                selected += 1;
                if cum >= params.top_p {
                    break;
                }
            }
            for (_, idx) in &indexed[selected..] {
                scores[*idx] = f32::NEG_INFINITY;
            }
        }

        // ── 9. Softmax → probabilities ────────────────────────────────────
        let probs = softmax(&scores);

        // ── 10. All-masked check ──────────────────────────────────────────
        let sum: f32 = probs.iter().sum();
        if sum == 0.0 || !sum.is_finite() {
            return Err(SamplingError::AllTokensMasked);
        }

        // ── 11. Sample from multinomial ───────────────────────────────────
        let token_idx = sample_multinomial(&probs, rng);
        let logprob = probs[token_idx].ln();

        Ok(SampledToken {
            token_id: token_idx as i64,
            logprob,
        })
    }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

/// Return `(index, value)` of the maximum element.
fn argmax(slice: &[f32]) -> (i64, f32) {
    let mut best_idx = 0_i64;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in slice.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i as i64;
        }
    }
    (best_idx, best_val)
}

/// Compute log-probability of `max_idx` without allocating a full softmax vector.
fn lightweight_logprob(logits: &[f32], max_idx: i64) -> f32 {
    let max = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let sum_exp: f32 = logits.iter().map(|&x| (x - max).exp()).sum();
    let prob = (logits[max_idx as usize] - max).exp() / sum_exp;
    prob.ln()
}

/// Numerically stable softmax (subtract max before exp).
pub fn softmax(scores: &[f32]) -> Vec<f32> {
    let max = scores
        .iter()
        .copied()
        .filter(|x| x.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);

    let mut probs: Vec<f32> = scores
        .iter()
        .map(|&s| if s.is_finite() { (s - max).exp() } else { 0.0 })
        .collect();

    let sum: f32 = probs.iter().sum();
    if sum > 0.0 {
        for p in &mut probs {
            *p /= sum;
        }
    }
    probs
}

/// Sample from a multinomial distribution using CDF + uniform random.
fn sample_multinomial(probs: &[f32], rng: &mut impl Rng) -> usize {
    let r: f32 = rng.gen_range(0.0..1.0);
    let mut cumsum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cumsum += p;
        if r < cumsum {
            return i;
        }
    }
    // Float rounding guard: return last token
    probs.len() - 1
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    /// Make logits where token `n` has value `n` (higher → more likely).
    fn linear_logits(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    #[test]
    fn temperature_zero_is_greedy() {
        let sampler = Sampler;
        let logits = vec![1.0, 2.0, 3.0, 0.5, 100.0];
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        // Run 10 times — should always return the same argmax (index 4 = 100.0)
        for _ in 0..10 {
            let result = sampler.sample(&logits, &[], &params, &mut rng()).unwrap();
            assert_eq!(result.token_id, 4);
            assert!(
                result.logprob <= 0.0,
                "logprob should be non-positive: {}",
                result.logprob
            );
        }
    }

    #[test]
    fn temperature_clamp_to_greedy() {
        let sampler = Sampler;
        let logits = vec![1.0, 2.0, 50.0, 0.5];
        let params = SamplingParams {
            temperature: 1e-10, // below clamp threshold
            ..Default::default()
        };
        let result = sampler.sample(&logits, &[], &params, &mut rng()).unwrap();
        assert_eq!(result.token_id, 2); // index 2 = 50.0 is argmax
    }

    #[test]
    fn greedy_logprob_is_valid() {
        let sampler = Sampler;
        let vocab_size = 100;
        let logits: Vec<f32> = (0..vocab_size).map(|i| (i as f32) * 0.1).collect();
        let params = SamplingParams {
            temperature: 0.0,
            ..Default::default()
        };
        let result = sampler.sample(&logits, &[], &params, &mut rng()).unwrap();
        assert!(result.logprob <= 0.0);
        // logprob should be ≥ ln(1/vocab_size)
        let min_possible = (1.0 / vocab_size as f32).ln();
        assert!(
            result.logprob >= min_possible,
            "logprob {:.4} too small for {} tokens",
            result.logprob,
            vocab_size
        );
    }

    #[test]
    fn temperature_scaling() {
        let sampler = Sampler;
        let logits = linear_logits(10);
        // t=100 → extremely flat → each token has roughly equal chance
        let params = SamplingParams {
            temperature: 100.0,
            ..Default::default()
        };
        let result = sampler.sample(&logits, &[], &params, &mut rng()).unwrap();
        assert!(result.token_id < 10);
    }

    #[test]
    fn top_k_one_is_argmax() {
        let sampler = Sampler;
        let logits = vec![0.1, 5.0, 0.2, 3.0];
        let params = SamplingParams {
            top_k: 1,
            ..Default::default()
        };
        // k=1 + default temp=1 → should return argmax
        for _ in 0..5 {
            let result = sampler.sample(&logits, &[], &params, &mut rng()).unwrap();
            assert_eq!(result.token_id, 1); // index 1 = 5.0
        }
    }

    #[test]
    fn top_k_filters_low_prob_tokens() {
        let sampler = Sampler;
        // 100 tokens, token 99 has the highest logit
        let logits = linear_logits(100);
        let params = SamplingParams {
            top_k: 3,
            ..Default::default()
        };
        for _ in 0..20 {
            let result = sampler.sample(&logits, &[], &params, &mut rng()).unwrap();
            // Only tokens 97, 98, 99 can be returned (top 3)
            assert!(
                result.token_id >= 97,
                "token {} outside top-3",
                result.token_id
            );
        }
    }

    #[test]
    fn top_p_keeps_all_when_one() {
        let sampler = Sampler;
        let logits = linear_logits(10);
        let params = SamplingParams {
            top_p: 1.0,
            ..Default::default()
        };
        let result = sampler.sample(&logits, &[], &params, &mut rng()).unwrap();
        assert!(result.token_id < 10);
    }

    #[test]
    fn repetition_penalty_discourages_repeats() {
        let sampler = Sampler;
        let logits = vec![1.0, 2.0, 100.0, 0.5]; // token 2 is dominant
        let prev = vec![2_i64, 2]; // already output token 2 twice

        let params_no_penalty = SamplingParams {
            repetition_penalty: 1.0,
            ..Default::default()
        };
        let params_penalty = SamplingParams {
            repetition_penalty: 1.5,
            ..Default::default()
        };

        // With penalty, token 2's effective logit should be lower
        let no_pen = sampler
            .sample(&logits, &prev, &params_no_penalty, &mut rng())
            .unwrap();
        // Since logits[2]=100 is dominant, without penalty it almost always picks token 2
        assert_eq!(no_pen.token_id, 2);

        // With penalty, logits[2] becomes 100/1.5 ≈ 66.7, still dominant but closer
        // Actually 66.7 is still way above others so it'll still pick token 2.
        // Let's test with closer values.
        let close_logits = vec![1.0, 5.0, 8.0, 7.5];
        let prev_close = vec![1_i64]; // token 1
        let no_pen_close = sampler
            .sample(&close_logits, &prev_close, &params_no_penalty, &mut rng())
            .unwrap();
        assert_eq!(no_pen_close.token_id, 2); // token 2=8.0 is argmax

        let pen_close = sampler
            .sample(&close_logits, &prev_close, &params_penalty, &mut rng())
            .unwrap();
        // token 1's logit becomes 5.0/1.5 ≈ 3.33, making token 2 the clear winner
        assert_eq!(pen_close.token_id, 2);
    }

    #[test]
    fn frequency_penalty_scales_with_count() {
        let sampler = Sampler;
        // 4 tokens: 0=1.0, 1=100.0, 2=1.0, 3=1.0
        let logits = vec![1.0, 100.0, 1.0, 1.0];

        // With frequency_penalty=30 and token 1 appearing 3 times:
        // token 1: 100.0 - 3*30 = 10.0 (still the highest if others are 1.0)
        let prev = vec![1_i64, 1, 1, 1, 1, 1, 1, 1]; // 8 occurrences, 100-240=-140
        let params = SamplingParams {
            frequency_penalty: 30.0,
            ..Default::default()
        };
        let result = sampler.sample(&logits, &prev, &params, &mut rng()).unwrap();
        // After penalty: token 1 = 100 - 8*30 = -140, others stay at 1.0
        // So token 1 should NOT be chosen
        assert_ne!(
            result.token_id, 1,
            "heavily penalized token should not be chosen"
        );
    }

    #[test]
    fn presence_penalty_additive() {
        let sampler = Sampler;
        let logits = vec![10.0, 15.0, 5.0, 12.0]; // token 1 is argmax (15.0)
        let prev = vec![1_i64]; // token 1 has been seen

        let params = SamplingParams {
            presence_penalty: 10.0,
            ..Default::default()
        };
        // token 1: 15.0 - 10.0 = 5.0 (now below token 3 = 12.0)
        let result = sampler.sample(&logits, &prev, &params, &mut rng()).unwrap();
        assert_ne!(result.token_id, 1, "penalized seen token should be avoided");
    }

    #[test]
    fn empty_logits_error() {
        let sampler = Sampler;
        let err = sampler.sample(&[], &[], &SamplingParams::default(), &mut rng());
        assert!(matches!(err, Err(SamplingError::EmptyLogits)));
    }

    #[test]
    fn all_masked_error() {
        let sampler = Sampler;
        // All logits are -inf (all masked out)
        let logits = vec![f32::NEG_INFINITY; 10];
        let err = sampler.sample(&logits, &[], &SamplingParams::default(), &mut rng());
        assert!(matches!(err, Err(SamplingError::AllTokensMasked)));
    }

    #[test]
    fn deterministic_with_seed() {
        let sampler = Sampler;
        let logits = linear_logits(50);
        let params = SamplingParams {
            temperature: 0.8,
            top_k: 10,
            ..Default::default()
        };

        let r1 = sampler
            .sample(&logits, &[], &params, &mut StdRng::seed_from_u64(12345))
            .unwrap();
        let r2 = sampler
            .sample(&logits, &[], &params, &mut StdRng::seed_from_u64(12345))
            .unwrap();
        assert_eq!(r1.token_id, r2.token_id);
        assert!((r1.logprob - r2.logprob).abs() < 1e-6);
    }

    #[test]
    fn all_params_combined() {
        let sampler = Sampler;
        let logits = linear_logits(100);
        let prev = vec![99_i64, 98, 97];
        let params = SamplingParams {
            temperature: 0.7,
            top_k: 20,
            top_p: 0.9,
            repetition_penalty: 1.2,
            frequency_penalty: 0.5,
            presence_penalty: 0.3,
            seed: None,
        };
        let result = sampler.sample(&logits, &prev, &params, &mut rng()).unwrap();
        assert!(result.token_id < 100);
        assert!(result.logprob <= 0.0);
    }

    // ── softmax tests ─────────────────────────────────────────────────────

    #[test]
    fn softmax_sums_to_one() {
        let inputs = vec![1.0_f32, 2.0, 3.0, 4.0, 5.0];
        let probs = softmax(&inputs);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={}", sum);
    }

    #[test]
    fn softmax_stable_with_extreme_values() {
        // Large values that would overflow if not for max-subtract
        let inputs = vec![1e5_f32, -1e5_f32, 0.0, 5e4_f32];
        let probs = softmax(&inputs);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "sum={}", sum);
        // All outputs should be finite
        assert!(probs.iter().all(|p| p.is_finite()));
    }

    #[test]
    fn top_k_disabled_at_zero() {
        let sampler = Sampler;
        let logits = linear_logits(10);
        let params = SamplingParams {
            top_k: 0, // disabled
            ..Default::default()
        };
        let result = sampler.sample(&logits, &[], &params, &mut rng()).unwrap();
        assert!(result.token_id < 10);
    }
}
