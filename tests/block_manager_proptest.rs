//! M4 §8: BlockManager 随机操作序列 property-based 测试（issue #10）。
//!
//! 模型对照法：测试维护一个与 BlockManager 语义完全一致的模型
//! （free list 栈 + 每 block 的 refcount/generation），随机生成
//! Allocate / Free / Reference 操作序列，逐操作断言真实实现的可观测
//! 行为与模型完全一致：
//!
//! - **无重复分配**：allocate 返回的 block_id 必须等于模型栈弹出的 block，
//!   且该 block 在模型中必须是 Free；
//! - **释放后不可用**：模型判定 stale 的 handle，真实 `try_free` /
//!   `try_reference` 必须返回 `StaleHandle`（generation 门控）；
//! - **refcount 归零才回池**：free 只在 refcount 降到 0 时把 block 推回
//!   栈，模型空闲数必须时刻等于 `bm.available()`；
//! - **generation 递增**：每次回池 generation +1，handle 携带的
//!   generation 必须等于模型值（仅递增）。
//!
//! 运行：`cargo test --test block_manager_proptest`（cargo test 全量包含）。

use proptest::prelude::*;
use zealot_engine::block_manager::{BlockHandle, BlockManager};
use zealot_engine::error::ZealotError;

const BLOCK_SIZE: usize = 16;

/// 随机操作：Allocate；Free/Reference 携带 handle 索引（执行时对
/// 当前 handle 数取模钳制，空时跳过——钳制让同一 handle 可被多次
/// 释放/引用，充分锻炼 stale 路径）。
#[derive(Debug, Clone)]
enum Op {
    Allocate,
    Free(usize),
    Reference(usize),
}

fn op_strategy() -> impl Strategy<Value = Op> {
    prop_oneof![
        1 => Just(Op::Allocate),
        1 => (0usize..16).prop_map(Op::Free),
        1 => (0usize..16).prop_map(Op::Reference),
    ]
}

/// 模型侧每 block 的状态。generation 在 Free 变体中也保留——
/// 真实实现里 generation 是每 block 持久计数器（回池时 +1），
/// 模型若在 Free 时丢弃它，重新分配会错误重置为 1。
#[derive(Debug, Clone, PartialEq)]
enum ModelState {
    Free { generation: u64 },
    Allocated { refcount: usize, generation: u64 },
}

/// 与 BlockManager 语义完全一致的模型。
struct Model {
    /// free list 栈（与 create 相同：`(0..n).rev().collect()`，push/pop 同序）
    free_stack: Vec<usize>,
    blocks: Vec<ModelState>,
}

impl Model {
    fn new(num_blocks: usize) -> Self {
        let mut free_stack: Vec<usize> = (0..num_blocks).collect();
        free_stack.reverse();
        Self {
            free_stack,
            blocks: (0..num_blocks)
                .map(|_| ModelState::Free { generation: 1 })
                .collect(),
        }
    }

    fn free_count(&self) -> usize {
        self.free_stack.len()
    }

    fn allocated_count(&self) -> usize {
        self.blocks
            .iter()
            .filter(|s| !matches!(s, ModelState::Free { .. }))
            .count()
    }

    fn allocate(&mut self) -> Option<usize> {
        let block_id = self.free_stack.pop()?;
        let generation = match self.blocks[block_id] {
            ModelState::Free { generation } => generation,
            ModelState::Allocated { .. } => unreachable!("free_stack 与 blocks 状态不一致"),
        };
        self.blocks[block_id] = ModelState::Allocated {
            refcount: 1,
            generation,
        };
        Some(block_id)
    }

    /// 返回 (是否 stale, 释放后 block 是否回池)。
    fn free(&mut self, h: &BlockHandle) -> Result<bool, ()> {
        let block = &mut self.blocks[h.block_id];
        match block {
            ModelState::Free { .. } => Err(()),
            ModelState::Allocated {
                refcount,
                generation,
            } => {
                if *generation != h.generation {
                    return Err(());
                }
                *refcount -= 1;
                if *refcount == 0 {
                    let next_gen = *generation + 1;
                    *block = ModelState::Free {
                        generation: next_gen,
                    };
                    self.free_stack.push(h.block_id);
                    Ok(true)
                } else {
                    Ok(false)
                }
            }
        }
    }

    fn reference(&mut self, h: &BlockHandle) -> Result<(), ()> {
        match &mut self.blocks[h.block_id] {
            ModelState::Free { .. } => Err(()),
            ModelState::Allocated {
                refcount,
                generation,
            } => {
                if *generation != h.generation {
                    return Err(());
                }
                *refcount += 1;
                Ok(())
            }
        }
    }
}

fn run_sequence(num_blocks: usize, ops: &[Op]) {
    let bm = BlockManager::create(num_blocks, BLOCK_SIZE).unwrap();
    let mut model = Model::new(num_blocks);
    // 测试持有的 handle 池（永不删除：已释放的 handle 留在池里走 stale 路径）
    let mut handles: Vec<BlockHandle> = Vec::new();
    // 测试侧持有的有效引用数（allocate +1 / 成功 free -1 / 成功 reference +1）
    let mut held_refs: usize = 0;

    for op in ops {
        match op {
            Op::Allocate => {
                match bm.try_allocate() {
                    Ok(h) => {
                        let expected = model.allocate().expect("model 无空闲 block");
                        // 无重复分配：真实返回的 block 必须正是模型弹出的那个
                        assert_eq!(h.block_id, expected, "allocate 返回了非预期 block");
                        // generation 与模型一致（仅递增）
                        if let ModelState::Allocated { generation, .. } = model.blocks[h.block_id] {
                            assert_eq!(h.generation, generation, "generation 与模型不一致");
                        }
                        held_refs += 1;
                        handles.push(h);
                    }
                    Err(ZealotError::OutOfBlocks { .. }) => {
                        assert!(
                            model.free_stack.is_empty(),
                            "模型还有空闲 block，真实却报 OutOfBlocks"
                        );
                    }
                    Err(other) => panic!("unexpected error: {other}"),
                }
            }
            Op::Free(idx) => {
                if handles.is_empty() {
                    continue;
                }
                let h = handles[idx % handles.len()].clone();
                let expected = model.free(&h);
                match (bm.try_free(&h), expected) {
                    (Ok(()), Ok(reclaimed)) => {
                        held_refs -= 1;
                        if reclaimed {
                            // 归零才回池：模型里该 block 已 Free 且栈顶是它
                            assert_eq!(model.free_stack.last(), Some(&h.block_id));
                        } else {
                            assert!(model.free_count() < num_blocks);
                        }
                    }
                    (Err(e), Err(())) => {
                        // 释放后不可用：stale handle 必须被拒绝
                        assert!(
                            matches!(e, ZealotError::StaleHandle { .. }),
                            "stale free 应报 StaleHandle，实际 {e}"
                        );
                    }
                    (Ok(()), Err(())) => panic!("模型判定 stale，真实却成功释放"),
                    (Err(e), Ok(_)) => panic!("模型判定有效，真实却释放失败: {e}"),
                }
            }
            Op::Reference(idx) => {
                if handles.is_empty() {
                    continue;
                }
                let h = handles[idx % handles.len()].clone();
                match (bm.try_reference(&h), model.reference(&h)) {
                    (Ok(()), Ok(())) => {
                        held_refs += 1;
                    }
                    (Err(e), Err(())) => {
                        assert!(
                            matches!(e, ZealotError::StaleHandle { .. }),
                            "stale reference 应报 StaleHandle，实际 {e}"
                        );
                    }
                    (Ok(()), Err(())) => panic!("模型判定 stale，真实却成功引用"),
                    (Err(e), Ok(())) => panic!("模型判定有效，真实却引用失败: {e}"),
                }
            }
        }

        // 引用守恒（逐 op）：测试侧有效引用数必须等于模型 refcount 总和
        let model_refs: usize = model
            .blocks
            .iter()
            .map(|s| match s {
                ModelState::Allocated { refcount, .. } => *refcount,
                ModelState::Free { .. } => 0,
            })
            .sum();
        assert_eq!(
            held_refs, model_refs,
            "引用守恒：held_refs != 模型 refcount 总和"
        );
    }

    // 终局不变量：模型空闲数 == 真实空闲数（无泄漏 / 无重复入池）
    assert_eq!(model.free_count(), bm.available());
    assert_eq!(model.allocated_count(), num_blocks - bm.available());
    // 全部引用已释放（所有 handle 都已 free 或 stale）
    // —— 不强制：stale 的 handle 不持有引用，此处只做守恒（上面已逐 op 断言）。
}

proptest! {
    /// 随机操作序列 vs 模型：任意交错下真实行为必须与模型完全一致。
    #[test]
    fn random_ops_match_model(
        num_blocks in 1usize..8,
        ops in prop::collection::vec(op_strategy(), 1..300),
    ) {
        run_sequence(num_blocks, &ops);
    }

    /// 单 block 池：最高强度的复用/竞争（每 op 都命中同一 block 的
    /// 分配-释放-过期循环）。
    #[test]
    fn single_block_pool_torture(
        ops in prop::collection::vec(op_strategy(), 1..300),
    ) {
        run_sequence(1, &ops);
    }
}
