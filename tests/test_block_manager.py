"""M4 §9: BlockManager / BlockHandle PyO3 接口一致性 + 错误转换 + StaleHandle 检测。

覆盖 §8 测试矩阵「Python 集成」行：接口一致性、错误转换、StaleHandle 检测。
运行：`pytest tests/`（前置：`.venv/bin/maturin develop` 安装扩展）。
"""

import pytest

import zealot_engine as ze


def test_module_exposes_all_classes():
    for name in ("BlockManager", "BlockHandle", "JsonSchemaCompiler", "ConstrainedGrammar"):
        assert hasattr(ze, name), f"missing {name}"


def test_allocate_returns_handle_and_tracks_free_blocks():
    bm = ze.BlockManager(num_gpu_blocks=4, block_size=16)
    assert bm.free_blocks == 4
    assert bm.num_gpu_blocks == 4
    assert bm.block_size == 16

    h = bm.allocate()
    assert isinstance(h, ze.BlockHandle)
    assert isinstance(h.block_id, int)
    assert isinstance(h.generation, int)
    assert h.block_id == 0  # LIFO: create 时 free_list 反转，先分配 id=0
    assert h.generation == 1
    assert bm.free_blocks == 3


def test_handle_attributes_and_repr():
    h = ze.BlockHandle(block_id=7, generation=3)
    assert h.block_id == 7
    assert h.generation == 3
    assert "7" in repr(h) and "3" in repr(h)


def test_free_returns_block_to_pool_and_bumps_generation():
    bm = ze.BlockManager(4, 16)
    h1 = bm.allocate()
    gen1 = h1.generation
    bm.free(h1)
    assert bm.free_blocks == 4

    h2 = bm.allocate()
    # LIFO 复用同一 block_id，但 generation 必须递增
    assert h2.block_id == h1.block_id
    assert h2.generation == gen1 + 1


def test_reference_counting_prevents_early_reclaim():
    bm = ze.BlockManager(2, 16)
    h = bm.allocate()
    bm.reference(h)  # refcount 1 → 2
    bm.free(h)       # 2 → 1：不得回池
    assert bm.free_blocks == 1
    bm.free(h)       # 1 → 0：回池
    assert bm.free_blocks == 2


def test_stale_handle_detected():
    """§12-1 核心防御：对已释放 handle 的操作必须报 StaleHandle 错误。"""
    bm = ze.BlockManager(2, 16)
    h = bm.allocate()
    bm.free(h)
    with pytest.raises(RuntimeError, match="Stale handle"):
        bm.free(h)
    # handle 已过期：reference 同样被拒
    with pytest.raises(RuntimeError, match="Stale handle"):
        bm.reference(h)


def test_stale_handle_after_reallocation():
    """block 被重新分配后，旧 handle 依然过期（generation 门控）。"""
    bm = ze.BlockManager(1, 16)
    h1 = bm.allocate()
    bm.free(h1)
    h2 = bm.allocate()  # 拿到同一 block（新 generation）
    assert h2.generation > h1.generation
    with pytest.raises(RuntimeError, match="Stale handle"):
        bm.free(h1)


def test_out_of_blocks_error():
    bm = ze.BlockManager(1, 16)
    bm.allocate()
    with pytest.raises(RuntimeError, match="Out of blocks"):
        bm.allocate()


def test_invalid_block_size_error():
    with pytest.raises(RuntimeError, match="Invalid block size"):
        ze.BlockManager(4, 0)


def test_errors_convert_to_runtime_error():
    """错误转换：所有 ZealotError 变体在 Python 面均为 RuntimeError（含详情）。"""
    bm = ze.BlockManager(1, 16)
    h = bm.allocate()
    bm.free(h)
    try:
        bm.free(h)
    except RuntimeError as e:
        assert "Stale handle" in str(e)
        assert "block 0" in str(e)
    else:
        pytest.fail("expected RuntimeError")


def test_large_pool_roundtrip():
    bm = ze.BlockManager(1024, 16)
    handles = [bm.allocate() for _ in range(1024)]
    assert bm.free_blocks == 0
    ids = {h.block_id for h in handles}
    assert len(ids) == 1024  # 无重复分配
    for h in handles:
        bm.free(h)
    assert bm.free_blocks == 1024
