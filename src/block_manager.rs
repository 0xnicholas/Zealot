use pyo3::prelude::*;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::error::ZealotError;

#[pyclass]
#[derive(Clone, Debug)]
pub struct BlockHandle {
    #[pyo3(get)]
    pub block_id: usize,

    #[pyo3(get)]
    pub generation: u64,
}

#[pymethods]
impl BlockHandle {
    #[new]
    fn new(block_id: usize, generation: u64) -> Self {
        Self { block_id, generation }
    }

    fn __repr__(&self) -> String {
        format!("BlockHandle(id={}, gen={})", self.block_id, self.generation)
    }
}

#[pyclass]
#[derive(Debug)]
pub struct BlockManager {
    num_gpu_blocks: usize,
    block_size: usize,
    free_list: Mutex<Vec<usize>>,
    refcount: Vec<AtomicUsize>,
    generation: Vec<AtomicU64>,
}

#[pymethods]
impl BlockManager {
    #[new]
    fn new(num_gpu_blocks: usize, block_size: usize) -> PyResult<Self> {
        if block_size == 0 {
            return Err(ZealotError::InvalidBlockSize(block_size).into());
        }

        let mut free_list: Vec<usize> = (0..num_gpu_blocks).collect();
        // LIFO for cache locality: pop from end gives recently-freed blocks
        free_list.reverse();

        Ok(Self {
            num_gpu_blocks,
            block_size,
            free_list: Mutex::new(free_list),
            refcount: (0..num_gpu_blocks).map(|_| AtomicUsize::new(0)).collect(),
            generation: (0..num_gpu_blocks).map(|_| AtomicU64::new(1)).collect(),
        })
    }

    fn allocate(&self) -> PyResult<BlockHandle> {
        let mut free = self.free_list.lock().unwrap();
        let block_id = free.pop().ok_or(ZealotError::OutOfBlocks {
            requested: 1,
            available: free.len(),
        })?;
        drop(free);

        let gen = self.generation[block_id].load(Ordering::Acquire);
        self.refcount[block_id].store(1, Ordering::Release);
        Ok(BlockHandle {
            block_id,
            generation: gen,
        })
    }

    fn free(&self, handle: &BlockHandle) -> PyResult<()> {
        self.check_handle(handle)?;

        let prev = self.refcount[handle.block_id].fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            self.generation[handle.block_id].fetch_add(1, Ordering::AcqRel);
            let mut free = self.free_list.lock().unwrap();
            free.push(handle.block_id);
        }
        Ok(())
    }

    fn reference(&self, handle: &BlockHandle) -> PyResult<()> {
        self.check_handle(handle)?;
        self.refcount[handle.block_id].fetch_add(1, Ordering::AcqRel);
        Ok(())
    }

    #[getter]
    fn free_blocks(&self) -> PyResult<usize> {
        Ok(self.free_list.lock().unwrap().len())
    }

    #[getter]
    fn num_gpu_blocks(&self) -> usize {
        self.num_gpu_blocks
    }

    #[getter]
    fn block_size(&self) -> usize {
        self.block_size
    }
}

impl BlockManager {
    fn check_handle(&self, handle: &BlockHandle) -> PyResult<()> {
        if handle.block_id >= self.num_gpu_blocks {
            return Err(ZealotError::BlockNotAllocated(handle.block_id).into());
        }
        let current_gen = self.generation[handle.block_id].load(Ordering::Acquire);
        if handle.generation != current_gen {
            return Err(ZealotError::StaleHandle {
                block_id: handle.block_id,
                handle_gen: handle.generation,
                current_gen,
            }
            .into());
        }
        Ok(())
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BlockManager>()?;
    m.add_class::<BlockHandle>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocate_and_free() {
        let bm = BlockManager::new(4, 16).unwrap();
        assert_eq!(bm.free_blocks().unwrap(), 4);

        let h1 = bm.allocate().unwrap();
        assert_eq!(bm.free_blocks().unwrap(), 3);

        let h2 = bm.allocate().unwrap();
        assert_eq!(bm.free_blocks().unwrap(), 2);

        bm.free(&h2).unwrap();
        assert_eq!(bm.free_blocks().unwrap(), 3);

        bm.free(&h1).unwrap();
        assert_eq!(bm.free_blocks().unwrap(), 4);
    }

    #[test]
    fn test_reference_counting() {
        let bm = BlockManager::new(2, 16).unwrap();
        let h = bm.allocate().unwrap();

        bm.reference(&h).unwrap();
        // free once — refcount goes 2→1, block stays allocated
        bm.free(&h).unwrap();
        assert_eq!(bm.free_blocks().unwrap(), 1);

        // free again — refcount goes 1→0, block returns to pool
        bm.free(&h).unwrap();
        assert_eq!(bm.free_blocks().unwrap(), 2);
    }

    #[test]
    fn test_stale_handle() {
        let bm = BlockManager::new(2, 16).unwrap();
        let h = bm.allocate().unwrap();
        bm.free(&h).unwrap();

        // handle is stale after free
        let err = bm.free(&h).unwrap_err();
        assert!(err.to_string().contains("Stale handle"));
    }

    #[test]
    fn test_out_of_blocks() {
        let bm = BlockManager::new(1, 16).unwrap();
        let _h = bm.allocate().unwrap();
        let err = bm.allocate().unwrap_err();
        assert!(err.to_string().contains("Out of blocks"));
    }

    #[test]
    fn test_invalid_block_size() {
        let err = BlockManager::new(4, 0).unwrap_err();
        assert!(err.to_string().contains("Invalid block size"));
    }

    #[test]
    fn test_reuse_after_free() {
        let bm = BlockManager::new(2, 16).unwrap();
        let h1 = bm.allocate().unwrap();
        let id1 = h1.block_id;
        bm.free(&h1).unwrap();

        let h2 = bm.allocate().unwrap();
        // Should get the same block_id back (LIFO free list)
        assert_eq!(h2.block_id, id1);
        // But generation should be different
        assert!(h2.generation > h1.generation);
    }
}
