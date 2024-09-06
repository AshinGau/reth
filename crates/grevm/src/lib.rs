use lazy_static::lazy_static;
use revm_primitives::{Address, U256};
use tokio::runtime::{Builder, Runtime};

mod storage;
mod scheduler;
mod partition;

lazy_static! {
    static ref TK_RUNTIME: Runtime = Builder::new_multi_thread()
        .worker_threads(std::thread::available_parallelism()
            .map(|n| n.get() * 2)
            .unwrap_or(8))
        .enable_all()
        .build()
        .unwrap();
}

static MAX_NUM_ROUND: usize = 3;

type PartitionId = usize;

type TxId = usize;

#[derive(PartialEq)]
enum LocationAndType {
    Basic(Address),
    Storage(Address, U256),
    Code(Address),
}
