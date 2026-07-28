//! Parallel Ethereum block execution using grevm.

use crate::{EthEvmConfig, RethReceiptBuilder};
use alloc::{boxed::Box, sync::Arc, vec::Vec};
use alloy_consensus::{BlockHeader, Transaction};
use alloy_eips::eip7685::Requests;
use alloy_evm::{
    block::{
        state_changes::post_block_balance_increments, BlockExecutor, BlockValidationError,
        SystemCaller,
    },
    eth::{eip6110, spec::EthExecutorSpec, EthBlockExecutorFactory},
    precompiles::PrecompilesMap,
    Evm, EvmEnv, EvmFactory as AlloyEvmFactory,
};
use core::fmt::Debug;
use grevm::{DelegatedSafetyConfig, GrevmConfig, ParallelState, Scheduler, TxExecutionOutcome};
use reth_chainspec::{EthChainSpec, EthereumHardfork, Hardforks};
use reth_ethereum_primitives::{EthPrimitives, Receipt};
use reth_evm::{
    execute::{BlockExecutionError, Executor},
    ConfigureEvm, OnStateHook, ParallelDatabase,
};
use reth_execution_types::BlockExecutionResult;
use reth_primitives_traits::{NodePrimitives, RecoveredBlock, SignedTransaction};
use revm::{
    context::{result::EVMError, BlockEnv, TxEnv},
    database::{states::bundle_state::BundleRetention, State, WrapDatabaseRef},
    primitives::hardfork::SpecId,
    Database, DatabaseRef,
};

struct SharedDatabase<DB>(parking_lot::Mutex<DB>);

impl<DB> SharedDatabase<DB> {
    const fn new(db: DB) -> Self {
        Self(parking_lot::Mutex::new(db))
    }

    fn into_inner(self) -> DB {
        self.0.into_inner()
    }
}

impl<DB> Debug for SharedDatabase<DB> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("SharedDatabase").finish_non_exhaustive()
    }
}

impl<DB: Database + Send> DatabaseRef for SharedDatabase<DB> {
    type Error = DB::Error;

    fn basic_ref(
        &self,
        address: alloy_primitives::Address,
    ) -> Result<Option<revm::state::AccountInfo>, Self::Error> {
        self.0.lock().basic(address)
    }

    fn code_by_hash_ref(
        &self,
        code_hash: alloy_primitives::B256,
    ) -> Result<revm::state::Bytecode, Self::Error> {
        self.0.lock().code_by_hash(code_hash)
    }

    fn storage_ref(
        &self,
        address: alloy_primitives::Address,
        index: alloy_primitives::U256,
    ) -> Result<alloy_primitives::U256, Self::Error> {
        self.0.lock().storage(address, index)
    }

    fn block_hash_ref(&self, number: u64) -> Result<alloy_primitives::B256, Self::Error> {
        self.0.lock().block_hash(number)
    }
}

/// Ethereum block executor backed by grevm.
#[expect(missing_debug_implementations)]
pub struct GrevmExecutor<DB, ChainSpec, EvmFactory> {
    evm_config: EthEvmConfig<ChainSpec, EvmFactory>,
    state: Option<ExecutionState<DB>>,
    grevm_config: GrevmConfig,
}

enum ExecutionState<DB> {
    Parallel(ParallelState<SharedDatabase<DB>>),
    Sequential(State<DB>),
}

impl<DB, ChainSpec, EvmFactory> GrevmExecutor<DB, ChainSpec, EvmFactory>
where
    DB: ParallelDatabase,
    ChainSpec: EthExecutorSpec + EthChainSpec + Hardforks + 'static,
    EvmFactory: AlloyEvmFactory<
            Tx = TxEnv,
            Spec = SpecId,
            BlockEnv = BlockEnv,
            Precompiles = PrecompilesMap,
        > + Clone
        + Debug
        + Send
        + Sync
        + Unpin
        + 'static,
    EthEvmConfig<ChainSpec, EvmFactory>: ConfigureEvm<
        Primitives = EthPrimitives,
        BlockExecutorFactory = EthBlockExecutorFactory<
            RethReceiptBuilder,
            Arc<ChainSpec>,
            EvmFactory,
        >,
    >,
{
    /// Creates a block executor using grevm's environment-configurable scheduler settings.
    pub fn new(evm_config: EthEvmConfig<ChainSpec, EvmFactory>, db: DB) -> Self {
        let grevm_config =
            GrevmConfig::from_env().with_delegated_safety(DelegatedSafetyConfig::disabled());
        Self {
            evm_config,
            state: Some(ExecutionState::Parallel(ParallelState::new(
                SharedDatabase::new(db),
                true,
                false,
            ))),
            grevm_config,
        }
    }

    fn parallel_state_mut(&mut self) -> &mut ParallelState<SharedDatabase<DB>> {
        match self.state.as_mut().expect("executor state is always present") {
            ExecutionState::Parallel(state) => state,
            ExecutionState::Sequential(_) => unreachable!("executor is in sequential mode"),
        }
    }

    fn switch_to_sequential(&mut self) {
        let state = self.state.take().expect("executor state is always present");
        self.state = Some(match state {
            ExecutionState::Parallel(mut state) => {
                let bundle = state.take_bundle();
                let state = State::builder()
                    .with_database(state.database.into_inner())
                    .with_bundle_prestate(bundle)
                    .with_bundle_update()
                    .build();
                ExecutionState::Sequential(state)
            }
            state @ ExecutionState::Sequential(_) => state,
        });
    }

    fn execute_sequential(
        &mut self,
        block: &RecoveredBlock<<EthPrimitives as NodePrimitives>::Block>,
        state_hook: Option<Box<dyn OnStateHook>>,
    ) -> Result<BlockExecutionResult<Receipt>, BlockExecutionError> {
        self.switch_to_sequential();
        let ExecutionState::Sequential(state) =
            self.state.as_mut().expect("executor state is always present")
        else {
            unreachable!("executor was switched to sequential mode")
        };

        let mut executor =
            self.evm_config.executor_for_block(state, block).map_err(BlockExecutionError::other)?;
        executor.evm_mut().db_mut().set_state_hook(state_hook);
        let result = executor.execute_block(block.transactions_recovered());
        state.set_state_hook(None);
        state.merge_transitions(BundleRetention::Reverts);
        result
    }

    fn apply_pre_execution_changes(
        &mut self,
        block: &RecoveredBlock<<EthPrimitives as NodePrimitives>::Block>,
    ) -> Result<(), BlockExecutionError> {
        let evm_config = self.evm_config.clone();
        let mut system_caller = SystemCaller::new(evm_config.chain_spec().clone());
        let mut evm = evm_config
            .evm_for_block(self.parallel_state_mut(), block.header())
            .map_err(BlockExecutionError::other)?;
        system_caller.apply_pre_execution_changes(block.header(), &mut evm)
    }

    fn execute_transactions(
        &mut self,
        block: &RecoveredBlock<<EthPrimitives as NodePrimitives>::Block>,
    ) -> Result<(Vec<Receipt>, u64, u64), BlockExecutionError> {
        let evm_env =
            self.evm_config.evm_env(block.header()).map_err(BlockExecutionError::other)?;
        let block_gas_limit = block.header().gas_limit();
        let tx_gas_limit_cap = evm_env.cfg_env.tx_gas_limit_cap;
        for transaction in block.body().transactions() {
            let transaction_gas_limit = transaction.gas_limit();
            let max_tx_gas_usage = tx_gas_limit_cap
                .map_or(transaction_gas_limit, |cap| cap.min(transaction_gas_limit));
            if max_tx_gas_usage > block_gas_limit {
                return Err(BlockValidationError::TransactionGasLimitMoreThanAvailableBlockGas {
                    transaction_gas_limit,
                    block_available_gas: block_gas_limit,
                }
                .into())
            }
        }

        let txs = Arc::new(
            block
                .transactions_recovered()
                .map(|tx| self.evm_config.tx_env(tx))
                .collect::<Vec<TxEnv>>(),
        );

        let custom_precompiles = {
            let evm_config = self.evm_config.clone();
            let evm = evm_config
                .evm_with_env(WrapDatabaseRef(&*self.parallel_state_mut()), evm_env.clone());
            Arc::new(
                evm.precompiles()
                    .clone()
                    .into_dyn_precompiles()
                    .into_precompiles()
                    .collect::<Vec<_>>(),
            )
        };

        let state = match self.state.take().expect("executor state is always present") {
            ExecutionState::Parallel(state) => state,
            ExecutionState::Sequential(_) => unreachable!("executor is in parallel mode"),
        };
        let EvmEnv { cfg_env, block_env } = evm_env;
        let scheduler = Scheduler::new_with_runtime_config(
            cfg_env,
            block_env,
            txs,
            state,
            Some(custom_precompiles),
            self.grevm_config.clone(),
        );
        let execution_result = scheduler.execute();
        let (outcomes, state) = scheduler.take_result_and_state();
        self.state = Some(ExecutionState::Parallel(state));

        if let Err(error) = execution_result {
            let hash = block
                .transactions_with_sender()
                .nth(error.txid)
                .map(|(_, tx)| tx.recalculate_hash())
                .unwrap_or_default();
            return Err(BlockExecutionError::evm(error.error, hash))
        }

        if outcomes.len() != block.transaction_count() {
            return Err(BlockExecutionError::msg(alloc::format!(
                "grevm outcome count mismatch: got {}, expected {}",
                outcomes.len(),
                block.transaction_count()
            )))
        }

        let mut receipts = Vec::with_capacity(outcomes.len());
        let mut cumulative_gas_used = 0u64;
        let mut blob_gas_used = 0u64;
        let is_cancun_active =
            self.evm_config.chain_spec().is_cancun_active_at_timestamp(block.timestamp());
        for (outcome, transaction) in outcomes.into_iter().zip(block.body().transactions()) {
            let block_available_gas =
                block.header().gas_limit().saturating_sub(cumulative_gas_used);
            let transaction_gas_limit = transaction.gas_limit();
            let max_tx_gas_usage = tx_gas_limit_cap
                .map_or(transaction_gas_limit, |cap| cap.min(transaction_gas_limit));
            if max_tx_gas_usage > block_available_gas {
                return Err(BlockValidationError::TransactionGasLimitMoreThanAvailableBlockGas {
                    transaction_gas_limit,
                    block_available_gas,
                }
                .into())
            }

            let result = match outcome {
                TxExecutionOutcome::Executed(result) => result,
                TxExecutionOutcome::Skipped(error) => {
                    return Err(BlockExecutionError::evm(
                        EVMError::<<DB as revm::Database>::Error>::Transaction(error),
                        transaction.recalculate_hash(),
                    ))
                }
            };

            cumulative_gas_used = cumulative_gas_used.saturating_add(result.tx_gas_used());
            if is_cancun_active {
                blob_gas_used =
                    blob_gas_used.saturating_add(transaction.blob_gas_used().unwrap_or_default());
            }
            receipts.push(Receipt {
                tx_type: transaction.tx_type(),
                success: result.is_success(),
                cumulative_gas_used,
                logs: result.into_logs(),
            });
        }

        Ok((receipts, cumulative_gas_used, blob_gas_used))
    }

    fn apply_post_execution_changes(
        &mut self,
        block: &RecoveredBlock<<EthPrimitives as NodePrimitives>::Block>,
        receipts: &[Receipt],
    ) -> Result<Requests, BlockExecutionError> {
        let evm_config = self.evm_config.clone();
        let chain_spec = evm_config.chain_spec().clone();
        let mut system_caller = SystemCaller::new(chain_spec.clone());
        let mut evm = evm_config
            .evm_for_block(self.parallel_state_mut(), block.header())
            .map_err(BlockExecutionError::other)?;

        let requests = if chain_spec.is_prague_active_at_timestamp(block.timestamp()) {
            let deposit_requests = eip6110::parse_deposits_from_receipts(&chain_spec, receipts)?;
            let mut requests = Requests::default();
            if !deposit_requests.is_empty() {
                requests.push_request_with_type(eip6110::DEPOSIT_REQUEST_TYPE, deposit_requests);
            }
            system_caller.append_post_execution_changes(&mut evm, &mut requests)?;
            requests
        } else {
            Requests::default()
        };

        let balance_increments = post_block_balance_increments(
            &chain_spec,
            evm.block(),
            &block.body().ommers,
            block.body().withdrawals.as_deref().map(Vec::as_slice),
        );
        if !balance_increments.is_empty() {
            evm.db_mut()
                .increment_balances(balance_increments)
                .map_err(|_| BlockValidationError::IncrementBalanceFailed)?;
        }

        Ok(requests)
    }

    fn execute_parallel(
        &mut self,
        block: &RecoveredBlock<<EthPrimitives as NodePrimitives>::Block>,
    ) -> Result<BlockExecutionResult<Receipt>, BlockExecutionError> {
        self.apply_pre_execution_changes(block)?;
        let (receipts, gas_used, blob_gas_used) = if block.transaction_count() == 0 {
            (Vec::new(), 0, 0)
        } else {
            self.execute_transactions(block)?
        };
        let requests = self.apply_post_execution_changes(block, &receipts)?;
        self.parallel_state_mut().merge_transitions(BundleRetention::Reverts);
        Ok(BlockExecutionResult { receipts, requests, gas_used, blob_gas_used })
    }
}

impl<DB, ChainSpec, EvmFactory> Executor<DB> for GrevmExecutor<DB, ChainSpec, EvmFactory>
where
    DB: ParallelDatabase,
    ChainSpec: EthExecutorSpec + EthChainSpec + Hardforks + 'static,
    EvmFactory: AlloyEvmFactory<
            Tx = TxEnv,
            Spec = SpecId,
            BlockEnv = BlockEnv,
            Precompiles = PrecompilesMap,
        > + Clone
        + Debug
        + Send
        + Sync
        + Unpin
        + 'static,
    EthEvmConfig<ChainSpec, EvmFactory>: ConfigureEvm<
        Primitives = EthPrimitives,
        BlockExecutorFactory = EthBlockExecutorFactory<
            RethReceiptBuilder,
            Arc<ChainSpec>,
            EvmFactory,
        >,
    >,
{
    type Primitives = EthPrimitives;
    type Error = BlockExecutionError;

    fn execute_one(
        &mut self,
        block: &RecoveredBlock<<Self::Primitives as NodePrimitives>::Block>,
    ) -> Result<BlockExecutionResult<Receipt>, Self::Error> {
        let evm_env =
            self.evm_config.evm_env(block.header()).map_err(BlockExecutionError::other)?;
        let is_dao_transition = self
            .evm_config
            .chain_spec()
            .fork(EthereumHardfork::Dao)
            .transitions_at_block(block.number());
        // Grevm materializes missing accounts while draining DAO balances. Keep the one-off
        // migration on revm so dev chains and historical replay produce the canonical bundle.
        // Amsterdam also stays on revm until grevm exposes state-gas and BAL accounting.
        if evm_env.cfg_env.enable_amsterdam_eip8037 ||
            is_dao_transition ||
            matches!(self.state, Some(ExecutionState::Sequential(_)))
        {
            self.execute_sequential(block, None)
        } else {
            self.execute_parallel(block)
        }
    }

    fn execute_one_with_state_hook<H>(
        &mut self,
        block: &RecoveredBlock<<Self::Primitives as NodePrimitives>::Block>,
        state_hook: H,
    ) -> Result<BlockExecutionResult<Receipt>, Self::Error>
    where
        H: OnStateHook + 'static,
    {
        self.execute_sequential(block, Some(Box::new(state_hook)))
    }

    fn into_state(mut self) -> State<DB> {
        self.switch_to_sequential();
        match self.state.expect("executor state is always present") {
            ExecutionState::Sequential(state) => state,
            ExecutionState::Parallel(_) => unreachable!("executor was switched to sequential mode"),
        }
    }

    fn size_hint(&self) -> usize {
        match self.state.as_ref().expect("executor state is always present") {
            ExecutionState::Parallel(state) => state.bundle_size_hint(),
            ExecutionState::Sequential(state) => state.bundle_size_hint(),
        }
    }

    fn take_bal(&mut self) -> Option<alloy_eip7928::BlockAccessList> {
        match self.state.as_mut().expect("executor state is always present") {
            ExecutionState::Parallel(_) => None,
            ExecutionState::Sequential(state) => state.take_built_alloy_bal(),
        }
    }
}
