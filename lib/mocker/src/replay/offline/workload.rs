// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use dynamo_kv_router::config::KvRouterConfig;

use super::agg::{AggRuntime, ReplayMode as AggReplayMode};
use super::disagg::{DisaggRuntime, ReplayMode as DisaggReplayMode};
use super::single::{SingleReplayMode, SingleRuntime};
use super::{finish_with_replay_wall_time, use_single_runtime};
use crate::common::perf_model::{
    ReplayDecodeLatencyModel, ReplayLatencyModel, ReplayPrefillLatencyModel,
};
use crate::common::protocols::MockEngineArgs;
use crate::loadgen::{AgenticTrace, Trace, WorkloadDriver};
use crate::replay::OfflineDisaggReplayConfig;
use crate::replay::{ReplayPrefillLoadEstimator, ReplayRouterMode, TraceSimulationReport};

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_workload(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
) -> Result<TraceSimulationReport> {
    let latency_model = Arc::clone(&args.perf_model);
    simulate_trace_workload_with_latency_model(
        args,
        latency_model,
        router_config,
        prefill_load_estimator,
        trace,
        num_workers,
        router_mode,
        record_per_request,
        max_sim_time_ms,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_workload_with_latency_model<M: ReplayLatencyModel>(
    args: MockEngineArgs,
    latency_model: Arc<M>,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
) -> Result<TraceSimulationReport> {
    simulate_trace_workload_with_latency_model_and_delta_mode(
        args,
        latency_model,
        router_config,
        prefill_load_estimator,
        trace,
        num_workers,
        router_mode,
        false,
        record_per_request,
        max_sim_time_ms,
    )
}

pub(crate) fn simulate_agentic_trace_workload(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: AgenticTrace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let latency_model = Arc::clone(&args.perf_model);
    simulate_agentic_trace_workload_with_latency_model(
        args,
        latency_model,
        router_config,
        prefill_load_estimator,
        trace,
        num_workers,
        router_mode,
    )
}

pub(crate) fn simulate_agentic_trace_workload_with_latency_model<M: ReplayLatencyModel>(
    args: MockEngineArgs,
    latency_model: Arc<M>,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: AgenticTrace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    if use_single_runtime(num_workers, router_mode) {
        simulate_agentic_trace_workload_single_with_latency_model(args, latency_model, trace)
    } else {
        simulate_agentic_trace_workload_multi_with_latency_model(
            args,
            latency_model,
            router_config,
            prefill_load_estimator,
            trace,
            num_workers,
            router_mode,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_workload_accumulating_deltas(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
) -> Result<TraceSimulationReport> {
    let latency_model = Arc::clone(&args.perf_model);
    simulate_trace_workload_accumulating_deltas_with_latency_model(
        args,
        latency_model,
        router_config,
        prefill_load_estimator,
        trace,
        num_workers,
        router_mode,
        record_per_request,
        max_sim_time_ms,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_workload_accumulating_deltas_with_latency_model<
    M: ReplayLatencyModel,
>(
    args: MockEngineArgs,
    latency_model: Arc<M>,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
) -> Result<TraceSimulationReport> {
    simulate_trace_workload_with_latency_model_and_delta_mode(
        args,
        latency_model,
        router_config,
        prefill_load_estimator,
        trace,
        num_workers,
        router_mode,
        true,
        record_per_request,
        max_sim_time_ms,
    )
}

#[allow(clippy::too_many_arguments)]
fn simulate_trace_workload_with_latency_model_and_delta_mode<M: ReplayLatencyModel>(
    args: MockEngineArgs,
    latency_model: Arc<M>,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    accumulate_session_deltas: bool,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
) -> Result<TraceSimulationReport> {
    if use_single_runtime(num_workers, router_mode) {
        simulate_trace_workload_single_with_latency_model(
            args,
            latency_model,
            trace,
            accumulate_session_deltas,
            record_per_request,
            max_sim_time_ms,
        )
    } else {
        simulate_trace_workload_multi_with_latency_model(
            args,
            latency_model,
            router_config,
            prefill_load_estimator,
            trace,
            num_workers,
            router_mode,
            accumulate_session_deltas,
            record_per_request,
            max_sim_time_ms,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_concurrency_workload(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
) -> Result<TraceSimulationReport> {
    let latency_model = Arc::clone(&args.perf_model);
    simulate_concurrency_workload_with_latency_model(
        args,
        latency_model,
        router_config,
        prefill_load_estimator,
        trace,
        max_in_flight,
        num_workers,
        router_mode,
        record_per_request,
        max_sim_time_ms,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_concurrency_workload_with_latency_model<M: ReplayLatencyModel>(
    args: MockEngineArgs,
    latency_model: Arc<M>,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
) -> Result<TraceSimulationReport> {
    simulate_concurrency_workload_with_latency_model_and_delta_mode(
        args,
        latency_model,
        router_config,
        prefill_load_estimator,
        trace,
        max_in_flight,
        num_workers,
        router_mode,
        false,
        record_per_request,
        max_sim_time_ms,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_concurrency_workload_accumulating_deltas(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
) -> Result<TraceSimulationReport> {
    let latency_model = Arc::clone(&args.perf_model);
    simulate_concurrency_workload_accumulating_deltas_with_latency_model(
        args,
        latency_model,
        router_config,
        prefill_load_estimator,
        trace,
        max_in_flight,
        num_workers,
        router_mode,
        record_per_request,
        max_sim_time_ms,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_concurrency_workload_accumulating_deltas_with_latency_model<
    M: ReplayLatencyModel,
>(
    args: MockEngineArgs,
    latency_model: Arc<M>,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
) -> Result<TraceSimulationReport> {
    simulate_concurrency_workload_with_latency_model_and_delta_mode(
        args,
        latency_model,
        router_config,
        prefill_load_estimator,
        trace,
        max_in_flight,
        num_workers,
        router_mode,
        true,
        record_per_request,
        max_sim_time_ms,
    )
}

#[allow(clippy::too_many_arguments)]
fn simulate_concurrency_workload_with_latency_model_and_delta_mode<M: ReplayLatencyModel>(
    args: MockEngineArgs,
    latency_model: Arc<M>,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    accumulate_session_deltas: bool,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
) -> Result<TraceSimulationReport> {
    if use_single_runtime(num_workers, router_mode) {
        simulate_concurrency_workload_single_with_latency_model(
            args,
            latency_model,
            trace,
            max_in_flight,
            accumulate_session_deltas,
            record_per_request,
            max_sim_time_ms,
        )
    } else {
        simulate_concurrency_workload_multi_with_latency_model(
            args,
            latency_model,
            router_config,
            prefill_load_estimator,
            trace,
            max_in_flight,
            num_workers,
            router_mode,
            accumulate_session_deltas,
            record_per_request,
            max_sim_time_ms,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_workload_disagg(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
) -> Result<TraceSimulationReport> {
    let prefill_latency_model = Arc::clone(&config.prefill_args.perf_model);
    let decode_latency_model = Arc::clone(&config.decode_args.perf_model);
    simulate_trace_workload_disagg_with_latency_models(
        config,
        prefill_latency_model,
        decode_latency_model,
        router_config,
        prefill_load_estimator,
        trace,
        router_mode,
        record_per_request,
        max_sim_time_ms,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_workload_disagg_with_latency_models<
    P: ReplayPrefillLatencyModel,
    D: ReplayDecodeLatencyModel,
>(
    config: OfflineDisaggReplayConfig,
    prefill_latency_model: Arc<P>,
    decode_latency_model: Arc<D>,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
) -> Result<TraceSimulationReport> {
    let started_at = Instant::now();
    let driver = WorkloadDriver::new_trace(trace, config.prefill_args.block_size)?;
    let (collector, _) = DisaggRuntime::new_workload_with_latency_models(
        &config,
        prefill_latency_model,
        decode_latency_model,
        router_config,
        prefill_load_estimator,
        driver,
        DisaggReplayMode::Trace,
        router_mode,
    )?
    .with_per_request_records(record_per_request)
    .with_max_sim_time_ms(max_sim_time_ms)
    .run()?;
    Ok(finish_with_replay_wall_time(collector, started_at))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_concurrency_workload_disagg(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
) -> Result<TraceSimulationReport> {
    let prefill_latency_model = Arc::clone(&config.prefill_args.perf_model);
    let decode_latency_model = Arc::clone(&config.decode_args.perf_model);
    simulate_concurrency_workload_disagg_with_latency_models(
        config,
        prefill_latency_model,
        decode_latency_model,
        router_config,
        prefill_load_estimator,
        trace,
        max_in_flight,
        router_mode,
        record_per_request,
        max_sim_time_ms,
    )
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_concurrency_workload_disagg_with_latency_models<
    P: ReplayPrefillLatencyModel,
    D: ReplayDecodeLatencyModel,
>(
    config: OfflineDisaggReplayConfig,
    prefill_latency_model: Arc<P>,
    decode_latency_model: Arc<D>,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
) -> Result<TraceSimulationReport> {
    let started_at = Instant::now();
    let driver =
        WorkloadDriver::new_concurrency(trace, config.prefill_args.block_size, max_in_flight)?;
    let (collector, _) = DisaggRuntime::new_workload_with_latency_models(
        &config,
        prefill_latency_model,
        decode_latency_model,
        router_config,
        prefill_load_estimator,
        driver,
        DisaggReplayMode::Concurrency { max_in_flight },
        router_mode,
    )?
    .with_per_request_records(record_per_request)
    .with_max_sim_time_ms(max_sim_time_ms)
    .run()?;
    Ok(finish_with_replay_wall_time(collector, started_at))
}

pub(crate) fn simulate_trace_workload_single_with_latency_model<M: ReplayLatencyModel>(
    args: MockEngineArgs,
    latency_model: Arc<M>,
    trace: Trace,
    accumulate_session_deltas: bool,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
) -> Result<TraceSimulationReport> {
    let started_at = Instant::now();
    let args = args.normalized()?;
    let engine_block_size = args.block_size;
    let driver = if accumulate_session_deltas {
        trace.into_delta_accumulating_trace_driver_with_block_size(engine_block_size)?
    } else {
        trace.into_trace_driver_with_block_size(engine_block_size)?
    };
    let collector = SingleRuntime::new_workload_with_latency_model(
        args,
        latency_model,
        driver,
        SingleReplayMode::Trace,
    )
    .with_per_request_records(record_per_request)
    .with_max_sim_time_ms(max_sim_time_ms)
    .run()?;
    Ok(finish_with_replay_wall_time(collector, started_at))
}

pub(crate) fn simulate_agentic_trace_workload_single_with_latency_model<M: ReplayLatencyModel>(
    args: MockEngineArgs,
    latency_model: Arc<M>,
    trace: AgenticTrace,
) -> Result<TraceSimulationReport> {
    let started_at = Instant::now();
    let args = args.normalized()?;
    let engine_block_size = args.block_size;
    let driver = trace.into_trace_driver_with_block_size(engine_block_size)?;
    let collector = SingleRuntime::new_workload_with_latency_model(
        args,
        latency_model,
        driver,
        SingleReplayMode::Trace,
    )
    .run()?;
    Ok(finish_with_replay_wall_time(collector, started_at))
}

pub(crate) fn simulate_concurrency_workload_single_with_latency_model<M: ReplayLatencyModel>(
    args: MockEngineArgs,
    latency_model: Arc<M>,
    trace: Trace,
    max_in_flight: usize,
    accumulate_session_deltas: bool,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
) -> Result<TraceSimulationReport> {
    let started_at = Instant::now();
    let args = args.normalized()?;
    let engine_block_size = args.block_size;
    let driver = if accumulate_session_deltas {
        trace.into_delta_accumulating_concurrency_driver_with_block_size(
            engine_block_size,
            max_in_flight,
        )?
    } else {
        trace.into_concurrency_driver_with_block_size(engine_block_size, max_in_flight)?
    };
    let collector = SingleRuntime::new_workload_with_latency_model(
        args,
        latency_model,
        driver,
        SingleReplayMode::Concurrency { max_in_flight },
    )
    .with_per_request_records(record_per_request)
    .with_max_sim_time_ms(max_sim_time_ms)
    .run()?;
    Ok(finish_with_replay_wall_time(collector, started_at))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_trace_workload_multi_with_latency_model<M: ReplayLatencyModel>(
    args: MockEngineArgs,
    latency_model: Arc<M>,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    accumulate_session_deltas: bool,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
) -> Result<TraceSimulationReport> {
    let started_at = Instant::now();
    let args = args.normalized()?;
    let driver = if accumulate_session_deltas {
        trace.into_delta_accumulating_trace_driver_with_block_size(args.block_size)?
    } else {
        trace.into_trace_driver_with_block_size(args.block_size)?
    };
    let (collector, _) = AggRuntime::new_workload_with_latency_model(
        &args,
        latency_model,
        router_config,
        prefill_load_estimator,
        driver,
        num_workers,
        AggReplayMode::Trace,
        router_mode,
    )?
    .with_per_request_records(record_per_request)
    .with_max_sim_time_ms(max_sim_time_ms)
    .run()?;
    Ok(finish_with_replay_wall_time(collector, started_at))
}

pub(crate) fn simulate_agentic_trace_workload_multi_with_latency_model<M: ReplayLatencyModel>(
    args: MockEngineArgs,
    latency_model: Arc<M>,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: AgenticTrace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let started_at = Instant::now();
    let args = args.normalized()?;
    let driver = trace.into_trace_driver_with_block_size(args.block_size)?;
    let (collector, _) = AggRuntime::new_workload_with_latency_model(
        &args,
        latency_model,
        router_config,
        prefill_load_estimator,
        driver,
        num_workers,
        AggReplayMode::Trace,
        router_mode,
    )?
    .run()?;
    Ok(finish_with_replay_wall_time(collector, started_at))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn simulate_concurrency_workload_multi_with_latency_model<M: ReplayLatencyModel>(
    args: MockEngineArgs,
    latency_model: Arc<M>,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
    accumulate_session_deltas: bool,
    record_per_request: bool,
    max_sim_time_ms: Option<f64>,
) -> Result<TraceSimulationReport> {
    let started_at = Instant::now();
    let args = args.normalized()?;
    let driver = if accumulate_session_deltas {
        trace.into_delta_accumulating_concurrency_driver_with_block_size(
            args.block_size,
            max_in_flight,
        )?
    } else {
        trace.into_concurrency_driver_with_block_size(args.block_size, max_in_flight)?
    };
    let (collector, _) = AggRuntime::new_workload_with_latency_model(
        &args,
        latency_model,
        router_config,
        prefill_load_estimator,
        driver,
        num_workers,
        AggReplayMode::Concurrency { max_in_flight },
        router_mode,
    )?
    .with_per_request_records(record_per_request)
    .with_max_sim_time_ms(max_sim_time_ms)
    .run()?;
    Ok(finish_with_replay_wall_time(collector, started_at))
}
