// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use anyhow::Result;
use dynamo_kv_router::config::KvRouterConfig;

use super::online;
use super::validate::{
    validate_offline_concurrency_args, validate_offline_disagg_concurrency_args,
    validate_offline_disagg_replay_args, validate_offline_replay_args,
    validate_online_concurrency_args, validate_online_replay_args,
};
use super::{
    OfflineDisaggReplayConfig, ReplayPrefillLoadEstimator, ReplayRouterMode, TraceSimulationReport,
};
use crate::common::perf_model::{
    ReplayDecodeLatencyModel, ReplayLatencyModel, ReplayPrefillLatencyModel,
};
use crate::common::protocols::MockEngineArgs;
use crate::loadgen::Trace;

pub fn simulate_trace_workload(
    args: MockEngineArgs,
    trace: Trace,
    num_workers: usize,
) -> Result<TraceSimulationReport> {
    simulate_trace_workload_with_router_mode(
        args,
        None,
        None,
        trace,
        num_workers,
        ReplayRouterMode::RoundRobin,
    )
}

pub fn simulate_trace_workload_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let latency_model = Arc::clone(&args.perf_model);
    simulate_trace_workload_with_latency_model(
        latency_model,
        args,
        router_config,
        prefill_load_estimator,
        trace,
        num_workers,
        router_mode,
    )
}

#[allow(clippy::too_many_arguments)]
/// Simulate a workload using a caller-supplied aggregated latency model.
pub fn simulate_trace_workload_with_latency_model<M: ReplayLatencyModel>(
    latency_model: Arc<M>,
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_offline_replay_args(&args, num_workers, router_mode)?;
    crate::replay::offline::simulate_trace_workload_with_latency_model(
        args,
        latency_model,
        router_config,
        prefill_load_estimator,
        trace,
        num_workers,
        router_mode,
        false,
        None,
    )
}

pub fn simulate_trace_workload_disagg_with_router_mode(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let prefill_latency_model = Arc::clone(&config.prefill_args.perf_model);
    let decode_latency_model = Arc::clone(&config.decode_args.perf_model);
    simulate_trace_workload_disagg_with_latency_models(
        prefill_latency_model,
        decode_latency_model,
        config,
        router_config,
        prefill_load_estimator,
        trace,
        router_mode,
    )
}

#[allow(clippy::too_many_arguments)]
/// Simulate a disaggregated workload with independent stage latency models.
pub fn simulate_trace_workload_disagg_with_latency_models<
    P: ReplayPrefillLatencyModel,
    D: ReplayDecodeLatencyModel,
>(
    prefill_latency_model: Arc<P>,
    decode_latency_model: Arc<D>,
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    validate_offline_disagg_replay_args(&config, router_mode)?;
    crate::replay::offline::simulate_trace_workload_disagg_with_latency_models(
        config,
        prefill_latency_model,
        decode_latency_model,
        router_config,
        prefill_load_estimator,
        trace,
        router_mode,
        false,
        None,
    )
}

pub fn simulate_trace_live_workload(
    args: MockEngineArgs,
    trace: Trace,
    num_workers: usize,
) -> Result<TraceSimulationReport> {
    simulate_trace_live_workload_with_router_mode(
        args,
        None,
        None,
        trace,
        num_workers,
        ReplayRouterMode::RoundRobin,
    )
}

pub fn simulate_trace_live_workload_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_online_replay_args(&args, num_workers)?;
    online::simulate_trace_workload(
        args,
        router_config,
        prefill_load_estimator,
        trace,
        num_workers,
        router_mode,
    )
}

pub fn simulate_concurrency_workload(
    args: MockEngineArgs,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
) -> Result<TraceSimulationReport> {
    simulate_concurrency_workload_with_router_mode(
        args,
        None,
        None,
        trace,
        max_in_flight,
        num_workers,
        ReplayRouterMode::RoundRobin,
    )
}

pub fn simulate_concurrency_workload_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let latency_model = Arc::clone(&args.perf_model);
    simulate_concurrency_workload_with_latency_model(
        latency_model,
        args,
        router_config,
        prefill_load_estimator,
        trace,
        max_in_flight,
        num_workers,
        router_mode,
    )
}

#[allow(clippy::too_many_arguments)]
/// Simulate a fixed-concurrency workload using a caller-supplied aggregated model.
pub fn simulate_concurrency_workload_with_latency_model<M: ReplayLatencyModel>(
    latency_model: Arc<M>,
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_offline_concurrency_args(&args, num_workers, max_in_flight, router_mode)?;
    crate::replay::offline::simulate_concurrency_workload_with_latency_model(
        args,
        latency_model,
        router_config,
        prefill_load_estimator,
        trace,
        max_in_flight,
        num_workers,
        router_mode,
        false,
        None,
    )
}

pub fn simulate_concurrency_workload_disagg_with_router_mode(
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let prefill_latency_model = Arc::clone(&config.prefill_args.perf_model);
    let decode_latency_model = Arc::clone(&config.decode_args.perf_model);
    simulate_concurrency_workload_disagg_with_latency_models(
        prefill_latency_model,
        decode_latency_model,
        config,
        router_config,
        prefill_load_estimator,
        trace,
        max_in_flight,
        router_mode,
    )
}

#[allow(clippy::too_many_arguments)]
/// Simulate a fixed-concurrency disaggregated workload with independent stage models.
pub fn simulate_concurrency_workload_disagg_with_latency_models<
    P: ReplayPrefillLatencyModel,
    D: ReplayDecodeLatencyModel,
>(
    prefill_latency_model: Arc<P>,
    decode_latency_model: Arc<D>,
    config: OfflineDisaggReplayConfig,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let config = config.normalized()?;
    validate_offline_disagg_concurrency_args(&config, max_in_flight, router_mode)?;
    crate::replay::offline::simulate_concurrency_workload_disagg_with_latency_models(
        config,
        prefill_latency_model,
        decode_latency_model,
        router_config,
        prefill_load_estimator,
        trace,
        max_in_flight,
        router_mode,
        false,
        None,
    )
}

pub fn simulate_concurrency_live_workload(
    args: MockEngineArgs,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
) -> Result<TraceSimulationReport> {
    simulate_concurrency_live_workload_with_router_mode(
        args,
        None,
        None,
        trace,
        max_in_flight,
        num_workers,
        ReplayRouterMode::RoundRobin,
    )
}

pub fn simulate_concurrency_live_workload_with_router_mode(
    args: MockEngineArgs,
    router_config: Option<KvRouterConfig>,
    prefill_load_estimator: Option<ReplayPrefillLoadEstimator>,
    trace: Trace,
    max_in_flight: usize,
    num_workers: usize,
    router_mode: ReplayRouterMode,
) -> Result<TraceSimulationReport> {
    let args = args.normalized()?;
    validate_online_concurrency_args(&args, num_workers, max_in_flight)?;
    online::simulate_concurrency_workload(
        args,
        router_config,
        prefill_load_estimator,
        trace,
        max_in_flight,
        num_workers,
        router_mode,
    )
}
