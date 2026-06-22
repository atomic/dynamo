// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Instant;

use crate::replay::{ReplayRouterMode, TraceCollector, TraceSimulationReport};

pub(crate) use crate::replay::normalize_trace_requests;

pub(crate) mod agg;
pub(crate) mod components;
pub(crate) mod core;
pub(crate) mod disagg;
mod entrypoints;
pub(crate) mod events;
mod progress;
pub(crate) mod runtime_utils;
pub(crate) mod single;
pub(crate) mod state;
mod workload;

pub(crate) use entrypoints::{
    generate_trace_worker_artifacts, generate_trace_worker_artifacts_with_visibility,
    simulate_concurrency_disagg_with_latency_models, simulate_concurrency_with_latency_model,
    simulate_trace, simulate_trace_disagg, simulate_trace_disagg_with_latency_models,
    simulate_trace_with_latency_model,
};

pub(crate) use workload::{
    simulate_agentic_trace_workload, simulate_concurrency_workload,
    simulate_concurrency_workload_accumulating_deltas, simulate_concurrency_workload_disagg,
    simulate_concurrency_workload_disagg_with_latency_models,
    simulate_concurrency_workload_with_latency_model, simulate_trace_workload,
    simulate_trace_workload_accumulating_deltas, simulate_trace_workload_disagg,
    simulate_trace_workload_disagg_with_latency_models, simulate_trace_workload_with_latency_model,
};

pub(super) fn finish_with_replay_wall_time(
    collector: TraceCollector,
    started_at: Instant,
) -> TraceSimulationReport {
    // Capture elapsed time before final report aggregation so bookkeeping such
    // as latency sorting is not counted as replay execution.
    let wall_time_ms = started_at.elapsed().as_secs_f64() * 1000.0;
    collector.finish().with_wall_time_ms(wall_time_ms)
}

pub(super) fn use_single_runtime(num_workers: usize, router_mode: ReplayRouterMode) -> bool {
    num_workers == 1 && router_mode != ReplayRouterMode::KvRouter
}
