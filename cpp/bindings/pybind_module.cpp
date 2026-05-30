#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "estimation_engine.h"
#include "hardware_matcher.h"
#include "calibration.h"

namespace py = pybind11;
using namespace model_compute;

PYBIND11_MODULE(model_compute, m) {
    m.doc() = "C++ estimation engine for LLM compute requirements";

    // ── Enums ──────────────────────────────────────────────────────────

    py::enum_<ModelType>(m, "ModelType")
        .value("DENSE", ModelType::DENSE)
        .value("MOE", ModelType::MOE)
        .value("O1_REASONING", ModelType::O1_REASONING)
        .value("MULTIMODAL", ModelType::MULTIMODAL)
        .value("RECOMMENDATION", ModelType::RECOMMENDATION)
        .export_values();

    py::enum_<Quantization>(m, "Quantization")
        .value("FP16", Quantization::FP16)
        .value("INT8", Quantization::INT8)
        .value("INT4", Quantization::INT4)
        .export_values();

    // ── ModelParams ────────────────────────────────────────────────────

    py::class_<ModelParams>(m, "ModelParams")
        .def(py::init<>())
        .def_readwrite("type", &ModelParams::type)
        .def_readwrite("param_billions", &ModelParams::param_billions)
        .def_readwrite("quant", &ModelParams::quant)
        .def_readwrite("concurrency", &ModelParams::concurrency)
        .def_readwrite("max_tokens", &ModelParams::max_tokens)
        .def_readwrite("num_experts", &ModelParams::num_experts)
        .def_readwrite("active_experts", &ModelParams::active_experts)
        .def_readwrite("reasoning_depth", &ModelParams::reasoning_depth)
        .def_readwrite("image_resolution", &ModelParams::image_resolution)
        .def_readwrite("num_images", &ModelParams::num_images)
        .def_readwrite("num_sparse_features", &ModelParams::num_sparse_features)
        .def_readwrite("vocab_size_per_feature", &ModelParams::vocab_size_per_feature)
        .def_readwrite("embed_dim", &ModelParams::embed_dim)
        .def_readwrite("mlp_dims", &ModelParams::mlp_dims)
        .def_readwrite("num_kv_heads", &ModelParams::num_kv_heads)
        .def_readwrite("head_dim", &ModelParams::head_dim)
        .def_readwrite("use_swiglu", &ModelParams::use_swiglu);

    // ── EstimationResult ───────────────────────────────────────────────

    py::class_<EstimationResult>(m, "EstimationResult")
        .def_readwrite("memory_gb", &EstimationResult::memory_gb)
        .def_readwrite("flops_total", &EstimationResult::flops_total)
        .def_readwrite("bandwidth_gbs", &EstimationResult::bandwidth_gbs)
        .def_readwrite("kv_cache_gb", &EstimationResult::kv_cache_gb)
        .def_readonly("weight_memory_gb", &EstimationResult::weight_memory_gb)
        .def_readwrite("runtime_overhead_gb", &EstimationResult::runtime_overhead_gb)
        .def_readwrite("fragmentation_gb", &EstimationResult::fragmentation_gb);

    // ── EstimationEngine ───────────────────────────────────────────────

    py::class_<EstimationEngine>(m, "EstimationEngine")
        .def(py::init<>())
        .def("estimate", &EstimationEngine::estimate,
             py::arg("params"),
             "Estimate compute requirements for the given model parameters");

    // ── HardwareSpec ───────────────────────────────────────────────────

    py::class_<HardwareSpec>(m, "HardwareSpec")
        .def(py::init<>())
        .def_readwrite("name", &HardwareSpec::name)
        .def_readwrite("vendor", &HardwareSpec::vendor)
        .def_readwrite("architecture", &HardwareSpec::architecture)
        .def_readwrite("type", &HardwareSpec::type)
        .def_readwrite("fp16_tflops", &HardwareSpec::fp16_tflops)
        .def_readwrite("int8_tops", &HardwareSpec::int8_tops)
        .def_readwrite("fp32_tflops", &HardwareSpec::fp32_tflops)
        .def_readwrite("memory_gb", &HardwareSpec::memory_gb)
        .def_readwrite("memory_type", &HardwareSpec::memory_type)
        .def_readwrite("memory_bandwidth_gbs", &HardwareSpec::memory_bandwidth_gbs)
        .def_readwrite("nvlink_bandwidth_gbs", &HardwareSpec::nvlink_bandwidth_gbs)
        .def_readwrite("pcie_version", &HardwareSpec::pcie_version)
        .def_readwrite("max_tdp_watts", &HardwareSpec::max_tdp_watts)
        .def_readwrite("cost_per_unit", &HardwareSpec::cost_per_unit);

    // ── HardwareConfig ─────────────────────────────────────────────────

    py::class_<HardwareConfig>(m, "HardwareConfig")
        .def_readonly("hardware", &HardwareConfig::hardware)
        .def_readonly("num_cards", &HardwareConfig::num_cards)
        .def_readonly("estimated_throughput", &HardwareConfig::estimated_throughput)
        .def_readonly("estimated_latency_ms", &HardwareConfig::estimated_latency_ms)
        .def_readonly("bottleneck_type", &HardwareConfig::bottleneck_type)
        .def_readonly("parallel_strategy", &HardwareConfig::parallel_strategy)
        .def_readonly("meets_baseline", &HardwareConfig::meets_baseline);

    // ── HardwareMatcher ────────────────────────────────────────────────

    py::class_<HardwareMatcher>(m, "HardwareMatcher")
        .def(py::init<>())
        .def("match", &HardwareMatcher::match,
             py::arg("estimation"),
             py::arg("model_params"),
             py::arg("hardware_pool"),
             py::arg("baseline_throughput") = 10.0,
             "Match hardware options to the estimated requirements");

    // ── CalibrationPoint ───────────────────────────────────────────────

    py::class_<CalibrationPoint>(m, "CalibrationPoint")
        .def(py::init<>())
        .def_readwrite("model_type", &CalibrationPoint::model_type)
        .def_readwrite("hardware_name", &CalibrationPoint::hardware_name)
        .def_readwrite("predicted_throughput", &CalibrationPoint::predicted_throughput)
        .def_readwrite("actual_throughput", &CalibrationPoint::actual_throughput)
        .def_readwrite("predicted_memory", &CalibrationPoint::predicted_memory)
        .def_readwrite("actual_memory", &CalibrationPoint::actual_memory);

    // ── CalibrationFactor ──────────────────────────────────────────────

    py::class_<CalibrationFactor>(m, "CalibrationFactor")
        .def_readonly("throughput_factor", &CalibrationFactor::throughput_factor)
        .def_readonly("memory_factor", &CalibrationFactor::memory_factor)
        .def_readonly("num_points", &CalibrationFactor::num_points);

    // ── Calibration ────────────────────────────────────────────────────

    py::class_<Calibration>(m, "Calibration")
        .def(py::init<>())
        .def("add_point", &Calibration::add_point,
             py::arg("point"),
             "Add a calibration data point")
        .def("get_factor", &Calibration::get_factor,
             py::arg("model_type"),
             py::arg("hardware_name"),
             "Get calibration factor for a model/hardware pair")
        .def("adjust_throughput", &Calibration::adjust_throughput,
             py::arg("predicted"),
             py::arg("model_type"),
             py::arg("hardware_name"),
             "Adjust a predicted throughput using calibration")
        .def("adjust_memory", &Calibration::adjust_memory,
             py::arg("predicted"),
             py::arg("model_type"),
             py::arg("hardware_name"),
             "Adjust a predicted memory value using calibration")
        .def("load_from_file", &Calibration::load_from_file,
             py::arg("path"),
             "Load calibration data from a CSV file")
        .def("save_to_file", &Calibration::save_to_file,
             py::arg("path"),
             "Save calibration data to a CSV file");
}
