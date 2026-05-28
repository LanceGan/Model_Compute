#include "calibration.h"
#include <fstream>
#include <numeric>

namespace model_compute {

void Calibration::add_point(const CalibrationPoint& point) {
    auto key = std::make_pair(point.model_type, point.hardware_name);
    points_[key].push_back(point);
    recompute_factor(point.model_type, point.hardware_name);
}

CalibrationFactor Calibration::get_factor(const std::string& model_type, const std::string& hardware_name) const {
    auto key = std::make_pair(model_type, hardware_name);
    auto it = factors_.find(key);
    if (it != factors_.end()) {
        return it->second;
    }
    return {1.0, 1.0, 0};
}

double Calibration::adjust_throughput(double predicted, const std::string& model_type, const std::string& hardware_name) const {
    auto factor = get_factor(model_type, hardware_name);
    return predicted * factor.throughput_factor;
}

double Calibration::adjust_memory(double predicted, const std::string& model_type, const std::string& hardware_name) const {
    auto factor = get_factor(model_type, hardware_name);
    return predicted * factor.memory_factor;
}

void Calibration::load_from_file(const std::string& path) {
    // Stub - will be implemented in Task 4
}

void Calibration::save_to_file(const std::string& path) const {
    // Stub - will be implemented in Task 4
}

void Calibration::recompute_factor(const std::string& model_type, const std::string& hardware_name) {
    auto key = std::make_pair(model_type, hardware_name);
    auto it = points_.find(key);
    if (it == points_.end() || it->second.empty()) return;

    double throughput_sum = 0.0;
    double memory_sum = 0.0;
    int count = 0;

    for (const auto& p : it->second) {
        if (p.predicted_throughput > 0 && p.actual_throughput > 0) {
            throughput_sum += p.actual_throughput / p.predicted_throughput;
            memory_sum += p.actual_memory / p.predicted_memory;
            count++;
        }
    }

    CalibrationFactor factor;
    if (count > 0) {
        factor.throughput_factor = throughput_sum / count;
        factor.memory_factor = memory_sum / count;
    } else {
        factor.throughput_factor = 1.0;
        factor.memory_factor = 1.0;
    }
    factor.num_points = count;
    factors_[key] = factor;
}

} // namespace model_compute
