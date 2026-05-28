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
    std::ifstream file(path);
    if (!file.is_open()) return;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        CalibrationPoint pt;
        size_t pos = 0;
        auto next = [&]() -> std::string {
            size_t end = line.find(',', pos);
            std::string val = line.substr(pos, end - pos);
            pos = (end == std::string::npos) ? end : end + 1;
            return val;
        };
        pt.model_type = next();
        pt.hardware_name = next();
        pt.predicted_throughput = std::stod(next());
        pt.actual_throughput = std::stod(next());
        pt.predicted_memory = std::stod(next());
        pt.actual_memory = std::stod(next());
        add_point(pt);
    }
}

void Calibration::save_to_file(const std::string& path) const {
    std::ofstream file(path);
    file << "# model_type,hardware_name,predicted_tp,actual_tp,predicted_mem,actual_mem\n";
    for (const auto& [key, pts] : points_) {
        for (const auto& p : pts) {
            file << p.model_type << ","
                 << p.hardware_name << ","
                 << p.predicted_throughput << ","
                 << p.actual_throughput << ","
                 << p.predicted_memory << ","
                 << p.actual_memory << "\n";
        }
    }
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
