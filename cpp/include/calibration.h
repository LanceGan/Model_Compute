#pragma once
#include <string>
#include <vector>
#include <map>

namespace model_compute {

struct CalibrationPoint {
    std::string model_type;
    std::string hardware_name;
    double predicted_throughput;
    double actual_throughput;
    double predicted_memory;
    double actual_memory;
};

struct CalibrationFactor {
    double throughput_factor;
    double memory_factor;
    int num_points;
};

class Calibration {
public:
    void add_point(const CalibrationPoint& point);
    CalibrationFactor get_factor(const std::string& model_type, const std::string& hardware_name) const;
    double adjust_throughput(double predicted, const std::string& model_type, const std::string& hardware_name) const;
    double adjust_memory(double predicted, const std::string& model_type, const std::string& hardware_name) const;
    void load_from_file(const std::string& path);
    void save_to_file(const std::string& path) const;

private:
    std::map<std::pair<std::string, std::string>, std::vector<CalibrationPoint>> points_;
    std::map<std::pair<std::string, std::string>, CalibrationFactor> factors_;
    void recompute_factor(const std::string& model_type, const std::string& hardware_name);
};

} // namespace model_compute
