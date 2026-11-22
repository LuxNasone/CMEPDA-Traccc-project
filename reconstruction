#pragma once   

#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <numbers>
#include <chrono>

namespace utils{

    /**
     * @brief Cuts all spaces in a string
     * @param text : (const std::string) input string
     * @return The substring without any spacing left
     */

    inline std::string trim(const std::string& text) {
        const auto first = text.find_first_not_of(" \t\r");
        if (first == std::string::npos) {
            return "";
        }
        const auto last = text.find_last_not_of(" \t\r");
        return text.substr(first, last - first + 1u);
    }

    inline std::vector<std::string> split(const std::string& line) {
        std::vector<std::string> fields;
        std::stringstream stream(line);
        std::string field;
        while (std::getline(stream, field, ',')) {
            fields.push_back(field);
        }
        return fields;
    }

    inline double to_double(const std::string& key, const std::string& value) {
        std::size_t used = 0;
        double result = 0.;
        try {
            result = std::stod(value, &used);
        } catch (const std::exception&) {
            used = 0;
        }
        if (used == 0 || used != value.size() || !std::isfinite(result)) {
            throw std::runtime_error("Invalid value for " + key + ": " + value);
        }
        return result;
    }

    inline unsigned int to_unsigned(const std::string& key, const std::string& value) {
        std::size_t used = 0;
        unsigned long result = 0;
        try {
            result = value.empty() || value.front() == '-' ? 0 : std::stoul(value, &used);
        } catch (const std::exception&) {
            used = 0;
        }
        if (used == 0 || used != value.size() ||
            result > std::numeric_limits<unsigned int>::max()) {
            throw std::runtime_error("Invalid value for " + key + ": " + value);
        }
        return static_cast<unsigned int>(result);
    }

    inline double wrap(double angle) {
        constexpr double two_pi = 2. * std::numbers::pi;
        angle = std::fmod(angle + std::numbers::pi, two_pi);
        if (angle < 0.) {
            angle += two_pi;
        }
        return angle - std::numbers::pi;
    }

    using clock_type = std::chrono::steady_clock;

    inline double ms_since(clock_type::time_point start) {
        return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
    }

    inline double ms_between(clock_type::time_point from, clock_type::time_point to) {
        return std::chrono::duration<double, std::milli>(to - from).count();
    }
}