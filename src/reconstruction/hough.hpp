/**
 * @file hough.hpp
 * @brief Track finding with the Hough transform, and fit of the tracks found.
 *
 * In the transverse plane a track coming from the origin is a circle through
 * the origin. A hit at radius r and angle phi lies on the track with initial
 * angle phi0 and curvature kappa = 1/R (signed by the charge) if
 *     phi0 = phi - asin(r * kappa / 2).
 * The hits are first split into overlapping slices of z/r (cot theta for a
 * track from the origin). In every slice each hit votes, for each kappa cell,
 * for the phi0 cell given by this relation. The cells that are local maxima of the accumulator, with the votes
 * of at least min_modules different modules, are the track candidates. The
 * hits of a candidate are then fitted with a circle through the origin
 * (transverse plane) and with a straight line z(s) (s = transverse arc length);
 * the hits far from the fit are removed and the fit repeated. Candidates that
 * share most of their hits with a better one are dropped.
 *
 * The code depends only on the standard library: it works on plain points, so
 * that it can be used with the spacepoints of any reconstruction.
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <numbers>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#include "../utils.hpp"

namespace hough {

/// Parameters of the Hough transform and of the fit (see config/hough.conf).
struct config {
    double cot_theta_bin = 0.1;         ///< Width of a z/r slice of the hits
    double phi_bin_deg = 0.5;           ///< Width of a phi0 cell, in degrees
    double kappa_bin_per_mm = 1.2e-4;   ///< Width of a kappa cell, in 1/mm
    double min_pt_gev = 0.6;            ///< Minimum pT, sets the largest kappa
    double bfield_tesla = 2.0;          ///< Magnetic field along z, in T
    unsigned int min_modules = 3;       ///< Minimum different modules per track
    double max_residual_xy_mm = 0.5;    ///< Largest distance from the circle
    double max_residual_z_mm = 2.0;     ///< Largest distance from the z(s) line
    double max_z0_mm = 250.0;           ///< Largest |z| at the beam line
    double max_shared_fraction = 0.5;   ///< Largest fraction of shared hits

    /// Largest curvature searched, in 1/mm: kappa = 0.3 B / pT in 1/m.
    double kappa_max_per_mm() const {
        return 0.3 * std::abs(bfield_tesla) / min_pt_gev / 1000.0;
    }
};

/// One hit given to the track finder.
struct point {
    double x = 0.;              ///< Global position, in mm
    double y = 0.;
    double z = 0.;
    std::uint64_t module = 0;   ///< Module (surface) of the hit
    std::size_t index = 0;      ///< Index of the hit in the caller's collection
};

/// One track found and fitted.
struct track {
    std::vector<std::size_t> hits;  ///< Positions in the input vector, by radius
    int charge = 0;                 ///< +1, -1 (0 for a straight track)
    double pt_gev = 0.;             ///< Transverse momentum
    double phi = 0.;                ///< Direction at the first hit (rad)
    double theta = 0.;              ///< Polar angle (rad)
    double qop = 0.;                ///< charge / momentum (1/GeV)
    double z0_mm = 0.;              ///< z at the beam line
    double radius_mm = 0.;          ///< Radius of the circle
    double rms_xy_mm = 0.;          ///< RMS distance of the hits from the circle
    double rms_z_mm = 0.;           ///< RMS distance of the hits from the line
};

namespace detail{
/// Circle through the origin: centre and radius.
struct circle {
    double xc = 0.;
    double yc = 0.;
    double radius = std::numeric_limits<double>::infinity();
    bool straight = true;  ///< Points on a line through the origin
};

/**
 * @brief Least squares circle through the origin: x^2 + y^2 + D x + E y = 0
 *        is linear in D and E (Kasa method with the origin constraint).
 * @param pts All the points; only those listed in @p hits are used.
 */
inline circle fit_circle(const std::vector<point>& pts,
                         const std::vector<std::size_t>& hits) {
    double sxx = 0., sxy = 0., syy = 0., sxw = 0., syw = 0.;
    for (std::size_t h : hits) {
        const point& p = pts[h];
        const double w = p.x * p.x + p.y * p.y;
        sxx += p.x * p.x;
        sxy += p.x * p.y;
        syy += p.y * p.y;
        sxw += p.x * w;
        syw += p.y * w;
    }
    circle c;
    const double det = sxx * syy - sxy * sxy;
    // Points (almost) on a line through the origin: no finite circle
    if (std::abs(det) <= 1e-12 * (sxx * syy + 1.)) {
        return c;
    }
    const double d = -(syy * sxw - sxy * syw) / det;
    const double e = -(sxx * syw - sxy * sxw) / det;
    c.xc = -d / 2.;
    c.yc = -e / 2.;
    c.radius = std::hypot(c.xc, c.yc);
    c.straight = false;
    return c;
}

/// Transverse distance of a point from a circle (or from the straight line).
inline double residual_xy(const point& p, const circle& c, double line_phi) {
    if (c.straight) {
        // Distance from the line through the origin with angle line_phi
        return std::abs(p.y * std::cos(line_phi) - p.x * std::sin(line_phi));
    }
    return std::abs(std::hypot(p.x - c.xc, p.y - c.yc) - c.radius);
}

/// Transverse arc length from the origin to a point on the circle.
inline double arc_length(const point& p, const circle& c) {
    const double r = std::hypot(p.x, p.y);
    if (c.straight) {
        return r;
    }
    return 2. * c.radius * std::asin(std::min(1., r / (2. * c.radius)));
}

/// Straight line z = z0 + slope * s, by least squares.
struct line {
    double z0 = 0.;
    double slope = 0.;
    bool valid = false;
};

inline line fit_line(const std::vector<double>& s, const std::vector<double>& z) {
    const double n = static_cast<double>(s.size());
    double ss = 0., sz = 0., sss = 0., ssz = 0.;
    for (std::size_t i = 0; i < s.size(); ++i) {
        ss += s[i];
        sz += z[i];
        sss += s[i] * s[i];
        ssz += s[i] * z[i];
    }
    line l;
    const double det = n * sss - ss * ss;
    if (s.size() < 2 || std::abs(det) < 1e-9) {
        return l;
    }
    l.slope = (n * ssz - ss * sz) / det;
    l.z0 = (sz - l.slope * ss) / n;
    l.valid = true;
    return l;
}

/// Number of different modules among some hits.
inline unsigned int count_modules(const std::vector<point>& pts,
                                  const std::vector<std::size_t>& hits) {
    std::vector<std::uint64_t> modules;
    modules.reserve(hits.size());
    for (std::size_t h : hits) {
        modules.push_back(pts[h].module);
    }
    std::sort(modules.begin(), modules.end());
    return static_cast<unsigned int>(
        std::unique(modules.begin(), modules.end()) - modules.begin());
}

/// Circle through the origin with initial direction phi0 and curvature kappa.
inline circle cell_circle(double phi0, double kappa) {
    circle c;
    if (kappa == 0.) {
        return c;
    }
    // Centre at distance 1/kappa, perpendicular to the initial direction
    c.xc = -std::sin(phi0) / kappa;
    c.yc = std::cos(phi0) / kappa;
    c.radius = std::abs(1. / kappa);
    c.straight = false;
    return c;
}

/**
 * @brief Fits the hits of a candidate cell: keeps, for every module, the hit
 *        closest to the circle of the cell, then fits, removes the outliers
 *        and fits again; fills the track parameters.
 * @param phi0, kappa, phi_bin Parameters and width of the cell
 * @return false if too few hits are left or the track is outside the vertex region
 */
inline bool fit_track(const std::vector<point>& pts, std::vector<std::size_t> hits,
                      double phi0, double kappa, double phi_bin, const config& cfg,
                      track& out) {
    // First selection with the circle of the cell: tolerance of one cell
    // width in phi0 at the radius of the hit.
    circle c = cell_circle(phi0, kappa);
    double line_phi = phi0;
    {
        std::sort(hits.begin(), hits.end(), [&](std::size_t a, std::size_t b) {
            if (pts[a].module != pts[b].module) {
                return pts[a].module < pts[b].module;
            }
            return residual_xy(pts[a], c, line_phi) < residual_xy(pts[b], c, line_phi);
        });
        std::vector<std::size_t> kept;
        for (std::size_t i = 0; i < hits.size(); ++i) {
            const bool first_of_module =
                i == 0 || pts[hits[i]].module != pts[hits[i - 1]].module;
            const double tolerance = std::hypot(pts[hits[i]].x, pts[hits[i]].y) * phi_bin;
            if (first_of_module && residual_xy(pts[hits[i]], c, line_phi) <= tolerance) {
                kept.push_back(hits[i]);
            }
        }
        hits = std::move(kept);
    }

    // Transverse plane: fit, keep the best hit of every module, drop the
    // outliers, and repeat until nothing changes.
    for (int iteration = 0; iteration < 5; ++iteration) {
        if (hits.size() < 2) {
            return false;
        }
        c = fit_circle(pts, hits);
        const point& last = pts[hits.back()];
        line_phi = std::atan2(last.y, last.x);

        std::vector<std::size_t> kept;
        std::sort(hits.begin(), hits.end(), [&](std::size_t a, std::size_t b) {
            if (pts[a].module != pts[b].module) {
                return pts[a].module < pts[b].module;
            }
            return residual_xy(pts[a], c, line_phi) < residual_xy(pts[b], c, line_phi);
        });
        for (std::size_t i = 0; i < hits.size(); ++i) {
            const bool first_of_module =
                i == 0 || pts[hits[i]].module != pts[hits[i - 1]].module;
            if (first_of_module &&
                residual_xy(pts[hits[i]], c, line_phi) <= cfg.max_residual_xy_mm) {
                kept.push_back(hits[i]);
            }
        }
        const bool changed = kept.size() != hits.size();
        hits = std::move(kept);
        if (!changed) {
            break;
        }
    }
    if (count_modules(pts, hits) < cfg.min_modules) {
        return false;
    }

    // Longitudinal plane: z against the arc length, with outlier removal.
    line l;
    for (int iteration = 0; iteration < 5; ++iteration) {
        std::vector<double> s, z;
        for (std::size_t h : hits) {
            s.push_back(arc_length(pts[h], c));
            z.push_back(pts[h].z);
        }
        l = fit_line(s, z);
        if (!l.valid) {
            return false;
        }
        std::vector<std::size_t> kept;
        for (std::size_t i = 0; i < hits.size(); ++i) {
            if (std::abs(z[i] - (l.z0 + l.slope * s[i])) <= cfg.max_residual_z_mm) {
                kept.push_back(hits[i]);
            }
        }
        const bool changed = kept.size() != hits.size();
        hits = std::move(kept);
        if (!changed) {
            break;
        }
    }
    if (count_modules(pts, hits) < cfg.min_modules || std::abs(l.z0) > cfg.max_z0_mm) {
        return false;
    }

    // Hits ordered by radius, from the beam line outwards
    std::sort(hits.begin(), hits.end(), [&](std::size_t a, std::size_t b) {
        return std::hypot(pts[a].x, pts[a].y) < std::hypot(pts[b].x, pts[b].y);
    });
    const point& first = pts[hits.front()];
    const point& last = pts[hits.back()];

    // Direction of motion at the first hit and charge. For B along +z a
    // positive particle turns clockwise.
    double tx = first.x, ty = first.y;
    int charge = 0;
    if (!c.straight) {
        const double ux = first.x - c.xc, uy = first.y - c.yc;
        const double vx = last.x - c.xc, vy = last.y - c.yc;
        const bool clockwise = ux * vy - uy * vx < 0.;
        tx = clockwise ? uy : -uy;
        ty = clockwise ? -ux : ux;
        charge = (clockwise == (cfg.bfield_tesla > 0.)) ? +1 : -1;
    }

    double sum_xy = 0., sum_z = 0.;
    for (std::size_t h : hits) {
        const double rxy = residual_xy(pts[h], c, line_phi);
        const double rz = pts[h].z - (l.z0 + l.slope * arc_length(pts[h], c));
        sum_xy += rxy * rxy;
        sum_z += rz * rz;
    }
    const double n = static_cast<double>(hits.size());

    out.hits = std::move(hits);
    out.charge = charge;
    out.radius_mm = c.radius;
    // pT [GeV] = 0.3 * B [T] * R [m]
    out.pt_gev = c.straight ? std::numeric_limits<double>::infinity()
                            : 0.3 * std::abs(cfg.bfield_tesla) * c.radius / 1000.;
    out.phi = std::atan2(ty, tx);
    out.theta = std::atan2(1., l.slope);  // cot(theta) = dz/ds
    out.qop = c.straight ? 0. : charge * std::sin(out.theta) / out.pt_gev;
    out.z0_mm = l.z0;
    out.rms_xy_mm = std::sqrt(sum_xy / n);
    out.rms_z_mm = std::sqrt(sum_z / n);
    return true;
}

}  // namespace detail

/**
 * @brief Reads a configuration file of "key = value" lines ("#" for comments).
 * @param path File to read, e.g. config/hough.conf
 * @return The configuration; missing keys keep their default value.
 * @throws std::runtime_error if the file cannot be read, or a line, a key or
 *         a value is not valid.
 */
inline config read_config(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw std::runtime_error("Cannot open the Hough configuration " + path);
    }
    config cfg;
    std::string raw;
    unsigned int number = 0;
    while (std::getline(file, raw)) {
        ++number;
        const std::string line = utils::trim(raw.substr(0, raw.find('#')));
        if (line.empty()) {
            continue;
        }
        const auto equal = line.find('=');
        if (equal == std::string::npos) {
            throw std::runtime_error(path + ":" + std::to_string(number) +
                                     ": invalid line: " + line);
        }
        const std::string key = utils::trim(line.substr(0, equal));
        const std::string value = utils::trim(line.substr(equal + 1));
        if (key == "cot-theta-bin") {
            cfg.cot_theta_bin = utils::to_double(key, value);
        } else if (key == "phi-bin-deg") {
            cfg.phi_bin_deg = utils::to_double(key, value);
        } else if (key == "kappa-bin-per-mm") {
            cfg.kappa_bin_per_mm = utils::to_double(key, value);
        } else if (key == "min-pt-gev") {
            cfg.min_pt_gev = utils::to_double(key, value);
        } else if (key == "bfield-tesla") {
            cfg.bfield_tesla = utils::to_double(key, value);
        } else if (key == "min-modules") {
            cfg.min_modules = utils::to_unsigned(key, value);
        } else if (key == "max-residual-xy-mm") {
            cfg.max_residual_xy_mm = utils::to_double(key, value);
        } else if (key == "max-residual-z-mm") {
            cfg.max_residual_z_mm = utils::to_double(key, value);
        } else if (key == "max-z0-mm") {
            cfg.max_z0_mm = utils::to_double(key, value);
        } else if (key == "max-shared-fraction") {
            cfg.max_shared_fraction = utils::to_double(key, value);
        } else {
            throw std::runtime_error(path + ":" + std::to_string(number) +
                                     ": unknown key: " + key);
        }
    }
    if (cfg.cot_theta_bin <= 0. || cfg.phi_bin_deg <= 0. || cfg.kappa_bin_per_mm <= 0. ||
        cfg.min_pt_gev <= 0. ||
        cfg.bfield_tesla == 0. || cfg.min_modules < 2) {
        throw std::runtime_error(path + ": bin widths and pT must be positive, B not 0, "
                                 "min-modules at least 2");
    }
    return cfg;
}

/**
 * @brief Finds and fits the tracks of one event.
 * @param pts The hits of the event (global positions, modules).
 * @param cfg Parameters of the Hough transform and of the fit.
 * @return The tracks, best first (more hits, then smaller residuals).
 */
inline std::vector<track> find_tracks(const std::vector<point>& pts, const config& cfg) {
    constexpr double pi = std::numbers::pi;

    // Hits in slices of z/r (= cot(theta) for a track from the origin): tracks
    // with the same transverse parameters but different polar angles end up in
    // different slices, so the accumulators stay sparse. A hit goes in its
    // slice and in the neighbour closer to its value (overlapping slices).
    std::map<long, std::vector<std::size_t>> slices;
    for (std::size_t h = 0; h < pts.size(); ++h) {
        const double r = std::hypot(pts[h].x, pts[h].y);
        if (r <= 0.) {
            continue;
        }
        const double position = pts[h].z / r / cfg.cot_theta_bin;
        const long slice = static_cast<long>(std::floor(position));
        slices[slice].push_back(h);
        slices[position - std::floor(position) < 0.5 ? slice - 1 : slice + 1].push_back(h);
    }

    // Accumulator of one slice: n_kappa rows (cell centres from -kappa_max to
    // +kappa_max) times n_phi columns covering [-pi, pi).
    const std::size_t n_phi = static_cast<std::size_t>(
        std::ceil(360. / cfg.phi_bin_deg));
    const double phi_bin = 2. * pi / static_cast<double>(n_phi);
    const std::size_t half = static_cast<std::size_t>(
        std::ceil(cfg.kappa_max_per_mm() / cfg.kappa_bin_per_mm));
    const std::size_t n_kappa = 2 * half + 1;
    std::vector<std::vector<std::uint32_t>> cells(n_kappa * n_phi);
    std::vector<unsigned int> score(cells.size(), 0u);

    std::vector<track> candidates;
    for (const auto& [slice, members] : slices) {
        if (members.size() < cfg.min_modules) {
            continue;
        }
        for (auto& cell : cells) {
            cell.clear();
        }

        // Votes. A hit votes in its phi0 cell and in the neighbour closer to
        // its exact phi0, so that a track on a cell border is not split.
        for (std::size_t h : members) {
            const double r = std::hypot(pts[h].x, pts[h].y);
            const double phi = std::atan2(pts[h].y, pts[h].x);
            for (std::size_t k = 0; k < n_kappa; ++k) {
                const double kappa = (static_cast<double>(k) - static_cast<double>(half)) *
                                     cfg.kappa_bin_per_mm;
                const double sine = r * kappa / 2.;
                if (std::abs(sine) >= 1.) {
                    continue;  // the circle never reaches this radius
                }
                const double position =
                    (utils::wrap(phi - std::asin(sine)) + pi) / phi_bin;
                const std::size_t j = static_cast<std::size_t>(position) % n_phi;
                const std::size_t other =
                    position - std::floor(position) < 0.5 ? (j + n_phi - 1) % n_phi
                                                          : (j + 1) % n_phi;
                cells[k * n_phi + j].push_back(static_cast<std::uint32_t>(h));
                cells[k * n_phi + other].push_back(static_cast<std::uint32_t>(h));
            }
        }

        // Score of a cell: number of different modules among its votes.
        for (std::size_t c = 0; c < cells.size(); ++c) {
            score[c] = 0u;
            if (cells[c].size() >= cfg.min_modules) {
                std::vector<std::size_t> hits(cells[c].begin(), cells[c].end());
                score[c] = detail::count_modules(pts, hits);
            }
        }

        // Candidates: local maxima of the score (3 x 3 neighbourhood, phi
        // wraps). On a plateau only the first cell (lowest index) is kept.
        for (std::size_t k = 0; k < n_kappa; ++k) {
            for (std::size_t j = 0; j < n_phi; ++j) {
                const std::size_t c = k * n_phi + j;
                if (score[c] < cfg.min_modules) {
                    continue;
                }
                bool peak = true;
                for (int dk = -1; dk <= 1 && peak; ++dk) {
                    const long kk = static_cast<long>(k) + dk;
                    if (kk < 0 || kk >= static_cast<long>(n_kappa)) {
                        continue;
                    }
                    for (int dj = -1; dj <= 1 && peak; ++dj) {
                        if (dk == 0 && dj == 0) {
                            continue;
                        }
                        const std::size_t jj =
                            (j + n_phi + static_cast<std::size_t>(dj + 1) - 1) % n_phi;
                        const std::size_t n = static_cast<std::size_t>(kk) * n_phi + jj;
                        if (score[n] > score[c] || (score[n] == score[c] && n < c)) {
                            peak = false;
                        }
                    }
                }
                if (!peak) {
                    continue;
                }
                std::vector<std::size_t> hits(cells[c].begin(), cells[c].end());
                std::sort(hits.begin(), hits.end());
                hits.erase(std::unique(hits.begin(), hits.end()), hits.end());
                const double kappa = (static_cast<double>(k) - static_cast<double>(half)) *
                                     cfg.kappa_bin_per_mm;
                const double phi0 = -pi + (static_cast<double>(j) + 0.5) * phi_bin;
                track t;
                if (detail::fit_track(pts, std::move(hits), phi0, kappa, phi_bin, cfg, t)) {
                    candidates.push_back(std::move(t));
                }
            }
        }
    }

    // Duplicates: best candidates first, then drop those sharing too many hits
    // with the tracks already accepted.
    std::sort(candidates.begin(), candidates.end(), [](const track& a, const track& b) {
        if (a.hits.size() != b.hits.size()) {
            return a.hits.size() > b.hits.size();
        }
        return a.rms_xy_mm < b.rms_xy_mm;
    });
    std::vector<track> accepted;
    std::unordered_set<std::size_t> used;
    for (track& t : candidates) {
        std::size_t shared = 0;
        for (std::size_t h : t.hits) {
            shared += used.count(h);
        }
        if (static_cast<double>(shared) >
            cfg.max_shared_fraction * static_cast<double>(t.hits.size())) {
            continue;
        }
        used.insert(t.hits.begin(), t.hits.end());
        accepted.push_back(std::move(t));
    }
    return accepted;
    
} //namespace hough

}  // namespace hough
