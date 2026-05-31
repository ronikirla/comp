#ifndef SIMULATION_H
#define SIMULATION_H

#include "pdf.h"
#include "time.h"
#include <stddef.h>

/**
 * Precomputed lookup table for a single segment.
 * Maps percentile index (0-100) to Duration, for fast simulation.
 */
typedef struct {
    Duration times[101];  // indexed by round(percentile * 100)
    int valid;            // whether this segment has data
} SegmentLookup;

/**
 * Precompute lookup tables for all segments.
 * Caller must free the array with free_lookups.
 */
SegmentLookup *precompute_lookups(const SegmentPDF *segments, int num_segments);

void free_lookups(SegmentLookup *lookups, int num_segments);

/**
 * Run Monte Carlo simulations to estimate success percentage.
 *
 * @param segments       Array of SegmentPDFs
 * @param num_segments   Number of segments
 * @param start_segment  Index of the first segment to simulate (0 = from start)
 * @param start_time     Accumulated time before start_segment
 * @param goal           Goal time to beat
 * @param reset_times    Array of reset threshold times (one per segment except last), or NULL
 * @param target_pct     Target percentage for early stopping, or -1 for default convergence
 * @param result         Output: percentage of successful runs (0-100)
 * @return               Number of simulations run
 */
long long simulate_runs(const SegmentPDF *segments, int num_segments,
                        int start_segment, const Duration *start_time,
                        const Duration *goal, const Duration *reset_times,
                        double target_pct, double *result);

/**
 * Compute total time at a given percentile across all segments.
 */
Duration finish_at_percentile(const SegmentPDF *segments, int num_segments,
                               double percentile);

/**
 * Binary search for the percentile that produces the goal time.
 * Returns the percentile, or -1.0 if not found.
 */
double find_percentile_for_goal(const SegmentPDF *segments, int num_segments,
                                 const Duration *goal);

/**
 * Print split times at the given percentile (used by find_goal_splits).
 */
void print_goal_splits(const SegmentPDF *segments, int num_segments,
                       double percentile, const Duration *goal);

/**
 * Simulated annealing / binary search to find reset comparison splits.
 * For each split from index 1..n-1, find the time such that the odds of
 * beating the goal from that reset point equal the base odds.
 */
void find_reset_splits(const SegmentPDF *segments, int num_segments,
                       const Duration *goal, double max_iterations);

/**
 * Find a goal time that matches a target success percentage,
 * then print the corresponding splits.
 */
void find_goal_by_percentage(const SegmentPDF *segments, int num_segments,
                              const Duration *search_start, double target_pct);

/**
 * Run a simple simulation and print the percentage.
 */
void run_sim(const SegmentPDF *segments, int num_segments,
             int start_segment, const Duration *start_time,
             const Duration *goal);

#endif /* SIMULATION_H */