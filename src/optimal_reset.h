#ifndef OPTIMAL_RESET_H
#define OPTIMAL_RESET_H

#include "pdf.h"
#include "time.h"
#include <stddef.h>

/**
 * Discretized segment distribution for DP-based value iteration.
 * Each segment's time distribution is represented as a set of
 * (time, probability) outcome pairs.
 */
typedef struct {
    double *times;       // possible segment times (seconds)
    double *probs;       // corresponding probabilities
    size_t count;        // number of outcomes
} DiscreteSegment;

/**
 * Result of the optimal reset computation.
 */
typedef struct {
    Duration *thresholds;     // optimal reset threshold at each split (index s = threshold after segment s)
    int num_thresholds;       // = num_segments - 1
    Duration goal;            // the goal time (printed as last split)
    double v00;               // optimal expected time to PB from start (seconds)
    double baseline_etpb;     // baseline expected time to PB without resets (seconds)
    double success_prob;      // probability of PB per attempt under optimal policy
    double expected_attempt_time; // expected time per attempt under optimal policy
    int converged;            // whether the value iteration converged
    int iterations;           // number of iterations performed
} OptimalResetResult;

/**
 * Discretize a SegmentPDF into a discrete distribution with up to max_bins outcomes.
 * Uses the same kernel-density interpolation as pdf_time_at_percentile.
 * Caller must free with free_discrete_segment.
 */
DiscreteSegment discretize_segment(const SegmentPDF *pdf, int max_bins);

void free_discrete_segment(DiscreteSegment *seg);

/**
 * Compute the optimal reset policy that minimizes expected time to PB.
 *
 * Algorithm:
 * 1. Discretize each segment's time distribution
 * 2. Iterate until convergence:
 *    a. Given V(0,0), compute thresholds: reset at split s if E[V(s+1, t+Δt)] > V(0,0)
 *    b. Given thresholds, compute P(PB per attempt) and E[attempt time]
 *    c. Update V(0,0) = E[attempt time] / P(PB per attempt)
 *
 * @param segments       Array of SegmentPDFs
 * @param num_segments   Number of segments
 * @param goal           Target time to beat
 * @param max_bins       Max discretization bins per segment
 * @param max_iter       Maximum value iteration steps
 * @param tol            Convergence tolerance (relative change in V(0,0))
 * @return               OptimalResetResult (caller must free with free_optimal_reset_result)
 */
OptimalResetResult compute_optimal_reset(const SegmentPDF *segments, int num_segments,
                                          const Duration *goal,
                                          int max_bins, int max_iter, double tol);

/**
 * Free resources allocated by compute_optimal_reset.
 */
void free_optimal_reset_result(OptimalResetResult *result);

/**
 * Print the optimal reset results in a human-readable format.
 */
void print_optimal_reset_result(const OptimalResetResult *result,
                                 const SegmentPDF *segments, int num_segments,
                                 const Duration *goal);

/**
 * Print optimal reset thresholds in LiveSplit-pastable format
 * (one time per line, followed by the goal time).
 */
void print_optimal_reset_splits(const OptimalResetResult *result);

#endif /* OPTIMAL_RESET_H */