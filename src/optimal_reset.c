/**
 * optimal_reset.c
 *
 * Computes optimal reset thresholds by matching the Python comp.py --reset algorithm.
 *
 * Algorithm (matching Python find_reset_splits):
 *
 *   For each iteration:
 *     1. Compute base_percentage = P(PB) from (0,0) with current reset policy
 *     2. For each split s (forward from 1 to n-1):
 *        Binary search for threshold t* where:
 *          run_progress_factor = goal / (goal - t)
 *          effective_p_pb = repeated_odds(simulate_p_pb(s, t), run_progress_factor)
 *        The threshold is where effective_p_pb ≈ base_percentage
 *
 *   repeated_odds(p, n) = 1 - (1-p)^n
 *     Probability of at least one success in n independent attempts.
 */

#include "optimal_reset.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <omp.h>

/* ------------------------------------------------------------------ */
/*  Fast per-thread RNG (xoshiro128**)                                */
/* ------------------------------------------------------------------ */

typedef struct {
    uint64_t s[2];
} RNG;

static inline uint64_t rng_next(RNG *r) {
    uint64_t s0 = r->s[0];
    uint64_t s1 = r->s[1];

    uint64_t result = (s0 + s1) * 0x2545F4914F6CDD1FULL;
    result = (result ^ (result >> 13)) * 0x2545F4914F6CDD1FULL;
    result = result ^ (result >> 27);

    s0 ^= s1;
    r->s[0] = s1 ^ (s1 << 5) ^ (s1 << 13);
    r->s[1] = s0 + s1;

    return result;
}

static double rng_double(RNG *r) {
    return ((rng_next(r) >> 11) / (double)(1ULL << 53));
}

static RNG rng_seed(uint64_t seed) {
    RNG r;
    r.s[0] = seed ^ 0x6A09E667F3BCC908ULL;
    r.s[1] = seed ^ 0xBB67AE8584CAA73BULL;
    for (int i = 0; i < 10; i++) {
        rng_next(&r);
    }
    return r;
}

/* ------------------------------------------------------------------ */
/*  Discretize segment distribution                                    */
/* ------------------------------------------------------------------ */

DiscreteSegment discretize_segment(const SegmentPDF *pdf, int max_bins) {
    DiscreteSegment seg;
    seg.count = (size_t)max_bins;
    seg.times = calloc((size_t)max_bins, sizeof(double));
    seg.probs = calloc((size_t)max_bins, sizeof(double));

    if (!seg.times || !seg.probs) {
        seg.count = 0;
        return seg;
    }

    for (int i = 0; i < max_bins; i++) {
        double pct = (i + 0.5) / (double)max_bins;
        Duration d = pdf_time_at_percentile(pdf, pct);
        seg.times[i] = d.seconds;
        seg.probs[i] = 1.0 / (double)max_bins;
    }

    return seg;
}

void free_discrete_segment(DiscreteSegment *seg) {
    free(seg->times);
    free(seg->probs);
    seg->times = NULL;
    seg->probs = NULL;
    seg->count = 0;
}

/* ------------------------------------------------------------------ */
/*  Sample from a discretized segment                                  */
/* ------------------------------------------------------------------ */

static double sample_segment(const DiscreteSegment *seg, RNG *rng) {
    double r = rng_double(rng);
    size_t idx = (size_t)(r * (double)seg->count);
    if (idx >= seg->count) idx = seg->count - 1;
    return seg->times[idx];
}

/* ------------------------------------------------------------------ */
/*  Simulate: estimate P(PB) from (start_seg, start_time)              */
/* ------------------------------------------------------------------ */

/**
 * Run Monte Carlo simulations from state (start_seg, start_time).
 * Returns p_pb = probability of achieving PB in one cycle.
 *
 * A cycle ends when:
 *   - All remaining segments completed and total time < goal (PB)
 *   - All remaining segments completed and total time >= goal (failure)
 *   - Accumulated time exceeds reset threshold at some split (reset)
 */
static double simulate_p_pb(const DiscreteSegment *segments,
                              int num_segments,
                              int start_seg,
                              double start_time,
                              const double *thresholds,
                              double goal_time,
                              int num_sims,
                              uint64_t seed_base) {
     int total_pb = 0;
     int total_sims = 0;

#pragma omp parallel reduction(+ : total_pb) reduction(+ : total_sims)
     {
         int tid = omp_get_thread_num();
         RNG rng = rng_seed(seed_base + (uint64_t)tid * 6364136223846793005ULL);

         int local_pb = 0;

         for (int sim = 0; sim < num_sims; sim++) {
             double t = start_time;
             int pb = 0;

             for (int j = start_seg; j < num_segments; j++) {
                 double delta = sample_segment(&segments[j], &rng);
                 t += delta;

                 if (j < num_segments - 1) {
                     if (thresholds && thresholds[j] < goal_time && t > thresholds[j]) {
                         break; /* reset */
                     }
                 } else if (j == num_segments - 1) {
                     if (t < goal_time) {
                         pb = 1;
                     }
                 }
             }

             if (pb) local_pb++;
         }

         total_pb += local_pb;
         total_sims += num_sims;
     }

     return (double)total_pb / (double)total_sims;
}

/* ------------------------------------------------------------------ */
/*  repeated_odds: P(at least one success in n attempts)               */
/* ------------------------------------------------------------------ */

static double repeated_odds(double p_pb, double n) {
    if (p_pb <= 0.0) return 0.0;
    if (p_pb >= 1.0) return 1.0;
    if (n <= 0.0) return 0.0;
    if (n >= 1e10) return 1.0; /* avoid underflow */
    return 1.0 - pow(1.0 - p_pb, n);
}

/* ------------------------------------------------------------------ */
/*  Find threshold at split s (Python algorithm)                       */
/* ------------------------------------------------------------------ */

/**
 * Binary search for the threshold t* at split s where
 * effective_p_pb(s, t*) ≈ base_p_pb.
 *
 * effective_p_pb(s, t) = repeated_odds(simulate_p_pb(s, t), run_progress_factor)
 * run_progress_factor = goal / (goal - t)
 *
 * If effective_p_pb > base_p_pb: continuing is better, threshold is higher
 * If effective_p_pb < base_p_pb: continuing is worse, threshold is lower
 */
static double find_threshold_python(const DiscreteSegment *segments,
                                     int num_segments,
                                     int s,
                                     const double *thresholds,
                                     double goal_time,
                                     double base_p_pb,
                                     int num_sims,
                                     uint64_t seed_base) {
    double lo = 0.0;
    double hi = goal_time;

    /* Quick check: at t=0, if effective_p_pb < base_p_pb, always reset */
    {
        double p0 = simulate_p_pb(segments, num_segments, s, 0.0,
                                   thresholds, goal_time, num_sims, seed_base);
        double rpf = goal_time / (goal_time - 0.0); /* = 1.0 */
        double eff0 = repeated_odds(p0, rpf);
        if (eff0 < base_p_pb * 0.99) {
            return 0.0;
        }
        if (eff0 >= base_p_pb * 1.01) {
            /* At t=0 continuing is still better, need to search */
        }
    }

    /* Binary search */
    for (int iter = 0; iter < 50; iter++) {
        double mid = (lo + hi) / 2.0;

        if (mid >= goal_time - 0.5) {
            /* Very close to goal: continuing is essentially doomed */
            hi = mid;
            continue;
        }

        double run_progress_factor = goal_time / (goal_time - mid);
        double actual_p_pb = simulate_p_pb(segments, num_segments, s, mid,
                                            thresholds, goal_time, num_sims,
                                            seed_base + (uint64_t)iter * 7919);
        double effective_p_pb = repeated_odds(actual_p_pb, run_progress_factor);

        /* Compare to base_p_pb with tolerance */
        double ratio = effective_p_pb / (base_p_pb > 0.0001 ? base_p_pb : 0.0001);

        if (ratio > 1.005) {
            lo = mid; /* continuing is better, raise threshold */
        } else if (ratio < 0.995) {
            hi = mid; /* continuing is worse, lower threshold */
        } else {
            return mid; /* within tolerance */
        }
    }

    return (lo + hi) / 2.0;
}

/* ------------------------------------------------------------------ */
/*  Compute baseline ETPB (no resets)                                  */
/* ------------------------------------------------------------------ */

static double compute_baseline_etpb(const DiscreteSegment *segments,
                                      int num_segments,
                                      double goal_time,
                                      int num_sims) {
     double total_time = 0.0;
     int total_pb = 0;
     int total_sims = 0;

#pragma omp parallel reduction(+ : total_time) reduction(+ : total_pb) reduction(+ : total_sims)
     {
         int tid = omp_get_thread_num();
         RNG rng = rng_seed((uint64_t)tid * 6364136223846793005ULL + 999);

         double local_time = 0.0;
         int local_pb = 0;

         for (int sim = 0; sim < num_sims; sim++) {
             double t = 0.0;
             for (int j = 0; j < num_segments; j++) {
                 t += sample_segment(&segments[j], &rng);
             }
             local_time += t;
             if (t < goal_time) local_pb++;
         }

         total_time += local_time;
         total_pb += local_pb;
         total_sims += num_sims;
     }

     double p_pb = (double)total_pb / (double)total_sims;
     double avg_time = total_time / (double)total_sims;

     if (p_pb < 1e-10) return 1e18;
     return avg_time / p_pb;
}

/* ------------------------------------------------------------------ */
/*  Main: compute_optimal_reset                                        */
/* ------------------------------------------------------------------ */

OptimalResetResult compute_optimal_reset(const SegmentPDF *segments,
                                          int num_segments,
                                          const Duration *goal,
                                          int max_bins,
                                          int max_iter,
                                          double tol) {
    OptimalResetResult result;
    memset(&result, 0, sizeof(result));

    int num_thresholds = num_segments - 1;
    double goal_time = goal->seconds;

    /* Step 0: Discretize all segment distributions */
    DiscreteSegment *disc = calloc((size_t)num_segments, sizeof(DiscreteSegment));
    if (!disc) return result;

    for (int i = 0; i < num_segments; i++) {
        disc[i] = discretize_segment(&segments[i], max_bins);
        if (disc[i].count == 0) {
            for (int j = 0; j < i; j++)
                free_discrete_segment(&disc[j]);
            free(disc);
            return result;
        }
    }

    /* Step 1: Compute baseline ETPB (no resets) */
    int num_sims = 200000;
    result.baseline_etpb = compute_baseline_etpb(disc, num_segments, goal_time, num_sims);

    /* Initialize thresholds to goal (never reset) */
    double *thresholds = calloc((size_t)num_thresholds, sizeof(double));
    if (!thresholds) {
        for (int i = 0; i < num_segments; i++)
            free_discrete_segment(&disc[i]);
        free(disc);
        return result;
    }

    for (int i = 0; i < num_thresholds; i++)
        thresholds[i] = goal_time;

    result.thresholds = calloc((size_t)num_thresholds, sizeof(Duration));
    result.num_thresholds = num_thresholds;
    result.goal = *goal;
    result.iterations = 0;

    /* Baseline P(PB) with no resets */
    double base_p_pb = simulate_p_pb(disc, num_segments, 0, 0.0,
                                      thresholds, goal_time, num_sims, 12345);
    result.success_prob = base_p_pb;

    printf("Baseline P(PB): %.4f%%\n", base_p_pb * 100.0);
    printf("Baseline E[time to PB]: %.1f seconds (%.1f minutes)\n",
           result.baseline_etpb, result.baseline_etpb / 60.0);

    /* Use fewer sims per threshold for speed */
    int threshold_sims = 100000;

    printf("\n=== Value Iteration ===\n");

    double *best_thresholds = calloc((size_t)num_thresholds, sizeof(double));
    for (int i = 0; i < num_thresholds; i++)
        best_thresholds[i] = goal_time;
    double best_v00 = result.baseline_etpb;

    for (int iter = 0; iter < max_iter; iter++) {
        result.iterations = iter + 1;

        /* Recompute base P(PB) with current thresholds */
        base_p_pb = simulate_p_pb(disc, num_segments, 0, 0.0,
                                   thresholds, goal_time, num_sims,
                                   (uint64_t)(iter + 1) * 7777);

        printf("\nIteration %d (base P(PB) = %.4f%%):\n",
               iter + 1, base_p_pb * 100.0);

        if (base_p_pb < 0.001) {
            printf("  Warning: P(PB) too low, estimates may be unreliable\n");
        }

        /* Compute thresholds forward (split 1 first, matching Python)
         * Python: for idx, start_split in enumerate(range(1, len(segments))):
         *   times[idx] = find_threshold for split start_split
         *   simulate_runs(start_split, t, goal, times, ...)
         *
         * Python reset_times[idx] is checked at segment idx in simulate_runs.
         * So thresholds[s] is checked at segment s, and we simulate from segment s+1
         * to find threshold[s] (which is checked at segment s).
         */
        for (int s = 0; s < num_thresholds; s++) {
            // Python: simulate from (start_seg=s+1, start_time=t)
            // to find threshold for split s+1 (thresholds[s], checked at segment s)
            int start_seg = s + 1;

            double t_star = find_threshold_python(disc, num_segments, start_seg,
                                                   thresholds, goal_time,
                                                   base_p_pb, threshold_sims,
                                                   (uint64_t)(s + 1) * 31337 + (uint64_t)iter * 1000000);

            thresholds[s] = t_star;
            result.thresholds[s].seconds = t_star;

            char buf[64];
            duration_format(&result.thresholds[s], buf, sizeof(buf));
            printf("  Split %2d threshold: %s", s + 1, buf);

            if (t_star <= 0.5) {
                printf(" [always reset]");
            } else if (t_star >= goal_time - 0.5) {
                printf(" [never reset]");
            }
            printf("\n");
        }

        /* Compute V(0,0) with new thresholds */
        double new_p_pb = simulate_p_pb(disc, num_segments, 0, 0.0,
                                         thresholds, goal_time, num_sims,
                                         (uint64_t)(iter + 1) * 54321);

         /* Compute avg cycle time and PB count together */
         double total_time = 0.0;
         int total_pb = 0;
         int total_sims = 0;
#pragma omp parallel reduction(+ : total_time) reduction(+ : total_pb) reduction(+ : total_sims)
         {
             int tid = omp_get_thread_num();
             RNG rng = rng_seed(((uint64_t)(iter + 1) * 54321) + (uint64_t)tid * 6364136223846793005ULL);
             double local_time = 0.0;
             int local_pb = 0;
             for (int sim = 0; sim < num_sims; sim++) {
                 double t = 0.0;
                 int reached_end = 1;
                 for (int j = 0; j < num_segments; j++) {
                     t += sample_segment(&disc[j], &rng);
                     if (j < num_segments - 1 && thresholds[j] < goal_time && t > thresholds[j]) {
                         reached_end = 0; /* reset */
                         break;
                     }
                 }
                 local_time += t;
                 if (reached_end && t < goal_time) local_pb++;
             }
             total_time += local_time;
             total_pb += local_pb;
             total_sims += num_sims;
         }
         double avg_cycle = total_time / (double)total_sims;
         double new_v00 = new_p_pb > 1e-10 ? avg_cycle / new_p_pb : 1e18;

        printf("  New P(PB): %.4f%%, New V(0,0): %.1f sec = %.2f min\n",
               new_p_pb * 100.0, new_v00, new_v00 / 60.0);
        printf("  Improvement: %.1f sec = %.1f min\n",
               best_v00 - new_v00, (best_v00 - new_v00) / 60.0);

        /* Only accept if it improves */
        if (new_v00 >= best_v00 || new_v00 >= 1e17) {
            printf("  New policy is not better. Reverting to previous thresholds.\n");
            for (int i = 0; i < num_thresholds; i++) {
                thresholds[i] = best_thresholds[i];
                result.thresholds[i].seconds = best_thresholds[i];
            }
            printf("\nConverged after %d iterations (no improvement).\n", iter + 1);
            result.converged = 1;
            break;
        }

        /* Check convergence BEFORE updating best */
        double relative_change = fabs(new_v00 - best_v00) / (best_v00 > 0 ? best_v00 : 1.0);

        for (int i = 0; i < num_thresholds; i++)
            best_thresholds[i] = thresholds[i];
        best_v00 = new_v00;
        if (relative_change < tol) {
            printf("\nConverged after %d iterations (relative change: %.6f)\n",
                   iter + 1, relative_change);
            result.converged = 1;
            break;
        }
    }

    if (!result.converged) {
        printf("\nDid not converge after %d iterations.\n", max_iter);
    }

    result.v00 = best_v00;

    /* Compute final statistics consistently */
     double final_avg_cycle, final_p_pb_val;
     {
         double total_time = 0.0;
         int total_pb = 0;
         int total_sims = 0;
#pragma omp parallel reduction(+ : total_time) reduction(+ : total_pb) reduction(+ : total_sims)
         {
             int tid = omp_get_thread_num();
             RNG rng = rng_seed(54321 + (uint64_t)tid * 6364136223846793005ULL);
             double local_time = 0.0;
             int local_pb = 0;
             for (int sim = 0; sim < num_sims; sim++) {
                 double t = 0.0;
                 int reached_end = 1;
                 for (int j = 0; j < num_segments; j++) {
                     t += sample_segment(&disc[j], &rng);
                     if (j < num_segments - 1 && thresholds[j] < goal_time && t > thresholds[j]) {
                         reached_end = 0;
                         break;
                     }
                 }
                 local_time += t;
                 if (reached_end && t < goal_time) local_pb++;
             }
             total_time += local_time;
             total_pb += local_pb;
             total_sims += num_sims;
         }
         final_avg_cycle = total_time / (double)total_sims;
         final_p_pb_val = (double)total_pb / (double)total_sims;
     }

    result.expected_attempt_time = final_avg_cycle;
    result.success_prob = final_p_pb_val;
    /* Ensure v00 is consistent with final stats */
    result.v00 = final_p_pb_val > 1e-10 ? final_avg_cycle / final_p_pb_val : 1e18;

    /* Clean up */
    for (int i = 0; i < num_segments; i++)
        free_discrete_segment(&disc[i]);
    free(disc);
    free(thresholds);
    free(best_thresholds);

    return result;
}

/* ------------------------------------------------------------------ */
/*  Free result                                                        */
/* ------------------------------------------------------------------ */

void free_optimal_reset_result(OptimalResetResult *result) {
    free(result->thresholds);
    result->thresholds = NULL;
    result->num_thresholds = 0;
}

/* ------------------------------------------------------------------ */
/*  Print results                                                      */
/* ------------------------------------------------------------------ */

void print_optimal_reset_result(const OptimalResetResult *result,
                                 const SegmentPDF *segments,
                                 int num_segments,
                                 const Duration *goal) {
    (void)segments;
    (void)num_segments;
    (void)goal;

    printf("\n=== Optimal Reset Policy Results ===\n\n");

    printf("Baseline (no resets):\n");
    printf("  Expected time to PB: %.1f seconds (%.2f minutes)\n",
           result->baseline_etpb, result->baseline_etpb / 60.0);

    printf("\nWith optimal resets:\n");
    printf("  Expected time to PB: %.1f seconds (%.2f minutes)\n",
           result->v00, result->v00 / 60.0);
    printf("  Improvement: %.1f seconds (%.2f minutes) = %.1f%%\n",
           result->baseline_etpb - result->v00,
           (result->baseline_etpb - result->v00) / 60.0,
           (1.0 - result->v00 / result->baseline_etpb) * 100.0);
    printf("  P(PB per cycle): %.4f%%\n", result->success_prob * 100.0);
    printf("  E[cycle time]: %.1f seconds (%.2f minutes)\n",
           result->expected_attempt_time,
           result->expected_attempt_time / 60.0);

    printf("\n  Converged: %s (%d iterations)\n",
           result->converged ? "yes" : "no",
           result->iterations);

    printf("\nReset thresholds (LiveSplit format):\n");
    print_optimal_reset_splits(result);
}

void print_optimal_reset_splits(const OptimalResetResult *result) {
    char buf[64];
    for (int i = 0; i < result->num_thresholds; i++) {
        duration_format(&result->thresholds[i], buf, sizeof(buf));
        printf("%s\n", buf);
    }
    /* Print the goal time as the last split */
    duration_format(&result->goal, buf, sizeof(buf));
    printf("%s\n", buf);
}
