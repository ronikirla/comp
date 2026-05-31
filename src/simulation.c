#include "simulation.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
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
    // rng_next returns 64-bit; take top 53 bits, divide by 2^53
    return ((rng_next(r) >> 11) / (double)(1ULL << 53));
}

static RNG rng_seed(uint64_t seed) {
    RNG r;
    // Simple seed splitting
    r.s[0] = seed ^ 0x6A09E667F3BCC908ULL;
    r.s[1] = seed ^ 0xBB67AE8584CAA73BULL;
    for (int i = 0; i < 10; i++) {
        rng_next(&r);
    }
    return r;
}

/* ------------------------------------------------------------------ */
/*  Lookup table precomputation                                        */
/* ------------------------------------------------------------------ */

SegmentLookup *precompute_lookups(const SegmentPDF *segments, int num_segments) {
    SegmentLookup *lookups = calloc(num_segments, sizeof(SegmentLookup));
    if (!lookups) return NULL;

    for (int i = 0; i < num_segments; i++) {
        lookups[i].valid = (segments[i].count > 0);
        if (!lookups[i].valid) continue;
        for (int p = 0; p <= 100; p++) {
            lookups[i].times[p] = pdf_time_at_percentile(&segments[i], p / 100.0);
        }
    }
    return lookups;
}

void free_lookups(SegmentLookup *lookups, int num_segments) {
    (void)num_segments;
    free(lookups);
}

/* ------------------------------------------------------------------ */
/*  finish_at_percentile                                               */
/* ------------------------------------------------------------------ */

Duration finish_at_percentile(const SegmentPDF *segments, int num_segments,
                               double percentile) {
    Duration total = duration_zero();
    for (int i = 0; i < num_segments; i++) {
        Duration t = pdf_time_at_percentile(&segments[i], percentile);
        total.seconds += t.seconds;
    }
    return total;
}

/* ------------------------------------------------------------------ */
/*  find_percentile_for_goal  (iterative bisection)                    */
/* ------------------------------------------------------------------ */

double find_percentile_for_goal(const SegmentPDF *segments, int num_segments,
                                 const Duration *goal) {
    double lo = 0.0, hi = 1.0;
    for (int iter = 0; iter < 100; iter++) {
        double mid = (lo + hi) / 2.0;
        Duration result = finish_at_percentile(segments, num_segments, mid);
        double diff = result.seconds - goal->seconds;
        if (fabs(diff) < 0.1)
            return mid;
        if (diff > 0)
            hi = mid;
        else
            lo = mid;
    }
    return (lo + hi) / 2.0;
}

/* ------------------------------------------------------------------ */
/*  print_goal_splits                                                  */
/* ------------------------------------------------------------------ */

void print_goal_splits(const SegmentPDF *segments, int num_segments,
                       double percentile, const Duration *goal) {
    char buf[64];
    printf("Found percentile: %f\n", percentile);
    Duration accum = duration_zero();
    for (int i = 0; i < num_segments - 1; i++) {
        Duration t = pdf_time_at_percentile(&segments[i], percentile);
        accum.seconds += t.seconds;
        duration_format(&accum, buf, sizeof(buf));
        printf("%s\n", buf);
    }
    duration_format(goal, buf, sizeof(buf));
    printf("%s\n", buf);
}

/* ------------------------------------------------------------------ */
/*  simulate_runs  (parallel Monte Carlo via OpenMP)                   */
/* ------------------------------------------------------------------ */

long long simulate_runs(const SegmentPDF *segments, int num_segments,
                        int start_segment, const Duration *start_time,
                        const Duration *goal, const Duration *reset_times,
                        double target_pct, double *result) {

    SegmentLookup *lookups = precompute_lookups(segments, num_segments);
    if (!lookups) {
        *result = 0.0;
        return 0;
    }

    double goal_sec = goal->seconds;
    double start_sec = start_time->seconds;

    long long count = 0;
    long long success = 0;
    int convergence = 0;
    double prev_pct = -1.0;

    // Process in batches for progress tracking and convergence detection
    const long long BATCH = 64;
    const int MAX_BATCHES = 100000;  // safety limit
    int batches = 0;

    while (batches < MAX_BATCHES) {
        long long batch_count = 0;
        long long batch_success = 0;

#pragma omp parallel reduction(+ : batch_count, batch_success)
        {
            long long tid = omp_get_thread_num();
            RNG rng = rng_seed((uint64_t)(count + tid + 1) * 2654435761ULL);

            long long local_count = 0;
            long long local_success = 0;

            // Each thread does BATCH simulations independently
            for (long long b = 0; b < BATCH; b++) {
                double sum = start_sec;
                int skip = start_segment;
                int hit_reset = 0;

                // Each thread gets its own RNG stream
                double percentile = rng_double(&rng);
                int pidx = (int)(percentile * 100.0 + 0.5);
                if (pidx > 100) pidx = 100;
                if (pidx < 0) pidx = 0;

                int reroll = 1;  // chunk_size = 1 (no chunking)

                for (int idx = 0; idx < num_segments; idx++) {
                    // Reset check before skip (matches Python order)
                    if (reset_times && idx < num_segments - 1) {
                        if (reset_times[idx].seconds > 0 &&
                            sum > reset_times[idx].seconds) {
                            hit_reset = 1;
                            break;
                        }
                    }

                    if (skip > 0) {
                        skip--;
                        continue;
                    }

                    if (!lookups[idx].valid) continue;
                    sum += lookups[idx].times[pidx].seconds;

                    reroll--;
                    if (reroll <= 0) {
                        percentile = rng_double(&rng);
                        pidx = (int)(percentile * 100.0 + 0.5);
                        if (pidx > 100) pidx = 100;
                        if (pidx < 0) pidx = 0;
                        reroll = 1;
                    }
                }

                local_count++;
                // Python uses "s <= g" (inclusive)
                if (!hit_reset && sum <= goal_sec)
                    local_success++;
            }
            batch_count += local_count;
            batch_success += local_success;
        }

        count += batch_count;
        success += batch_success;
        batches++;

        // Convergence check after 10000 simulations
        if (count >= 10000) {
            double pct = (success / (double)count) * 100.0;

            if (target_pct > 0) {
                if (pct <= target_pct * 0.5 || pct >= target_pct * 1.5) {
                    break;
                }
                if (prev_pct < 0) {
                    prev_pct = pct;
                    convergence = 0;
                } else if (prev_pct == 0.0) {
                    // Don't converge on zero until we've seen enough runs
                    // (Python requires 10000+ total before convergence check)
                    if (pct < 0.05 && count > 100000) {
                        convergence++;
                        if (convergence >= 20) goto done;
                    } else {
                        prev_pct = pct;
                        convergence = 0;
                    }
                } else if (pct <= prev_pct * 1.001 && pct >= prev_pct * 0.999) {
                    convergence++;
                    if (convergence >= 50) goto done;
                } else {
                    prev_pct = pct;
                    convergence = 0;
                }
            } else {
                if (prev_pct < 0) {
                    prev_pct = pct;
                    convergence = 0;
                } else if (prev_pct == 0.0) {
                    if (pct < 0.05 && count > 200000) {
                        convergence++;
                        if (convergence >= 50) goto done;
                    } else {
                        prev_pct = pct;
                        convergence = 0;
                    }
                } else if (pct <= prev_pct * 1.001 && pct >= prev_pct * 0.999) {
                    convergence++;
                    if (convergence >= 100) goto done;
                } else {
                    prev_pct = pct;
                    convergence = 0;
                }
            }
        }
    }

done:
    *result = (count > 0) ? (success / (double)count) * 100.0 : 0.0;
    free_lookups(lookups, num_segments);
    return count;
}

/* ------------------------------------------------------------------ */
/*  repeated_odds  helper                                              */
/* ------------------------------------------------------------------ */

static double repeated_odds(double p0, double n) {
    return (1.0 - pow(1.0 - p0 / 100.0, (int)n)) * 100.0;
}

/* ------------------------------------------------------------------ */
/*  find_reset_splits                                                  */
/* ------------------------------------------------------------------ */

void find_reset_splits(const SegmentPDF *segments, int num_segments,
                       const Duration *goal, double max_iterations) {

    // Python: rt = [None] * len(segments), rt[start_split - 1] = mid
    // start_split = si + 1, so rt[si] = mid
    // Checks reset_times[idx] at idx = si+1 (first processed segment)
    // which is rt[si+1] = None, so we match Python exactly: times[si] = mid
    int num_reset_slots = num_segments;
    Duration *times = calloc(num_reset_slots, sizeof(Duration));
    if (!times) return;

    for (double iter = 0; iter < max_iterations; iter++) {
        Duration *times_prev = calloc(num_reset_slots, sizeof(Duration));
        memcpy(times_prev, times, num_reset_slots * sizeof(Duration));

        double base_pct;
        Duration zero = duration_zero();
        simulate_runs(segments, num_segments, 0, &zero,
                      goal, NULL, -1, &base_pct);

        printf("Base percentage: %.4f%%\n", base_pct);

        if (base_pct < 0.1) {
            fprintf(stderr, "Warning: Too low odds, estimates may be wrong\n");
        }

        for (int si = 0; si < num_reset_slots; si++) {
            int start_split = si + 1;
            printf("Split: %d / %d\n", start_split, num_segments);

            Duration lo = duration_zero();
            Duration hi = *goal;

            for (int search_iter = 0; search_iter < 50; search_iter++) {
                Duration sum_lo_hi = duration_add(&lo, &hi);
                Duration mid = duration_div(&sum_lo_hi, 2.0);

                char buf[64];
                duration_format(&mid, buf, sizeof(buf));
                printf("  Time: %s\n", buf);

                if (fabs(lo.seconds - hi.seconds) < 1.0) {
                    times[si] = mid;
                    break;
                }

                // run_progress_factor
                double run_progress_factor = goal->seconds / (goal->seconds - mid.seconds);
                if (run_progress_factor <= 1.0 || run_progress_factor > 100.0) {
                    times[si] = mid;
                    break;
                }

                double target = repeated_odds(base_pct, run_progress_factor);
                double actual;
                const Duration *reset_arr = times_prev;
                simulate_runs(segments, num_segments, start_split, &mid,
                              goal, reset_arr, target, &actual);

                // Python uses adjusted = repeated_odds(actual, run_progress_factor)
                // then compares adjusted to base_pct
                double adjusted = repeated_odds(actual, run_progress_factor);

                if (adjusted <= base_pct * 1.01 && adjusted >= base_pct * 0.99) {
                    times[si] = mid;
                    break;
                } else if (adjusted < base_pct) {
                    hi.seconds = mid.seconds;
                } else {
                    lo.seconds = mid.seconds;
                }
            }
        }

        printf("Iteration: %.0f\n", iter + 1);
        {
        char buf[64];
        for (int i = 0; i < num_segments - 1; i++) {
            duration_format(&times[i], buf, sizeof(buf));
            printf("%s\n", buf);
        }
        duration_format(goal, buf, sizeof(buf));
        printf("%s\n", buf);
        }

        free(times_prev);
    }

    free(times);
}

/* ------------------------------------------------------------------ */
/*  find_goal_by_percentage                                            */
/* ------------------------------------------------------------------ */

void find_goal_by_percentage(const SegmentPDF *segments, int num_segments,
                              const Duration *search_start, double target_pct) {
    Duration lo = duration_zero();
    Duration hi = *search_start;
    Duration new_goal = *search_start;

    for (int iter = 0; iter < 20; iter++) {
        Duration sum_lo_hi = duration_add(&lo, &hi);
        new_goal = duration_div(&sum_lo_hi, 2.0);
        double pct;
        Duration zero2 = duration_zero();
        simulate_runs(segments, num_segments, 0, &zero2,
                      &new_goal, NULL, -1, &pct);

        // Compare with some tolerance
        if (fabs(pct - target_pct) < 1.0)
            break;

        if (pct < target_pct)
            lo.seconds = new_goal.seconds;
        else
            hi.seconds = new_goal.seconds;
    }

    double pctile = find_percentile_for_goal(segments, num_segments, &new_goal);
    if (pctile >= 0)
        print_goal_splits(segments, num_segments, pctile, &new_goal);
}

/* ------------------------------------------------------------------ */
/*  run_sim                                                            */
/* ------------------------------------------------------------------ */

void run_sim(const SegmentPDF *segments, int num_segments,
             int start_segment, const Duration *start_time,
             const Duration *goal) {
    double pct;
    simulate_runs(segments, num_segments, start_segment, start_time,
                  goal, NULL, -1, &pct);
    printf("%.2f%%\n", pct);
}