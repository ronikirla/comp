#include "parser.h"
#include "simulation.h"
#include "optimal_reset.h"
#include "time.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <omp.h>

static void print_usage(const char *prog) {
    fprintf(stderr, "\n"
        "Usage: %s <splits.lss> <target_time> [options]\n"
        "\n"
        "This program creates balanced LiveSplit comparisons with customizable\n"
        "parameters such as goal time and recency weighing. Outputs split times\n"
        "in text that can be copy-pasted into a custom LiveSplit comparison.\n"
        "\n"
        "Target time should be given as hh:mm:ss, mm:ss, or ss (fractional allowed).\n"
        "\n"
        "Examples:\n"
        "  %s splits.lss 0:12:34\n"
        "  %s splits.lss 0:12:34 --linear\n"
        "  %s splits.lss 0:12:34 -w 0.9 --reset\n"
        "  %s splits.lss 0:12:34 --sim 2 1:30.5\n"
        "\n"
        "Options:\n"
        "  -w, --weight <value>\n"
        "      Recency weighing with geometric decay. Default: 0.75 (LiveSplit default).\n"
        "      Recommended for small amounts of data (after a new route discovery).\n"
        "\n"
        "  --linear\n"
        "      Recency weighing with linear decay. Most recent = 1.0, oldest = 0.0.\n"
        "      Recommended long-term when most data is from your current route.\n"
        "\n"
        "  --sim [start_split] [start_time]\n"
        "      Simulate runs using existing split data and print success odds.\n"
        "      Optionally start from a given split number and accumulated time.\n"
        "\n"
        "  --reset [iterations]\n"
        "      Create a reset comparison: find times where it's equally likely to\n"
        "      hit the goal from the current split as from a fresh reset.\n"
        "      Accounts for IRL time investment. Default iterations: 1.\n"
        "\n"
         "  --findgoal <percentage>\n"
         "      Find the goal time that would give the specified success percentage,\n"
         "      then output the corresponding split times.\n"
         "\n"
         "  --out <file>\n"
         "      Write --findgoal result to the specified file.\n"
         "\n"
         "  --reset-optimal\n"
         "      Compute optimal reset thresholds via value iteration: for each split,\n"
         "      binary-search the time where continuing vs resetting gives equal PB odds.\n"
         "      Minimizes expected time to PB (V(0,0) = E[cycle] / P(PB per cycle)).\n"
         "\n"
         "  --optimal-bins <n>\n"
         "      Number of discretization bins per segment distribution for --reset-optimal.\n"
         "      More bins = finer resolution but slower. Default: 100.\n"
         "\n"
         "  --optimal-iter <n>\n"
         "      Maximum value-iteration rounds for --reset-optimal. Each round recomputes\n"
         "      all thresholds then evaluates the policy. Stops early on convergence.\n"
         "      Default: 50.\n"
         "\n"
         "  --optimal-tol <value>\n"
         "      Relative-change convergence tolerance for --reset-optimal.\n"
         "      Default: 0.001.\n"
        "\n"
        "  -t, --threads <count>\n"
        "      Number of threads for parallel simulation. Default: all available.\n"
        "\n"
        "  --help\n"
        "      Show this help message.\n"
        "\n",
        prog, prog, prog, prog, prog);
}

int main(int argc, char *argv[]) {
    if (argc < 2 || strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
        print_usage(argv[0]);
        return (argc < 2) ? 1 : 0;
    }

    // Parse required positional arguments
    const char *file_path = NULL;
    const char *goal_str = NULL;
    int pos_idx = 1;

    while (pos_idx < argc && argv[pos_idx][0] == '-') {
        pos_idx++;
    }
    if (pos_idx < argc) file_path = argv[pos_idx++];
    if (pos_idx < argc) goal_str = argv[pos_idx++];

    if (!file_path || !goal_str) {
        fprintf(stderr, "Error: Missing file path or goal time\n");
        print_usage(argv[0]);
        return 1;
    }

    // Parse optional arguments
    WeightMode weight_mode = WEIGHT_GEOMETRIC;
    double weight_mul = 0.75;
    int mode_sim = 0;
    int mode_reset = 0;
    int mode_reset_optimal = 0;
    int mode_findgoal = 0;
    int reset_optimal_bins = 100;
    int reset_optimal_max_iter = 50;
    double reset_optimal_tol = 0.001;
    int sim_start_split = 0;
    Duration sim_start_time = duration_zero();
    double reset_iterations = 1.0;
    double findgoal_pct = -1.0;
    const char *out_file = NULL;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-w") == 0 || strcmp(argv[i], "--weight") == 0) {
            if (i + 1 < argc) {
                weight_mul = atof(argv[++i]);
            }
        } else if (strcmp(argv[i], "--linear") == 0) {
            weight_mode = WEIGHT_LINEAR;
        } else if (strcmp(argv[i], "--sim") == 0) {
            mode_sim = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                sim_start_split = atoi(argv[++i]);
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    duration_parse(argv[++i], &sim_start_time);
                }
            }
        } else if (strcmp(argv[i], "--reset") == 0) {
            mode_reset = 1;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                reset_iterations = atof(argv[++i]);
            }
        } else if (strcmp(argv[i], "--reset-optimal") == 0) {
            mode_reset_optimal = 1;
        } else if (strcmp(argv[i], "--findgoal") == 0) {
            mode_findgoal = 1;
            if (i + 1 < argc) {
                findgoal_pct = atof(argv[++i]);
            }
        } else if (strcmp(argv[i], "--out") == 0) {
            if (i + 1 < argc) {
                out_file = argv[++i];
            }
        } else if (strcmp(argv[i], "--optimal-bins") == 0) {
            if (i + 1 < argc) {
                reset_optimal_bins = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "--optimal-iter") == 0) {
            if (i + 1 < argc) {
                reset_optimal_max_iter = atoi(argv[++i]);
            }
        } else if (strcmp(argv[i], "--optimal-tol") == 0) {
            if (i + 1 < argc) {
                reset_optimal_tol = atof(argv[++i]);
            }
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--threads") == 0) {
            if (i + 1 < argc) {
                omp_set_num_threads(atoi(argv[++i]));
            }
        }
    }

    // Parse goal time
    Duration goal;
    if (!duration_parse(goal_str, &goal)) {
        fprintf(stderr, "Error: Invalid goal time format '%s'\n", goal_str);
        return 1;
    }

    // Parse LSS file
    SegmentPDF *segments = NULL;
    int num_segments = parse_lss_file(file_path, &segments, weight_mode, weight_mul);
    if (num_segments < 0 || !segments) {
        fprintf(stderr, "Error: Failed to parse '%s'\n", file_path);
        return 1;
    }

    printf("Loaded %d segments from '%s'\n", num_segments, file_path);
    printf("Weight mode: %s (%s)\n",
           weight_mode == WEIGHT_GEOMETRIC ? "geometric" : "linear",
           weight_mode == WEIGHT_GEOMETRIC ? "default -w 0.75" : "linear decay");

    /* ---------------------------------------------------------------- */
    /*  Execute requested mode                                           */
    /* ---------------------------------------------------------------- */

    if (mode_sim) {
        run_sim(segments, num_segments, sim_start_split, &sim_start_time, &goal);
    } else if (mode_findgoal) {
        if (findgoal_pct <= 0) {
            fprintf(stderr, "Error: --findgoal requires a target percentage\n");
            free_segments(segments, num_segments);
            return 1;
        }
        find_goal_by_percentage(segments, num_segments, &goal, findgoal_pct);

        if (out_file) {
            FILE *f = fopen(out_file, "w");
            if (f) {
                double total_sec = goal.seconds;
                int hours = (int)(total_sec / 3600.0);
                int minutes = (int)((total_sec - hours * 3600.0) / 60.0);
                int secs = (int)(total_sec - hours * 3600.0 - minutes * 60.0);
                char hrs_prefix[16] = {0};
                if (hours > 0) snprintf(hrs_prefix, sizeof(hrs_prefix), "%02d:", hours);
                fprintf(f, "Simulated top %.1f%% time: %s%d:%02d\n",
                        findgoal_pct, hrs_prefix, minutes, secs);
                fclose(f);
                printf("Result written to '%s'\n", out_file);
            }
        }
    } else if (mode_reset) {
        find_reset_splits(segments, num_segments, &goal, reset_iterations);
    } else if (mode_reset_optimal) {
        OptimalResetResult result = compute_optimal_reset(segments, num_segments,
                                                           &goal,
                                                           reset_optimal_bins,
                                                           reset_optimal_max_iter,
                                                           reset_optimal_tol);
        print_optimal_reset_result(&result, segments, num_segments, &goal);
        free_optimal_reset_result(&result);
    } else {
        // Default: find_goal_splits
        double pctile = find_percentile_for_goal(segments, num_segments, &goal);
        if (pctile >= 0.0) {
            print_goal_splits(segments, num_segments, pctile, &goal);
        } else {
            fprintf(stderr, "Error: Could not find target percentile. Check goal time.\n");
            free_segments(segments, num_segments);
            return 1;
        }
    }

    free_segments(segments, num_segments);
    return 0;
}