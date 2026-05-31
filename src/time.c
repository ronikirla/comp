#include "time.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

static double parse_number(const char *start, const char **end) {
    char *stop;
    double val = strtod(start, &stop);
    *end = stop;
    return val;
}

bool duration_parse_float(const char *str, Duration *out) {
    if (!str || !out) return false;

    double total_seconds = 0.0;
    const char *p = str;
    int parts = 0;  // track how many colon-separated parts we've processed

    while (*p) {
        double val = 0.0;
        const char *num_end;

        // Skip whitespace
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;

        val = parse_number(p, &num_end);
        p = num_end;

        // Skip whitespace between number and next part
        while (*p && isspace((unsigned char)*p)) p++;

        if (*p == ':') {
            p++;
            if (parts == 0) {
                total_seconds += val * 3600.0;  // hours
            } else if (parts == 1) {
                total_seconds += val * 60.0;     // minutes
            }
            parts++;
        } else {
            total_seconds += val;  // seconds (last part)
            break;
        }
    }

    out->seconds = total_seconds;
    return true;
}

bool duration_parse(const char *str, Duration *out) {
    return duration_parse_float(str, out);
}

void duration_format(const Duration *d, char *buf, int bufsize) {
    if (!d || !buf || bufsize <= 0) return;

    double abs_sec = fabs(d->seconds);
    int hours = (int)(abs_sec / 3600.0);
    int minutes = (int)((abs_sec - hours * 3600.0) / 60.0);
    double secs = abs_sec - hours * 3600.0 - minutes * 60.0;

    const char *sign = d->seconds < 0 ? "-" : "";

    if (hours > 0) {
        snprintf(buf, bufsize, "%s%d:%02d:%06.3f", sign, hours, minutes, secs);
    } else {
        snprintf(buf, bufsize, "%s%d:%06.3f", sign, minutes, secs);
    }
}

Duration duration_add(const Duration *a, const Duration *b) {
    Duration result;
    result.seconds = a->seconds + b->seconds;
    return result;
}

Duration duration_sub(const Duration *a, const Duration *b) {
    Duration result;
    result.seconds = a->seconds - b->seconds;
    return result;
}

Duration duration_mul(const Duration *a, double factor) {
    Duration result;
    result.seconds = a->seconds * factor;
    return result;
}

Duration duration_div(const Duration *a, double divisor) {
    Duration result;
    result.seconds = a->seconds / divisor;
    return result;
}

Duration duration_zero(void) {
    Duration result = {0.0};
    return result;
}

bool duration_lt(const Duration *a, const Duration *b) {
    return a->seconds < b->seconds;
}

bool duration_gt(const Duration *a, const Duration *b) {
    return a->seconds > b->seconds;
}

bool duration_eq_approx(const Duration *a, const Duration *b, double epsilon) {
    return fabs(a->seconds - b->seconds) < epsilon;
}