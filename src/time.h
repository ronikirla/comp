#ifndef TIME_H
#define TIME_H

#include <stdbool.h>

typedef struct {
    double seconds;
} Duration;

/**
 * Parse a duration string in hh:mm:ss or mm:ss or ss format.
 * Returns true on success, false on failure.
 */
bool duration_parse(const char *str, Duration *out);

/**
 * Parse a duration string that may contain fractional seconds, e.g. "12:34.567"
 */
bool duration_parse_float(const char *str, Duration *out);

/**
 * Format a duration as a human-readable string like "1:23:45.678"
 * Writes into buf (must be at least 32 bytes).
 */
void duration_format(const Duration *d, char *buf, int bufsize);

/**
 * Basic arithmetic operations on Duration.
 */
Duration duration_add(const Duration *a, const Duration *b);
Duration duration_sub(const Duration *a, const Duration *b);
Duration duration_mul(const Duration *a, double factor);
Duration duration_div(const Duration *a, double divisor);
Duration duration_zero(void);

bool duration_lt(const Duration *a, const Duration *b);
bool duration_gt(const Duration *a, const Duration *b);
bool duration_eq_approx(const Duration *a, const Duration *b, double epsilon);

#endif /* TIME_H */