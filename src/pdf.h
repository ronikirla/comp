#ifndef PDF_H
#define PDF_H

#include "time.h"
#include <stddef.h>

/**
 * A single weighted sample point in a segment's distribution.
 */
typedef struct {
    Duration time;
    double weight;
} Sample;

/**
 * Probability distribution for a single segment.
 * Samples are sorted by time and weights are normalized to sum to 1.0.
 */
typedef struct {
    Sample *samples;
    size_t count;
} SegmentPDF;

/**
 * Initialize an empty PDF. Caller must free with pdf_free when done.
 */
void pdf_init(SegmentPDF *pdf);

/**
 * Free memory allocated for a PDF.
 */
void pdf_free(SegmentPDF *pdf);

/**
 * Add a sample to the PDF.
 */
void pdf_add_sample(SegmentPDF *pdf, Duration time, double weight);

/**
 * Sort samples by time and normalize weights so they sum to 1.0.
 * Must be called before using pdf_time_at_percentile.
 */
void pdf_finalize(SegmentPDF *pdf);

/**
 * Compute the time at a given percentile (0.0 to 1.0) using the
 * kernel-density-style interpolation from the Python script.
 *
 * Edge handling:
 *   - If percentile <= first_weight/2, return first sample's time
 *   - If percentile >= 1 - last_weight/2, return last sample's time
 *   - Otherwise interpolate between samples
 */
Duration pdf_time_at_percentile(const SegmentPDF *pdf, double percentile);

#endif /* PDF_H */