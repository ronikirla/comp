#ifndef PARSER_H
#define PARSER_H

#include "pdf.h"
#include <stddef.h>

/**
 * Weighting mode for recency weights.
 */
typedef enum {
    WEIGHT_GEOMETRIC,  // geometric decay with -w multiplier (default 0.75)
    WEIGHT_LINEAR      // linear decay, most recent = 1, oldest = 0
} WeightMode;

/**
 * Parse an .lss file and build an array of SegmentPDFs.
 *
 * Returns number of segments on success, -1 on failure.
 * Caller must free each pdf with pdf_free() and free the segments array.
 *
 * skip_ids_out: optional output array of per-segment skip ID lists.
 */
int parse_lss_file(const char *path,
                   SegmentPDF **segments_out,
                   WeightMode mode,
                   double geometric_weight);

/**
 * Free the segment array (does not free individual PDFs).
 */
void free_segments(SegmentPDF *segments, size_t count);

#endif /* PARSER_H */