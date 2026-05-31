#include "pdf.h"
#include <stdlib.h>
#include <string.h>

static int compare_samples_by_time(const void *a, const void *b) {
    const Sample *sa = (const Sample *)a;
    const Sample *sb = (const Sample *)b;
    if (sa->time.seconds < sb->time.seconds) return -1;
    if (sa->time.seconds > sb->time.seconds) return 1;
    return 0;
}

void pdf_init(SegmentPDF *pdf) {
    pdf->samples = NULL;
    pdf->count = 0;
}

void pdf_free(SegmentPDF *pdf) {
    free(pdf->samples);
    pdf->samples = NULL;
    pdf->count = 0;
}

void pdf_add_sample(SegmentPDF *pdf, Duration time, double weight) {
    Sample *tmp = realloc(pdf->samples, (pdf->count + 1) * sizeof(Sample));
    if (!tmp) return;
    pdf->samples = tmp;
    pdf->samples[pdf->count].time = time;
    pdf->samples[pdf->count].weight = weight;
    pdf->count++;
}

void pdf_finalize(SegmentPDF *pdf) {
    if (pdf->count == 0) return;

    // Sort by time
    qsort(pdf->samples, pdf->count, sizeof(Sample), compare_samples_by_time);

    // Normalize weights
    double sum = 0.0;
    for (size_t i = 0; i < pdf->count; i++) {
        sum += pdf->samples[i].weight;
    }
    if (sum > 0.0) {
        for (size_t i = 0; i < pdf->count; i++) {
            pdf->samples[i].weight /= sum;
        }
    }
}

Duration pdf_time_at_percentile(const SegmentPDF *pdf, double percentile) {
    if (!pdf || pdf->count == 0) return duration_zero();
    if (pdf->count == 1) return pdf->samples[0].time;

    // Edge cases
    if (percentile <= pdf->samples[0].weight / 2.0) {
        return pdf->samples[0].time;
    }
    if (percentile >= 1.0 - pdf->samples[pdf->count - 1].weight / 2.0) {
        return pdf->samples[pdf->count - 1].time;
    }

    // Linear scan through samples
    Sample prev = pdf->samples[0];
    double accuml = prev.weight / 2.0;

    for (size_t i = 1; i < pdf->count; i++) {
        Sample curr = pdf->samples[i];
        double step = (prev.weight + curr.weight) / 2.0;

        if (percentile >= accuml && percentile <= accuml + step) {
            double frac = (percentile - accuml) / step;
            Duration diff = duration_sub(&curr.time, &prev.time);
            Duration scaled = duration_mul(&diff, frac);
            return duration_add(&prev.time, &scaled);
        }

        accuml += step;
        prev = curr;
    }

    // Fallback: return the closest sample
    return pdf->samples[pdf->count - 1].time;
}