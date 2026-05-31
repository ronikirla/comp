#include "parser.h"
#include "time.h"
#include <libxml/parser.h>
#include <libxml/tree.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/**
 * Simple string set for tracking skip IDs.
 */
typedef struct {
    int *ids;
    size_t count;
    size_t capacity;
} IdSet;

static void id_set_init(IdSet *set) {
    set->ids = NULL;
    set->count = 0;
    set->capacity = 0;
}

static void id_set_add(IdSet *set, int id) {
    if (set->count >= set->capacity) {
        size_t new_cap = set->capacity == 0 ? 16 : set->capacity * 2;
        int *tmp = realloc(set->ids, new_cap * sizeof(int));
        if (!tmp) return;
        set->ids = tmp;
        set->capacity = new_cap;
    }
    set->ids[set->count++] = id;
}

static int id_set_contains(const IdSet *set, int id) {
    for (size_t i = 0; i < set->count; i++) {
        if (set->ids[i] == id) return 1;
    }
    return 0;
}

static void id_set_free(IdSet *set) {
    free(set->ids);
    set->ids = NULL;
    set->count = 0;
    set->capacity = 0;
}

/**
 * Get text content of the first child element with the given name.
 */
static char *get_child_text(xmlNodePtr parent, const char *child_name) {
    xmlNodePtr child = parent->children;
    while (child) {
        if (child->type == XML_ELEMENT_NODE &&
            strcmp((const char *)child->name, child_name) == 0) {
            xmlChar *txt = xmlNodeGetContent(child);
            if (txt) {
                // Trim whitespace
                char *s = (char *)txt;
                while (*s && isspace((unsigned char)*s)) s++;
                char *end = s + strlen(s) - 1;
                while (end > s && isspace((unsigned char)*end)) *end = '\0';
                return s;
            }
        }
        child = child->next;
    }
    return NULL;
}

/**
 * Iterate over all child elements with the given name.
 */
static xmlNodePtr get_first_child(xmlNodePtr parent, const char *name) {
    xmlNodePtr child = parent->children;
    while (child) {
        if (child->type == XML_ELEMENT_NODE &&
            strcmp((const char *)child->name, name) == 0) {
            return child;
        }
        child = child->next;
    }
    return NULL;
}

static int count_segments(xmlNodePtr segments_node) {
    int count = 0;
    xmlNodePtr child = segments_node->children;
    while (child) {
        if (child->type == XML_ELEMENT_NODE &&
            strcmp((const char *)child->name, "Segment") == 0) {
            count++;
        }
        child = child->next;
    }
    return count;
}

int parse_lss_file(const char *path, SegmentPDF **segments_out,
                   WeightMode mode, double geometric_weight) {
    *segments_out = NULL;

    xmlKeepBlanksDefault(0);
    xmlDocPtr doc = xmlReadFile(path, NULL, 0);
    if (!doc) {
        fprintf(stderr, "Error: Could not parse XML file '%s'\n", path);
        return -1;
    }

    xmlNodePtr root = xmlDocGetRootElement(doc);
    if (!root || (strcmp((const char *)root->name, "LiveSplitSave") != 0 &&
                  strcmp((const char *)root->name, "Run") != 0)) {
        fprintf(stderr, "Error: Not a valid LiveSplit save file\n");
        xmlFreeDoc(doc);
        return -1;
    }

    // Navigate to Segments
    xmlNodePtr segments_node = get_first_child(root, "Segments");
    if (!segments_node) {
        fprintf(stderr, "Error: No Segments found\n");
        xmlFreeDoc(doc);
        return -1;
    }

    int seg_count = count_segments(segments_node);
    if (seg_count == 0) {
        fprintf(stderr, "Error: No segments found\n");
        xmlFreeDoc(doc);
        return -1;
    }

    SegmentPDF *segments = calloc(seg_count, sizeof(SegmentPDF));
    if (!segments) {
        xmlFreeDoc(doc);
        return -1;
    }

    IdSet skip_prev;
    id_set_init(&skip_prev);

    int seg_idx = 0;
    xmlNodePtr seg_node = segments_node->children;

    while (seg_node && seg_idx < seg_count) {
        if (seg_node->type == XML_ELEMENT_NODE &&
            strcmp((const char *)seg_node->name, "Segment") == 0) {

            xmlNodePtr history = get_first_child(seg_node, "SegmentHistory");
            if (history) {
                // First pass: collect times and skip IDs
                typedef struct { Duration time; double weight; int id; } TempSample;
                TempSample *temp_samples = NULL;
                size_t temp_count = 0;
                size_t temp_cap = 0;

                IdSet current_skips;
                id_set_init(&current_skips);

                xmlNodePtr time_node = history->children;
                while (time_node) {
                    if (time_node->type == XML_ELEMENT_NODE &&
                        strcmp((const char *)time_node->name, "Time") == 0) {

                        // Get id attribute
                        int time_id = atoi((const char *)
                            xmlGetProp(time_node, (xmlChar *)"id"));

                        char *rt_text = get_child_text(time_node, "RealTime");

                        if (rt_text) {
                            // Has RealTime - it's a valid sample
                            // But check if this id is in skip_prev
                            if (!id_set_contains(&skip_prev, time_id)) {
                                if (temp_count >= temp_cap) {
                                    size_t new_cap = temp_cap == 0 ? 32 : temp_cap * 2;
                                    TempSample *tmp = realloc(temp_samples,
                                        new_cap * sizeof(TempSample));
                                    if (!tmp) { free(rt_text); continue; }
                                    temp_samples = tmp;
                                    temp_cap = new_cap;
                                }
                                Duration d;
                                if (duration_parse(rt_text, &d)) {
                                    temp_samples[temp_count].time = d;
                                    temp_samples[temp_count].id = time_id;
                                    temp_samples[temp_count].weight = 1.0;
                                    temp_count++;
                                }
                            }
                            free((void *)rt_text);
                        } else {
                            // No RealTime - mark as skip
                            id_set_add(&current_skips, time_id);
                        }
                    }
                    time_node = time_node->next;
                }

                // Apply weights based on mode
                // In the Python code, weights are applied in reverse order
                // (most recent first). The temp_samples are in document order
                // which is the same as the order they appear in the XML,
                // which is oldest-first. So we need to iterate backwards.
                if (mode == WEIGHT_GEOMETRIC) {
                    double w = 1.0;
                    for (size_t i = temp_count; i > 0; i--) {
                        w *= geometric_weight;
                        temp_samples[i - 1].weight = w;
                    }
                } else {
                    // Linear: most recent gets 1.0, oldest gets 0.0
                    // The Python code does: weight -= 1/len for each element
                    // iterating reverse, starting at weight = 1
                    // So first (most recent) gets weight = 1 - 1/len
                    // Wait, let me re-read the Python:
                    //   weight = 1
                    //   for split in reversed(...):
                    //       if --linear: weight -= 1/len
                    //       else: weight *= weight_mul
                    // So for linear: starting at 1, subtract 1/len each step
                    // Most recent (first in reversed) gets weight after subtracting
                    size_t len = temp_count;
                    double w = 1.0;
                    for (size_t i = 0; i < len; i++) {
                        w -= 1.0 / (double)len;
                        // i=0 is most recent (end of array)
                        temp_samples[len - 1 - i].weight = w;
                    }
                }

                // Add to PDF
                for (size_t i = 0; i < temp_count; i++) {
                    pdf_add_sample(&segments[seg_idx],
                                   temp_samples[i].time,
                                   temp_samples[i].weight);
                }
                free(temp_samples);

                // Update skip_prev for next segment
                id_set_free(&skip_prev);
                skip_prev = current_skips;
            }

            seg_idx++;
        }
        seg_node = seg_node->next;
    }

    // Finalize all PDFs (sort + normalize)
    for (int i = 0; i < seg_count; i++) {
        pdf_finalize(&segments[i]);
    }

    *segments_out = segments;
    id_set_free(&skip_prev);
    xmlFreeDoc(doc);
    return seg_count;
}

void free_segments(SegmentPDF *segments, size_t count) {
    if (!segments) return;
    for (size_t i = 0; i < count; i++) {
        pdf_free(&segments[i]);
    }
    free(segments);
}