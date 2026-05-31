CC       = gcc
CFLAGS   = -O2 -Wall -Wextra -std=c11 -fopenmp -Isrc -I/usr/include/libxml2
LDFLAGS  = -lm -lxml2 -fopenmp
SRCDIR   = src
BUILDDIR = build
TARGET   = comp

SRCS = $(SRCDIR)/main.c \
       $(SRCDIR)/time.c \
       $(SRCDIR)/pdf.c \
       $(SRCDIR)/parser.c \
       $(SRCDIR)/simulation.c

OBJS = $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(SRCS))

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c -o $@ $<

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -rf $(BUILDDIR) $(TARGET)

# Dependencies (headers)
$(BUILDDIR)/main.o:    $(SRCDIR)/main.c    $(SRCDIR)/parser.h $(SRCDIR)/simulation.h $(SRCDIR)/time.h $(SRCDIR)/pdf.h
$(BUILDDIR)/time.o:    $(SRCDIR)/time.c    $(SRCDIR)/time.h
$(BUILDDIR)/pdf.o:     $(SRCDIR)/pdf.c     $(SRCDIR)/pdf.h $(SRCDIR)/time.h
$(BUILDDIR)/parser.o:  $(SRCDIR)/parser.c  $(SRCDIR)/parser.h $(SRCDIR)/pdf.h $(SRCDIR)/time.h
$(BUILDDIR)/simulation.o: $(SRCDIR)/simulation.c $(SRCDIR)/simulation.h $(SRCDIR)/pdf.h $(SRCDIR)/time.h