
ORIG_CXXFLAGS := $(CXXFLAGS)
CXXFLAGS := -Wall -std=c++17 -g $(ORIG_CXXFLAGS) \
	-Werror=return-type -pthread -mtune=znver2 \
	-mavx -mavx2 -mfma -mbmi -mbmi2 -ffast-math -fno-plt -flto=16

ifneq ($(SANITIZE),)
CXXFLAGS += -fsanitize=address
endif

ifeq ($(DEBUG),)
CXXFLAGS += -O3 -ftree-vectorize
endif

ifneq ($(GPROF_PROFILE),)
$(info Building for profiling)
CXXFLAGS += -pg
endif

CXXFLAGS += -MMD

all: funrender

clean:
	rm -f funrender funrender.o funsdl.o *.d

-include *.d

SDL_CFLAGS = $(shell pkg-config --cflags sdl2) \
	 $(shell pkg-config --cflags SDL2_ttf)
SDL_LIBS = $(shell pkg-config --libs sdl2) \
	$(shell pkg-config --libs SDL2_ttf)

CXXFLAGS += $(SDL_CFLAGS)

funrender: funsdl.o funrender.o affinity.o
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS) $(SDL_LIBS)

run: ./funrender
	./funrender

.PHONY: all clean run
