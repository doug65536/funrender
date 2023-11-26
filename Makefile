
-include config.mk

PROGRAM = ./funrender

HOST ?=
PKG_CONFIG ?= $(HOST)pkg-config
UNAME ?= uname
ARCH ?= $(shell $(UNAME) -m)
NPROC ?= nproc
RM ?= rm
SUDO ?= sudo
SETCAP ?= setcap

CPU_COUNT := $(shell $(NPROC))


ifeq ($(ARCH),x86_64)
CXXFLAGS += -DARCH_X86_64
VECFLAGS = -mtune=znver2 -mavx -mavx2 -mfma -mbmi -mbmi2 \
	-DHAVE_HUGEPAGES=1 -DHAVE_VEC256=1
else ifeq ($(ARCH),aarch64)
CXXFLAGS += -DARCH_AARCH64
VECFLAGS = -march=armv8-a -mcpu=cortex-a72 -DHAVE_VEC128=1
else
$(error Unknown architecture)
endif

ORIG_CXXFLAGS := $(CXXFLAGS)
CXXFLAGS := -Wall -std=c++17 -g $(ORIG_CXXFLAGS) \
	-Werror=return-type -pthread \
	$(VECFLAGS) -ffast-math -fno-plt


#=$(CPU_COUNT)

#CXXFLAGS += -fsanitize=thread
#CXXFLAGS += -fsanitize=address
#CXXFLAGS += -fsanitize=undefined

ifneq ($(SANITIZE),)
CXXFLAGS += -fsanitize=$(SANITIZE)
endif

ifeq ($(DEBUG),)
CXXFLAGS += -Ofast -ftree-vectorize -flto
else
CXXFLAGS += -O0
endif

ifneq ($(GPROF),)
CXXFLAGS += -pg
endif

ifneq ($(GPROF_PROFILE),)
$(info Building for profiling)
CXXFLAGS += -pg
endif

CXXFLAGS += -MMD

OBJS := pool.o funsdl.o funrender.o affinity.o stb_image_impl.o

all: funrender

clean:
	$(RM) -f funrender $(OBJS) *.d

-include *.d

SDL_CFLAGS = $(shell $(PKG_CONFIG) --cflags sdl2) \
	 $(shell $(PKG_CONFIG) --cflags SDL2_ttf)
SDL_LIBS = $(shell $(PKG_CONFIG) --libs sdl2) \
	$(shell $(PKG_CONFIG) --libs SDL2_ttf)

CXXFLAGS += $(SDL_CFLAGS)

funrender: $(OBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS) $(SDL_LIBS)

grantlargepages: $(PROGRAM)
	$(SUDO) $(SETCAP) cap_ipc_lock=+ep $(PROGRAM)

run: $(PROGRAM)
	$(PROGRAM)

scanview-clang:
	CXX=clang++ CC=clang scan-build --use-c++=clang++ make -B

.PHONY: all clean run grantlargepages scanview-clang
