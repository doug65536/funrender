
-include config.mk

# TC__PROGRAMS =

# # $(1) name
# # $(2) prefix
# define add_toolchain=
# TC_$(1)_PREFIX=$(2)
# TC_$(1)_
# endef


PROGRAM = ./funrender

WHICH ?= $(shell which which)
HOST ?=
PKG_CONFIG ?= $(shell $(WHICH) $(HOST)pkg-config)
UNAME ?= $(shell $(WHICH) uname)
ARCH ?= $(shell $(UNAME) -m)
NPROC ?= $(shell $(WHICH) nproc)
RM ?= $(shell $(WHICH) rm)
MKDIR ?= $(shell $(WHICH) mkdir)
SUDO ?= $(shell $(WHICH) sudo)
SETCAP ?= $(shell $(WHICH) setcap)
LLVM_PROFDATA ?= $(shell $(WHICH) llvm-profdata)
LLVM_COV ?= $(shell $(WHICH) llvm-cov)

CPU_COUNT := $(shell $(NPROC))

ifeq ($(ARCH),x86_64)
VECFLAGS += -DARCH_X86_64
#VECFLAGS = -DHAVE_HUGEPAGES=1

VECFLAGS += -mtune=znver2 \
	-msse3 -mssse3 -msse4.1 -msse4.2 \
	-mavx -mavx2 -mfma -mbmi -mbmi2 \
	-DHAVE_HUGEPAGES=1 -mavx512f -mavx512bw -mavx512dq
#-mavx512er
#-march=native

#VECFLAGS = -DHAVE_VEC128=1
#VECFLAGS += -DHAVE_VEC256=1
VECFLAGS += -DHAVE_VEC512=1
else ifeq ($(ARCH),aarch64)
VECFLAGS += -DARCH_AARCH64
VECFLAGS = -march=armv8-a -mcpu=cortex-a72 -DHAVE_VEC128=1
else
$(error Unknown architecture)
endif

#	-fvar-tracking -fvar-tracking-assignments

CXXFLAGS := -Wall -Wextra \
	-std=c++17 \
	-ggdb3 \
	-Werror=return-type \
	-Werror=reorder \
	-Werror=format \
	-pthread \
	-fno-plt \
	-I$(SRC_DIR) \
	$(VECFLAGS) \
	$(CXXFLAGS)


#=$(CPU_COUNT)

#CXXFLAGS += -fsanitize=thread
#CXXFLAGS += -fsanitize=address
#CXXFLAGS += -fsanitize=undefined

ifneq ($(SANITIZE),)
CXXFLAGS += -fsanitize=$(SANITIZE)
endif

ifeq ($(DEBUG),)
CXXFLAGS += -Ofast -ftree-vectorize
CXXFLAGS += -flto -ffast-math -fassociative-math -fstrict-aliasing
else
CXXFLAGS += -O0
endif

ifneq ($(GPROF_PROFILE_HEAVY),)
$(info Building for extreme profiling)
#CXXFLAGS += -pg -fprofile-arcs -ftest-coverage -fprofile-instr-generate
CXXFLAGS += -pg -fprofile-instr-generate -fcoverage-mapping
else
ifneq ($(GPROF_PROFILE),)
$(info Building for light profiling)
CXXFLAGS += -pg
else
$(info No profiling option)
endif
endif

CXXFLAGS += -MMD

OBJS := pool.o funsdl.o funrender.o affinity.o \
	stb_image_impl.o text.o objmodel.o \
	3ds.o cpu_usage.o huge_alloc.o

TESTOBJS := test/test_vector.o

all: funrender test-funrender

clean:
	$(RM) -f funrender \
		$(OBJS) \
		$(OBJS:.o=.d) \
		$(OBJS:.o=.gcda) \
		$(OBJS:.o=.gcno) \
		$(TESTOBJS) \
		$(TESTOBJS:.o=.d) \
		default.profdata \
		funrender.profdata \
		gmon.out \
		build.log

-include $(OBJS:.o=.d)
-include $(TESTOBJS:.o=.d)

GTEST_CFLAGS = $(shell $(PKG_CONFIG) --cflags gtest)
GTEST_LIBS = $(shell $(PKG_CONFIG) --libs gtest)

SDL_CFLAGS = $(shell $(PKG_CONFIG) --cflags sdl2) \
	 $(shell $(PKG_CONFIG) --cflags SDL2_ttf)
SDL_LIBS = $(shell $(PKG_CONFIG) --libs sdl2) \
	$(shell $(PKG_CONFIG) --libs SDL2_ttf)

GLM_CFLAGS = $(shell $(PKG_CONFIG) --cflags glm)
GLM_LIBS = $(shell $(PKG_CONFIG) --libs glm)

CXXFLAGS += $(SDL_CFLAGS) $(GLM_CFLAGS)
LDFLAGS += $(SDL_LIBS) $(GLM_LIBS)

$(BUILD_DIR)/funrender: $(patsubst %,$(BUILD_DIR)/%,$(OBJS))
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

grantlargepages: $(PROGRAM)
	$(SUDO) $(SETCAP) cap_ipc_lock=+ep $(PROGRAM)

run: $(PROGRAM)
	$(PROGRAM)

test: test-funrender
	./$<

test-funrender: $(TESTOBJS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS) \
		$(GTEST_CFLAGS) $(GTEST_LIBS)

line-profile: default.profdata
	$(LLVM_COV) show -instr-profile=$< $(PROGRAM)

default.profdata: default.profdata
	$(LLVM_PROFDATA) merge -o $@ $^

scanview-clang:
	CXX=clang++ CC=clang scan-build --use-c++=clang++ make -B

.PHONY: all clean run grantlargepages scanview-clang line-profile

#$(info BUILD_DIR=$(BUILD_DIR) and SRC_DIR=$(SRC_DIR))
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) -o $@ -c $< $(CXXFLAGS)

$(BUILD_DIR)/test/%.o: $(SRC_DIR)/test/%.cpp
	$(MKDIR) -p $(@D)
	$(CXX) -o $@ -c $< $(CXXFLAGS)
