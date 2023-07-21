###########################################################

## USER SPECIFIC DIRECTORIES ##

# CUDA directory:
CUDA_ROOT_DIR=/usr/local/cuda

##########################################################

## CC COMPILER OPTIONS ##

# CC compiler options:
CC=g++
CC_FLAGS=-std=c++17 -IIncludes -IHeaders -I$(CUDA_ROOT_DIR)/include -O3
CC_FLAGS2=$(CC_FLAGS) -DRAY_TRACING
CC_LIBS=-lpthread

##########################################################

## NVCC COMPILER OPTIONS ##

# NVCC compiler options:
NVCC=nvcc
NVCC_FLAGS=-IIncludes -IHeaders -O3 -dc
NVCC_FLAGS2= $(NVCC_FLAGS) -DRAY_TRACING
NVCC_LIBS=

# CUDA library directory:
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
# CUDA include directory:
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
# CUDA linking libraries:
CUDA_LINK_LIBS= -lcudart -pthread -lpthread -lcurand -lcudadevrt

##########################################################

## Project file structure ##

# Source file directory:
SRC_DIR = Sources

# Object file directory:
OBJ_DIR = bin
OBJ_DIR2 = bin_rtx

# Include header file diretory:
INC_DIR = Headers

##########################################################

## Make variables ##

# Target executable name:
EXE = mandelbrot

EXE2 = ray-tracing

# Object files:
OBJS = $(OBJ_DIR)/Main.o
OBJS += $(OBJ_DIR)/CudaUtil.o
OBJS += $(OBJ_DIR)/EncoderThread.o
OBJS += $(OBJ_DIR)/Kernel.o
OBJS += $(OBJ_DIR)/RenderThread.o
OBJS += $(OBJ_DIR)/Signal.o
OBJS += $(OBJ_DIR)/Maths/Maths.o
OBJS += $(OBJ_DIR)/RayTracing/FrameBuffer.o
OBJS += $(OBJ_DIR)/RayTracing/Mesh.o
OBJS += $(OBJ_DIR)/RayTracing/ModelLoader.o
OBJS += $(OBJ_DIR)/RayTracing/RayTracing.o
OBJS += $(OBJ_DIR)/RayTracing/Texture.o

OBJS2 = $(OBJ_DIR2)/Main.o
OBJS2 += $(OBJ_DIR2)/CudaUtil.o
OBJS2 += $(OBJ_DIR2)/EncoderThread.o
OBJS2 += $(OBJ_DIR2)/Kernel.o
OBJS2 += $(OBJ_DIR2)/RenderThread.o
OBJS2 += $(OBJ_DIR2)/Signal.o
OBJS2 += $(OBJ_DIR2)/Maths/Maths.o
OBJS2 += $(OBJ_DIR2)/RayTracing/FrameBuffer.o
OBJS2 += $(OBJ_DIR2)/RayTracing/Mesh.o
OBJS2 += $(OBJ_DIR2)/RayTracing/ModelLoader.o
OBJS2 += $(OBJ_DIR2)/RayTracing/RayTracing.o
OBJS2 += $(OBJ_DIR2)/RayTracing/Texture.o

##########################################################

## Compile ##

TARGETS = $(EXE) $(EXE2) clean

all : $(TARGETS)

# Link c++ and CUDA compiled object files to target executable:
$(EXE) : $(OBJS)
	$(NVCC) $(NVCC_FLAGS) -dlink $(OBJS) -o $(OBJ_DIR)/link.o $(NVCC_LIBS)
	$(CC) $(CC_FLAGS) $(OBJS) $(OBJ_DIR)/link.o -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)
	
$(EXE2) : $(OBJS2)
	$(NVCC) $(NVCC_FLAGS2) -dlink $(OBJS2) -o $(OBJ_DIR2)/link.o $(NVCC_LIBS)
	$(CC) $(CC_FLAGS2) $(OBJS2) $(OBJ_DIR2)/link.o -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .cpp file to object files:
$(OBJ_DIR)/Main.o : $(SRC_DIR)/Main.cpp
	mkdir -p $(OBJ_DIR)/Maths $(OBJ_DIR)/RayTracing
	$(CC) $(CC_FLAGS) -c $< -o $@ $(CC_LIBS)
	
$(OBJ_DIR2)/Main.o : $(SRC_DIR)/Main.cpp
	mkdir -p $(OBJ_DIR2)/Maths $(OBJ_DIR2)/RayTracing
	$(CC) $(CC_FLAGS2) -c $< -o $@ $(CC_LIBS)

# Compile C++ source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cpp $(INC_DIR)/%.hpp
	$(CC) $(CC_FLAGS) -c $< -o $@ $(CC_LIBS)

# Compile CUDA source files to object files:
$(OBJ_DIR)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@ $(NVCC_LIBS)
	
# Compile C++ source files to object files:
$(OBJ_DIR2)/%.o : $(SRC_DIR)/%.cpp $(INC_DIR)/%.hpp
	$(CC) $(CC_FLAGS2) -c $< -o $@ $(CC_LIBS)

# Compile CUDA source files to object files:
$(OBJ_DIR2)/%.o : $(SRC_DIR)/%.cu $(INC_DIR)/%.cuh
	$(NVCC) $(NVCC_FLAGS2) -c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory.
clean:
	$(RM) -r $(OBJ_DIR)/* $(OBJ_DIR2)/* $(EXE) $(EXE2)

