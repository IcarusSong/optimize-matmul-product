# 编译器设置
CC      = gcc
CFLAGS  = -O3 -mavx512f -fopenmp -Wall -Wextra
LDFLAGS = -fopenmp

# 目录设置
SRC_DIR = src
OBJ_DIR = obj

# 获取所有 .c 文件，并生成对应的可执行文件路径
SRCS     = $(wildcard $(SRC_DIR)/*.c)
TARGETS  = $(patsubst $(SRC_DIR)/%.c, $(OBJ_DIR)/%, $(SRCS))

# 确保 obj 目录存在
$(shell mkdir -p $(OBJ_DIR))

.PHONY: all clean

all: $(TARGETS)

# 编译规则：每个 .c 文件生成一个同名可执行文件
$(OBJ_DIR)/%: $(SRC_DIR)/%.c
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $<

clean:
	rm -rf $(OBJ_DIR)/*