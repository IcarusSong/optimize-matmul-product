# 编译器设置
CC = gcc
CFLAGS = -O3 -mavx512f -fopenmp -Wall -Wextra
LDFLAGS = -fopenmp

# 获取所有.c文件
SRCS = $(wildcard *.c)
# 生成对应的.o文件
OBJS = $(SRCS:.c=.o)
# 生成可执行文件名列表（与.c文件同名）
TARGETS = $(SRCS:.c=)

.PHONY: all clean

all: $(TARGETS)

# 模式规则：从.c生成可执行文件
%: %.o
	$(CC) $(LDFLAGS) -o $@ $^

# 模式规则：从.c生成.o
%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f $(OBJS) $(TARGETS)