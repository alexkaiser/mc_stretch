EXECUTABLES = stretch_move_main

all: $(EXECUTABLES)
	

UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
	CFLAGS := -Wall -O3 -std=c99 
	LDLIBS := -framework OpenCL
endif

ifeq ($(UNAME), Linux)
	CFLAGS := -Wall -O3 -std=gnu99 
	LDLIBS := -lrt -lOpenCL -lm
endif


stretch_move_main: stretch_move_main.o cl-helper.o stretch_move_util.o stretch_move_sampler.o
	gcc $(LDLIBS) $(CFLAGS) -o stretch_move_main $^

stretch_move_main.o: stretch_move_main.c 
	gcc -c $(CFLAGS) stretch_move_main.c

stretch_move_sampler.o: stretch_move_sampler.c 
	gcc -c $(CFLAGS) stretch_move_sampler.c

stretch_move_util.o: stretch_move_util.c 
	gcc -c $(CFLAGS) stretch_move_util.c

cl-helper.o: cl-helper.c 
	gcc -c $(CFLAGS) cl-helper.c


clean:
	rm -f $(EXECUTABLES) *.o
