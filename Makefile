BIN = ./bin/md-mpi
CC = mpicc
CFLAGS = -std=c99 -g -O5 -DDOUBLE -DDO_MPI
INC = -I ./src 
SRC = $(wildcard src/*.c)
LIB = -lm

all: $(BIN)

$(BIN):$(SRC)
	$(CC) $(CFLAGS) -o $@ $^ $(INC) $(LIB)

.PHONY:clean
clean:
	rm -rf $(BIN)
