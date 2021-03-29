PROGRAM = lexTri
COMMON = common
CC = g++
CFLAGS = -g -O3

${PROGRAM}:	${COMMON}.o ${PROGRAM}.o main.cpp
	${CC} ${CFLAGS} main.cpp ${PROGRAM}.o ${COMMON}.o -llapack -lblas -lpthread -o ${PROGRAM}

${PROGRAM}.o:	${PROGRAM}.cpp ${PROGRAM}.hpp
	${CC} ${CFLAGS} -c -o ${PROGRAM}.o ${PROGRAM}.cpp

${COMMON}.o:	${COMMON}.cpp ${COMMON}.hpp
	${CC} ${CFLAGS} -c -o ${COMMON}.o ${COMMON}.cpp

clean:
	rm -f *.o ${PROGRAM}
