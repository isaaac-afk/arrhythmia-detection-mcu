CC = gcc
CFLAGS = -std=c99 -O2 -Wall -Wextra
detector: main.c detector.c detector.h
	$(CC) $(CFLAGS) -o detector main.c detector.c -lm
clean:
	rm -f detector
