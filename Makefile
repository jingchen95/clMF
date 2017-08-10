CC = g++
INCLUDE = -I/home/intel/opencl-1.2-3.1.1.11385/include/
LIB = -lOpenCL
TARGET = clmf
OBJECTS= opencl.o util.o cj.o 
#.SUFFIXES:.o .cpp .h
#.cpp.o:
#	g++ $(INCLUDE) -c -g -o $@ $<
$(TARGET): $(OBJECTS)
	$(CC) -g -o $@ $^ $(INCLUDE) $(LIB)

opencl.o: OpenCL.cpp cj.h util.h tools.h 
	$(CC) $(INCLUDE) -c -g -o $@ $<

util.o: util.cpp util.h
	$(CC) $(INCLUDE) -c -g -o $@ $<

cj.o: cj.cpp cj.h
	$(CC) $(INCLUDE) -c -g -o $@ $<

.PHONY: clean

clean:
	rm -rf *.o $(TARGET)
