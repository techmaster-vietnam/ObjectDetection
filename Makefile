.PHONY: all clean test debug release

all: release

debug:
	mkdir -p build/debug
	cd build/debug && cmake -DCMAKE_BUILD_TYPE=Debug ../.. && make
	cp build/debug/yolo_ncnn ./yolo_ncnn_d

release:
	mkdir -p build/release
	cd build/release && cmake -DCMAKE_BUILD_TYPE=Release ../.. && make
	cp build/release/yolo_ncnn ./yolo_ncnn_r
	ln -sf yolo_ncnn_r yolo_ncnn

clean:
	rm -rf build 
	rm -f yolo_ncnn yolo_ncnn_d yolo_ncnn_r