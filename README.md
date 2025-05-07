# ObjectDetection (ncnn-yolo12)
Repository này chứa ví dụ C++ về cách sử dụng framework `ncnn` với các mô hình YOLO12. Nếu bạn sử dụng `ncnn` và `opencv-mobile` đã được build sẵn, dự án sẽ được build mà không cần các thư viện động.

## Tổng quan
Dự án này minh họa cách sử dụng mô hình YOLO (You Only Look Once) để phát hiện đối tượng với framework NCNN trong C++. Nó cung cấp một ví dụ thực tế về việc chạy các mô hình YOLO để phát hiện đối tượng thời gian thực sử dụng NCNN, một framework suy luận mạng nơ-ron hiệu suất cao được tối ưu hóa cho các nền tảng di động.

## Cấu trúc dự án
```
.
├── main.cpp                   # Mã nguồn chính
├── CMakeLists.txt            # Cấu hình build CMake
└── run.sh                    # Script build và chạy
```

## Yêu cầu
1. **CMake** (phiên bản 3.20 trở lên)
2. **Framework NCNN**
   - Có thể build từ mã nguồn hoặc sử dụng bản build sẵn
   - Phiên bản hiện tại: 20250503
3. **OpenCV**
   - Các thành phần cần thiết: core, imgproc, videoio, imgcodecs, highgui, objdetect
   - Có thể sử dụng OpenCV đầy đủ hoặc opencv-mobile

## Cài đặt và Build

### Bước 1: Cài đặt các công cụ cần thiết
1. Cài đặt CMake:
   ```bash
   sudo apt-get install cmake
   ```

2. Cài đặt OpenCV:
   ```bash
   sudo apt-get install libopencv-dev
   ```
   Hoặc sử dụng opencv-mobile cho phiên bản nhẹ hơn.

3. Cài đặt NCNN:
   ```bash
   wget https://github.com/Tencent/ncnn/releases/download/20250503/ncnn-20250503-ubuntu-2404-shared.zip
   unzip ncnn-20250503-ubuntu-2404-shared.zip
   ```

### Bước 2: Chuẩn bị môi trường Python và chuyển đổi mô hình
1. Tạo và kích hoạt môi trường Python:
   ```bash
   python -m venv env
   source env/bin/activate
   pip install ultralytics
   ```

2. Chuyển đổi mô hình YOLO sang định dạng NCNN:
   ```bash
   yolo export model=yolo12n.pt format=ncnn imgsz=320 half=True
   ```

3. Tối ưu hóa mô hình NCNN:
   ```bash
   ./ncnn-20250503-ubuntu-2204-shared/bin/ncnnoptimize yolo12n_ncnn_model/model.ncnn.param yolo12n_ncnn_model/model.ncnn.bin yolo12n_opt.param yolo12n_opt.bin 65536
   ```

### Bước 3: Build dự án
1. Clone repository:
   ```bash
   git clone https://github.com/techmaster-vietnam/ObjectDetection.git
   cd ObjectDetection
   ```

2. Cấu hình build:
   - Chỉnh sửa `CMakeLists.txt` để đặt đường dẫn chính xác cho NCNN và OpenCV
   - Cập nhật các đường dẫn sau nếu cần:
     ```cmake
     link_directories(${CMAKE_CURRENT_SOURCE_DIR}/ncnn-20250503-ubuntu-2204-shared/lib)
     target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/ncnn-20250503-ubuntu-2204-shared/include)
     ```

3. Build dự án:
   ```bash
   ./run.sh
   ```
   Hoặc thủ công:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```

## Chạy ứng dụng

### Cách sử dụng cơ bản
```bash
./build/yolo_ncnn <file_param_mô_hình> <file_bin_mô_hình>
```

Ví dụ:
```bash
./build/yolo_ncnn yolo12n_opt.param yolo12n_opt.bin
```

### Sử dụng mô hình YOLO của riêng bạn
1. Xuất mô hình YOLO của bạn sang định dạng NCNN:
   ```bash
   yolo export model=your_model.pt format=ncnn
   ```

2. Cập nhật tên các lớp trong `main.cpp`:
   ```cpp
   static const char* class_names[] = {
       "person", "bicycle", "car",
       // Thêm các lớp của mô hình của bạn vào đây
   };
   ```

## Cấu trúc mã nguồn

### Các thành phần chính
1. **Tải và khởi tạo mô hình**
   - Tải các file mô hình NCNN (.param và .bin)
   - Khởi tạo mạng nơ-ron

2. **Xử lý ảnh**
   - Xử lý tải và tiền xử lý ảnh
   - Thực hiện suy luận sử dụng mô hình YOLO
   - Hậu xử lý kết quả phát hiện

3. **Phát hiện đối tượng**
   - Triển khai thuật toán phát hiện YOLO
   - Xử lý tính toán bounding box
   - Quản lý điểm tin cậy và dự đoán lớp

### Các hàm chính
- `load_model()`: Tải mô hình NCNN
- `detect()`: Thực hiện phát hiện đối tượng
- `draw_detections()`: Hiển thị kết quả phát hiện

## Xử lý sự cố
1. **Vấn đề khi build**
   - Đảm bảo tất cả các dependency đã được cài đặt đúng
   - Kiểm tra các đường dẫn trong CMakeLists.txt
   - Kiểm tra tính tương thích phiên bản OpenCV và NCNN

2. **Vấn đề khi chạy**
   - Kiểm tra định dạng file mô hình
   - Kiểm tra định dạng và kích thước ảnh đầu vào
   - Đảm bảo đủ tài nguyên hệ thống
