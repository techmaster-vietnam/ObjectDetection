Dưới đây là hướng dẫn chi tiết về các công cụ trong thư mục `ncnn-20250503-ubuntu-2204-shared/bin/` của NCNN framework, bao gồm chức năng, cách sử dụng, cú pháp lệnh, các bước thực hiện, lưu ý và ví dụ cụ thể cho từng công cụ: `caffe2ncnn`, `mxnet2ncnn`, `ncnn2mem`, `ncnnmerge`, `onnx2ncnn`, `darknet2ncnn`, `ncnn2int8`, `ncnn2table`, và `ncnnoptimize`. Các công cụ này được sử dụng để chuyển đổi, tối ưu hóa và triển khai mô hình học sâu trong NCNN, một framework tối ưu cho thiết bị di động và nhúng.

---

### 1. **`caffe2ncnn`**
#### **Chức năng**
- Chuyển đổi mô hình từ định dạng **Caffe** (`.prototxt` và `.caffemodel`) sang định dạng NCNN (`.param` và `.bin`).
- Dùng để triển khai các mô hình được huấn luyện bằng Caffe trên các nền tảng hỗ trợ NCNN (như Android, iOS, hoặc thiết bị nhúng).

#### **Cú pháp**
```bash
./caffe2ncnn model.prototxt model.caffemodel output.param output.bin
```
- **model.prototxt**: File mô tả kiến trúc mạng của Caffe.
- **model.caffemodel**: File chứa trọng số của mô hình Caffe.
- **output.param**: File NCNN mô tả kiến trúc mạng.
- **output.bin**: File NCNN chứa trọng số.

#### **Các bước sử dụng**
1. **Chuẩn bị mô hình Caffe**:
   - Đảm bảo bạn có file `.prototxt` (mô tả mạng) và `.caffemodel` (trọng số).
   - Ví dụ: Mô hình MobileNet (`mobilenet.proTOTxt` và `mobilenet.caffemodel`).
2. **Chạy lệnh chuyển đổi**:
   ```bash
   ./caffe2ncnn mobilenet.prototxt mobilenet.caffemodel mobilenet.param mobilenet.bin
   ```
3. **Kiểm tra đầu ra**:
   - Kiểm tra file `mobilenet.param` và `mobilenet.bin` được tạo.
   - Sử dụng các công cụ NCNN khác (như `ncnnoptimize`) để tối ưu hóa hoặc triển khai mô hình.

#### **Lưu ý**
- **Phiên bản Caffe**: Đảm bảo mô hình Caffe tương thích với NCNN (một số layer tùy chỉnh có thể không được hỗ trợ).
- **Tối ưu hóa**: Sau khi chuyển đổi, nên dùng `ncnnoptimize` để giảm kích thước mô hình và tăng tốc suy luận.
- **Lỗi thường gặp**: Nếu gặp lỗi về layer không hỗ trợ, kiểm tra danh sách layer được NCNN hỗ trợ trên GitHub hoặc chỉnh sửa file `.prototxt`.

#### **Ví dụ**
Chuyển đổi mô hình MobileNet từ Caffe sang NCNN:
```bash
./caffe2ncnn mobilenet.prototxt mobilenet.caffemodel mobilenet.param mobilenet.bin
```

---

### 2. **`mxnet2ncnn`**
#### **Chức năng**
- Chuyển đổi mô hình từ định dạng **MXNet** (`.json` và `.params`) sang định dạng NCNN (`.param` và `.bin`).
- Dùng để triển khai các mô hình được huấn luyện bằng MXNet trên các nền tảng NCNN.

#### **Cú pháp**
```bash
./mxnet2ncnn model-symbol.json model.params output.param output.bin
```
- **model-symbol.json**: File mô tả kiến trúc mạng của MXNet.
- **model.params**: File chứa trọng số của mô hình MXNet.
- **output.param**: File NCNN mô tả kiến trúc mạng.
- **output.bin**: File NCNN chứa trọng số.

#### **Các bước sử dụng**
1. **Chuẩn bị mô hình MXNet**:
   - Xuất mô hình từ MXNet thành file `.json` (kiến trúc) và `.params` (trọng số).
   - Ví dụ: Mô hình ResNet-18 (`resnet18-symbol.json` và `resnet18.params`).
2. **Chạy lệnh chuyển đổi**:
   ```bash
   ./mxnet2ncnn resnet18-symbol.json resnet18.params resnet18.param resnet18.bin
   ```
3. **Kiểm tra và tối ưu hóa**:
   - Kiểm tra file `resnet18.param` và `resnet18.bin`.
   - Dùng `ncnnoptimize` để tối ưu hóa mô hình nếu cần.

#### **Lưu ý**
- **Layer hỗ trợ**: NCNN chỉ hỗ trợ một tập hợp layer của MXNet. Kiểm tra tài liệu NCNN để đảm bảo mô hình của bạn tương thích.
- **Phiên bản MXNet**: Sử dụng phiên bản MXNet tương thích với công cụ này (thường là các phiên bản cũ hơn).
- **Lỗi JSON/params không khớp**: Đảm bảo file `.json` và `.params` được xuất từ cùng một mô hình.

#### **Ví dụ**
Chuyển đổi mô hình ResNet-18 từ MXNet sang NCNN:
```bash
./mxnet2ncnn resnet18-symbol.json resnet18.params resnet18.param resnet18.bin
```

---

### 3. **`ncnn2mem`**
#### **Chức năng**
- Chuyển đổi mô hình NCNN (`.param` và `.bin`) thành mã C (file `.cpp` và `.h`) để nhúng trực tiếp trọng số và cấu trúc mạng vào mã nguồn.
- Giảm sự phụ thuộc vào file mô hình bên ngoài, hữu ích cho các ứng dụng nhúng hoặc khi triển khai trên thiết bị không có hệ thống file.

#### **Cú pháp**
```bash
./ncnn2mem model.param model.bin output.cpp output.h
```
- **model.param**: File NCNN mô tả kiến trúc mạng.
- **model.bin**: File NCNN chứa trọng số.
- **output.cpp**: File mã C chứa dữ liệu trọng số và cấu trúc.
- **output.h**: File header định nghĩa các biến và hàm liên quan.

#### **Các bước sử dụng**
1. **Chuẩn bị mô hình NCNN**:
   - Đảm bảo bạn có file `.param` và `.bin` (ví dụ, từ `onnx2ncnn` hoặc `caffe2ncnn`).
2. **Chạy lệnh chuyển đổi**:
   ```bash
   ./ncnn2mem mobilenet.param mobilenet.bin mobilenet.cpp mobilenet.h
   ```
3. **Tích hợp vào mã nguồn**:
   - Bao gồm `mobilenet.h` trong mã C++ của bạn.
   - Sử dụng hàm được tạo trong `mobilenet.cpp` để tải mô hình mà không cần file `.param` hoặc `.bin`.
4. **Biên dịch**:
   - Biên dịch mã với NCNN library:
     ```bash
     g++ -o app main.cpp mobilenet.cpp -I/path/to/ncnn/include -L/path/to/ncnn/lib -lncnn
     ```

#### **Lưu ý**
- **Kích thước mã**: File `.cpp` có thể rất lớn nếu mô hình phức tạp, vì trọng số được mã hóa dưới dạng mảng tĩnh.
- **Tối ưu trước**: Dùng `ncnnoptimize` trước khi chạy `ncnn2mem` để giảm kích thước mô hình.
- **Ứng dụng nhúng**: Công cụ này lý tưởng cho các thiết bị không có hệ thống file hoặc cần bảo mật mô hình.

#### **Ví dụ**
Chuyển đổi mô hình MobileNet thành mã C:
```bash
./ncnn2mem mobilenet.param mobilenet.bin mobilenet.cpp mobilenet.h
```

---

### 4. **`ncnnmerge`**
#### **Chức năng**
- Gộp nhiều mô hình NCNN thành một mô hình duy nhất.
- Hữu ích khi cần kết hợp các mạng con (sub-networks) hoặc tích hợp nhiều mô hình vào một pipeline suy luận.

#### **Cú pháp**
```bash
./ncnnmerge model1.param model1.bin model2.param model2.bin ... output.param output.bin
```
- **model1.param, model1.bin**: File mô hình NCNN thứ nhất.
- **model2.param, model2.bin**: File mô hình NCNN thứ hai (và tiếp tục cho các mô hình khác).
- **output.param**: File NCNN mô tả kiến trúc mạng gộp.
- **output.bin**: File NCNN chứa trọng số gộp.

#### **Các bước sử dụng**
1. **Chuẩn bị các mô hình NCNN**:
   - Đảm bảo các mô hình cần gộp đã ở định dạng NCNN (`.param` và `.bin`).
   - Ví dụ: `model1.param`, `model1.bin` và `model2.param`, `model2.bin`.
2. **Chạy lệnh gộp**:
   ```bash
   ./ncnnmerge model1.param model1.bin model2.param model2.bin merged.param merged.bin
   ```
3. **Kiểm tra mô hình gộp**:
   - Sử dụng mô hình `merged.param` và `merged.bin` để suy luận hoặc tối ưu hóa thêm.

#### **Lưu ý**
- **Tương thích**: Các mô hình cần có cấu trúc tương thích để gộp (ví dụ, cùng định dạng đầu vào/đầu ra).
- **Tối ưu hóa**: Sau khi gộp, dùng `ncnnoptimize` để tối ưu hóa mô hình.
- **Ứng dụng**: Thường dùng trong các pipeline phức tạp như phát hiện và phân loại kết hợp.

#### **Ví dụ**
Gộp hai mô hình NCNN:
```bash
./ncnnmerge model1.param model1.bin model2.param model2.bin merged.param merged.bin
```

---

### 5. **`onnx2ncnn`**
#### **Chức năng**
- Chuyển đổi mô hình từ định dạng **ONNX** (`.onnx`) sang định dạng NCNN (`.param` và `.bin`).
- ONNX là định dạng trung gian phổ biến, hỗ trợ chuyển đổi từ PyTorch, TensorFlow, và nhiều framework khác.

#### **Cú pháp**
```bash
./onnx2ncnn model.onnx output.param output.bin
```
- **model.onnx**: File mô hình ONNX.
- **output.param**: File NCNN mô tả kiến trúc mạng.
- **output.bin**: File NCNN chứa trọng số.

#### **Các bước sử dụng**
1. **Chuẩn bị mô hình ONNX**:
   - Xuất mô hình từ framework như PyTorch hoặc TensorFlow sang ONNX.
   - Ví dụ: Mô hình YOLO11 từ Ultralytics:
     ```python
     from ultralytics import YOLO
     model = YOLO("yolo11n.pt")
     model.export(format="onnx")
     ```
     Kết quả: File `yolo11n.onnx`.
2. **Chạy lệnh chuyển đổi**:
   ```bash
   ./onnx2ncnn yolo11n.onnx yolo11n.param yolo11n.bin
   ```
3. **Kiểm tra và tối ưu hóa**:
   - Kiểm tra file `yolo11n.param` và `yolo11n.bin`.
   - Dùng `ncnnoptimize` để tối ưu hóa:
     ```bash
     ./ncnnoptimize yolo11n.param yolo11n.bin yolo11n_opt.param yolo11n_opt.bin 65536
     ```

#### **Lưu ý**
- **Phiên bản ONNX**: NCNN hỗ trợ ONNX opset từ 7 đến 13. Đảm bảo mô hình ONNX tương thích.
- **Layer không hỗ trợ**: Một số layer ONNX phức tạp có thể không được hỗ trợ. Kiểm tra danh sách layer trên tài liệu NCNN.
- **ONNX Simplifier**: Nếu gặp lỗi, dùng `onnx-simplifier` để đơn giản hóa mô hình trước:
  ```bash
  pip install onnx-simplifier
  onnxsim input.onnx simplified.onnx
  ```

#### **Ví dụ**
Chuyển đổi mô hình YOLO11 từ ONNX sang NCNN:
```bash
./onnx2ncnn yolo11n.onnx yolo11n.param yolo11n.bin
```

---

### 6. **`darknet2ncnn`**
#### **Chức năng**
- Chuyển đổi mô hình từ định dạng **Darknet** (`.cfg` và `.weights`) sang định dạng NCNN (`.param` và `.bin`).
- Dùng để triển khai các mô hình YOLO (YOLOv3, YOLOv4, v.v.) được huấn luyện bằng Darknet.

#### **Cú pháp**
```bash
./darknet2ncnn model.cfg model.weights output.param output.bin
```
- **model.cfg**: File cấu hình mô hình Darknet.
- **model.weights**: File chứa trọng số Darknet.
- **output.param**: File NCNN mô tả kiến trúc mạng.
- **output.bin**: File NCNN chứa trọng số.

#### **Các bước sử dụng**
1. **Chuẩn bị mô hình Darknet**:
   - Tải mô hình Darknet, ví dụ YOLOv4 (`yolov4.cfg` và `yolov4.weights`).
2. **Chạy lệnh chuyển đổi**:
   ```bash
   ./darknet2ncnn yolov4.cfg yolov4.weights yolov4.param yolov4.bin
   ```
3. **Kiểm tra và tối ưu hóa**:
   - Kiểm tra file `yolov4.param` và `yolov4.bin`.
   - Tối ưu hóa với `ncnnoptimize`:
     ```bash
     ./ncnnoptimize yolov4.param yolov4.bin yolov4_opt.param yolov4_opt.bin 65536
     ```

#### **Lưu ý**
- **YOLO phiên bản**: Công cụ này hỗ trợ các mô hình YOLO được huấn luyện bằng Darknet (như YOLOv3, YOLOv4). YOLO11 từ Ultralytics nên dùng `onnx2ncnn`.
- **Layer tùy chỉnh**: Một số layer Darknet tùy chỉnh có thể không được hỗ trợ.
- **Kích thước đầu vào**: Đảm bảo kích thước đầu vào trong `.cfg` phù hợp với NCNN.

#### **Ví dụ**
Chuyển đổi mô hình YOLOv4 từ Darknet sang NCNN:
```bash
./darknet2ncnn yolov4.cfg yolov4.weights yolov4.param yolov4.bin
```

---

### 7. **`ncnn2int8`**
#### **Chức năng**
- Chuyển đổi mô hình NCNN từ định dạng FP32/FP16 sang định dạng **INT8** (8-bit integer) để tối ưu hóa hiệu suất suy luận trên thiết bị nhúng.
- Giảm kích thước mô hình và tăng tốc độ suy luận, nhưng có thể làm giảm nhẹ độ chính xác.

#### **Cú pháp**
```bash
./ncnn2int8 input.param input.bin output.param output.bin calibration.table
```
- **input.param**: File NCNN mô tả kiến trúc mạng (FP32/FP16).
- **input.bin**: File NCNN chứa trọng số (FP32/FP16).
- **output.param**: File NCNN mô tả kiến trúc mạng INT8.
- **output.bin**: File NCNN chứa trọng số INT8.
- **calibration.table**: File bảng hiệu chuẩn (tạo bằng `ncnn2table`).

#### **Các bước sử dụng**
1. **Chuẩn bị mô hình NCNN**:
   - Có file `.param` và `.bin` (ví dụ, từ `onnx2ncnn`).
2. **Tạo bảng hiệu chuẩn**:
   - Chuẩn bị tập dữ liệu hiệu chuẩn (ảnh đại diện).
   - Chạy `ncnn2table`:
     ```bash
     ./ncnn2table yolo11n.param yolo11n.bin images/ calibration.table \
         mean=0.0,0.0,0.0 norm=1.0/255.0,1.0/255.0,1.0/255.0 shape=640,640,3 swapRB=1
     ```
3. **Chuyển đổi sang INT8**:
   ```bash
   ./ncnn2int8 yolo11n.param yolo11n.bin yolo11n_int8.param yolo11n_int8.bin calibration.table
   ```
4. **Kiểm tra mô hình INT8**:
   - Sử dụng mô hình INT8 để suy luận và kiểm tra độ chính xác.

#### **Lưu ý**
- **Tập dữ liệu hiệu chuẩn**: Nên dùng 100-1000 ảnh đại diện để đảm bảo độ chính xác.
- **Mất mát độ chính xác**: INT8 quantization có thể làm giảm độ chính xác. Tinh chỉnh bảng hiệu chuẩn nếu cần.
- **Tối ưu hóa**: Dùng `ncnnoptimize` trước khi chuyển sang INT8 để giảm chi phí tính toán.

#### **Ví dụ**
Chuyển đổi mô hình YOLO11n sang INT8:
```bash
./ncnn2table yolo11n.param yolo11n.bin images/ calibration.table \
    mean=0.0,0.0,0.0 norm=1.0/255.0,1.0/255.0,1.0/255.0 shape=640,640,3 swapRB=1
./ncnn2int8 yolo11n.param yolo11n.bin yolo11n_int8.param yolo11n_int8.bin calibration.table
```

---

### 8. **`ncnn2table`**
#### **Chức năng**
- Tạo bảng hiệu chuẩn (calibration table) để sử dụng trong quá trình quantization INT8 với `ncnn2int8`.
- Phân tích phân phối trọng số và đầu ra của mô hình trên tập dữ liệu hiệu chuẩn để đảm bảo độ chính xác sau quantization.

#### **Cú pháp**
```bash
./ncnn2table model.param model.bin image_dir output.table \
    mean=mean_r,mean_g,mean_b norm=norm_r,norm_g,norm_b shape=width,height,channels swapRB=0|1
```
- **model.param**: File NCNN mô tả kiến trúc mạng.
- **model.bin**: File NCNN chứa trọng số.
- **image_dir**: Thư mục chứa ảnh hiệu chuẩn (`.jpg`, `.png`).
- **output.table**: File bảng hiệu chuẩn đầu ra.
- **mean**: Giá trị trung bình để chuẩn hóa ảnh (per channel).
- **norm**: Hệ số chuẩn hóa ảnh (thường là 1/255).
- **shape**: Kích thước đầu vào của mô hình (width, height, channels).
- **swapRB**: Chuyển đổi kênh RGB thành BGR (1 nếu cần, 0 nếu không).

#### **Các bước sử dụng**
1. **Chuẩn bị tập dữ liệu hiệu chuẩn**:
   - Tạo thư mục chứa các ảnh đại diện (ví dụ: `images/` với 100-1000 ảnh từ tập COCO).
2. **Chạy lệnh tạo bảng**:
   ```bash
   ./ncnn2table yolo11n.param yolo11n.bin images/ calibration.table \
       mean=0.0,0.0,0.0 norm=1.0/255.0,1.0/255.0,1.0/255.0 shape=640,640,3 swapRB=1
   ```
3. **Sử dụng bảng hiệu chuẩn**:
   - Dùng file `calibration.table` với `ncnn2int8`.

#### **Lưu ý**
- **Số lượng ảnh**: Sử dụng đủ ảnh (100-1000) để bảng hiệu chuẩn chính xác.
- **Tham số chuẩn hóa**: Đảm bảo `mean` và `norm` khớp với cách mô hình được huấn luyện.
- **Lỗi đọc ảnh**: Đảm bảo thư mục `image_dir` chứa các ảnh hợp lệ và NCNN được liên kết với OpenCV.

#### **Ví dụ**
Tạo bảng hiệu chuẩn cho YOLO11n:
```bash
./ncnn2table yolo11n.param yolo11n.bin images/ calibration.table \
    mean=0.0,0.0,0.0 norm=1.0/255.0,1.0/255.0,1.0/255.0 shape=640,640,3 swapRB=1
```

---

### 9. **`ncnnoptimize`**
#### **Chức năng**
- Tối ưu hóa mô hình NCNN để giảm kích thước, tăng tốc độ suy luận và cải thiện hiệu suất trên thiết bị nhúng.
- Thực hiện các kỹ thuật như gộp layer, loại bỏ layer dư thừa, và chuyển đổi sang định dạng FP16 (nếu cần).

#### **Cú pháp**
```bash
./ncnnoptimize input.param input.bin output.param output.bin flag
```
- **input.param**: File NCNN mô tả kiến trúc mạng.
- **input.bin**: File NCNN chứa trọng số.
- **output.param**: File NCNN mô tả kiến trúc mạng tối ưu.
- **output.bin**: File NCNN chứa trọng số tối ưu.
- **flag**: 
  - `65536`: Chuyển đổi sang FP16 (mặc định).
  - `0`: Giữ nguyên FP32.

#### **Các bước sử dụng**
1. **Chuẩn bị mô hình NCNN**:
   - Có file `.param` và `.bin` (ví dụ, từ `onnx2ncnn`).
2. **Chạy lệnh tối ưu hóa**:
   ```bash
   ./ncnnoptimize yolo11n.param yolo11n.bin yolo11n_opt.param yolo11n_opt.bin 65536
   ```
3. **Kiểm tra mô hình tối ưu**:
   - Sử dụng mô hình tối ưu để suy luận hoặc tiếp tục xử lý (như `ncnn2int8`).

#### **Lưu ý**
- **FP16 vs FP32**: Sử dụng FP16 (`flag=65536`) để tăng tốc trên thiết bị hỗ trợ FP16 (như GPU di động).
- **Tương thích**: Một số layer sau khi tối ưu có thể không tương thích với phiên bản NCNN cũ.
- **Kiểm tra hiệu suất**: So sánh tốc độ và độ chính xác trước/sau khi tối ưu.

#### **Ví dụ**
Tối ưu hóa mô hình YOLO11n:
```bash
./ncnnoptimize yolo11n.param yolo11n.bin yolo11n_opt.param yolo11n_opt.bin 65536
```

---

### **Lưu ý chung**
- **Môi trường**: Đảm bảo NCNN được cài đặt đúng trên Ubuntu 22.04, với các thư viện phụ thuộc như OpenCV và `libncnn.so` được liên kết:
  ```bash
  export LD_LIBRARY_PATH=/path/to/ncnn/lib:$LD_LIBRARY_PATH
  ```
- **Tài liệu**: Tham khảo https://github.com/Tencent/ncnn/wiki để biết danh sách layer hỗ trợ và các vấn đề thường gặp.
- **Lỗi layer không hỗ trợ**: Nếu một công cụ báo lỗi về layer, thử đơn giản hóa mô hình (ví dụ, với `onnx-simplifier`) hoặc kiểm tra mã nguồn công cụ để thêm hỗ trợ layer.
- **Pipeline điển hình**:
  1. Chuyển đổi mô hình sang NCNN (dùng `onnx2ncnn`, `caffe2ncnn`, v.v.).
  2. Tối ưu hóa mô hình với `ncnnoptimize`.
  3. Tạo bảng hiệu chuẩn với `ncnn2table`.
  4. Chuyển sang INT8 với `ncnn2int8`.
  5. Nhúng mô hình vào mã C với `ncnn2mem` (nếu cần).

### **Ví dụ pipeline đầy đủ (YOLO11)**
```bash
# Chuyển từ ONNX sang NCNN
./onnx2ncnn yolo11n.onnx yolo11n.param yolo11n.bin

# Tối ưu hóa mô hình
./ncnnoptimize yolo11n.param yolo11n.bin yolo11n_opt.param yolo11n_opt.bin 65536

# Tạo bảng hiệu chuẩn
./ncnn2table yolo11n_opt.param yolo11n_opt.bin images/ calibration.table \
    mean=0.0,0.0,0.0 norm=1.0/255.0,1.0/255.0,1.0/255.0 shape=640,640,3 swapRB=1

# Chuyển sang INT8
./ncnn2int8 yolo11n_opt.param yolo11n_opt.bin yolo11n_int8.param yolo11n_int8.bin calibration.table

# Nhúng vào mã C (tùy chọn)
./ncnn2mem yolo11n_int8.param yolo11n_int8.bin yolo11n_int8.cpp yolo11n_int8.h
```
