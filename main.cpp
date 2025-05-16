#include "ncnn/layer.h"
#include "ncnn/net.h"
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <float.h>
#include <memory>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <vector>
#include <string>

#define MAX_STRIDE 32

// Hàm đọc danh sách tên các lớp từ file metadata YAML
std::vector<std::string> get_class_names(const std::string& metadata_path) {
    std::vector<std::string> class_names;
    try {
        YAML::Node config = YAML::LoadFile(metadata_path);
        if (config["names"]) {
            for (const auto& pair : config["names"]) {
                int index = pair.first.as<int>();
                std::string name = pair.second.as<std::string>();
                if (index >= class_names.size()) {
                    class_names.resize(index + 1);
                }
                class_names[index] = name;
            }
        }
    } catch (const std::exception& e) {
        fprintf(stderr, "Error reading metadata file: %s\n", e.what());
    }
    return class_names;
}

// Cấu trúc lưu thông tin về đối tượng được phát hiện
struct Object
{
    cv::Rect_<float> rect;
    int              label;
    float            prob;
};

// Hàm tính diện tích giao nhau giữa hai đối tượng
static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

// Hàm sắp xếp các đối tượng theo xác suất giảm dần (quicksort)
static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int   i = left;
    int   j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p) i++;

        while (objects[j].prob < p) j--;

        if (i <= j)
        {
            // swap
            std::swap(objects[i], objects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(objects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(objects, i, right);
        }
    }
}

// Hàm wrapper cho quicksort
static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty()) return;

    qsort_descent_inplace(objects, 0, objects.size() - 1);
}

// Hàm thực hiện Non-Maximum Suppression (NMS) để loại bỏ các bounding box trùng lặp
static void nms_sorted_bboxes(const std::vector<Object>& faceobjects,
                              std::vector<int>&          picked,
                              float                      nms_threshold,
                              bool                       agnostic = false)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int) picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            if (!agnostic && a.label != b.label) continue;

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold) keep = 0;
        }

        if (keep) picked.push_back(i);
    }
}

// Hàm tính sigmoid
static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

// Hàm giới hạn giá trị trong khoảng [min, max]
static inline float clampf(float d, float min, float max)
{
    const float t = d < min ? min : d;
    return t > max ? max : t;
}

// Hàm phân tích kết quả dự đoán từ mô hình YOLOv8
static void parse_yolov8_detections(float*               inputs,
                                    float                confidence_threshold,
                                    int                  num_channels,
                                    int                  num_anchors,
                                    int                  num_labels,
                                    int                  infer_img_width,
                                    int                  infer_img_height,
                                    std::vector<Object>& objects)
{
    std::vector<Object> detections;
    cv::Mat             output = cv::Mat((int) num_channels, (int) num_anchors, CV_32F, inputs).t();

    for (int i = 0; i < num_anchors; i++)
    {
        const float* row_ptr    = output.row(i).ptr<float>();
        const float* bboxes_ptr = row_ptr;
        const float* scores_ptr = row_ptr + 4;
        const float* max_s_ptr  = std::max_element(scores_ptr, scores_ptr + num_labels);
        float        score      = *max_s_ptr;
        if (score > confidence_threshold)
        {
            float x = *bboxes_ptr++;
            float y = *bboxes_ptr++;
            float w = *bboxes_ptr++;
            float h = *bboxes_ptr;

            float x0 = clampf((x - 0.5f * w), 0.f, (float) infer_img_width);
            float y0 = clampf((y - 0.5f * h), 0.f, (float) infer_img_height);
            float x1 = clampf((x + 0.5f * w), 0.f, (float) infer_img_width);
            float y1 = clampf((y + 0.5f * h), 0.f, (float) infer_img_height);

            cv::Rect_<float> bbox;
            bbox.x      = x0;
            bbox.y      = y0;
            bbox.width  = x1 - x0;
            bbox.height = y1 - y0;
            Object object;
            object.label = max_s_ptr - scores_ptr;
            object.prob  = score;
            object.rect  = bbox;
            detections.push_back(object);
        }
    }
    objects = detections;
}

// Hàm chính thực hiện phát hiện đối tượng sử dụng mô hình YOLOv8
static int detect_yolov8(ncnn::Net&           yolov8,
                         const cv::Mat&       bgr,
                         std::vector<Object>& objects,
                         const std::vector<std::string>& class_names)
{
    const int   target_size    = 320;
    const float prob_threshold = 0.25f;
    const float nms_threshold  = 0.45f;

    int img_w = bgr.cols;
    int img_h = bgr.rows;

    // letterbox pad to multiple of MAX_STRIDE
    int   w     = img_w;
    int   h     = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float) target_size / w;
        w     = target_size;
        h     = h * scale;
    }
    else
    {
        scale = (float) target_size / h;
        h     = target_size;
        w     = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);

    int       wpad = (target_size + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - w;
    int       hpad = (target_size + MAX_STRIDE - 1) / MAX_STRIDE * MAX_STRIDE - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(
        in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov8.create_extractor();

    ex.input("in0", in_pad);

    std::vector<Object> proposals;

    // stride 32
    {
        ncnn::Mat out;
        ex.extract("out0", out);

        std::vector<Object> objects32;
        const int num_labels = class_names.size();
        parse_yolov8_detections(
            (float*) out.data, prob_threshold, out.h, out.w, num_labels, in_pad.w, in_pad.h, objects32);
        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }

    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);

    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    int count = picked.size();

    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];

        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        // clip
        x0 = std::max(std::min(x0, (float) (img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float) (img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float) (img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float) (img_h - 1)), 0.f);

        objects[i].rect.x      = x0;
        objects[i].rect.y      = y0;
        objects[i].rect.width  = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

// Hàm vẽ các đối tượng được phát hiện lên ảnh
static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, const std::vector<std::string>& class_names)
{
    static const unsigned char colors[19][3] = {{54, 67, 244},
                                                {99, 30, 233},
                                                {176, 39, 156},
                                                {183, 58, 103},
                                                {181, 81, 63},
                                                {243, 150, 33},
                                                {244, 169, 3},
                                                {212, 188, 0},
                                                {136, 150, 0},
                                                {80, 175, 76},
                                                {74, 195, 139},
                                                {57, 220, 205},
                                                {59, 235, 255},
                                                {7, 193, 255},
                                                {0, 152, 255},
                                                {34, 87, 255},
                                                {72, 85, 121},
                                                {158, 158, 158},
                                                {139, 125, 96}};

    int color_index = 0;

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        const unsigned char* color = colors[color_index % 19];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        fprintf(stderr,
                "%d = %.5f at %.2f %.2f %.2f x %.2f\n",
                obj.label,
                obj.prob,
                obj.rect.x,
                obj.rect.y,
                obj.rect.width,
                obj.rect.height);

        cv::rectangle(image, obj.rect, cc, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label].c_str(), obj.prob * 100);

        int      baseLine   = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0) y = 0;
        if (x + label_size.width > image.cols) x = image.cols - label_size.width;

        cv::rectangle(
            image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cc, -1);

        cv::putText(
            image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
    }

    cv::imshow("YOLOv12 Detection", image);
}

// Hàm main - điểm khởi đầu của chương trình
int main(int argc, char** argv)
{
    std::string model_dir;
    if (argc == 2) {
        model_dir = argv[1];
    } else if (argc == 1) {
        model_dir = "."; // Thư mục hiện tại
    } else {
        fprintf(stderr, "Usage: %s [model_directory]\n", argv[0]);
        return -1;
    }

    // Tạo đường dẫn đầy đủ cho param và model file
    std::string parampath = model_dir + "/model.ncnn.param";
    std::string modelpath = model_dir + "/model.ncnn.bin";
    std::string metadata_path = model_dir + "/metadata.yaml";

    // Đọc danh sách class names từ file metadata
    std::vector<std::string> class_names = get_class_names(metadata_path);
    if (class_names.empty()) {
        fprintf(stderr, "Error: No class names found in metadata file\n");
        return -1;
    }

    const int cam = 0; // your usb cam device

    // Open video capture
    cv::VideoCapture cap(0, cv::CAP_AVFOUNDATION); 
    
    if (!cap.isOpened())
    {
        std::cerr << "Error: Could not open the camera!\n";
        return -1;
    }
    // Set kích thước video (nếu muốn)
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // Print actual camera properties
    fprintf(stderr, "Camera properties:\n");
    fprintf(stderr, "Width: %d\n", (int) cap.get(cv::CAP_PROP_FRAME_WIDTH));
    fprintf(stderr, "Height: %d\n", (int) cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    fprintf(stderr, "FPS: %d\n", (int) cap.get(cv::CAP_PROP_FPS));

    // Khởi tạo mạng nơ-ron một lần duy nhất
    ncnn::Net yolov8;
    yolov8.opt.use_vulkan_compute = true;
    yolov8.opt.num_threads = 4;
    yolov8.opt.use_fp16_packed = true;
    yolov8.opt.use_fp16_storage = true;
    yolov8.load_param(parampath.c_str());
    yolov8.load_model(modelpath.c_str());

    while (true)
    {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty())
        {
            fprintf(stderr, "Failed to capture frame\n");
            break;
        }

        std::vector<Object> objects;
        detect_yolov8(yolov8, frame, objects, class_names);

        draw_objects(frame, objects, class_names);

        // Press 'q' to quit
        char key = cv::waitKey(1);
        if (key == 'q' || key == 27) // 27 is ESC key
            break;
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
