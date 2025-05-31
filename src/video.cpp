#include <filesystem>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "YoloOnnxModel.h"

int main(const int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "ERROR - Please format as such:\n$ " << argv[0] << " <path to file>" << std::endl;
        return -1;
    }

    std::string pathToModel = "include/model/yolo11m.onnx";
    std::string pathToNames = "include/coco.names";
    constexpr bool isGPU = false; // set to false if your GPU does not support CUDA 😭😭😭

    YoloOnnxModel yolo(pathToModel, pathToNames, isGPU);

    std::string video_path = std::filesystem::absolute(argv[1]).string(); // Specify your video file path here
    cv::VideoCapture cap(video_path);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the video file." << std::endl;
        return -1;
    }

    cv::Mat frame;
    std::vector<cv::Mat> processed;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cout << "no frame captured, breaking" << std::endl;
            break;
        }

        auto detections = yolo.infer(frame, 0.5f, 0.4f);

        for (const auto& [box, label] : detections) {
            cv::rectangle(frame, box, cv::Scalar(0, 0, 255), 4);
            cv::putText(frame, label, cv::Point(box.x, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 0, 255), 3);
        }

        processed.emplace_back(frame.clone());
    }

    cv::namedWindow("Video Feed", cv::WINDOW_AUTOSIZE);
    cv::imshow("Video Feed", processed[0]);
    cv::setWindowProperty("Video Feed", cv::WND_PROP_TOPMOST, 1);
    cv::waitKey(1);
    double fps = cv::max(10.0, cap.get(cv::CAP_PROP_FPS));
    int frameDelayInMs = static_cast<int>(1000.0 / fps);

    auto start = std::chrono::high_resolution_clock::now();

    for (const auto& frame : processed) {
        const auto frameStart = std::chrono::high_resolution_clock::now();
        cv::Mat displayFrame;
        cv::resize(frame, displayFrame, cv::Size(1280, 720));
        cv::imshow("Video Feed", displayFrame);
        const auto currentTime = std::chrono::high_resolution_clock::now();
        const int delay = frameDelayInMs - static_cast<int>(std::chrono::duration_cast<std::chrono::milliseconds>(
                                 currentTime - frameStart).count());
        if (cv::waitKey(std::max(1, delay)) == 'q') break;
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << std::endl;

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
