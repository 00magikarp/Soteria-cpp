#include <filesystem>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "YoloOnnxModel.h"

int main(const int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "ERROR - Please format as such:\n$ " << argv[0] << " <path to file>" << std::endl;
        return -1;
    }

    std::string pathToModel = "include/model/yolo11s.onnx";
    std::string pathToNames = "include/coco.names";
    constexpr bool isGPU = false; // set to false if your GPU does not support CUDA 😭😭😭

    YoloOnnxModel yolo(pathToModel, pathToNames, isGPU);

    std::string video_path = std::filesystem::absolute(argv[1]).string(); // Specify your video file path here
    cv::VideoCapture cap(video_path);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the video file." << std::endl;
        return -1;
    }

    std::filesystem::path inputPath(video_path);
    std::string fileName = inputPath.stem().string();
    std::string fileExtension = inputPath.extension().string();

    double fps = cap.get(cv::CAP_PROP_FPS);
    int frameWidth = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frameHeight = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    cv::Size frameSize(frameWidth, frameHeight);

    auto outputFilePath = std::filesystem::current_path() / "out" / (fileName + "-processed" + fileExtension);
    std::cout << outputFilePath;

    cv::VideoWriter videoWriter(
        outputFilePath.string(),
        cv::VideoWriter::fourcc('X', '2', '6', '4'),
        fps,
        frameSize
        );

    cv::Mat frame;

    while (true) {
        cap >> frame;
        if (frame.empty()) {
            std::cout << "no frame captured, breaking" << std::endl;
            break;
        }

        auto detections = yolo.infer(frame, 0.5f, 0.4f);

        for (const auto& [box, label] : detections) {
            cv::rectangle(frame, box, cv::Scalar(255, 0, 0), 2);
            cv::putText(frame, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0));
        }

        videoWriter.write(frame);
    }

    cap.release();
    videoWriter.release();

    return 0;
}
