#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <iostream>

#include <opencv2/opencv.hpp>
#include "YoloOnnxModel.h"

std::queue<cv::Mat> frameQueue;
std::mutex frameMutex;
std::condition_variable frameCV;
bool done = false;

void captureFrames(cv::VideoCapture& cap) {
    cv::Mat frame;
    while (!done) {
        cap >> frame;
        if (frame.empty()) continue;

        std::lock_guard<std::mutex> lock(frameMutex);
        if (frameQueue.size() < 10) {
            frameQueue.push(frame);
            frameCV.notify_one();
        }
    }
}

void processFrames(YoloOnnxModel& yolo) {
    while (!done) {
        std::unique_lock<std::mutex> lock(frameMutex);
        frameCV.wait(lock, [] { return !frameQueue.empty() || done; });

        if (frameQueue.empty() && done) break;

        if (!frameQueue.empty()) {
            cv::Mat frame = frameQueue.front();
            frameQueue.pop();
            lock.unlock();

            auto detections = yolo.infer(frame, 0.4f, 0.4f);
            for (const auto& [box, label] : detections) {
                cv::rectangle(frame, box, cv::Scalar(255, 0, 0), 2);
                cv::putText(frame, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
            }

            cv::imshow("Live Camera Feed (RTSP)", frame);
            if (cv::waitKey(1) == 'q') {
                done = true;
            }
        }
    }
}

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "ERROR\nPlease format as such:\n $ " << argv[0] << " <stream ip>" << std::endl;
        return -1;
    }

    std::string pathToModel = "include/model/yolo11s.onnx";
    std::string pathToNames = "include/coco.names";
    constexpr bool isGPU = false; // set to false if your GPU does not support CUDA 😭😭😭

    YoloOnnxModel yolo(pathToModel, pathToNames, isGPU);
    cv::VideoCapture cap(argv[1], cv::CAP_FFMPEG); // make filename args in future
    cap.set(cv::CAP_PROP_BUFFERSIZE, 3);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the camera." << std::endl;
        return -1;
    }

    std::thread captureThread(captureFrames, std::ref(cap));
    std::thread processThread(processFrames, std::ref(yolo));

    captureThread.join();
    processThread.join();

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
