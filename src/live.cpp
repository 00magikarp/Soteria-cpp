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

void processFrames(YoloOnnxModel& yolo, const bool& debug) {
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
                cv::putText(frame, label, box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 0));
            }

            if (debug) {
                cv::imshow("Live Camera Feed", frame);
                if (cv::waitKey(1) == 'q') {
                    done = true;
                }
            }
        }
    }
}

bool isNumber(const std::string& str) {
    return !str.empty() && std::all_of(str.begin(), str.end(), ::isdigit);
}

int main(const int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Error - Please format as such:" << std::endl;
        std::cerr << "$ " << argv[0] << " <identifier> <debug [y/n]>" << std::endl;
        std::cerr << "Identifier can be one of the following:" << std::endl;
        std::cerr << "- Camera location\n indices starting on 0 on Windows\n /dev/ folder on Linux" << std::endl;
        std::cerr << "- URL on local network" << std::endl;
        return -1;
    }

    const std::string pathToModel = "include/model/yolo11s.onnx";
    const std::string pathToNames = "include/coco.names";
    constexpr bool isGPU = false; // set to false if your GPU does not support CUDA 😭😭😭

    YoloOnnxModel yolo(pathToModel, pathToNames, isGPU);
    cv::VideoCapture cap;
    const std::string input = argv[1];
    if (input.find("://") != std::string::npos) {
        cap.open(input, cv::CAP_FFMPEG);
    } else if (isNumber(input)) {
        cap.open(std::stoi(input), cv::CAP_DSHOW);
    } else {
        cap.open(input, cv::CAP_V4L2);
    }
    cap.set(cv::CAP_PROP_BUFFERSIZE, 3);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open the camera." << std::endl;
        return -1;
    }

    std::thread captureThread(captureFrames, std::ref(cap));
    std::thread processThread(processFrames, std::ref(yolo), std::string(argv[2]) == "y");

    captureThread.join();
    processThread.join();

    cap.release();
    cv::destroyAllWindows();

    return 0;
}
