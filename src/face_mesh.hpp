#ifndef FACE_MESH_HPP
#define FACE_MESH_HPP
#include <string>
#include <cstdint>
#include <opencv2/core.hpp>
#include "tensorflow/lite/interpreter.h"
#include <tensorflow/lite/model.h>

namespace CLFML::FaceMesh
{
    inline constexpr size_t NUM_OF_FACE_MESH_POINTS  = 468;
    inline constexpr size_t NUM_OF_FACE_MESH_OUTPUT_TENSORS = 2;

    enum class face_mesh_delegate {
        CPU
    };

    class FaceMesh
    {
    public:
    FaceMesh();

    void load_model(const std::string model_path, const face_mesh_delegate delegate_type = face_mesh_delegate::CPU, const uint8_t num_of_threads = 4);

    void load_image(cv::Mat &frame, cv::Rect roi_offset = cv::Rect(0, 0, 0 ,0));

    std::array<cv::Point3f, NUM_OF_FACE_MESH_POINTS> get_face_mesh_points();

    ~FaceMesh();

    private:
        /* Model input frame width and height */
        int32_t m_input_frame_size_x = 192;

        int32_t m_input_frame_size_y = 192;

        /*
         * Model inputs and outputs
         */
        TfLiteTensor *m_input_tensor;

        std::array<TfLiteTensor *, NUM_OF_FACE_MESH_OUTPUT_TENSORS> m_output_tensors;

        std::array<float, NUM_OF_FACE_MESH_POINTS*3> m_model_regressors;

        /* Intermediary variable which contains grid-aligned Landmarks (after model inference) */
        std::array<cv::Point3f, NUM_OF_FACE_MESH_POINTS> m_model_landmarks;

        /*
         * Handles to the model and model_inpreter runtime
         */
        std::unique_ptr<tflite::FlatBufferModel> m_model;

        std::unique_ptr<tflite::Interpreter> m_model_interpreter;
        cv::Mat preprocess_image(const cv::Mat &in);
        void get_regressor();
    };
}


#endif /* FACE_MESH_HPP */