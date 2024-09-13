#ifndef FACE_MESH_HPP
#define FACE_MESH_HPP
#include <string>
#include <cstdint>
#include <opencv2/core.hpp>
#include "tensorflow/lite/interpreter.h"
#include <tensorflow/lite/model.h>

namespace CLFML::FaceMesh
{
    /* Number of face landmarks */
    inline constexpr size_t NUM_OF_FACE_MESH_POINTS = 468;

    /* Number of model output tensors */
    inline constexpr size_t NUM_OF_FACE_MESH_OUTPUT_TENSORS = 2;

    /*
     * The used delegate for model inference;
     * In the future TPU support might be added
     */
    enum class face_mesh_delegate
    {
        CPU
    };

    class FaceMesh
    {
    public:
        FaceMesh();

        /**
         * @brief Loads model and initializes the inference runtime
         * @param model_path Path to the Mediapipe Face Mesh model (.tflite) file
         * @param delegate_type The delegate to use for inference (CPU only for now) (default = CPU)
         * @param num_of_threads The number of CPU threads which can be used by the inference runtime (default= 4 threads)
         */
        void load_model(const std::string model_path, const face_mesh_delegate delegate_type = face_mesh_delegate::CPU, const uint8_t num_of_threads = 4);

        /**
         * @brief Loads image into model and does inference
         * @param frame Any frame which is formatted in CV_8UC3 or CV_8UC4 format
         * @param roi_offset If the input_frame is a cropped ROI frame, 
         *                   the face_mesh_points can be adjusted to the original frame.
         */
        void load_image(cv::Mat &frame, cv::Rect roi_offset = cv::Rect(0, 0, 0, 0));

        /**
         * @brief Get the 3D landmarks from the model
         * @return Array with 468 3D Facial landmarks;
         */
        std::array<cv::Point3f, NUM_OF_FACE_MESH_POINTS> get_face_mesh_points();

        ~FaceMesh();

    private:
        /* Model input frame width and height (set to defaults from the model-card) 
         * See: https://drive.google.com/file/d/1QvwWNfFoweGVjsXF3DXzcrCnz-mx-Lha/preview
         */
        int32_t m_input_frame_size_x = 192;

        int32_t m_input_frame_size_y = 192;

        /*
         * Model inputs and outputs
         */
        TfLiteTensor *m_input_tensor;

        /*
         * The output tensors of the model, there are two outputs; 
         * - The 3D Landmark points (468 x 32-bit floats)
         * - The confidence score (1 x 32-bit float)
         */
        std::array<TfLiteTensor *, NUM_OF_FACE_MESH_OUTPUT_TENSORS> m_output_tensors;

        /* Array that contains the unprocessed Face Mesh landmarks */
        std::array<float, NUM_OF_FACE_MESH_POINTS * 3> m_model_regressors;

        /* Internal variable that contains the 3D Landmarks */
        std::array<cv::Point3f, NUM_OF_FACE_MESH_POINTS> m_model_landmarks;

        /* Internal variable that contains the Confidence score */
        float m_model_confidence_score = 0.0f;

        /*
         * Handles to the model and model_inpreter runtime
         */
        std::unique_ptr<tflite::FlatBufferModel> m_model;
        std::unique_ptr<tflite::Interpreter> m_model_interpreter;

        /**
         * @brief Preprocess any incoming image to a 192x192px 24-bit RGB image
         */
        cv::Mat preprocess_image(const cv::Mat &in);

        /**
         * @brief Copies the results from the tensors into our m_model_regressors array
         */
        void get_regressor();
    };
}

#endif /* FACE_MESH_HPP */