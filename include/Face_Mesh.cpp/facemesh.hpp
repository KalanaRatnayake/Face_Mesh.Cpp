/*
 *  Copyright 2024 (C) Jeroen Veen <ducroq> & Victor Hogeweij <Hoog-V>
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 * This file is part of the Face_Mesh.Cpp library
 *
 * Author:          Victor Hogeweij <Hoog-V>
 * Modified by:     Kalana Ratnayake
 *
 */
#ifndef FACE_MESH_HPP
#define FACE_MESH_HPP
#include <string>
#include <cstdint>
#include <opencv2/core.hpp>
#include "tensorflow/lite/interpreter.h"
#include <tensorflow/lite/model.h>
#include "tensorflow/lite/kernels/register.h"
#include "opencv2/imgproc.hpp"

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

/* Mapping from model output tensor name to their array index */
enum output_tensor_id
{
  OUTPUT_TENSOR_REGRESSOR,
  OUTPUT_TENSOR_CONFIDENCE_SCORE
};

class FaceMesh
{
public:
  FaceMesh()
  {
  }

  /**
   * @brief Loads model and initializes the inference runtime
   * @param model_path Path to the Mediapipe Face Mesh model (.tflite) file
   * @param delegate_type The delegate to use for inference (CPU only for now) (default = CPU)
   * @param num_of_threads The number of CPU threads which can be used by the inference runtime (default= 4 threads)
   */
  void load_model(const std::string model_path, const face_mesh_delegate delegate_type = face_mesh_delegate::CPU,
                  const uint8_t num_of_threads = 4)
  {
    /* Load the model in to memory */
    m_model = tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
    if (m_model == nullptr)
    {
      fprintf(stderr, "File \"%s\" ERROR: Can't build flatbuffer from model: %s \n", __FILE__, model_path.c_str());
      exit(1);
    }

    /*
     * We want to use the default tensorflow operations (for CPU inference):
     * See: https://www.tensorflow.org/lite/guide/ops_compatibility
     */
    tflite::ops::builtin::BuiltinOpResolver resolver;

    /*
     * Build the model interpreter with our model and the resolver;
     * This gives a handle and saves it into m_model_intepreter which is later used for doing inference
     */
    if (tflite::InterpreterBuilder(*m_model, resolver)(&m_model_interpreter) != kTfLiteOk)
    {
      fprintf(stderr, "File \"%s\" ERROR: Can't initialize the interpreter \n", __FILE__);
      exit(1);
    }

    /*
     * We can set the amount of CPU threads we want to dedicate to our model_interpreter engine.
     * This is only useful for CPU inference,
     * Using multiple threads with TPU inference will slow down the program, due to synchronisation between threads!
     * Default = 4 threads
     */
    m_model_interpreter->SetNumThreads(num_of_threads);

    /* Allocate memory for model inference */
    if (m_model_interpreter->AllocateTensors() != kTfLiteOk)
    {
      fprintf(stderr, "File \"%s\" ERROR: Can't allocate tensors for face detector model interpreter \n", __FILE__);
      exit(1);
    }

    /*
     * Get the amount of input tensors (vectors) of the model, which should be one
     * As this is where the 128x128 input frame will be copied to before doing inference
     */
    const std::vector<int>& inputs = m_model_interpreter->inputs();

    /* Save the handle to this input vector for later */
    m_input_tensor = m_model_interpreter->tensor(inputs.at(0));

    /* Get the input frame size (should be 128x128 pixels!) */
    m_input_frame_size_x = m_input_tensor->dims->data[1];
    m_input_frame_size_y = m_input_tensor->dims->data[2];

    /*
     * Get the amount of output tensors of the model, which should be two;
     *  - one for the list of regressors(boxes)
     *  - one for the list of classifiers (confidence score)
     */
    const std::vector<int>& outputs = m_model_interpreter->outputs();

    if (outputs.size() != m_output_tensors.size())
    {
      fprintf(stderr, "File \"%s\" ERROR: Model tensor quantity does not match expected tensors! \n", __FILE__);
      exit(1);
    }

    /* Save the output tensor handles for later! (With bound-check)*/
    for (uint8_t i = 0; i < m_output_tensors.size(); i++)
    {
      int tensor_index = outputs.at(i);
      m_output_tensors.at(i) = m_model_interpreter->tensor(tensor_index);
    }
  }

  /**
   * @brief Loads image into model and does inference
   * @param frame Any frame which is formatted in CV_8UC3 or CV_8UC4 format
   * @param roi_offset If the input_frame is a cropped ROI frame,
   *                   the face_mesh_points can be adjusted to the original frame.
   */
  void load_image(cv::Mat& frame, cv::Rect roi_offset = cv::Rect(0, 0, 0, 0))
  {
    int image_width = frame.size().width;
    int image_height = frame.size().height;

    /* Convert image to 128x128 pixels image with CV32_FC3 format */
    cv::Mat preprocessed_image = preprocess_image(frame);

    /* Copy the image data to the input tensor, which feeds it into the model */
    memcpy(m_input_tensor->data.f, preprocessed_image.data, m_input_tensor->bytes);

    /* Run inference! */
    m_model_interpreter->Invoke();

    /* Get model landmarks */
    get_regressor();

    /* Scale the model landmarks to original image dimensions */
    float _x;
    float _y;
    cv::Point3f* point;
    for (size_t point_idx = 0; point_idx < NUM_OF_FACE_MESH_POINTS; point_idx++)
    {
      point = &(m_model_landmarks.at(point_idx));

      /* Do some linear scaling :) */
      _x = m_model_regressors.at(point_idx * 3) / m_input_frame_size_x;
      _y = m_model_regressors.at((point_idx * 3) + 1) / m_input_frame_size_y;
      point->z = m_model_regressors.at((point_idx * 3) + 2);

      /* Add original ROI offset if needed */
      point->x = (_x * image_width) + roi_offset.x;
      point->y = (_y * image_height) + roi_offset.y;
    }
  }

  /**
   * @brief Get the 3D landmarks from the model
   * @return Array with 468 3D Facial landmarks;
   */
  std::array<cv::Point3f, NUM_OF_FACE_MESH_POINTS> get_face_mesh_points()
  {
    return m_model_landmarks;
  }

  ~FaceMesh()
  {
  }

private:
  /* Model input frame width and height (set to defaults from the model-card)
   * See: https://drive.google.com/file/d/1QvwWNfFoweGVjsXF3DXzcrCnz-mx-Lha/preview
   */
  int32_t m_input_frame_size_x = 192;

  int32_t m_input_frame_size_y = 192;

  /*
   * Model inputs and outputs
   */
  TfLiteTensor* m_input_tensor;

  /*
   * The output tensors of the model, there are two outputs;
   * - The 3D Landmark points (468 x 32-bit floats)
   * - The confidence score (1 x 32-bit float)
   */
  std::array<TfLiteTensor*, NUM_OF_FACE_MESH_OUTPUT_TENSORS> m_output_tensors;

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
  cv::Mat preprocess_image(const cv::Mat& in)
  {
    cv::Mat preprocessed_frame = convert_image_to_rgb(in);
    cv::Size input_frame_size = cv::Size(m_input_frame_size_x, m_input_frame_size_y);
    cv::resize(preprocessed_frame, preprocessed_frame, input_frame_size);

    const double alpha = 1 / 191.5f;
    const double beta = -191.5f / 191.5f;
    preprocessed_frame.convertTo(preprocessed_frame, CV_32FC3, alpha, beta);
    return preprocessed_frame;
  }

  /**
   * @brief Copies the results from the tensors into our m_model_regressors array
   */
  void get_regressor()
  {
    size_t num_of_bytes = m_output_tensors.at(OUTPUT_TENSOR_REGRESSOR)->bytes;
    size_t num_of_floats = num_of_bytes / sizeof(float);
    memcpy(&(m_model_regressors.at(0)), m_output_tensors.at(OUTPUT_TENSOR_REGRESSOR)->data.f, num_of_bytes);
  }

  /**
   * @brief This function converts images with other color-spaces to RGB. As the model expects RGB formatted images.
   *
   * @param in Image to be converted, Can be CV_8UC3; 8-bit int with 3 channels or CV_8UC4; 8-bit int with 4 channels
   * @return RGB formatted frame
   */
  cv::Mat convert_image_to_rgb(const cv::Mat& in)
  {
    cv::Mat rgb_frame;
    int frame_color_type = in.type();

    switch (frame_color_type)
    {
      case CV_8UC3: {
        cv::cvtColor(in, rgb_frame, cv::COLOR_BGR2RGB);
        break;
      }
      case CV_8UC4: {
        cv::cvtColor(in, rgb_frame, cv::COLOR_BGRA2RGB);
        break;
      }
      default: {
        fprintf(stderr, "ERROR: Image type %d is not supported by the face_detector library! \n", frame_color_type);
        exit(1);
      }
    };
    return rgb_frame;
  }
};
}  // namespace CLFML::FaceMesh