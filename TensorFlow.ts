import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-react-native';

export async function loadModel() {
  const model = await tf.loadLayersModel("https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/frameworks/TensorFlow2/variations/fpnlite-320x320/versions/1");
  return model;
}

export async function detectObjects(model: any, imageTensor: any) {
  const predictions = await model.executeAsync(imageTensor);
  return predictions;
}
