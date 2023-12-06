import React, { useEffect, useRef, useState } from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { Camera } from 'expo-camera';
import * as tf from '@tensorflow/tfjs';
import { cameraWithTensors } from '@tensorflow/tfjs-react-native';

import { loadModel, detectObjects } from '../TensorFlow';
import { ExpoWebGLRenderingContext } from 'expo-gl';

const TensorCamera = cameraWithTensors(Camera);

interface ObjectDetectionComponentProps {}

const ObjectDetectionComponent: React.FC<ObjectDetectionComponentProps> = () => {
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [predictions, setPredictions] = useState<any[]>([]);
  const cameraRef = useRef<Camera | null>(null);

  useEffect(() => {
    async function setupModel() {
      const loadedModel = await loadModel();
      setModel(loadedModel);
    }

    setupModel();
  }, []);

  const handleCameraStream = async (
    images: IterableIterator<tf.Tensor3D>,
    updateCameraPreview: () => void,
    gl: ExpoWebGLRenderingContext,
    cameraTexture: WebGLTexture
  ) => {
    const nextImage = images.next();
    const imageTensor = nextImage.value?.[0];

    if (imageTensor && model) {
      const predictions = await detectObjects(model, [imageTensor]);
      setPredictions(predictions || []);
    }
  };

  return (
    <View style={styles.container}>
      <TensorCamera
              style={styles.camera}
              type={1} // 1 para la cámara trasera, 0 para la cámara frontal
              resizeHeight={300}
              resizeWidth={300}
              resizeDepth={3}
              onReady={handleCameraStream}
              autorender={true}
              ref={(ref) => {
                  cameraRef.current = ref as unknown as Camera;
              } } useCustomShadersToResize={false} cameraTextureWidth={0} cameraTextureHeight={0}      />
      <Text>Predictions: {JSON.stringify(predictions)}</Text>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  camera: {
    width: 300,
    height: 300,
  },
});

export default ObjectDetectionComponent;

