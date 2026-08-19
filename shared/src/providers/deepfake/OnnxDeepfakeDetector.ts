/**
 * ONNX-based Deepfake Detector
 *
 * Uses an EfficientNet-B0 binary classifier (trained on FaceForensics++)
 * to detect AI-generated or manipulated face images.
 *
 * The model expects a 224x224 face crop with ImageNet normalization.
 * Runs ~50-150ms on CPU via onnxruntime-node.
 *
 * Lazy-init singleton pattern -- model loads once on first call, reused after.
 */

import { logger } from '@/utils/logger.js';
import sharp from 'sharp';
import path from 'path';
import fs from 'fs';
import type { FaceBufferDetectionResult } from '@/types/faceRecognition.js';

// Dynamic import to avoid hard crash if onnxruntime-node not available
let ort: typeof import('onnxruntime-node') | null = null;

export interface DeepfakeDetectionResult {
  /** Whether the face is likely real */
  isReal: boolean;
  /** Probability that the face is real (0-1) */
  realProbability: number;
  /** Probability that the face is fake/generated (0-1) */
  fakeProbability: number;
}

/** Reuse the canonical bounding box shape from face detection */
export type BoundingBox = FaceBufferDetectionResult['boundingBox'];

// Fallback preprocessing: ImageNet normalization, [fake, real] output order.
// Used when the model ships without a sidecar descriptor.
const IMAGENET_MEAN = [0.485, 0.456, 0.406]; // RGB
const IMAGENET_STD = [0.229, 0.224, 0.225];
const INPUT_SIZE = 224;

/**
 * Preprocessing and output layout of the loaded model, read from a sidecar
 * `<model>.json` written by scripts/models/export-deepfake-detector.py.
 *
 * These are not cosmetic details: a model trained with 0.5/0.5 normalization
 * fed ImageNet-normalized pixels answers noise, and a model whose first logit
 * is "real" read as "fake" inverts every verdict. The sidecar keeps the two
 * halves — exported weights and runtime preprocessing — in agreement, so the
 * model can be swapped without editing this file.
 */
export interface DeepfakeModelDescriptor {
  inputSize: number;
  mean: number[];
  std: number[];
  /** Class order of the output logits. */
  labels: ('real' | 'fake')[];
}

const DEFAULT_DESCRIPTOR: DeepfakeModelDescriptor = {
  inputSize: INPUT_SIZE,
  mean: IMAGENET_MEAN,
  std: IMAGENET_STD,
  labels: ['fake', 'real'],
};

function readDescriptor(modelPath: string): DeepfakeModelDescriptor {
  const sidecarPath = modelPath.replace(/\.onnx$/, '.json');
  try {
    if (!fs.existsSync(sidecarPath)) return DEFAULT_DESCRIPTOR;
    const parsed = JSON.parse(fs.readFileSync(sidecarPath, 'utf8')) as Partial<DeepfakeModelDescriptor>;
    const labels = Array.isArray(parsed.labels) && parsed.labels.length === 2 ? parsed.labels : DEFAULT_DESCRIPTOR.labels;
    return {
      inputSize: typeof parsed.inputSize === 'number' && parsed.inputSize > 0 ? parsed.inputSize : DEFAULT_DESCRIPTOR.inputSize,
      mean: Array.isArray(parsed.mean) && parsed.mean.length === 3 ? parsed.mean : DEFAULT_DESCRIPTOR.mean,
      std: Array.isArray(parsed.std) && parsed.std.length === 3 ? parsed.std : DEFAULT_DESCRIPTOR.std,
      labels: labels as DeepfakeModelDescriptor['labels'],
    };
  } catch (err) {
    logger.warn('Deepfake descriptor unreadable, using defaults', {
      sidecarPath,
      error: err instanceof Error ? err.message : 'Unknown',
    });
    return DEFAULT_DESCRIPTOR;
  }
}

export class OnnxDeepfakeDetector {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any -- ort.InferenceSession loaded dynamically
  private session: any = null;
  private initPromise: Promise<void> | null = null;
  private available = true;

  private modelPath: string;
  private descriptor: DeepfakeModelDescriptor = DEFAULT_DESCRIPTOR;

  constructor(modelPath?: string) {
    this.modelPath = modelPath || path.join(
      path.dirname(new URL(import.meta.url).pathname.replace(/^\/([A-Z]:)/, '$1')),
      '..', '..', '..', 'models', 'deepfake-detector.onnx'
    );
  }

  /**
   * Lazily initialize the ONNX runtime and load the model.
   */
  private async initialize(): Promise<void> {
    if (this.session) return;
    if (!this.available) return;
    if (this.initPromise) return this.initPromise;

    this.initPromise = (async () => {
      try {
        // Dynamic import -- avoids crash if onnxruntime-node isn't installed
        if (!ort) {
          ort = await import('onnxruntime-node');
        }

        // Check if model file exists
        if (!fs.existsSync(this.modelPath)) {
          logger.warn('Deepfake detector model not found, disabling', { modelPath: this.modelPath });
          this.available = false;
          return;
        }

        this.descriptor = readDescriptor(this.modelPath);

        this.session = await ort.InferenceSession.create(this.modelPath, {
          executionProviders: ['cpu'],
          graphOptimizationLevel: 'all',
        });

        logger.info('Deepfake detector model loaded', {
          modelPath: this.modelPath,
          inputNames: this.session.inputNames,
          outputNames: this.session.outputNames,
          labels: this.descriptor.labels,
          inputSize: this.descriptor.inputSize,
        });
      } catch (err) {
        logger.warn('Deepfake detector initialization failed, disabling', {
          error: err instanceof Error ? err.message : 'Unknown',
        });
        this.available = false;
        this.initPromise = null;
      }
    })();

    return this.initPromise;
  }

  /**
   * Detect whether a face crop is real or AI-generated.
   *
   * @param faceCropBuffer  Buffer containing a face-cropped image
   * @returns Detection result with real/fake probabilities
   */
  async detect(faceCropBuffer: Buffer): Promise<DeepfakeDetectionResult> {
    await this.initialize();

    if (!this.session || !this.available || !ort) {
      // Model not available -- return neutral result
      return { isReal: true, realProbability: 0.5, fakeProbability: 0.5 };
    }

    try {
      const tensor = await this.preprocessToTensor(faceCropBuffer);
      const feeds: Record<string, any> = {};
      feeds[this.session.inputNames[0]] = tensor;

      const results = await this.session.run(feeds);
      const output = results[this.session.outputNames[0]];
      const data = output.data as Float32Array;

      // Two logits in the order declared by the descriptor -- apply softmax
      let realProb: number;
      let fakeProb: number;

      if (data.length >= 2) {
        const maxVal = Math.max(data[0], data[1]);
        const exp0 = Math.exp(data[0] - maxVal);
        const exp1 = Math.exp(data[1] - maxVal);
        const sum = exp0 + exp1;
        const firstIsReal = this.descriptor.labels[0] === 'real';
        realProb = (firstIsReal ? exp0 : exp1) / sum;
        fakeProb = (firstIsReal ? exp1 : exp0) / sum;
      } else {
        // Single sigmoid output
        realProb = 1 / (1 + Math.exp(-data[0]));
        fakeProb = 1 - realProb;
      }

      return {
        isReal: realProb > 0.5,
        realProbability: realProb,
        fakeProbability: fakeProb,
      };
    } catch (err) {
      logger.warn('Deepfake detection inference failed', {
        error: err instanceof Error ? err.message : 'Unknown',
      });
      return { isReal: true, realProbability: 0.5, fakeProbability: 0.5 };
    }
  }

  /**
   * Extract a face crop from a full image using the detected bounding box.
   * Adds 20% margin around the face for context.
   */
  async extractFaceCrop(fullImage: Buffer, bbox: BoundingBox): Promise<Buffer> {
    const meta = await sharp(fullImage).metadata();
    const imgW = meta.width || 0;
    const imgH = meta.height || 0;

    // Add 20% margin
    const margin = 0.20;
    const marginX = Math.round(bbox.width * margin);
    const marginY = Math.round(bbox.height * margin);

    const left = Math.max(0, Math.round(bbox.x) - marginX);
    const top = Math.max(0, Math.round(bbox.y) - marginY);
    const right = Math.min(imgW, Math.round(bbox.x + bbox.width) + marginX);
    const bottom = Math.min(imgH, Math.round(bbox.y + bbox.height) + marginY);

    const width = right - left;
    const height = bottom - top;

    if (width <= 0 || height <= 0) {
      throw new Error('Invalid face crop dimensions');
    }

    return sharp(fullImage)
      .extract({ left, top, width, height })
      .resize(INPUT_SIZE, INPUT_SIZE, { fit: 'cover' })
      .removeAlpha()
      .toBuffer();
  }

  /**
   * Preprocess a face crop to an NCHW tensor with ImageNet normalization.
   *
   * Pipeline: resize 224x224 -> raw RGB -> normalize per-channel -> [1,3,224,224]
   */
  private async preprocessToTensor(crop: Buffer): Promise<any> {
    if (!ort) throw new Error('onnxruntime-node not loaded');

    const { inputSize, mean, std } = this.descriptor;

    const { data } = await sharp(crop)
      .resize(inputSize, inputSize, { fit: 'cover' })
      .removeAlpha()
      .raw()
      .toBuffer({ resolveWithObject: true });

    // Convert HWC uint8 -> NCHW float32 with the model's own normalization
    const float32Data = new Float32Array(3 * inputSize * inputSize);
    const pixelCount = inputSize * inputSize;

    for (let i = 0; i < pixelCount; i++) {
      const r = data[i * 3] / 255.0;
      const g = data[i * 3 + 1] / 255.0;
      const b = data[i * 3 + 2] / 255.0;

      // NCHW layout: channel-first
      float32Data[i] = (r - mean[0]) / std[0];                    // R channel
      float32Data[pixelCount + i] = (g - mean[1]) / std[1];       // G channel
      float32Data[2 * pixelCount + i] = (b - mean[2]) / std[2];   // B channel
    }

    return new ort.Tensor('float32', float32Data, [1, 3, inputSize, inputSize]);
  }

  /** Check if the model is loaded and ready */
  isAvailable(): boolean {
    return this.available && this.session !== null;
  }
}
