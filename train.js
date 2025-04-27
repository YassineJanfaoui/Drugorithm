const tf = require("@tensorflow/tfjs-node");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");

async function loadImages(dirPath, label) {
  const files = await fs.promises.readdir(dirPath);
  const images = [];
  const targetSize = 224; // Fixed size for all dimensions

  for (const file of files) {
    if (!/\.(jpg|jpeg|png)$/i.test(file)) continue;

    try {
      const imagePath = path.join(dirPath, file);
      const imageBuffer = await fs.promises.readFile(imagePath);

      // Process image with exact dimensions
      const processedImage = await sharp(imageBuffer)
        .resize(targetSize, targetSize, {
          fit: "fill", // Force exact size
          kernel: sharp.kernel.lanczos3,
          withoutEnlargement: false,
        })
        .toFormat("jpeg")
        .toBuffer();

      // Verify dimensions
      const { width, height } = await sharp(processedImage).metadata();
      if (width !== targetSize || height !== targetSize) {
        throw new Error(
          `Invalid dimensions after processing: ${width}x${height}`
        );
      }

      // Create tensor
      const tensor = tf.node
        .decodeImage(processedImage, 3)
        .toFloat()
        .div(255.0)
        .expandDims();

      // Verify tensor shape
      if (tensor.shape[1] !== targetSize || tensor.shape[2] !== targetSize) {
        throw new Error(`Invalid tensor shape: ${tensor.shape}`);
      }

      images.push(tensor);
    } catch (error) {
      console.error(`Error processing ${file}:`, error.message);
    }
  }

  if (images.length === 0) {
    console.warn(`No valid images found in ${dirPath}`);
    return { images: null, labels: null };
  }

  return {
    images: tf.concat(images),
    labels: tf.oneHot(tf.fill([images.length], label, "int32"), 2),
  };
}

async function trainModel() {
  try {
    // Load datasets
    const [class1, class2] = await Promise.all([
      loadImages(
        "/home/yassine/Documents/Drugorithm/Dataset_split/test/colon_adenocarcinoma",
        0
      ),
      loadImages(
        "/home/yassine/Documents/Drugorithm/Dataset_split/test/colon_benign",
        1
      ),
    ]);

    // Validate datasets
    if (!class1.images || !class2.images) {
      throw new Error("One or more classes have no valid images");
    }

    console.log("Class 1 shape:", class1.images.shape);
    console.log("Class 2 shape:", class2.images.shape);

    const xs = tf.concat([class1.images, class2.images]);
    const ys = tf.concat([class1.labels, class2.labels]);

    // Model definition
    const model = tf.sequential({
      layers: [
        tf.layers.conv2d({
          inputShape: [224, 224, 3],
          filters: 32,
          kernelSize: 3,
          activation: "relu",
        }),
        tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }),
        tf.layers.conv2d({ filters: 64, kernelSize: 3, activation: "relu" }),
        tf.layers.maxPooling2d({ poolSize: 2, strides: 2 }),
        tf.layers.flatten(),
        tf.layers.dense({ units: 128, activation: "relu" }),
        tf.layers.dense({ units: 2, activation: "softmax" }),
      ],
    });

    model.compile({
      optimizer: tf.train.adam(0.001),
      loss: "categoricalCrossentropy",
      metrics: ["accuracy"],
    });

    // Training with validation
    await model.fit(xs, ys, {
      epochs: 10,
      batchSize: 32,
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          console.log(`Epoch ${epoch + 1}: 
            Train Loss: ${logs.loss.toFixed(4)}
            Val Loss: ${logs.val_loss.toFixed(4)}
            Val Acc: ${logs.val_acc.toFixed(4)}`);
        },
      },
    });

    await model.save("file://./model");
    console.log("Model saved successfully!");
  } catch (error) {
    console.error("Training failed:", error);
  } finally {
    // Cleanup
    tf.disposeVariables();
  }
}

// Start training
trainModel();
