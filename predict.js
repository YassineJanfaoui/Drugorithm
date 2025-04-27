const express = require("express");
const tf = require("@tensorflow/tfjs-node");
const sharp = require("sharp");
const fs = require("fs");
const path = require("path");
const bodyParser = require("body-parser");
const cors = require("cors");
const multer = require("multer");

const app = express();
const port = process.env.PORT || 3000;

// Middleware
app.use(cors());
app.use(bodyParser.json({ limit: "50mb" }));
app.use(bodyParser.urlencoded({ extended: true, limit: "50mb" }));

// Initialize model and metrics calculator
let model;
const classes = ["lung_adenocarcinoma", "lung_benign"];
const upload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 10 * 1024 * 1024 }, // 10MB limit
});
// Load model on startup
async function initializeModel() {
  try {
   const modelPath = path.join(__dirname, "model_lung", "model.json");
model = await tf.loadLayersModel(`file://${modelPath}`);

    console.log("Model loaded successfully");
  } catch (error) {
    console.error("Error loading model:", error);
    process.exit(1);
  }
}

class MetricsCalculator {
  constructor(classes) {
    this.classes = classes;
    this.metrics = {
      confusionMatrix: Array.from({ length: classes.length }, () =>
        Array.from({ length: classes.length }, () => 0)
      ),
      classStats: classes.map(() => ({
        tp: 0,
        fp: 0,
        tn: 0,
        fn: 0,
      })),
      total: 0,
      correct: 0,
    };
  }

  updateMetrics(trueLabel, predictedLabel) {
    this.metrics.total++;
    this.metrics.confusionMatrix[trueLabel][predictedLabel]++;

    if (trueLabel === predictedLabel) {
      this.metrics.correct++;
    }

    this.classes.forEach((_, classIndex) => {
      const stats = this.metrics.classStats[classIndex];
      if (classIndex === trueLabel) {
        if (classIndex === predictedLabel) stats.tp++;
        else stats.fn++;
      } else {
        if (classIndex === predictedLabel) stats.fp++;
        else stats.tn++;
      }
    });
  }

  calculateMetrics() {
    const results = {
      overall: {
        accuracy: this.metrics.correct / this.metrics.total,
      },
      classes: {},
      macroAvg: {
        precision: 0,
        recall: 0,
        f1: 0,
      },
    };

    this.classes.forEach((className, classIndex) => {
      const { tp, fp, fn } = this.metrics.classStats[classIndex];
      const precision = tp / (tp + fp) || 0;
      const recall = tp / (tp + fn) || 0;
      const f1 = (2 * (precision * recall)) / (precision + recall) || 0;

      results.classes[className] = {
        precision,
        recall,
        f1,
        support: this.metrics.confusionMatrix[classIndex].reduce(
          (a, b) => a + b
        ),
      };

      results.macroAvg.precision += precision;
      results.macroAvg.recall += recall;
      results.macroAvg.f1 += f1;
    });

    const numClasses = this.classes.length;
    results.macroAvg.precision /= numClasses;
    results.macroAvg.recall /= numClasses;
    results.macroAvg.f1 /= numClasses;

    return results;
  }

  printMetrics() {
    const metrics = this.calculateMetrics();

    console.log("\nConfusion Matrix:");
    console.table(
      this.metrics.confusionMatrix.map((row, i) =>
        Object.assign(
          { "Actual Class": this.classes[i] },
          ...row.map((val, j) => ({ [this.classes[j]]: val }))
        )
      )
    );

    console.log("\nClassification Report:");
    console.table(
      Object.entries(metrics.classes).map(([className, stats]) => ({
        Class: className,
        Precision: stats.precision.toFixed(4),
        Recall: stats.recall.toFixed(4),
        "F1-Score": stats.f1.toFixed(4),
        Support: stats.support,
      }))
    );

    console.log("\nMacro Averages:");
    console.table({
      "Avg Precision": metrics.macroAvg.precision.toFixed(4),
      "Avg Recall": metrics.macroAvg.recall.toFixed(4),
      "Avg F1-Score": metrics.macroAvg.f1.toFixed(4),
    });

    console.log(
      `\nOverall Accuracy: ${(metrics.overall.accuracy * 100).toFixed(2)}%`
    );
  }
}
app.post("/predict", upload.single("image"), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: "No image file uploaded" });
    }

    // Process image
    const processedImage = await sharp(req.file.buffer)
      .resize(224, 224)
      .toFormat("jpeg")
      .toBuffer();

    // Create tensor
    const tensor = tf.tidy(() => {
      return tf.node
        .decodeImage(processedImage, 3)
        .toFloat()
        .div(255.0)
        .expandDims();
    });

    // Make prediction
    const prediction = model.predict(tensor);
    const results = await prediction.data();
    const predictedClass = tf.argMax(results).dataSync()[0];

    // Cleanup
    tf.dispose([tensor, prediction]);

    res.json({
      prediction: classes[predictedClass],
      confidence: results[predictedClass],
      probabilities: {
        [classes[0]]: results[0],
        [classes[1]]: results[1],
      },
    });
  } catch (error) {
    console.error("Prediction error:", error);
    res.status(500).json({
      error: "Prediction failed",
      details: error.message,
    });
  }
});

// Evaluation endpoint
app.post("/evaluate", async (req, res) => {
  try {
    const { testPaths } = req.body;

    if (!testPaths || !Array.isArray(testPaths)) {
      return res.status(400).json({ error: "Invalid test paths" });
    }

    const metrics = new MetricsCalculator(classes);

    for (const { path: dirPath, label: trueLabel } of testPaths) {
      const files = await fs.promises.readdir(dirPath);

      for (const file of files) {
        if (!/\.(jpg|jpeg|png)$/i.test(file)) continue;

        const imagePath = path.join(dirPath, file);
        const predictedLabel = await predictImage(model, imagePath);

        if (predictedLabel !== null) {
          metrics.updateMetrics(trueLabel, predictedLabel);
        }
      }
    }

    res.json(metrics.calculateMetrics());
  } catch (error) {
    console.error("Evaluation error:", error);
    res
      .status(500)
      .json({ error: "Evaluation failed", details: error.message });
  }
});

// Helper function (reusable prediction logic)
async function predictImage(model, imagePath) {
  try {
    const imageBuffer = await fs.promises.readFile(imagePath);
    const processedImage = await sharp(imageBuffer)
      .resize(224, 224)
      .toFormat("jpeg")
      .toBuffer();

    const tensor = tf.tidy(() => {
      return tf.node
        .decodeImage(processedImage, 3)
        .toFloat()
        .div(255.0)
        .expandDims();
    });

    const prediction = model.predict(tensor);
    const results = await prediction.data();
    const predictedClass = tf.argMax(results).dataSync()[0];

    tf.dispose([tensor, prediction]);
    return predictedClass;
  } catch (error) {
    console.error(
      `Error processing ${path.basename(imagePath)}: ${error.message}`
    );
    return null;
  }
}

// Start server
initializeModel().then(() => {
  app.listen(port, '0.0.0.0', () => {
    console.log(`Server running on http://0.0.0.0:${port}`);
  });
});
