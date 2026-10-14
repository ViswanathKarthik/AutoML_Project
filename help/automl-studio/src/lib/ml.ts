import { ProblemType, ClassificationMetrics, RegressionMetrics } from "../types";

// Simple CSV parser that handles basic commas and quotes
export function parseCSV(csvText: string): Record<string, string | number>[] {
  const lines = csvText.split(/\r?\n/).filter(line => line.trim().length > 0);
  if (lines.length === 0) return [];

  // Parse header
  const headers = parseCSVLine(lines[0]);
  const rows: Record<string, string | number>[] = [];

  for (let i = 1; i < lines.length; i++) {
    const values = parseCSVLine(lines[i]);
    const row: Record<string, string | number> = {};
    for (let j = 0; j < headers.length; j++) {
      const val = values[j] !== undefined ? values[j].trim() : "";
      // Convert to number if numeric
      if (val !== "" && !isNaN(Number(val))) {
        row[headers[j]] = Number(val);
      } else {
        row[headers[j]] = val;
      }
    }
    rows.push(row);
  }
  return rows;
}

function parseCSVLine(line: string): string[] {
  const result: string[] = [];
  let current = "";
  let inQuotes = false;
  for (let i = 0; i < line.length; i++) {
    const char = line[i];
    if (char === '"') {
      inQuotes = !inQuotes;
    } else if (char === ',' && !inQuotes) {
      result.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  result.push(current);
  return result;
}

// Preprocessing & Encoding Context
export interface PreprocessContext {
  features: string[];
  targetCol: string;
  problemType: ProblemType;
  classes: string[]; // for classification target
  categoricalMappings: Record<string, string[]>; // column name -> list of unique categories
  numericStats: Record<string, { mean: number; std: number; min: number; max: number }>;
}

export function preprocessDataset(
  rawRows: Record<string, string | number>[],
  targetCol: string,
  problemType: ProblemType,
  excludedCols: string[] = []
): {
  X: number[][]; // preprocessed feature matrix
  y: number[];   // target vector
  context: PreprocessContext;
} {
  // 1. Identify columns
  const allCols = Object.keys(rawRows[0] || {});
  const features = allCols.filter(col => col !== targetCol && !excludedCols.includes(col));

  // 2. Identify categorical mappings and compute numeric stats
  const categoricalMappings: Record<string, string[]> = {};
  const numericStats: Record<string, { mean: number; std: number; min: number; max: number }> = {};
  
  // Categorical targets for classification
  let classes: string[] = [];
  if (problemType === "classification") {
    const targetValues = new Set<string>();
    rawRows.forEach(row => {
      const val = String(row[targetCol] !== undefined ? row[targetCol] : "");
      if (val !== "") targetValues.add(val);
    });
    classes = Array.from(targetValues).sort();
  }

  // Pre-calculate mappings and means/std
  features.forEach(col => {
    // Check if the column is mostly numeric or categorical
    let isNumeric = true;
    let sum = 0;
    let count = 0;
    const values: number[] = [];

    rawRows.forEach(row => {
      const val = row[col];
      if (val !== undefined && val !== null && val !== "") {
        if (typeof val === "number") {
          sum += val;
          count++;
          values.push(val);
        } else {
          isNumeric = false;
        }
      }
    });

    if (isNumeric && count > 0) {
      const mean = sum / count;
      const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
      const variance = squaredDiffs.reduce((a, b) => a + b, 0) / count;
      const std = Math.sqrt(variance) || 1;
      const min = Math.min(...values);
      const max = Math.max(...values);
      numericStats[col] = { mean, std, min, max };
    } else {
      // Categorical feature mapping
      const uniqueVals = Array.from(new Set(rawRows.map(row => String(row[col] ?? ""))));
      categoricalMappings[col] = uniqueVals.sort();
    }
  });

  // 3. Build Matrices
  const X: number[][] = [];
  const y: number[] = [];

  rawRows.forEach(row => {
    const xRow: number[] = [];
    let isValidRow = true;

    // Process target
    let targetVal = 0;
    if (problemType === "classification") {
      const valStr = String(row[targetCol] ?? "");
      const idx = classes.indexOf(valStr);
      targetVal = idx !== -1 ? idx : 0;
    } else {
      // regression
      const valNum = Number(row[targetCol] ?? 0);
      targetVal = isNaN(valNum) ? 0 : valNum;
    }

    // Process features
    features.forEach(col => {
      const rawVal = row[col];
      if (numericStats[col]) {
        // Numeric column
        let val = typeof rawVal === "number" ? rawVal : Number(rawVal);
        if (isNaN(val)) {
          val = numericStats[col].mean; // impute with mean
        }
        // Normalize using Standard Scaling (Z-score)
        const normalized = (val - numericStats[col].mean) / numericStats[col].std;
        xRow.push(normalized);
      } else if (categoricalMappings[col]) {
        // Categorical column: index encoding
        const valStr = String(rawVal ?? "");
        const idx = categoricalMappings[col].indexOf(valStr);
        xRow.push(idx !== -1 ? idx : 0);
      } else {
        xRow.push(0);
      }
    });

    if (isValidRow) {
      X.push(xRow);
      y.push(targetVal);
    }
  });

  const context: PreprocessContext = {
    features,
    targetCol,
    problemType,
    classes,
    categoricalMappings,
    numericStats,
  };

  return { X, y, context };
}

// Single Out-Of-Sample prediction helper
export function predictSingle(
  rowInput: Record<string, string | number>,
  context: PreprocessContext,
  modelType: string,
  modelArtifact: any
): string | number {
  // Preprocess input row
  const xInput: number[] = [];
  context.features.forEach(col => {
    const rawVal = rowInput[col];
    if (context.numericStats[col]) {
      let val = typeof rawVal === "number" ? rawVal : Number(rawVal);
      if (isNaN(val)) val = context.numericStats[col].mean;
      const normalized = (val - context.numericStats[col].mean) / context.numericStats[col].std;
      xInput.push(normalized);
    } else if (context.categoricalMappings[col]) {
      const valStr = String(rawVal ?? "");
      const idx = context.categoricalMappings[col].indexOf(valStr);
      xInput.push(idx !== -1 ? idx : 0);
    } else {
      xInput.push(0);
    }
  });

  if (context.problemType === "classification") {
    const classIdx = runPredictionClassification(xInput, modelType, modelArtifact);
    return context.classes[classIdx] !== undefined ? context.classes[classIdx] : `Class ${classIdx}`;
  } else {
    return runPredictionRegression(xInput, modelType, modelArtifact);
  }
}

// --- CLASSIFICATION MODELS ---

// 1. Logistic Regression Classifier
export function trainLogisticRegression(
  X: number[][],
  y: number[],
  numClasses: number,
  epochs: number = 200,
  lr: number = 0.1
): any {
  const m = X.length;
  const n = X[0]?.length || 0;
  
  // Weights array: dimensions numClasses x (n + 1) for bias
  const weights: number[][] = Array.from({ length: numClasses }, () => Array(n + 1).fill(0));

  for (let epoch = 0; epoch < epochs; epoch++) {
    for (let i = 0; i < m; i++) {
      const xi = [1, ...X[i]]; // Add bias term
      const yi = y[i];

      // Calculate logits
      const scores = weights.map(w => {
        return w.reduce((acc, val, idx) => acc + val * xi[idx], 0);
      });

      // Softmax
      const maxScore = Math.max(...scores);
      const exps = scores.map(s => Math.exp(s - maxScore));
      const sumExps = exps.reduce((a, b) => a + b, 0);
      const probs = exps.map(e => e / (sumExps || 1));

      // Gradient Descent
      for (let c = 0; c < numClasses; c++) {
        const target = yi === c ? 1 : 0;
        const error = probs[c] - target;
        for (let j = 0; j < xi.length; j++) {
          weights[c][j] -= lr * error * xi[j];
        }
      }
    }
  }

  return { weights };
}

// 2. Gaussian Naive Bayes Classifier
export function trainNaiveBayes(X: number[][], y: number[], numClasses: number): any {
  const n = X[0]?.length || 0;
  const classCounts = Array(numClasses).fill(0);
  const classSums = Array.from({ length: numClasses }, () => Array(n).fill(0));
  const classMeans = Array.from({ length: numClasses }, () => Array(n).fill(0));
  const classVars = Array.from({ length: numClasses }, () => Array(n).fill(0));

  X.forEach((row, i) => {
    const c = y[i];
    classCounts[c]++;
    for (let j = 0; j < n; j++) {
      classSums[c][j] += row[j];
    }
  });

  const classPrior = classCounts.map(count => count / (X.length || 1));

  for (let c = 0; c < numClasses; c++) {
    const count = classCounts[c] || 1;
    for (let j = 0; j < n; j++) {
      classMeans[c][j] = classSums[c][j] / count;
    }
  }

  X.forEach((row, i) => {
    const c = y[i];
    for (let j = 0; j < n; j++) {
      classVars[c][j] += Math.pow(row[j] - classMeans[c][j], 2);
    }
  });

  for (let c = 0; c < numClasses; c++) {
    const count = classCounts[c] || 1;
    for (let j = 0; j < n; j++) {
      classVars[c][j] = (classVars[c][j] / count) + 1e-4; // small epsilon to prevent 0 variance
    }
  }

  return { classMeans, classVars, classPrior };
}

// 3. K-Nearest Neighbors Classifier
export function trainKNNClassifier(X: number[][], y: number[], k: number = 3): any {
  // Lazy model - KNN holds training data
  return { X_train: X, y_train: y, k };
}

// 4. Decision Tree Classifier
interface DecisionTreeNode {
  featureIdx?: number;
  threshold?: number;
  left?: DecisionTreeNode;
  right?: DecisionTreeNode;
  value?: number; // Target label for leaf node
}

export function trainDecisionTreeClassification(
  X: number[][],
  y: number[],
  maxDepth: number = 5,
  currentDepth: number = 0
): DecisionTreeNode {
  const numSamples = X.length;
  if (numSamples === 0) return { value: 0 };

  const classCounts = new Map<number, number>();
  y.forEach(label => classCounts.set(label, (classCounts.get(label) || 0) + 1));
  
  let bestClass = 0;
  let maxCount = 0;
  classCounts.forEach((count, cls) => {
    if (count > maxCount) {
      maxCount = count;
      bestClass = cls;
    }
  });

  // Stop splitting criteria
  if (classCounts.size === 1 || currentDepth >= maxDepth || numSamples < 4) {
    return { value: bestClass };
  }

  const n = X[0]?.length || 0;
  let bestGini = 1.0;
  let bestSplitFeature = -1;
  let bestSplitThreshold = 0;
  let bestLeftX: number[][] = [];
  let bestLeftY: number[] = [];
  let bestRightX: number[][] = [];
  let bestRightY: number[] = [];

  // Find best split
  for (let f = 0; f < n; f++) {
    const values = X.map(row => row[f]);
    const thresholds = Array.from(new Set(values)).sort();

    for (let tIdx = 0; tIdx < thresholds.length - 1; tIdx++) {
      const threshold = (thresholds[tIdx] + thresholds[tIdx + 1]) / 2;
      const leftX: number[][] = [];
      const leftY: number[] = [];
      const rightX: number[][] = [];
      const rightY: number[] = [];

      for (let i = 0; i < numSamples; i++) {
        if (X[i][f] <= threshold) {
          leftX.push(X[i]);
          leftY.push(y[i]);
        } else {
          rightX.push(X[i]);
          rightY.push(y[i]);
        }
      }

      if (leftY.length === 0 || rightY.length === 0) continue;

      // Calculate Gini Impurity
      const giniLeft = calculateGini(leftY);
      const giniRight = calculateGini(rightY);
      const totalGini = (leftY.length / numSamples) * giniLeft + (rightY.length / numSamples) * giniRight;

      if (totalGini < bestGini) {
        bestGini = totalGini;
        bestSplitFeature = f;
        bestSplitThreshold = threshold;
        bestLeftX = leftX;
        bestLeftY = leftY;
        bestRightX = rightX;
        bestRightY = rightY;
      }
    }
  }

  if (bestSplitFeature === -1) {
    return { value: bestClass };
  }

  return {
    featureIdx: bestSplitFeature,
    threshold: bestSplitThreshold,
    left: trainDecisionTreeClassification(bestLeftX, bestLeftY, maxDepth, currentDepth + 1),
    right: trainDecisionTreeClassification(bestRightX, bestRightY, maxDepth, currentDepth + 1),
  };
}

function calculateGini(labels: number[]): number {
  const counts: Record<number, number> = {};
  labels.forEach(l => counts[l] = (counts[l] || 0) + 1);
  const m = labels.length || 1;
  let sumSquares = 0;
  for (const k in counts) {
    sumSquares += Math.pow(counts[k] / m, 2);
  }
  return 1 - sumSquares;
}

// Prediction classifier router
function runPredictionClassification(x: number[], modelType: string, artifact: any): number {
  if (modelType === "logistic_regression") {
    const weights: number[][] = artifact.weights;
    const scores = weights.map(w => {
      let score = w[0]; // Bias term
      for (let i = 0; i < x.length; i++) {
        score += w[i + 1] * x[i];
      }
      return score;
    });
    return scores.indexOf(Math.max(...scores));
  } else if (modelType === "naive_bayes") {
    const { classMeans, classVars, classPrior } = artifact;
    const logProbs = classPrior.map((prior: number, c: number) => {
      let logProb = Math.log(prior || 1e-4);
      for (let j = 0; j < x.length; j++) {
        const mu = classMeans[c][j];
        const variance = classVars[c][j];
        const val = x[j];
        // log of Gaussian probability density
        const g = -0.5 * Math.log(2 * Math.PI * variance) - (Math.pow(val - mu, 2) / (2 * variance));
        logProb += g;
      }
      return logProb;
    });
    return logProbs.indexOf(Math.max(...logProbs));
  } else if (modelType === "knn") {
    const { X_train, y_train, k } = artifact;
    const distances = X_train.map((row: number[], idx: number) => {
      let distSq = 0;
      for (let j = 0; j < x.length; j++) {
        distSq += Math.pow(x[j] - row[j], 2);
      }
      return { idx, dist: Math.sqrt(distSq) };
    });
    distances.sort((a: any, b: any) => a.dist - b.dist);
    
    const votes: Record<number, number> = {};
    for (let i = 0; i < Math.min(k, distances.length); i++) {
      const cls = y_train[distances[i].idx];
      votes[cls] = (votes[cls] || 0) + 1;
    }

    let bestClass = 0;
    let maxVotes = 0;
    for (const cls in votes) {
      if (votes[cls] > maxVotes) {
        maxVotes = votes[cls];
        bestClass = Number(cls);
      }
    }
    return bestClass;
  } else if (modelType === "decision_tree") {
    let node: DecisionTreeNode = artifact;
    while (node.left && node.right) {
      if (node.featureIdx !== undefined && node.threshold !== undefined) {
        if (x[node.featureIdx] <= node.threshold) {
          node = node.left;
        } else {
          node = node.right;
        }
      } else {
        break;
      }
    }
    return node.value ?? 0;
  }
  return 0;
}


// --- REGRESSION MODELS ---

// 1. Linear Regression (Gradient Descent with bias)
export function trainLinearRegression(
  X: number[][],
  y: number[],
  epochs: number = 200,
  lr: number = 0.05
): any {
  const m = X.length;
  const n = X[0]?.length || 0;
  const weights = Array(n + 1).fill(0); // weights[0] is bias

  for (let epoch = 0; epoch < epochs; epoch++) {
    for (let i = 0; i < m; i++) {
      const xi = X[i];
      const yi = y[i];

      // Predict
      let yPred = weights[0];
      for (let j = 0; j < n; j++) {
        yPred += weights[j + 1] * xi[j];
      }

      const error = yPred - yi;

      // Update weights
      weights[0] -= lr * error;
      for (let j = 0; j < n; j++) {
        weights[j + 1] -= lr * error * xi[j];
      }
    }
  }

  return { weights };
}

// 2. KNN Regressor
export function trainKNNRegressor(X: number[][], y: number[], k: number = 3): any {
  return { X_train: X, y_train: y, k };
}

// 3. Decision Tree Regressor
export function trainDecisionTreeRegression(
  X: number[][],
  y: number[],
  maxDepth: number = 5,
  currentDepth: number = 0
): DecisionTreeNode {
  const numSamples = X.length;
  if (numSamples === 0) return { value: 0 };

  const mean = y.reduce((a, b) => a + b, 0) / numSamples;

  if (currentDepth >= maxDepth || numSamples < 5) {
    return { value: mean };
  }

  const n = X[0]?.length || 0;
  let bestMSEDecrease = 0;
  let bestSplitFeature = -1;
  let bestSplitThreshold = 0;
  let bestLeftX: number[][] = [];
  let bestLeftY: number[] = [];
  let bestRightX: number[][] = [];
  let bestRightY: number[] = [];

  const initialMSE = calculateMSE(y, mean);

  for (let f = 0; f < n; f++) {
    const values = X.map(row => row[f]);
    const thresholds = Array.from(new Set(values)).sort();

    for (let tIdx = 0; tIdx < thresholds.length - 1; tIdx++) {
      const threshold = (thresholds[tIdx] + thresholds[tIdx + 1]) / 2;
      const leftX: number[][] = [];
      const leftY: number[] = [];
      const rightX: number[][] = [];
      const rightY: number[] = [];

      for (let i = 0; i < numSamples; i++) {
        if (X[i][f] <= threshold) {
          leftX.push(X[i]);
          leftY.push(y[i]);
        } else {
          rightX.push(X[i]);
          rightY.push(y[i]);
        }
      }

      if (leftY.length === 0 || rightY.length === 0) continue;

      const meanLeft = leftY.reduce((a, b) => a + b, 0) / leftY.length;
      const meanRight = rightY.reduce((a, b) => a + b, 0) / rightY.length;

      const mseLeft = calculateMSE(leftY, meanLeft);
      const mseRight = calculateMSE(rightY, meanRight);
      const weightedMSE = (leftY.length / numSamples) * mseLeft + (rightY.length / numSamples) * mseRight;

      const decrease = initialMSE - weightedMSE;
      if (decrease > bestMSEDecrease) {
        bestMSEDecrease = decrease;
        bestSplitFeature = f;
        bestSplitThreshold = threshold;
        bestLeftX = leftX;
        bestLeftY = leftY;
        bestRightX = rightX;
        bestRightY = rightY;
      }
    }
  }

  if (bestSplitFeature === -1) {
    return { value: mean };
  }

  return {
    featureIdx: bestSplitFeature,
    threshold: bestSplitThreshold,
    left: trainDecisionTreeRegression(bestLeftX, bestLeftY, maxDepth, currentDepth + 1),
    right: trainDecisionTreeRegression(bestRightX, bestRightY, maxDepth, currentDepth + 1),
  };
}

function calculateMSE(y: number[], mean: number): number {
  return y.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / (y.length || 1);
}

// Prediction regression router
function runPredictionRegression(x: number[], modelType: string, artifact: any): number {
  if (modelType === "linear_regression") {
    const weights: number[] = artifact.weights;
    let yPred = weights[0];
    for (let i = 0; i < x.length; i++) {
      yPred += weights[i + 1] * x[i];
    }
    return yPred;
  } else if (modelType === "knn") {
    const { X_train, y_train, k } = artifact;
    const distances = X_train.map((row: number[], idx: number) => {
      let distSq = 0;
      for (let j = 0; j < x.length; j++) {
        distSq += Math.pow(x[j] - row[j], 2);
      }
      return { idx, dist: Math.sqrt(distSq) };
    });
    distances.sort((a: any, b: any) => a.dist - b.dist);

    let sum = 0;
    const actualK = Math.min(k, distances.length);
    for (let i = 0; i < actualK; i++) {
      sum += y_train[distances[i].idx];
    }
    return sum / (actualK || 1);
  } else if (modelType === "decision_tree") {
    let node: DecisionTreeNode = artifact;
    while (node.left && node.right) {
      if (node.featureIdx !== undefined && node.threshold !== undefined) {
        if (x[node.featureIdx] <= node.threshold) {
          node = node.left;
        } else {
          node = node.right;
        }
      } else {
        break;
      }
    }
    return node.value ?? 0;
  }
  return 0;
}


// --- EVALUATION METRICS ENGINE ---

export function evaluateClassifier(
  X_val: number[][],
  y_val: number[],
  modelType: string,
  artifact: any,
  classes: string[]
): ClassificationMetrics {
  let correct = 0;
  const m = y_val.length;
  
  const classCount = classes.length;
  const confusionMatrix = Array.from({ length: classCount }, () => Array(classCount).fill(0));

  for (let i = 0; i < m; i++) {
    const pred = runPredictionClassification(X_val[i], modelType, artifact);
    const actual = y_val[i];
    confusionMatrix[actual][pred]++;
    if (pred === actual) {
      correct++;
    }
  }

  const accuracy = correct / (m || 1);

  // Compute multi-class averages for Precision, Recall, F1
  let sumPrecision = 0;
  let sumRecall = 0;

  for (let c = 0; c < classCount; c++) {
    // True Positive
    const tp = confusionMatrix[c][c];
    
    // Predicted Positive (TP + FP)
    let predPos = 0;
    for (let r = 0; r < classCount; r++) predPos += confusionMatrix[r][c];
    
    // Actual Positive (TP + FN)
    let actPos = 0;
    for (let col = 0; col < classCount; col++) actPos += confusionMatrix[c][col];

    const precision = predPos > 0 ? tp / predPos : 0;
    const recall = actPos > 0 ? tp / actPos : 0;

    sumPrecision += precision;
    sumRecall += recall;
  }

  const precision = sumPrecision / classCount;
  const recall = sumRecall / classCount;
  const f1Score = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;

  return {
    accuracy,
    precision,
    recall,
    f1Score,
    confusionMatrix,
    classes,
  };
}

export function evaluateRegressor(
  X_val: number[][],
  y_val: number[],
  modelType: string,
  artifact: any
): RegressionMetrics {
  const m = y_val.length;
  let sumSquaredError = 0;
  let sumAbsoluteError = 0;
  let sumY = 0;

  const predictions: number[] = [];
  for (let i = 0; i < m; i++) {
    const pred = runPredictionRegression(X_val[i], modelType, artifact);
    predictions.push(pred);
    sumSquaredError += Math.pow(pred - y_val[i], 2);
    sumAbsoluteError += Math.abs(pred - y_val[i]);
    sumY += y_val[i];
  }

  const meanY = sumY / (m || 1);
  let sumTotalSquaredError = 0;
  for (let i = 0; i < m; i++) {
    sumTotalSquaredError += Math.pow(y_val[i] - meanY, 2);
  }

  const mse = sumSquaredError / (m || 1);
  const rmse = Math.sqrt(mse);
  const mae = sumAbsoluteError / (m || 1);
  
  // R-squared
  const r2 = sumTotalSquaredError > 0 ? 1 - (sumSquaredError / sumTotalSquaredError) : 1.0;

  return {
    r2,
    rmse,
    mae,
    mse,
  };
}
