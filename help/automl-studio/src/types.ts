export type ProblemType = "classification" | "regression";

export interface ColumnStat {
  name: string;
  type: "numeric" | "categorical";
  min?: number;
  max?: number;
  mean?: number;
  uniqueCount: number;
  missingCount: number;
  mostFrequent?: string | number;
}

export interface Dataset {
  name: string;
  rowCount: number;
  colCount: number;
  headers: string[];
  rawRows: Record<string, string | number>[];
  stats: ColumnStat[];
}

export interface PreprocessingConfig {
  trainRatio: number; // e.g. 0.8
  imputationStrategy: "mean_mode" | "zero" | "drop";
  scaling: boolean;
}

export interface ClassificationMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  confusionMatrix: number[][]; // 2x2 or classCount x classCount
  classes: string[];
}

export interface RegressionMetrics {
  r2: number;
  rmse: number;
  mae: number;
  mse: number;
}

export interface AlgorithmResult {
  algorithmId: string;
  algorithmName: string;
  metrics: ClassificationMetrics | RegressionMetrics;
  trainingTimeMs: number;
  modelArtifact: any; // Serialized model weights/trees/indices for prediction and download
}

export interface SampleDataset {
  id: string;
  name: string;
  description: string;
  csvData: string;
  targetCol: string;
  problemType: ProblemType;
}
