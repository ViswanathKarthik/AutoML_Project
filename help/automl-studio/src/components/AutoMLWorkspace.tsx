import React, { useState, useEffect, useRef } from "react";
import { 
  Upload, Database, Settings, Play, ArrowRight, CheckCircle, AlertCircle, 
  Download, RefreshCw, BarChart3, Binary, Table, HelpCircle, UserCheck, 
  Sparkles, FileText, ChevronRight, X, Eye, Code, Layers, Save, Cpu, Award
} from "lucide-react";
import { motion, AnimatePresence } from "motion/react";
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, 
  ResponsiveContainer, Cell, ScatterChart, Scatter, ZAxis
} from "recharts";

import { Dataset, ProblemType, ColumnStat, AlgorithmResult, ClassificationMetrics, RegressionMetrics } from "../types";
import { SAMPLE_DATASETS } from "../data/samples";
import { 
  parseCSV, preprocessDataset, trainLogisticRegression, trainNaiveBayes, 
  trainKNNClassifier, trainDecisionTreeClassification, trainLinearRegression, 
  trainKNNRegressor, trainDecisionTreeRegression, evaluateClassifier, 
  evaluateRegressor, predictSingle
} from "../lib/ml";

interface AutoMLWorkspaceProps {
  initialSampleId?: string;
  onGoBack: () => void;
}

export default function AutoMLWorkspace({ initialSampleId, onGoBack }: AutoMLWorkspaceProps) {
  // --- Workspace States ---
  const [csvInput, setCsvInput] = useState("");
  const [dragActive, setDragActive] = useState(false);
  
  const [dataset, setDataset] = useState<Dataset | null>(null);
  const [selectedTarget, setSelectedTarget] = useState<string>("");
  const [excludedCols, setExcludedCols] = useState<string[]>([]);
  const [problemType, setProblemType] = useState<ProblemType>("classification");
  const [trainRatio, setTrainRatio] = useState<number>(0.8);
  const [scaling, setScaling] = useState<boolean>(true);
  
  const [isTraining, setIsTraining] = useState(false);
  const [trainingStep, setTrainingStep] = useState("");
  const [modelResults, setModelResults] = useState<AlgorithmResult[]>([]);
  
  // Tabs
  const [activeTab, setActiveTab] = useState<"preview" | "comparison" | "playground" | "ai_insights">("preview");
  
  // Out of Sample Prediction Playground
  const [playgroundInputs, setPlaygroundInputs] = useState<Record<string, string>>({});
  const [testCsvInput, setTestCsvInput] = useState("");
  const [batchPredictions, setBatchPredictions] = useState<any[] | null>(null);
  
  // Gemini AI Insights
  const [aiInsights, setAiInsights] = useState<string>("");
  const [isLoadingAI, setIsLoadingAI] = useState(false);
  const [aiError, setAiError] = useState("");

  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load sample dataset on mount if selected
  useEffect(() => {
    if (initialSampleId) {
      const sample = SAMPLE_DATASETS.find(d => d.id === initialSampleId);
      if (sample) {
        handleLoadSample(sample.id);
      }
    }
  }, [initialSampleId]);

  // Infer metrics when target variable changes
  useEffect(() => {
    if (dataset && selectedTarget) {
      const targetStat = dataset.stats.find(s => s.name === selectedTarget);
      if (targetStat) {
        // If target is mostly strings, default to classification, otherwise regression
        if (targetStat.type === "categorical") {
          setProblemType("classification");
        } else {
          // If numeric but has very few unique values, could be binary classifier
          if (targetStat.uniqueCount < 6) {
            setProblemType("classification");
          } else {
            setProblemType("regression");
          }
        }
      }
    }
  }, [selectedTarget, dataset]);

  // Compute column statistics of raw records
  const calculateDatasetStats = (name: string, rawRows: Record<string, string | number>[]): Dataset => {
    if (rawRows.length === 0) {
      return { name, rowCount: 0, colCount: 0, headers: [], rawRows: [], stats: [] };
    }
    const headers = Object.keys(rawRows[0]);
    const stats: ColumnStat[] = headers.map(col => {
      let numericCount = 0;
      let stringCount = 0;
      let missingCount = 0;
      let sum = 0;
      const values: number[] = [];
      const freqMap: Record<string, number> = {};

      rawRows.forEach(row => {
        const val = row[col];
        if (val === undefined || val === null || val === "") {
          missingCount++;
          return;
        }
        
        freqMap[String(val)] = (freqMap[String(val)] || 0) + 1;

        if (typeof val === "number") {
          numericCount++;
          sum += val;
          values.push(val);
        } else {
          stringCount++;
        }
      });

      const isNumeric = numericCount > stringCount;
      const uniqueCount = Object.keys(freqMap).length;

      // Find most frequent
      let mostFrequent: string | number = "";
      let maxFreq = 0;
      Object.entries(freqMap).forEach(([k, v]) => {
        if (v > maxFreq) {
          maxFreq = v;
          mostFrequent = isNumeric && !isNaN(Number(k)) ? Number(k) : k;
        }
      });

      const stat: ColumnStat = {
        name: col,
        type: isNumeric ? "numeric" : "categorical",
        uniqueCount,
        missingCount,
        mostFrequent
      };

      if (isNumeric && values.length > 0) {
        stat.min = Math.min(...values);
        stat.max = Math.max(...values);
        stat.mean = sum / values.length;
      }

      return stat;
    });

    return {
      name,
      rowCount: rawRows.length,
      colCount: headers.length,
      headers,
      rawRows,
      stats
    };
  };

  // --- CSV Handlers ---
  const handleParseCsvText = (text: string, title = "Uploaded Dataset") => {
    try {
      const parsed = parseCSV(text);
      if (parsed.length === 0) {
        alert("The CSV file seems to be empty or invalid.");
        return;
      }
      const datasetObj = calculateDatasetStats(title, parsed);
      setDataset(datasetObj);
      
      // Auto-select last column as default target
      const lastCol = datasetObj.headers[datasetObj.headers.length - 1];
      setSelectedTarget(lastCol || "");
      setExcludedCols([]);
      setModelResults([]);
      setPlaygroundInputs({});
      setAiInsights("");
      setAiError("");
      setActiveTab("preview");
    } catch (e) {
      alert("Error parsing CSV. Please check formatting.");
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target?.result as string;
        handleParseCsvText(text, file.name);
      };
      reader.readAsText(file);
    }
  };

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    const file = e.dataTransfer.files?.[0];
    if (file && file.name.endsWith(".csv")) {
      const reader = new FileReader();
      reader.onload = (event) => {
        const text = event.target?.result as string;
        handleParseCsvText(text, file.name);
      };
      reader.readAsText(file);
    } else {
      alert("Please upload a valid .csv file.");
    }
  };

  const handleLoadSample = (sampleId: string) => {
    const sample = SAMPLE_DATASETS.find(d => d.id === sampleId);
    if (sample) {
      handleParseCsvText(sample.csvData, sample.name);
      setSelectedTarget(sample.targetCol);
      setProblemType(sample.problemType);
    }
  };

  const handleDownloadSampleCsv = () => {
    if (!dataset) return;
    // reconstruct csv text
    const headers = dataset.headers.join(",");
    const rows = dataset.rawRows.map(row => 
      dataset.headers.map(h => {
        const val = row[h];
        return typeof val === "string" && val.includes(",") ? `"${val}"` : val;
      }).join(",")
    ).join("\n");
    const fullCsv = `${headers}\n${rows}`;
    
    const blob = new Blob([fullCsv], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `${dataset.name.toLowerCase().replace(/\s+/g, "_")}_sample.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  // --- MODEL TRAINING ENGINE ---
  const handleTrainModels = async () => {
    if (!dataset || !selectedTarget) return;

    setIsTraining(true);
    setTrainingStep("Initializing preprocessing and encoding pipeline...");
    setModelResults([]);
    
    // Slight timeout to show beautiful state sequence
    await new Promise(resolve => setTimeout(resolve, 500));

    try {
      // 1. Preprocess & Normalize
      setTrainingStep("Performing matrix transformations, imputation, scaling, and category labeling...");
      const { X, y, context } = preprocessDataset(dataset.rawRows, selectedTarget, problemType, excludedCols);
      
      if (X.length < 5) {
        throw new Error("The dataset has too few rows after filtering to train. Minimum 5 rows needed.");
      }

      await new Promise(resolve => setTimeout(resolve, 500));

      // 2. Train-test split
      setTrainingStep(`Splitting indices (${Math.round(trainRatio * 100)}% train, ${Math.round((1 - trainRatio) * 100)}% test)...`);
      
      const numSamples = X.length;
      const indices = Array.from({ length: numSamples }, (_, i) => i);
      // shuffle indices
      for (let i = indices.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [indices[i], indices[j]] = [indices[j], indices[i]];
      }

      const trainSize = Math.floor(numSamples * trainRatio);
      const trainIdx = indices.slice(0, trainSize);
      const testIdx = indices.slice(trainSize);

      const X_train = trainIdx.map(i => X[i]);
      const y_train = trainIdx.map(i => y[i]);
      const X_test = testIdx.map(i => X[i]);
      const y_test = testIdx.map(i => y[i]);

      await new Promise(resolve => setTimeout(resolve, 400));

      const results: AlgorithmResult[] = [];

      if (problemType === "classification") {
        const numClasses = context.classes.length;

        // --- Model A: Logistic Regression ---
        setTrainingStep("Fitting Multi-Class Logistic Regression (Softmax with Gradient Descent)...");
        const t0_lr = performance.now();
        const lrArtifact = trainLogisticRegression(X_train, y_train, numClasses, 250, 0.08);
        const lrTime = performance.now() - t0_lr;
        const lrMetrics = evaluateClassifier(X_test, y_test, "logistic_regression", lrArtifact, context.classes);
        results.push({
          algorithmId: "logistic_regression",
          algorithmName: "Logistic Regression",
          metrics: lrMetrics,
          trainingTimeMs: Math.round(lrTime * 10) / 10,
          modelArtifact: lrArtifact
        });
        await new Promise(resolve => setTimeout(resolve, 200));

        // --- Model B: Gaussian Naive Bayes ---
        setTrainingStep("Estimating probability density vectors for Naive Bayes...");
        const t0_nb = performance.now();
        const nbArtifact = trainNaiveBayes(X_train, y_train, numClasses);
        const nbTime = performance.now() - t0_nb;
        const nbMetrics = evaluateClassifier(X_test, y_test, "naive_bayes", nbArtifact, context.classes);
        results.push({
          algorithmId: "naive_bayes",
          algorithmName: "Gaussian Naive Bayes",
          metrics: nbMetrics,
          trainingTimeMs: Math.round(nbTime * 10) / 10,
          modelArtifact: nbArtifact
        });
        await new Promise(resolve => setTimeout(resolve, 200));

        // --- Model C: KNN ---
        setTrainingStep("Structuring multi-dimensional distances in KNN...");
        const t0_knn = performance.now();
        const knnArtifact = trainKNNClassifier(X_train, y_train, 3);
        const knnTime = performance.now() - t0_knn;
        const knnMetrics = evaluateClassifier(X_test, y_test, "knn", knnArtifact, context.classes);
        results.push({
          algorithmId: "knn",
          algorithmName: "K-Nearest Neighbors (K=3)",
          metrics: knnMetrics,
          trainingTimeMs: Math.round(knnTime * 10) / 10,
          modelArtifact: knnArtifact
        });
        await new Promise(resolve => setTimeout(resolve, 200));

        // --- Model D: Decision Tree ---
        setTrainingStep("Partitioning recursive nodes for Gini impurity splits (Decision Tree)...");
        const t0_dt = performance.now();
        const dtArtifact = trainDecisionTreeClassification(X_train, y_train, 5);
        const dtTime = performance.now() - t0_dt;
        const dtMetrics = evaluateClassifier(X_test, y_test, "decision_tree", dtArtifact, context.classes);
        results.push({
          algorithmId: "decision_tree",
          algorithmName: "Decision Tree Classifier",
          metrics: dtMetrics,
          trainingTimeMs: Math.round(dtTime * 10) / 10,
          modelArtifact: dtArtifact
        });

      } else {
        // --- REGRESSION ---

        // --- Model A: Linear Regression ---
        setTrainingStep("Solving Least Squares optimization with gradient vectors...");
        const t0_lin = performance.now();
        const linArtifact = trainLinearRegression(X_train, y_train, 300, 0.05);
        const linTime = performance.now() - t0_lin;
        const linMetrics = evaluateRegressor(X_test, y_test, "linear_regression", linArtifact);
        results.push({
          algorithmId: "linear_regression",
          algorithmName: "Linear Regression (GD)",
          metrics: linMetrics,
          trainingTimeMs: Math.round(linTime * 10) / 10,
          modelArtifact: linArtifact
        });
        await new Promise(resolve => setTimeout(resolve, 200));

        // --- Model B: KNN Regressor ---
        setTrainingStep("Formulating nearest neighbors target weights (KNN Regression)...");
        const t0_knnr = performance.now();
        const knnrArtifact = trainKNNRegressor(X_train, y_train, 3);
        const knnrTime = performance.now() - t0_knnr;
        const knnrMetrics = evaluateRegressor(X_test, y_test, "knn", knnrArtifact);
        results.push({
          algorithmId: "knn",
          algorithmName: "K-Nearest Neighbors (K=3)",
          metrics: knnrMetrics,
          trainingTimeMs: Math.round(knnrTime * 10) / 10,
          modelArtifact: knnrArtifact
        });
        await new Promise(resolve => setTimeout(resolve, 200));

        // --- Model C: Decision Tree Regressor ---
        setTrainingStep("Growing tree nodes via recursive MSE reduction splitting...");
        const t0_dtr = performance.now();
        const dtrArtifact = trainDecisionTreeRegression(X_train, y_train, 5);
        const dtrTime = performance.now() - t0_dtr;
        const dtrMetrics = evaluateRegressor(X_test, y_test, "decision_tree", dtrArtifact);
        results.push({
          algorithmId: "decision_tree",
          algorithmName: "Decision Tree Regressor",
          metrics: dtrMetrics,
          trainingTimeMs: Math.round(dtrTime * 10) / 10,
          modelArtifact: dtrArtifact
        });
      }

      setTrainingStep("Synthesizing model evaluation parameters and accuracy artifacts...");
      await new Promise(resolve => setTimeout(resolve, 400));
      
      setModelResults(results);
      setIsTraining(false);
      setActiveTab("comparison");

      // Reset out of sample predicting context
      const defaultInputs: Record<string, string> = {};
      context.features.forEach(f => {
        const stat = dataset.stats.find(s => s.name === f);
        if (stat) {
          defaultInputs[f] = stat.type === "numeric" ? String(stat.mean || 0) : String(stat.mostFrequent || "");
        }
      });
      setPlaygroundInputs(defaultInputs);
      setAiInsights(""); // clear old AI insights so they load fresh
      setAiError("");

    } catch (e: any) {
      setIsTraining(false);
      alert(`Model training error: ${e.message}`);
    }
  };

  // --- OUT-OF-SAMPLE TEST FORM PLAYGROUND ---
  const handlePlaygroundValueChange = (feature: string, val: string) => {
    setPlaygroundInputs(prev => ({
      ...prev,
      [feature]: val
    }));
  };

  const getPlaygroundPredictions = (): Record<string, string | number> => {
    if (!dataset || modelResults.length === 0) return {};
    
    // reconstruct preprocess context
    const { context } = preprocessDataset(dataset.rawRows, selectedTarget, problemType, excludedCols);
    
    // Map string values back to appropriate numbers/strings
    const processedInput: Record<string, string | number> = {};
    context.features.forEach(f => {
      const rawVal = playgroundInputs[f];
      if (context.numericStats[f]) {
        processedInput[f] = Number(rawVal);
      } else {
        processedInput[f] = rawVal;
      }
    });

    const predictions: Record<string, string | number> = {};
    modelResults.forEach(res => {
      const pred = predictSingle(processedInput, context, res.algorithmId, res.modelArtifact);
      predictions[res.algorithmName] = pred;
    });

    return predictions;
  };

  const handleBatchTestCsv = () => {
    if (!dataset || !testCsvInput.trim() || modelResults.length === 0) return;
    try {
      const parsedRows = parseCSV(testCsvInput);
      if (parsedRows.length === 0) {
        alert("The pasted CSV contains no readable records.");
        return;
      }

      const { context } = preprocessDataset(dataset.rawRows, selectedTarget, problemType, excludedCols);
      
      const scored = parsedRows.map((row, idx) => {
        const predictions: Record<string, string | number> = {};
        modelResults.forEach(res => {
          predictions[res.algorithmName] = predictSingle(row, context, res.algorithmId, res.modelArtifact);
        });
        return {
          id: idx + 1,
          raw: row,
          predictions
        };
      });

      setBatchPredictions(scored);
    } catch (e: any) {
      alert(`Batch test parse error: ${e.message}`);
    }
  };

  // --- AI INSIGHTS GENERATION (GEMINI SERVER HANDLER) ---
  const handleGenerateAIReport = async () => {
    if (!dataset || modelResults.length === 0) return;
    
    setIsLoadingAI(true);
    setAiError("");
    setAiInsights("");

    try {
      const datasetSummary = {
        rowCount: dataset.rowCount,
        colCount: dataset.colCount,
        features: dataset.stats.map(s => ({
          name: s.name,
          type: s.type,
          missing: s.missingCount,
          uniques: s.uniqueCount
        }))
      };

      const metricsSummary = modelResults.map(m => ({
        algorithm: m.algorithmName,
        trainingTime: `${m.trainingTimeMs}ms`,
        scores: m.metrics
      }));

      const res = await fetch("/api/gemini/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          datasetSummary,
          modelResults: metricsSummary,
          problemType,
          selectedTarget
        })
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || "Internal Server Error calling Gemini API.");
      }

      const data = await res.json();
      setAiInsights(data.insights);
    } catch (e: any) {
      console.error(e);
      setAiError(e.message || "Could not generate AI insights at this time. Verify that the Gemini API Key is configured.");
    } finally {
      setIsLoadingAI(false);
    }
  };

  // --- ARTIFACT MODEL EXPORT DOWNLOAD ---
  const handleDownloadModelArtifact = (model: AlgorithmResult) => {
    if (!dataset) return;
    const { context } = preprocessDataset(dataset.rawRows, selectedTarget, problemType, excludedCols);
    
    const fullModelPayload = {
      modelName: model.algorithmName,
      algorithmId: model.algorithmId,
      problemType,
      targetCol: selectedTarget,
      metrics: model.metrics,
      trainingTimeMs: model.trainingTimeMs,
      preprocessContext: context,
      weightsOrNodes: model.modelArtifact,
      exportedAt: new Date().toISOString(),
      sourceDataset: dataset.name
    };

    const blob = new Blob([JSON.stringify(fullModelPayload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `${model.algorithmId}_model_deployment.json`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const activeTargetStat = dataset?.stats.find(s => s.name === selectedTarget);
  const bestModel = modelResults.length > 0 ? [...modelResults].sort((a, b) => {
    if (problemType === "classification") {
      // higher F1 or accuracy
      return ((b.metrics as any).f1Score || 0) - ((a.metrics as any).f1Score || 0);
    } else {
      // higher R2
      return ((b.metrics as any).r2 || 0) - ((a.metrics as any).r2 || 0);
    }
  })[0] : null;

  return (
    <div className="bg-slate-50 min-h-screen text-slate-900 pb-24 font-sans">
      {/* Workspace Header */}
      <header className="bg-white border-b border-slate-200 px-8 py-4 flex items-center justify-between sticky top-[73px] z-40 shadow-sm">
        <div className="flex items-center gap-3">
          <button 
            id="btn-nav-back"
            onClick={onGoBack}
            className="p-2 hover:bg-slate-100 rounded-lg text-slate-600 transition-colors cursor-pointer"
            title="Back to home"
          >
            <X className="w-5 h-5" />
          </button>
          <div className="h-6 w-px bg-slate-200" />
          <div className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-indigo-600" />
            <h2 className="text-lg font-bold tracking-tight text-slate-800 font-display">AutoML Workbench</h2>
          </div>
        </div>

        {dataset && (
          <div className="flex items-center gap-3">
            <span className="text-xs font-mono bg-indigo-50 text-indigo-700 px-3.5 py-1.5 rounded-xl font-semibold border border-indigo-100">
              📁 {dataset.name} ({dataset.rowCount} rows)
            </span>
            <button
              id="btn-header-csv-dl"
              onClick={handleDownloadSampleCsv}
              className="px-4 py-2 bg-white border border-slate-200 hover:bg-slate-50 text-slate-600 rounded-xl text-xs font-semibold cursor-pointer transition-all shadow-sm flex items-center gap-1.5"
            >
              <Download className="w-3.5 h-3.5 text-slate-400" />
              <span>Download Sample CSV</span>
            </button>
          </div>
        )}
      </header>

      {/* Main Body */}
      <div className="max-w-7xl mx-auto px-8 mt-8">
        
        {/* Step 1: No dataset uploaded */}
        {!dataset && (
          <div className="max-w-3xl mx-auto mt-10">
            <div className="bg-white rounded-2xl p-8 md:p-10 border border-slate-200 shadow-sm">
              <div className="text-center mb-8">
                <div className="w-16 h-16 rounded-2xl bg-indigo-50 flex items-center justify-center mx-auto mb-4">
                  <Upload className="w-8 h-8 text-indigo-600" />
                </div>
                <h3 className="text-2xl font-bold text-slate-850 tracking-tight font-display">Load Your Machine Learning Dataset</h3>
                <p className="text-slate-500 text-sm mt-2 max-w-lg mx-auto leading-relaxed">
                  Drag and drop a .csv file, upload from your file system, paste raw text, or run a simulated test with one of our high-quality sample datasets.
                </p>
              </div>

              {/* Upload Dropzone */}
              <div
                id="csv-dropzone"
                onDragEnter={handleDrag}
                onDragOver={handleDrag}
                onDragLeave={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`border-2 border-dashed rounded-2xl p-10 text-center cursor-pointer transition-all ${
                  dragActive 
                    ? "border-indigo-600 bg-indigo-50/40 scale-[0.99]" 
                    : "border-slate-200 hover:border-indigo-500 hover:bg-indigo-50/10 bg-slate-50/50"
                }`}
              >
                <Upload className="w-10 h-10 text-slate-400 mx-auto mb-3" />
                <p className="text-sm font-bold text-slate-750">Drag & Drop file here or Click to Browse</p>
                <p className="text-xs text-slate-400 mt-1">Supports standard Comma Separated Values (.csv)</p>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".csv"
                  onChange={handleFileUpload}
                  className="hidden"
                />
              </div>

              {/* Paste Text / Manual Entry */}
              <div className="mt-6 border-t border-slate-100 pt-6">
                <details className="group cursor-pointer">
                  <summary className="text-xs font-semibold text-slate-500 hover:text-indigo-600 flex items-center gap-1.5 select-none">
                    <ChevronRight className="w-4 h-4 group-open:rotate-90 transition-transform" />
                    <span>Or paste raw CSV text directly</span>
                  </summary>
                  <div className="mt-3 cursor-default">
                    <textarea
                      id="txt-csv-raw"
                      rows={6}
                      placeholder="sepal_length,sepal_width,petal_length,petal_width,species&#10;5.1,3.5,1.4,0.2,Setosa&#10;4.9,3.0,1.4,0.2,Setosa..."
                      value={csvInput}
                      onChange={(e) => setCsvInput(e.target.value)}
                      className="w-full text-xs font-mono p-4 bg-slate-50 border border-slate-200 rounded-xl focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                    />
                    <button
                      id="btn-parse-pasted"
                      onClick={() => handleParseCsvText(csvInput, "Pasted Dataset")}
                      disabled={!csvInput.trim()}
                      className="mt-3 px-5 py-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white text-xs font-bold rounded-xl shadow-lg shadow-indigo-150 transition-all cursor-pointer"
                    >
                      Process Raw Text
                    </button>
                  </div>
                </details>
              </div>

              {/* Try Samples */}
              <div className="mt-8 border-t border-slate-150 pt-8">
                <span className="text-[10px] font-bold text-slate-400 tracking-widest uppercase block mb-4">Or Quick Start with Pre-Built Scenarios</span>
                <div className="grid sm:grid-cols-3 gap-4">
                  {SAMPLE_DATASETS.map((sd) => (
                    <button
                      id={`btn-sample-${sd.id}`}
                      key={sd.id}
                      onClick={() => handleLoadSample(sd.id)}
                      className="bg-slate-50 hover:bg-indigo-50/20 hover:border-indigo-300 transition-all border border-slate-200 p-4 rounded-xl text-left cursor-pointer group"
                    >
                      <div className="flex items-center gap-2 mb-2">
                        <Database className="w-4 h-4 text-indigo-500" />
                        <span className="font-bold text-slate-850 text-xs truncate">{sd.name}</span>
                      </div>
                      <p className="text-slate-500 text-[11px] line-clamp-2 leading-relaxed">{sd.description}</p>
                      <div className="flex items-center gap-1 mt-3 text-[10px] font-semibold text-indigo-600 group-hover:translate-x-1 transition-transform">
                        <span>Load dataset</span>
                        <ChevronRight className="w-3 h-3" />
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Training Animation Overlay */}
        <AnimatePresence>
          {isTraining && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-slate-950/80 backdrop-blur-xs flex items-center justify-center z-50 p-6"
            >
              <div className="bg-white rounded-3xl p-8 max-w-md w-full shadow-2xl border border-slate-100 text-center">
                <RefreshCw className="w-12 h-12 text-indigo-600 animate-spin mx-auto mb-6" />
                <h4 className="text-lg font-bold text-slate-950 font-display">AutoML Pipeline Active</h4>
                <p className="text-slate-500 text-sm mt-1">Simulating target vector preprocessing, metric split calculations, and standard model coefficients estimation...</p>
                
                <div className="bg-slate-50 rounded-xl p-4 mt-6 text-left border border-slate-100">
                  <span className="text-[10px] font-mono text-indigo-600 font-bold tracking-widest block uppercase">CURRENT STAGE</span>
                  <p className="text-xs font-mono text-slate-700 mt-1 font-medium">{trainingStep}</p>
                </div>

                <div className="w-full bg-slate-100 rounded-full h-1.5 mt-6 overflow-hidden">
                  <div className="bg-gradient-to-r from-indigo-500 to-indigo-600 h-full w-4/5 rounded-full animate-pulse" />
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Step 2: Dataset loaded Workspace */}
        {dataset && (
          <div className="grid lg:grid-cols-4 gap-8">
            
            {/* Sidebar Controls (1 column) */}
            <div className="lg:col-span-1 flex flex-col gap-6">
              <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm">
                <div className="flex items-center gap-2 mb-6 border-b border-slate-100 pb-4">
                  <Settings className="w-4 h-4 text-slate-400" />
                  <h4 className="font-bold text-slate-800 text-sm font-display">AutoML Settings</h4>
                </div>

                {/* Target Selection */}
                <div className="mb-6">
                  <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1.5">Target Variable (Y)</label>
                  <select
                    id="select-target-col"
                    value={selectedTarget}
                    onChange={(e) => {
                      setSelectedTarget(e.target.value);
                      setModelResults([]);
                    }}
                    className="w-full bg-slate-50 text-slate-800 text-xs border border-slate-200 rounded-lg px-3 py-2.5 font-medium outline-none focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
                  >
                    {dataset.headers.map(h => (
                      <option key={h} value={h}>{h}</option>
                    ))}
                  </select>
                  <p className="text-[10px] text-slate-400 mt-1.5 leading-normal">This is the label column that our models will aim to predict.</p>
                </div>

                {/* Problem Type Inference */}
                <div className="mb-6">
                  <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-1.5">Problem Architecture</label>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      id="btn-problem-classification"
                      onClick={() => {
                        setProblemType("classification");
                        setModelResults([]);
                      }}
                      className={`px-3 py-2 text-xs font-bold rounded-lg border text-center cursor-pointer transition-all ${
                        problemType === "classification" 
                          ? "bg-indigo-50 text-indigo-600 border-indigo-200" 
                          : "bg-white text-slate-600 border-slate-200 hover:bg-slate-50"
                      }`}
                    >
                      Classification
                    </button>
                    <button
                      id="btn-problem-regression"
                      onClick={() => {
                        setProblemType("regression");
                        setModelResults([]);
                      }}
                      className={`px-3 py-2 text-xs font-bold rounded-lg border text-center cursor-pointer transition-all ${
                        problemType === "regression" 
                          ? "bg-indigo-50 text-indigo-600 border-indigo-200" 
                          : "bg-white text-slate-600 border-slate-200 hover:bg-slate-50"
                      }`}
                    >
                      Regression
                    </button>
                  </div>
                </div>

                {/* Train Test Split Slider */}
                <div className="mb-6">
                  <div className="flex justify-between items-center text-[10px] font-bold text-slate-400 uppercase tracking-wider mb-2">
                    <span>Train-Test Split</span>
                    <span className="text-indigo-600 font-bold">{Math.round(trainRatio * 100)}% / {Math.round((1 - trainRatio) * 100)}%</span>
                  </div>
                  <input
                    type="range"
                    min="0.5"
                    max="0.9"
                    step="0.05"
                    value={trainRatio}
                    onChange={(e) => {
                      setTrainRatio(parseFloat(e.target.value));
                      setModelResults([]);
                    }}
                    className="w-full h-1 bg-slate-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                  />
                  <div className="flex justify-between text-[9px] text-slate-400 mt-1 font-mono">
                    <span>More Test</span>
                    <span>More Train</span>
                  </div>
                </div>

                {/* Impute & Scaling Toggles */}
                <div className="mb-6 bg-slate-50 rounded-xl p-3.5 border border-slate-100">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs font-bold text-slate-700">Standardize Scaling</span>
                    <input
                      type="checkbox"
                      checked={scaling}
                      onChange={(e) => {
                        setScaling(e.target.checked);
                        setModelResults([]);
                      }}
                      className="w-4 h-4 text-indigo-600 border-slate-300 rounded-sm focus:ring-indigo-500 accent-indigo-600"
                    />
                  </div>
                  <div className="text-[10px] text-slate-400 leading-normal">
                    Apply Z-score standardization on numeric inputs to guarantee stable convergence rates.
                  </div>
                </div>

                {/* Exclude Features List */}
                <div className="mb-6 border-t border-slate-150 pt-4">
                  <label className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block mb-2.5">Include / Exclude Features</label>
                  <div className="max-h-48 overflow-y-auto pr-1 flex flex-col gap-1.5">
                    {dataset.headers.filter(h => h !== selectedTarget).map(f => {
                      const isExcluded = excludedCols.includes(f);
                      return (
                        <div key={f} className="flex items-center justify-between text-xs p-1.5 hover:bg-slate-50 rounded-lg">
                          <span className={`truncate font-mono ${isExcluded ? "text-slate-400 line-through" : "text-slate-750 font-semibold"}`}>
                            {f}
                          </span>
                          <button
                            id={`btn-toggle-feature-${f}`}
                            onClick={() => {
                              if (isExcluded) {
                                setExcludedCols(prev => prev.filter(col => col !== f));
                              } else {
                                setExcludedCols(prev => [...prev, f]);
                              }
                              setModelResults([]); // reset results
                            }}
                            className={`px-2 py-1 rounded text-[9px] font-bold cursor-pointer transition-colors ${
                              isExcluded ? "bg-slate-100 text-slate-400 hover:bg-slate-200" : "bg-indigo-50 text-indigo-600 hover:bg-indigo-100"
                            }`}
                          >
                            {isExcluded ? "Exclude" : "Include"}
                          </button>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Train Button */}
                <button
                  id="btn-train-pipeline"
                  onClick={handleTrainModels}
                  className="w-full py-3.5 bg-indigo-600 hover:bg-indigo-700 text-white font-bold rounded-xl shadow-lg shadow-indigo-150 hover:shadow-indigo-200 transition-all cursor-pointer flex items-center justify-center gap-2"
                >
                  <Play className="w-4 h-4 fill-white" />
                  <span>Execute AutoML Engine</span>
                </button>
              </div>

              {/* Reset Data Button */}
              <button
                id="btn-workspace-reset"
                onClick={() => {
                  setDataset(null);
                  setModelResults([]);
                }}
                className="w-full py-3.5 bg-white border border-slate-200 hover:bg-slate-50 hover:text-red-600 text-slate-500 text-xs font-bold rounded-xl cursor-pointer transition-all shadow-sm flex items-center justify-center gap-1.5"
              >
                <X className="w-4 h-4" />
                <span>Discard & Load Another CSV</span>
              </button>
            </div>

            {/* Core Tabs & Results Visualization Panel (3 columns) */}
            <div className="lg:col-span-3 flex flex-col gap-6">
              
              {/* Tabs Navigation */}
              <div className="bg-white border border-slate-200 rounded-2xl p-2 shadow-sm flex flex-wrap items-center gap-1.5">
                <button
                  id="tab-preview"
                  onClick={() => setActiveTab("preview")}
                  className={`px-4.5 py-2.5 rounded-xl text-xs font-bold flex items-center gap-2 cursor-pointer transition-all ${
                    activeTab === "preview" 
                      ? "bg-indigo-600 text-white shadow-md shadow-indigo-100" 
                      : "text-slate-500 hover:text-slate-800 hover:bg-slate-50"
                  }`}
                >
                  <Table className="w-4 h-4" />
                  <span>1. Preview & Statistics</span>
                </button>

                <button
                  id="tab-comparison"
                  onClick={() => {
                    if (modelResults.length === 0) {
                      alert("Execute AutoML training first to view comparison matrices.");
                      return;
                    }
                    setActiveTab("comparison");
                  }}
                  className={`px-4.5 py-2.5 rounded-xl text-xs font-bold flex items-center gap-2 cursor-pointer transition-all ${
                    modelResults.length === 0 ? "opacity-45 cursor-not-allowed" : ""
                  } ${
                    activeTab === "comparison" 
                      ? "bg-indigo-600 text-white shadow-md shadow-indigo-100" 
                      : "text-slate-500 hover:text-slate-800 hover:bg-slate-50"
                  }`}
                >
                  <BarChart3 className="w-4 h-4" />
                  <span>2. Comparison Dashboard</span>
                </button>

                <button
                  id="tab-playground"
                  onClick={() => {
                    if (modelResults.length === 0) {
                      alert("Execute AutoML training first to open out-of-sample predictor.");
                      return;
                    }
                    setActiveTab("playground");
                  }}
                  className={`px-4.5 py-2.5 rounded-xl text-xs font-bold flex items-center gap-2 cursor-pointer transition-all ${
                    modelResults.length === 0 ? "opacity-45 cursor-not-allowed" : ""
                  } ${
                    activeTab === "playground" 
                      ? "bg-indigo-600 text-white shadow-md shadow-indigo-100" 
                      : "text-slate-500 hover:text-slate-800 hover:bg-slate-50"
                  }`}
                >
                  <Binary className="w-4 h-4" />
                  <span>3. Prediction Playground</span>
                </button>

                <button
                  id="tab-ai-insights"
                  onClick={() => {
                    if (modelResults.length === 0) {
                      alert("Execute AutoML training first to fetch AI analysis.");
                      return;
                    }
                    setActiveTab("ai_insights");
                    if (!aiInsights) {
                      handleGenerateAIReport();
                    }
                  }}
                  className={`px-4.5 py-2.5 rounded-xl text-xs font-bold flex items-center gap-2 cursor-pointer transition-all ${
                    modelResults.length === 0 ? "opacity-45 cursor-not-allowed" : ""
                  } ${
                    activeTab === "ai_insights" 
                      ? "bg-indigo-600 text-white shadow-md shadow-indigo-100" 
                      : "text-slate-500 hover:text-slate-800 hover:bg-slate-50"
                  }`}
                >
                  <Sparkles className="w-4 h-4" />
                  <span>4. AI Insights Copilot</span>
                </button>
              </div>

              {/* Tab Contents */}
              <div className="bg-white border border-slate-200 rounded-2xl p-8 shadow-sm min-h-[520px]">
                
                {/* TAB 1: PREVIEW & COLUMN STATS */}
                {activeTab === "preview" && (
                  <div>
                    <div className="mb-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-slate-100 pb-5">
                      <div>
                        <h3 className="text-lg font-bold font-display text-slate-800">Dataset Preview</h3>
                        <p className="text-slate-500 text-xs mt-0.5">Explore parsed records and calculated summary stats below.</p>
                      </div>
                      <div className="text-xs text-slate-500 bg-slate-50 border border-slate-200 px-3.5 py-2 rounded-xl flex items-center gap-2">
                        <Layers className="w-4 h-4 text-indigo-600" />
                        <span>Problem suggestion: <strong className="text-indigo-700 capitalize">{problemType}</strong></span>
                      </div>
                    </div>

                    {/* Quick Stats Grid */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8">
                      <div className="bg-slate-50 p-5 rounded-2xl border border-slate-200 shadow-sm">
                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Total Rows</span>
                        <span className="text-2xl font-extrabold text-slate-800">{dataset.rowCount}</span>
                      </div>
                      <div className="bg-slate-50 p-5 rounded-2xl border border-slate-200 shadow-sm">
                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Total Features</span>
                        <span className="text-2xl font-extrabold text-slate-800">{dataset.colCount}</span>
                      </div>
                      <div className="bg-slate-50 p-5 rounded-2xl border border-slate-200 shadow-sm">
                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Target Selection</span>
                        <span className="text-xs font-mono font-bold bg-indigo-50 text-indigo-700 px-2 py-1 rounded-lg border border-indigo-100 inline-block truncate max-w-full mt-1.5">
                          {selectedTarget}
                        </span>
                      </div>
                      <div className="bg-slate-50 p-5 rounded-2xl border border-slate-200 shadow-sm">
                        <span className="text-[10px] font-bold text-slate-400 uppercase tracking-wider block">Excluded Columns</span>
                        <span className="text-xs font-mono font-bold text-slate-500 inline-block mt-2">
                          {excludedCols.length === 0 ? "None" : `${excludedCols.length} columns`}
                        </span>
                      </div>
                    </div>

                    {/* Table Preview */}
                    <div className="mb-8">
                      <h4 className="text-[10px] font-bold text-slate-400 tracking-wider uppercase block mb-3">Data Preview (First 5 Rows)</h4>
                      <div className="overflow-x-auto border border-slate-200 rounded-xl shadow-xs">
                        <table className="w-full text-left text-xs border-collapse bg-white">
                          <thead className="bg-slate-50 text-slate-600 font-bold border-b border-slate-200">
                            <tr>
                              <th className="p-3.5 w-12 text-center text-[10px] text-slate-400 font-mono border-r border-slate-200">Row</th>
                              {dataset.headers.map(h => (
                                <th key={h} className={`p-3.5 font-mono ${h === selectedTarget ? "text-indigo-600 bg-indigo-50/40 font-bold" : "text-slate-500"}`}>
                                  {h}
                                </th>
                              ))}
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-100">
                            {dataset.rawRows.slice(0, 5).map((row, idx) => (
                              <tr key={idx} className="hover:bg-slate-50/80 transition-colors">
                                <td className="p-3.5 text-center font-mono text-[10px] text-slate-400 bg-slate-50 border-r border-slate-200">{idx + 1}</td>
                                {dataset.headers.map(h => (
                                  <td key={h} className={`p-3.5 font-mono text-[11px] ${h === selectedTarget ? "bg-indigo-50/10 font-bold text-indigo-700" : "text-slate-600"}`}>
                                    {row[h]}
                                  </td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    {/* Statistics list */}
                    <div>
                      <h4 className="text-[10px] font-bold text-slate-400 tracking-wider uppercase block mb-3">Feature Statistics & Profiles</h4>
                      <div className="grid md:grid-cols-2 gap-4">
                        {dataset.stats.map(stat => {
                          const isTarget = stat.name === selectedTarget;
                          const isExcluded = excludedCols.includes(stat.name);
                          return (
                            <div 
                              key={stat.name} 
                              className={`p-5 rounded-2xl border transition-all ${
                                isTarget 
                                  ? "border-indigo-200 bg-indigo-50/20 shadow-xs" 
                                  : isExcluded 
                                    ? "border-slate-100 bg-slate-50/40 opacity-70" 
                                    : "border-slate-200 bg-white hover:border-slate-350 shadow-sm"
                              }`}
                            >
                              <div className="flex items-center justify-between gap-2 mb-3">
                                <div className="flex items-center gap-1.5 truncate">
                                  <span className={`font-bold font-mono text-xs ${isTarget ? "text-indigo-700" : "text-slate-800"}`}>
                                    {stat.name}
                                  </span>
                                  {isTarget && <span className="text-[8px] bg-indigo-600 text-white font-black px-1.5 py-0.5 rounded-md uppercase">Target</span>}
                                  {isExcluded && <span className="text-[8px] bg-slate-200 text-slate-500 font-bold px-1.5 py-0.5 rounded-md uppercase">Excluded</span>}
                                </div>
                                <span className="text-[10px] font-mono text-slate-400 bg-slate-100 px-2 py-0.5 rounded-md">
                                  {stat.type}
                                </span>
                              </div>

                              <div className="grid grid-cols-2 gap-2 text-[11px] text-slate-500">
                                <div>Missing values: <span className="font-mono text-slate-700 font-bold">{stat.missingCount}</span></div>
                                <div>Unique values: <span className="font-mono text-slate-700 font-bold">{stat.uniqueCount}</span></div>
                                {stat.type === "numeric" ? (
                                  <>
                                    <div>Mean: <span className="font-mono text-slate-700 font-bold">{stat.mean?.toFixed(2)}</span></div>
                                    <div>Range: <span className="font-mono text-slate-700 font-bold">[{stat.min}, {stat.max}]</span></div>
                                  </>
                                ) : (
                                  <div className="col-span-2 truncate">Most Frequent: <span className="font-mono text-slate-700 font-bold">"{stat.mostFrequent}"</span></div>
                                )}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  </div>
                )}

                {/* TAB 2: COMPARISON DASHBOARD */}
                {activeTab === "comparison" && modelResults.length > 0 && (
                  <div>
                    <div className="mb-6 flex flex-col md:flex-row items-start md:items-center justify-between gap-4 border-b border-slate-100 pb-5">
                      <div>
                        <h3 className="text-lg font-bold font-display text-slate-800">Model Evaluation & Comparison</h3>
                        <p className="text-slate-500 text-xs mt-0.5">Explore the validation outputs of multiple machine learning models trained on your data.</p>
                      </div>
                      
                      {bestModel && (
                        <div className="bg-emerald-50 border border-emerald-200 px-4 py-2.5 rounded-xl text-xs text-emerald-850 flex items-center gap-2 shadow-sm">
                          <Award className="w-5 h-5 text-emerald-600 shrink-0" />
                          <div>
                            Best Algorithm: <strong className="text-emerald-950 font-bold">{bestModel.algorithmName}</strong>
                            <span className="block text-[10px] text-emerald-600 mt-0.5">
                              {problemType === "classification" 
                                ? `Accuracy: ${Math.round((bestModel.metrics as any).accuracy * 1000) / 10}% | F1-Score: ${Math.round((bestModel.metrics as any).f1Score * 1000) / 10}%`
                                : `R²: ${Math.round((bestModel.metrics as any).r2 * 1000) / 1000} | RMSE: ${Math.round((bestModel.metrics as any).rmse * 100) / 100}`
                              }
                            </span>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* Visual Comparison Charts */}
                    <div className="grid md:grid-cols-2 gap-8 mb-8">
                      {/* Chart Panel */}
                      <div className="bg-slate-50 border border-slate-200 p-5 rounded-2xl shadow-sm">
                        <h4 className="text-[10px] font-bold text-slate-400 tracking-wider uppercase block mb-4">
                          {problemType === "classification" ? "Accuracy & F1 Score Comparison" : "R² Coeff Comparison (higher is better)"}
                        </h4>
                        
                        <div className="h-64">
                          <ResponsiveContainer width="100%" height="100%">
                            <BarChart
                              data={modelResults.map(res => ({
                                name: res.algorithmName,
                                score: problemType === "classification" 
                                  ? parseFloat((res.metrics as ClassificationMetrics).f1Score.toFixed(3))
                                  : parseFloat((res.metrics as RegressionMetrics).r2.toFixed(3)),
                                acc: problemType === "classification"
                                  ? parseFloat((res.metrics as ClassificationMetrics).accuracy.toFixed(3))
                                  : 0
                              }))}
                              margin={{ top: 20, right: 30, left: 0, bottom: 5 }}
                            >
                              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                              <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} tickLine={false} />
                              <YAxis domain={[0, 1]} stroke="#94a3b8" fontSize={11} tickLine={false} />
                              <Tooltip />
                              <Legend />
                              <Bar dataKey="score" name={problemType === "classification" ? "F1-Score" : "R² Variance"} fill="#4f46e5" radius={[4, 4, 0, 0]} />
                              {problemType === "classification" && (
                                <Bar dataKey="acc" name="Accuracy" fill="#10b981" radius={[4, 4, 0, 0]} />
                              )}
                            </BarChart>
                          </ResponsiveContainer>
                        </div>
                      </div>

                      {/* Side by side stats grid */}
                      <div className="flex flex-col gap-4 justify-between">
                        <div className="bg-slate-50 border border-slate-200 p-5 rounded-2xl flex-1 shadow-sm">
                          <h4 className="text-[10px] font-bold text-slate-400 tracking-wider uppercase block mb-4">
                            Training & Execution Statistics
                          </h4>
                          <div className="flex flex-col gap-3.5">
                            {modelResults.map(res => (
                              <div key={res.algorithmId} className="flex items-center justify-between border-b border-slate-200 pb-2 text-xs">
                                <span className="font-bold text-slate-800">{res.algorithmName}</span>
                                <div className="text-right font-mono text-slate-600 font-bold">
                                  <div>⏱️ {res.trainingTimeMs}ms</div>
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Note explaining metrics */}
                        <div className="bg-indigo-50/50 border border-indigo-100 p-4.5 rounded-xl text-xs text-indigo-900 leading-relaxed shadow-xs">
                          <h5 className="font-bold flex items-center gap-1.5 mb-1.5 text-indigo-950">
                            <HelpCircle className="w-4 h-4 text-indigo-600 shrink-0" />
                            <span>Quick Metric Guide</span>
                          </h5>
                          {problemType === "classification" ? (
                            <span>
                              <strong>Accuracy</strong> is the overall share of correct predictions. <strong>F1-Score</strong> is the harmonic mean of Precision and Recall—ideal for diagnosing imbalanced targets.
                            </span>
                          ) : (
                            <span>
                              <strong>R-Squared ($R^2$)</strong> explains the share of target variance captured by inputs. <strong>RMSE</strong> represents the standard deviation of residuals.
                            </span>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Side-by-side Table */}
                    <div className="mb-8">
                      <h4 className="text-[10px] font-bold text-slate-400 tracking-wider uppercase block mb-3">Model Performance Details</h4>
                      <div className="overflow-x-auto border border-slate-200 rounded-xl shadow-xs">
                        <table className="w-full text-left text-xs border-collapse bg-white">
                          <thead className="bg-slate-50 text-slate-600 font-bold border-b border-slate-200">
                            <tr>
                              <th className="p-3.5 border-r border-slate-100">Algorithm Name</th>
                              {problemType === "classification" ? (
                                <>
                                  <th className="p-3.5">Accuracy</th>
                                  <th className="p-3.5">Precision</th>
                                  <th className="p-3.5">Recall</th>
                                  <th className="p-3.5">F1-Score</th>
                                </>
                              ) : (
                                <>
                                  <th className="p-3.5">R² Score</th>
                                  <th className="p-3.5">RMSE</th>
                                  <th className="p-3.5">MAE</th>
                                  <th className="p-3.5">MSE</th>
                                </>
                              )}
                              <th className="p-3.5 text-center border-l border-slate-100">Export Artifact</th>
                            </tr>
                          </thead>
                          <tbody className="divide-y divide-slate-100 text-slate-700">
                            {modelResults.map(res => {
                              const isClass = problemType === "classification";
                              const clM = res.metrics as ClassificationMetrics;
                              const rgM = res.metrics as RegressionMetrics;
                              return (
                                <tr key={res.algorithmId} className="hover:bg-slate-50 transition-colors">
                                  <td className="p-3.5 font-bold text-slate-800 border-r border-slate-100">{res.algorithmName}</td>
                                  {isClass ? (
                                    <>
                                      <td className="p-3.5 font-mono font-medium">{clM.accuracy.toFixed(4)}</td>
                                      <td className="p-3.5 font-mono font-medium">{clM.precision.toFixed(4)}</td>
                                      <td className="p-3.5 font-mono font-medium">{clM.recall.toFixed(4)}</td>
                                      <td className="p-3.5 font-mono text-indigo-600 font-extrabold">{clM.f1Score.toFixed(4)}</td>
                                    </>
                                  ) : (
                                    <>
                                      <td className="p-3.5 font-mono text-indigo-600 font-extrabold">{rgM.r2.toFixed(4)}</td>
                                      <td className="p-3.5 font-mono font-medium">{rgM.rmse.toFixed(4)}</td>
                                      <td className="p-3.5 font-mono font-medium">{rgM.mae.toFixed(4)}</td>
                                      <td className="p-3.5 font-mono font-medium">{rgM.mse.toFixed(4)}</td>
                                    </>
                                  )}
                                  <td className="p-3.5 text-center border-l border-slate-100">
                                    <button
                                      id={`btn-export-${res.algorithmId}`}
                                      onClick={() => handleDownloadModelArtifact(res)}
                                      className="px-3.5 py-1.5 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 font-bold rounded-xl text-[10px] cursor-pointer transition-all inline-flex items-center gap-1 border border-indigo-100/40 shadow-xs"
                                    >
                                      <Download className="w-3.5 h-3.5" />
                                      <span>Download weights.json</span>
                                    </button>
                                  </td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    {/* Problem Specific Evaluation Accents */}
                    {problemType === "classification" && (
                      <div className="grid md:grid-cols-2 gap-8">
                        {modelResults.map(res => {
                          const clM = res.metrics as ClassificationMetrics;
                          if (!clM.confusionMatrix) return null;
                          return (
                            <div key={res.algorithmId} className="bg-slate-50 border border-slate-200 p-5 rounded-2xl shadow-sm">
                              <h5 className="font-bold text-xs text-slate-800 mb-3 font-display">{res.algorithmName} Confusion Matrix</h5>
                              <div className="grid grid-cols-3 gap-1.5 text-center font-mono text-xs">
                                <div className="bg-slate-200/60 p-2 font-sans font-bold text-slate-500 rounded-md">Act \ Pred</div>
                                {clM.classes.map(cls => (
                                  <div key={cls} className="bg-slate-200/40 p-2 font-sans font-bold text-slate-600 rounded-md truncate">{cls}</div>
                                ))}
                                
                                {clM.classes.map((rowCls, rIdx) => (
                                  <React.Fragment key={rowCls}>
                                    <div className="bg-slate-200/40 p-2 font-sans font-bold text-slate-600 rounded-md truncate">{rowCls}</div>
                                    {clM.classes.map((colCls, cIdx) => {
                                      const val = clM.confusionMatrix[rIdx]?.[cIdx] ?? 0;
                                      const isCorrect = rIdx === cIdx;
                                      return (
                                        <div 
                                          key={colCls} 
                                          className={`p-2 rounded-md font-extrabold ${
                                            isCorrect && val > 0 ? "bg-emerald-50 border border-emerald-200 text-emerald-800 shadow-xs" : "bg-white border border-slate-150 text-slate-400"
                                          }`}
                                        >
                                          {val}
                                        </div>
                                      );
                                    })}
                                  </React.Fragment>
                                ))}
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}

                {/* TAB 3: PLAYGROUND - OUT OF SAMPLE PREDICTOR */}
                {activeTab === "playground" && modelResults.length > 0 && (
                  <div>
                    <div className="mb-6 border-b border-slate-100 pb-5">
                      <h3 className="text-lg font-bold font-display text-slate-800">Out-Of-Sample Playground</h3>
                      <p className="text-slate-500 text-xs mt-0.5">Test the accuracies of your trained algorithms on custom values or out-of-sample datasets by yourself!</p>
                    </div>

                    <div className="grid md:grid-cols-2 gap-8">
                      {/* Column 1: Manual Feature Inputs Form */}
                      <div>
                        <div className="bg-slate-50 border border-slate-200 p-6 rounded-2xl mb-6 shadow-sm">
                          <h4 className="text-xs font-bold text-indigo-700 tracking-wider uppercase block mb-4 flex items-center gap-1.5 font-display">
                            <Binary className="w-4 h-4" />
                            <span>Manual Feature Inputs</span>
                          </h4>

                          <div className="flex flex-col gap-4">
                            {dataset.stats.filter(s => s.name !== selectedTarget && !excludedCols.includes(s.name)).map(f => {
                              const value = playgroundInputs[f] || "";
                              return (
                                <div key={f} className="flex flex-col gap-1.5 text-xs">
                                  <label className="font-bold text-slate-700 font-mono truncate">{f}</label>
                                  {f === selectedTarget ? null : (
                                    dataset.stats.find(s => s.name === f)?.type === "numeric" ? (
                                      <input
                                        type="number"
                                        step="any"
                                        value={value}
                                        onChange={(e) => handlePlaygroundValueChange(f, e.target.value)}
                                        placeholder="0.0"
                                        className="bg-white text-slate-800 text-xs border border-slate-200 rounded-xl px-3 py-2.5 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                                      />
                                    ) : (
                                      <select
                                        value={value}
                                        onChange={(e) => handlePlaygroundValueChange(f, e.target.value)}
                                        className="bg-white text-slate-800 text-xs border border-slate-200 rounded-xl px-3 py-2.5 focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500"
                                      >
                                        {(dataset.stats.find(s => s.name === f)?.mostFrequent !== undefined) && (
                                          <option value={dataset.stats.find(s => s.name === f)?.mostFrequent}>
                                            {dataset.stats.find(s => s.name === f)?.mostFrequent} (Default)
                                          </option>
                                        )}
                                        {/* Fallback to raw unique options */}
                                        {Array.from(new Set(dataset.rawRows.map(row => String(row[f] ?? "")))).slice(0, 8).map(opt => (
                                          <option key={opt} value={opt}>{opt}</option>
                                        ))}
                                      </select>
                                    )
                                  )}
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      </div>

                      {/* Column 2: Live Predictions Display */}
                      <div className="flex flex-col gap-6">
                        <div className="bg-gradient-to-br from-indigo-950 to-slate-900 text-white p-6 rounded-2xl border border-indigo-900/30 shadow-lg shadow-indigo-150">
                          <h4 className="text-[10px] font-bold text-indigo-300 tracking-wider uppercase block mb-4">
                            Live Trained Model Predictions
                          </h4>

                          <div className="flex flex-col gap-4">
                            {Object.entries(getPlaygroundPredictions()).map(([algoName, prediction]) => (
                              <div key={algoName} className="flex items-center justify-between border-b border-indigo-900/30 pb-3 last:border-0 last:pb-0">
                                <span className="text-xs text-indigo-200 font-bold">{algoName}</span>
                                <span className="font-mono font-black text-sm text-emerald-400 bg-emerald-950/40 border border-emerald-900/30 px-3 py-1 rounded-lg">
                                  {typeof prediction === "number" ? prediction.toFixed(4) : prediction}
                                </span>
                              </div>
                            ))}
                          </div>
                        </div>

                        {/* Batch Test Sub-section */}
                        <div className="bg-slate-50 border border-slate-200 p-6 rounded-2xl shadow-sm">
                          <h4 className="text-xs font-bold text-slate-600 tracking-wider uppercase block mb-2 flex items-center gap-1.5 font-display">
                            <Table className="w-4 h-4 text-slate-400" />
                            <span>Out-Of-Sample Batch Testing</span>
                          </h4>
                          <p className="text-slate-500 text-[11px] leading-relaxed mb-3">
                            Paste additional test rows in CSV format (including headers) to calculate batch predictions simultaneously.
                          </p>

                          <textarea
                            id="txt-test-batch"
                            rows={3}
                            placeholder={`${dataset.headers.filter(h => h !== selectedTarget).join(",")}\n...`}
                            value={testCsvInput}
                            onChange={(e) => setTestCsvInput(e.target.value)}
                            className="w-full text-[10px] font-mono p-3 bg-white border border-slate-200 rounded-lg focus:border-indigo-500 focus:outline-none focus:ring-1 focus:ring-indigo-500 mb-3"
                          />

                          <button
                            id="btn-evaluate-batch"
                            onClick={handleBatchTestCsv}
                            disabled={!testCsvInput.trim()}
                            className="px-4 py-2.5 bg-indigo-600 hover:bg-indigo-700 disabled:opacity-40 text-white text-xs font-bold rounded-xl cursor-pointer transition-all shadow-md shadow-indigo-100"
                          >
                            Calculate Batch Predictions
                          </button>

                          {batchPredictions && (
                            <div className="mt-4 max-h-48 overflow-y-auto border border-slate-200 rounded-lg shadow-xs">
                              <table className="w-full text-left text-[10px] border-collapse bg-white">
                                <thead className="bg-slate-100 text-slate-600 font-bold border-b border-slate-200 sticky top-0">
                                  <tr>
                                    <th className="p-2 w-8 text-center border-r border-slate-200">Row</th>
                                    <th className="p-2">Model Outputs</th>
                                  </tr>
                                </thead>
                                <tbody className="divide-y divide-slate-100 text-slate-750 bg-white">
                                  {batchPredictions.map((row) => (
                                    <tr key={row.id}>
                                      <td className="p-2 text-center text-slate-400 font-mono bg-slate-50 border-r border-slate-200">{row.id}</td>
                                      <td className="p-2">
                                        <div className="flex flex-col gap-1 font-mono">
                                          {Object.entries(row.predictions).map(([name, pred]: any) => (
                                            <div key={name} className="flex justify-between border-b border-slate-100 pb-0.5 last:border-0 last:pb-0">
                                              <span className="text-slate-400 font-medium">{name}:</span>
                                              <span className="font-bold text-indigo-700">{typeof pred === "number" ? pred.toFixed(2) : pred}</span>
                                            </div>
                                          ))}
                                        </div>
                                      </td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>
                )}

                {/* TAB 4: GEMINI AI INSIGHTS */}
                {activeTab === "ai_insights" && (
                  <div>
                    <div className="mb-6 flex items-center justify-between border-b border-slate-100 pb-5">
                      <div>
                        <h3 className="text-lg font-bold font-display text-slate-800 flex items-center gap-2">
                          <Sparkles className="w-5 h-5 text-indigo-500 animate-pulse" />
                          <span>AI Data Science Copilot Report</span>
                        </h3>
                        <p className="text-slate-500 text-xs mt-0.5">Gemini 3.5 Flash evaluates your dataset quality, explains model discrepancies, and suggests optimal hyperparameters.</p>
                      </div>

                      <button
                        id="btn-refresh-ai-report"
                        onClick={handleGenerateAIReport}
                        disabled={isLoadingAI}
                        className="px-4 py-2 bg-indigo-50 hover:bg-indigo-100 text-indigo-700 rounded-xl text-xs font-bold cursor-pointer transition-colors flex items-center gap-1.5 shadow-xs border border-indigo-100/40"
                      >
                        <RefreshCw className={`w-3.5 h-3.5 ${isLoadingAI ? "animate-spin" : ""}`} />
                        <span>Regenerate Report</span>
                      </button>
                    </div>

                    {isLoadingAI && (
                      <div className="py-20 text-center flex flex-col items-center justify-center">
                        <RefreshCw className="w-10 h-10 text-indigo-500 animate-spin mb-4" />
                        <p className="text-slate-700 text-sm font-bold">Generating AutoML Diagnostic & Optimizations Report...</p>
                        <p className="text-slate-400 text-xs mt-1">This takes about 5 seconds utilizing Gemini 3.5 Flash.</p>
                      </div>
                    )}

                    {aiError && (
                      <div className="bg-red-50 border border-red-200 text-red-800 p-5 rounded-2xl flex items-start gap-3">
                        <AlertCircle className="w-5 h-5 text-red-600 shrink-0 mt-0.5" />
                        <div>
                          <h4 className="font-bold text-sm">AI Insights Unavailable</h4>
                          <p className="text-xs text-red-700 mt-1">{aiError}</p>
                          <div className="mt-3 text-[11px] text-red-650 bg-white p-3 rounded-xl border border-red-100 shadow-xs">
                            Configure your <strong>GEMINI_API_KEY</strong> secret variable in the AI Studio sidebar to activate advanced full-stack AI evaluation.
                          </div>
                        </div>
                      </div>
                    )}

                    {aiInsights && !isLoadingAI && (
                      <div className="prose max-w-none text-slate-800 text-sm leading-relaxed">
                        <div className="bg-indigo-50/15 border border-indigo-100/60 p-7 rounded-2xl whitespace-pre-line font-sans text-slate-700 shadow-xs">
                          {aiInsights}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
