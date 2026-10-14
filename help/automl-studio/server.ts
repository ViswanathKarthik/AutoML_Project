import express from "express";
import path from "path";
import { createServer as createViteServer } from "vite";
import { GoogleGenAI } from "@google/genai";
import dotenv from "dotenv";

dotenv.config();

const app = express();
const PORT = 3000;

app.use(express.json({ limit: "50mb" }));

// Lazy initializer for GoogleGenAI
let aiClient: GoogleGenAI | null = null;
function getGeminiClient() {
  if (!aiClient) {
    const apiKey = process.env.GEMINI_API_KEY;
    if (!apiKey || apiKey === "MY_GEMINI_API_KEY") {
      throw new Error("GEMINI_API_KEY is not configured. Please add it to your secrets or .env file.");
    }
    aiClient = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          "User-Agent": "aistudio-build",
        },
      },
    });
  }
  return aiClient;
}

// API endpoint for AI Insights & Model Recommendations
app.post("/api/gemini/analyze", async (req, res) => {
  try {
    const { datasetSummary, modelResults, problemType, selectedTarget } = req.body;

    if (!datasetSummary || !modelResults) {
      res.status(400).json({ error: "Missing required parameters (datasetSummary and modelResults)." });
      return;
    }

    const ai = getGeminiClient();
    
    const prompt = `
      You are an expert AutoML and Data Science Copilot.
      Provide a highly professional, clear, and actionable Data Science & AutoML Report for the user's dataset and trained models.
      
      --- DATASET OVERVIEW ---
      Target Variable: "${selectedTarget}"
      Problem Type: ${problemType}
      Number of rows: ${datasetSummary.rowCount}
      Number of columns: ${datasetSummary.colCount}
      Columns list and types: ${JSON.stringify(datasetSummary.features)}
      
      --- MODEL METRICS ---
      Algorithms trained and their metrics:
      ${JSON.stringify(modelResults, null, 2)}
      
      Please structure your response with elegant Markdown headers and provide:
      1. **Dataset Quality Report**: Analyze the features, highlight any potential issues (e.g. imbalanced target, scale discrepancies, highly correlated variables), and explain what preprocessing we carried out (imputation, scaling, encoding).
      2. **Algorithm Performance Analysis**: Compare the results of the trained algorithms. Which one performed best and why? Explain what metrics like Accuracy, F1-Score, RMSE, or R² mean in this context.
      3. **Feature Importance Explanation**: Explain how features likely influence this target variable.
      4. **Next Steps & Deployment Advice**: Provide concrete hyperparameters tuning recommendations, advanced architectures suggestion, and how to deploy the downloaded model artifact.
      
      Keep the tone encouraging, data-driven, professional, and accessible to both beginners and pros.
    `;

    const response = await ai.models.generateContent({
      model: "gemini-3.5-flash",
      contents: prompt,
    });

    res.json({ insights: response.text });
  } catch (error: any) {
    console.error("Gemini API Error:", error);
    res.status(500).json({ 
      error: error.message || "An unexpected error occurred while generating AI insights." 
    });
  }
});

// Serve Vite dev server or static dist bundle
async function setupVite() {
  if (process.env.NODE_ENV !== "production") {
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(process.cwd(), "dist");
    app.use(express.static(distPath));
    app.get("*", (req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    console.log(`AutoML Studio Server running on port ${PORT}`);
  });
}

setupVite();
