import React from "react";
import { Cpu, Database, Award, CheckCircle, ArrowRight, Play, Download, Zap, BarChart3, HelpCircle } from "lucide-react";
import { motion } from "motion/react";

interface MainLandingProps {
  onStartWorkspace: (sampleId?: string) => void;
}

export default function MainLanding({ onStartWorkspace }: MainLandingProps) {
  return (
    <div className="bg-slate-50 text-slate-900 min-h-screen font-sans">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-white border-b border-slate-200 pt-20 pb-24 px-8">
        {/* Glow Effects */}
        <div className="absolute top-1/3 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-indigo-50/60 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-10 left-10 w-72 h-72 bg-emerald-50/30 rounded-full blur-3xl pointer-events-none" />

        <div className="max-w-5xl mx-auto text-center relative z-10">
          <motion.div
            initial={{ opacity: 0, y: 15 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            className="inline-flex items-center gap-2 px-3.5 py-1.5 bg-indigo-50 border border-indigo-100 rounded-full text-xs text-indigo-700 font-medium tracking-wide mb-8"
          >
            <Cpu className="w-3.5 h-3.5 text-indigo-600 animate-pulse" />
            <span>BROWSER-SAFE AUTOMATED MACHINE LEARNING</span>
          </motion.div>

          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
            className="text-5xl md:text-6xl font-extrabold tracking-tight font-display text-slate-900 leading-tight mb-6"
          >
            Automated Machine Learning <span className="text-indigo-600">Studio</span>
          </motion.h1>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="text-lg md:text-xl text-slate-500 max-w-3xl mx-auto mb-12 font-sans leading-relaxed"
          >
            Upload any CSV dataset, automatically preprocess and extract features, train robust machine learning algorithms instantly in your browser, evaluate side-by-side performance, and download production-ready models or run interactive in-browser predictions.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.5, delay: 0.3 }}
            className="flex flex-col sm:flex-row items-center justify-center gap-4"
          >
            <button
              id="btn-hero-try"
              onClick={() => onStartWorkspace()}
              className="w-full sm:w-auto px-8 py-4 bg-indigo-600 hover:bg-indigo-700 text-white font-semibold rounded-xl shadow-lg shadow-indigo-150 hover:shadow-indigo-200 transition-all flex items-center justify-center gap-2.5 cursor-pointer text-base group"
            >
              <Zap className="w-5 h-5 text-indigo-200" />
              <span>Launch AutoML Workbench</span>
              <ArrowRight className="w-4 h-4 group-hover:translate-x-1 transition-transform" />
            </button>
            
            <button
              id="btn-hero-demo"
              onClick={() => onStartWorkspace("iris")}
              className="w-full sm:w-auto px-8 py-4 bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 hover:text-slate-800 font-semibold rounded-xl shadow-sm transition-all flex items-center justify-center gap-2 cursor-pointer text-base"
            >
              <Play className="w-4 h-4 text-emerald-600 fill-emerald-600/10" />
              <span>Try with Iris Demo</span>
            </button>
          </motion.div>
        </div>
      </section>

      {/* Overview Cards / Core Capabilities */}
      <section className="py-24 px-8 max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold font-display text-slate-900 tracking-tight mb-4">
            A Complete ML Pipeline in the Browser
          </h2>
          <p className="text-slate-500 max-w-2xl mx-auto text-base">
            Experience 100% actual model training and feature computation on your local machine with our robust TypeScript client engines—backed by Gemini for intelligent insights.
          </p>
        </div>

        <div className="grid md:grid-cols-3 gap-8 mb-20">
          <div className="bg-white p-8 rounded-2xl border border-slate-200/80 shadow-sm hover:shadow-md transition-all">
            <div className="w-12 h-12 rounded-xl bg-indigo-50 flex items-center justify-center mb-6">
              <Database className="w-6 h-6 text-indigo-600" />
            </div>
            <h3 className="text-xl font-bold text-slate-800 mb-3">1. Preprocess & Preview</h3>
            <p className="text-slate-500 leading-relaxed text-sm">
              Upload custom CSV logs or use pre-loaded samples. The system instantly parses, computes metrics (means, mins, unique counts, missing entries), handles missing variables, and scales labels.
            </p>
          </div>

          <div className="bg-white p-8 rounded-2xl border border-slate-200/80 shadow-sm hover:shadow-md transition-all">
            <div className="w-12 h-12 rounded-xl bg-emerald-50 flex items-center justify-center mb-6">
              <Cpu className="w-6 h-6 text-emerald-600" />
            </div>
            <h3 className="text-xl font-bold text-slate-800 mb-3">2. In-Browser Auto-Fit</h3>
            <p className="text-slate-500 leading-relaxed text-sm">
              Train multiple models parallelly in real-time. Choose your target column, exclude features, select classification vs regression, and auto-fit algorithms with real-time speed metrics.
            </p>
          </div>

          <div className="bg-white p-8 rounded-2xl border border-slate-200/80 shadow-sm hover:shadow-md transition-all">
            <div className="w-12 h-12 rounded-xl bg-amber-50 flex items-center justify-center mb-6">
              <BarChart3 className="w-6 h-6 text-amber-600" />
            </div>
            <h3 className="text-xl font-bold text-slate-800 mb-3">3. Side-by-Side Evaluation</h3>
            <p className="text-slate-500 leading-relaxed text-sm">
              Compare F1-scores, accuracy rates, and error curves in intuitive visual matrices and charts. Download full JSON parameter artifacts to save model parameters or use them anywhere.
            </p>
          </div>
        </div>

        {/* Algorithm Catalog */}
        <div className="bg-white border border-slate-200 rounded-3xl p-10 md:p-14 relative overflow-hidden shadow-sm">
          <div className="absolute right-0 bottom-0 w-80 h-80 bg-indigo-50/40 rounded-full blur-3xl" />
          
          <div className="max-w-3xl relative z-10">
            <h3 className="text-2xl font-bold font-display tracking-tight text-slate-800 mb-8">
              Supported Classifiers and Regressors
            </h3>
            
            <div className="grid sm:grid-cols-2 gap-8 text-sm">
              <div className="flex items-start gap-3.5">
                <CheckCircle className="w-5 h-5 text-indigo-600 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-bold text-slate-800">Logistic Regression Classifier</h4>
                  <p className="text-slate-500 text-xs mt-1 leading-relaxed">Stochastic gradient descent, multi-class Softmax layer, custom learning rates.</p>
                </div>
              </div>

              <div className="flex items-start gap-3.5">
                <CheckCircle className="w-5 h-5 text-indigo-600 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-bold text-slate-800">Linear Regression (Least Squares)</h4>
                  <p className="text-slate-500 text-xs mt-1 leading-relaxed">Multi-variable regression models, standard scaling, learning progress graphs.</p>
                </div>
              </div>

              <div className="flex items-start gap-3.5">
                <CheckCircle className="w-5 h-5 text-indigo-600 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-bold text-slate-800">Decision Tree Classifier & Regressor</h4>
                  <p className="text-slate-500 text-xs mt-1 leading-relaxed">Recursive node partitioning, Gini impurity decrease splits, variance MSE minimization.</p>
                </div>
              </div>

              <div className="flex items-start gap-3.5">
                <CheckCircle className="w-5 h-5 text-indigo-600 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-bold text-slate-800">K-Nearest Neighbors (KNN)</h4>
                  <p className="text-slate-500 text-xs mt-1 leading-relaxed">Euclidean multi-dimensional distance calculations, custom neighbor count parameters.</p>
                </div>
              </div>

              <div className="flex items-start gap-3.5">
                <CheckCircle className="w-5 h-5 text-indigo-600 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-bold text-slate-800">Gaussian Naive Bayes</h4>
                  <p className="text-slate-500 text-xs mt-1 leading-relaxed">Prior class distribution analytics, probability density computations, fast training speeds.</p>
                </div>
              </div>

              <div className="flex items-start gap-3.5">
                <CheckCircle className="w-5 h-5 text-indigo-600 shrink-0 mt-0.5" />
                <div>
                  <h4 className="font-bold text-slate-800">Interactive Out-of-Sample Playground</h4>
                  <p className="text-slate-500 text-xs mt-1 leading-relaxed">Verify accuracy with manual feature input widgets or custom test dataset predictions instantly.</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Demo Quick Try Banner */}
      <section className="bg-slate-100/60 py-20 px-8 border-t border-b border-slate-200">
        <div className="max-w-4xl mx-auto flex flex-col md:flex-row items-center justify-between gap-8">
          <div>
            <span className="text-[10px] font-bold text-indigo-600 tracking-widest uppercase font-sans block mb-2">QUICK DEMO PLATFORM</span>
            <h3 className="text-2xl font-bold text-slate-900 font-display tracking-tight mb-2">
              Ready to test without uploading?
            </h3>
            <p className="text-slate-500 text-sm max-w-lg leading-relaxed">
              Click the Try Sample button to load the Fisher's Iris Flower dataset, configure "species" as the target variable, and train five models simultaneously in under a second.
            </p>
          </div>
          <div className="flex gap-4 shrink-0">
            <button
              id="btn-bottom-iris"
              onClick={() => onStartWorkspace("iris")}
              className="px-6 py-3.5 bg-indigo-600 hover:bg-indigo-700 text-white font-bold rounded-xl shadow-lg shadow-indigo-100 hover:shadow-indigo-200 cursor-pointer transition-all flex items-center gap-2"
            >
              <Zap className="w-4 h-4 text-indigo-200" />
              <span>Load Iris Dataset</span>
            </button>
            <button
              id="btn-bottom-housing"
              onClick={() => onStartWorkspace("housing")}
              className="px-6 py-3.5 bg-white hover:bg-slate-50 border border-slate-200 text-slate-700 font-bold rounded-xl shadow-sm cursor-pointer transition-all flex items-center gap-2"
            >
              <Database className="w-4 h-4 text-slate-400" />
              <span>Load Housing Dataset</span>
            </button>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="bg-white text-slate-500 py-12 px-8 text-center text-xs border-t border-slate-200">
        <p className="mb-2 font-medium">Ultimate AutoML Studio &copy; 2026. All operations are calculated local-first in your sandbox browser.</p>
        <p className="text-slate-400">Integrated with Gemini 3.5 Flash for Data Quality and Hyperparameter reports on demand.</p>
      </footer>
    </div>
  );
}
