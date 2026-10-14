import { useEffect, useMemo, useRef, useState } from 'react';
import {
  BarChart,
  Bar,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import './App.css';

const PREVIEW_URL = 'http://127.0.0.1:8000/api/preview/';
const RUN_URL = 'http://127.0.0.1:8000/api/run/';
const PREDICT_URL = 'http://127.0.0.1:8000/api/predict/';

function App() {
  const [view, setView] = useState('landing');
  const [file, setFile] = useState(null);
  const [previewColumns, setPreviewColumns] = useState([]);
  const [previewRows, setPreviewRows] = useState([]);
  const [targetSearch, setTargetSearch] = useState('');
  const [target, setTarget] = useState('');
  const [sessionId, setSessionId] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [activeTab, setActiveTab] = useState('summary');
  const [testFile, setTestFile] = useState(null);
  const [testResult, setTestResult] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [activeStory, setActiveStory] = useState('prep');
  const [activePanel, setActivePanel] = useState('overview');
  const [quickView, setQuickView] = useState('data');
  const [selectedHistogramFeature, setSelectedHistogramFeature] = useState('');
  const [previewHistograms, setPreviewHistograms] = useState({});
  const resultsRef = useRef(null);

  useEffect(() => {
    if (result && resultsRef.current) {
      requestAnimationFrame(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
        window.scrollTo({ top: 0, behavior: 'smooth' });
      });
    }
  }, [result]);

  useEffect(() => {
    if (previewColumns.length && !selectedHistogramFeature) {
      const firstNumeric = previewColumns.find((col) => {
        const vals = previewRows.map((row) => Number(row[col]));
        return vals.some((value) => !Number.isNaN(value));
      });
      setSelectedHistogramFeature(firstNumeric || previewColumns[0]);
    }
  }, [previewColumns, previewRows, selectedHistogramFeature]);

  const filteredTargets = useMemo(
    () => previewColumns.filter((col) => col.toLowerCase().includes(targetSearch.toLowerCase())),
    [previewColumns, targetSearch]
  );

  const runPipelineWithFile = async (selectedFile, selectedTarget) => {
    if (!selectedFile || !selectedTarget) {
      setError('Please upload a CSV file and select the target column.');
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('target', selectedTarget);

    try {
      setLoading(true);
      setError('');
      setResult(null);
      setTestResult(null);

      const response = await fetch(RUN_URL, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Run request failed');
      }
      setResult(data);
      setSessionId(data.session_id || '');
      setActiveTab('summary');
      setView('workspace');
      setActivePanel('results');
    } catch (err) {
      setError(err.message || 'Unable to run the AutoML pipeline.');
    } finally {
      setLoading(false);
    }
  };

  const previewDataset = async (selectedFile, options = {}) => {
    setFile(selectedFile);
    setPreviewColumns([]);
    setPreviewRows([]);
    setTarget('');
    setSessionId('');
    setResult(null);
    setTestResult(null);
    setPreviewHistograms({});
    setError('');

    if (!selectedFile) {
      return;
    }

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      setLoading(true);
      const response = await fetch(PREVIEW_URL, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Preview request failed');
      }
      const columns = data.columns || [];
      const rows = data.preview || [];
      const resolvedTarget = options.target || columns.find((col) => col.toLowerCase() === 'purchased') || columns[columns.length - 1] || '';

      setPreviewColumns(columns);
      setPreviewRows(rows);
      setPreviewHistograms(data.histograms || {});
      setTarget(resolvedTarget);
      setView('workspace');
      setActivePanel('data');
      setQuickView('run');

      if (options.autoRun && resolvedTarget) {
        await runPipelineWithFile(selectedFile, resolvedTarget);
      }
    } catch (err) {
      setError(err.message || 'Unable to preview the CSV.');
    } finally {
      setLoading(false);
    }
  };

  const handleFileChange = async (event) => {
    const selectedFile = event.target.files?.[0] || null;
    await previewDataset(selectedFile);
  };

  const handleTryDemo = async () => {
    try {
      const response = await fetch('/demo_dataset.csv');
      if (!response.ok) {
        throw new Error('Demo dataset could not be loaded.');
      }
      const blob = await response.blob();
      const demoFile = new File([blob], 'demo_dataset.csv', { type: 'text/csv' });
      await previewDataset(demoFile, { autoRun: true, target: 'purchased' });
    } catch (err) {
      setError(err.message || 'Unable to load the demo dataset.');
    }
  };

  const handleDrop = async (event) => {
    event.preventDefault();
    event.stopPropagation();
    setDragActive(false);
    const selectedFile = event.dataTransfer?.files?.[0] || null;
    if (!selectedFile) {
      return;
    }

    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.csv';
    input.onchange = (evt) => {
      const fileList = evt.target.files;
      if (fileList?.[0]) {
        handleFileChange({ target: { files: fileList } });
      }
    };
    input.click();
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    await runPipelineWithFile(file, target);
  };

  const handlePredict = async (e) => {
    e.preventDefault();
    if (!testFile || !sessionId) {
      setError('Upload a test file and ensure a session is available.');
      return;
    }

    const formData = new FormData();
    formData.append('file', testFile);
    formData.append('session_id', sessionId);

    try {
      setLoading(true);
      setError('');
      setTestResult(null);

      const response = await fetch(PREDICT_URL, {
        method: 'POST',
        body: formData,
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'Prediction request failed');
      }
      setTestResult(data);
      setActiveTab('prediction');
    } catch (err) {
      setError(err.message || 'Unable to create predictions.');
    } finally {
      setLoading(false);
    }
  };

  const downloadCsv = () => {
    if (!testResult) {
      return;
    }
    const csvRows = [testResult.columns.join(',')];
    testResult.predictions.forEach((row) => {
      csvRows.push([row.index, row.prediction, row.actual].filter((v) => v !== undefined).join(','));
    });
    const csv = csvRows.join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'prediction_results.csv';
    a.click();
    URL.revokeObjectURL(url);
  };

  const downloadReport = async (label, url) => {
    if (!url) {
      return;
    }

    try {
      const backendBase = 'http://127.0.0.1:8000';
      const absoluteUrl = url.startsWith('http') ? url : `${backendBase}${url.startsWith('/') ? url : `/${url}`}`;
      const response = await fetch(absoluteUrl);
      if (!response.ok) {
        throw new Error('Report could not be downloaded.');
      }
      const blob = await response.blob();
      const blobUrl = URL.createObjectURL(blob);
      const anchor = document.createElement('a');
      anchor.href = blobUrl;
      anchor.download = `${label}.${url.endsWith('.html') ? 'html' : url.split('.').pop() || 'report'}`;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(blobUrl);
    } catch (err) {
      setError(err.message || 'Unable to download the report.');
    }
  };

  const openReportInBrowser = (url) => {
    if (!url) {
      return;
    }

    const backendBase = 'http://127.0.0.1:8000';
    const absoluteUrl = url.startsWith('http') ? url : `${backendBase}${url.startsWith('/') ? url : `/${url}`}`;
    window.open(absoluteUrl, '_blank', 'noopener,noreferrer');
  };

  const histogramData = (histogram) => {
    if (!histogram) return [];

    if (histogram.type === 'categorical') {
      return (histogram.bins || []).map((label, index) => ({
        name: String(label),
        value: Number(histogram.counts?.[index] ?? 0),
      }));
    }

    return (histogram.bins || []).slice(0, (histogram.bins || []).length - 1).map((bin, index) => ({
      name: `${Number(bin).toFixed(1)} - ${Number((histogram.bins || [])[index + 1] ?? bin).toFixed(1)}`,
      value: Number(histogram.counts?.[index] ?? 0),
    }));
  };

  const histogramColumns = useMemo(
    () =>
      Object.keys(previewHistograms).length
        ? Object.keys(previewHistograms)
        : previewColumns.filter((column) => {
            const values = previewRows.map((row) => Number(row[column]));
            return values.some((value) => !Number.isNaN(value));
          }),
    [previewColumns, previewHistograms, previewRows]
  );

  const preRunHistogramData = useMemo(() => {
    if (!selectedHistogramFeature) {
      return [];
    }

    const hist = previewHistograms[selectedHistogramFeature];
    if (!hist) {
      return [];
    }

    return histogramData(hist);
  }, [previewHistograms, selectedHistogramFeature]);

  const importanceData = result?.feature_importance
    ? result.feature_importance.slice(0, 10).map((item) => ({
        feature: item.feature,
        importance: Number(item.importance),
      }))
    : [];

  const storyItems = {
    prep: {
      title: 'Prepare the data',
      body: 'Upload a CSV and let the studio inspect columns, target quality, and readiness in one guided flow.',
      accent: 'Bring structure to raw data',
    },
    train: {
      title: 'Train multiple models',
      body: 'Run your AutoML workflow and compare model outcomes without leaving the workspace.',
      accent: 'Parallel model evaluation',
    },
    ship: {
      title: 'Ship with confidence',
      body: 'Review model performance, download reports, and score fresh datasets with transparent outputs.',
      accent: 'Production-ready handoff',
    },
  };

  const modelComparisonData = useMemo(() => {
    if (!result?.results) return [];
    return Object.entries(result.results).map(([name, metrics]) => {
      const metricName = result.problem_type === 'classification' ? 'accuracy' : 'r2_score';
      const value = Number(metrics?.[metricName] ?? metrics?.score ?? metrics?.[Object.keys(metrics)[0]] ?? 0);
      return {
        name: name.replace(/_/g, ' '),
        score: Number.isFinite(value) ? value : 0,
      };
    });
  }, [result]);

  const htmlReportUrl = result?.files?.html || '';

  return (
    <div className="app-shell">
      <nav className="topbar">
        <div className="brand" onClick={() => setView('landing')} role="button" tabIndex={0} onKeyDown={(e) => e.key === 'Enter' && setView('landing')}>
          <div className="brand__mark" />
          <div className="brand__name">AutoML<span>Studio</span></div>
        </div>
        <div className="nav-links">
          <button className={`nav-btn ${view === 'landing' ? 'active' : ''}`} onClick={() => setView('landing')}>Capabilities</button>
          <button className={`nav-btn ${view === 'workspace' ? 'active' : ''}`} onClick={() => setView('workspace')}>Workbench</button>
          <button className="nav-btn primary" onClick={() => setView('workspace')}>Launch Studio</button>
        </div>
      </nav>

      <main className="page">
        {view === 'landing' ? (
          <>
            <section className="hero hero--premium">
              <div className="hero__card">
                <div className="eyebrow">Browser-safe automated machine learning</div>
                <h1>Build a complete ML workflow with a polished studio-style experience.</h1>
                <p>Upload a CSV, pick a target column, run the backend pipeline, inspect feature importance, and download predictions—all from a refined interface inspired by the reference design.</p>
                <div className="hero__actions">
                  <button className="hero__btn primary" onClick={() => setView('workspace')}>Launch workbench</button>
                  <button className="hero__btn secondary" onClick={handleTryDemo} disabled={loading}>
                    {loading ? 'Loading demo…' : 'Try with demo'}
                  </button>
                </div>
              </div>

              <div className="hero__panel hero__card">
                <div className="hero__mini hero__mini--highlight">
                  <strong>Live analytics workspace</strong>
                  <span>Guide users from upload to decision-ready insights with a premium, dashboard-oriented experience.</span>
                </div>
                <div className="hero__mini">
                  <strong>Upload any CSV</strong>
                  <span>Preview columns, choose targets, and run your training workflow in one place.</span>
                </div>
                <div className="hero__mini">
                  <strong>Inspect model results</strong>
                  <span>Review metrics, selected features, and result artifacts from the backend response.</span>
                </div>
                <div className="hero__mini">
                  <strong>Download predictions</strong>
                  <span>Send a test dataset through the same session and export the scored output.</span>
                </div>
              </div>
            </section>

            <section className="section-block">
              <div className="section-block__header">
                <div>
                  <div className="eyebrow">Product experience</div>
                  <h2>Built to feel like a real ML platform, not a demo shell.</h2>
                </div>
                <p>Every step is designed to guide a user from data upload to decision-ready insights with clarity and momentum.</p>
              </div>

              <div className="workflow-grid">
                <article className="workflow-card">
                  <div className="metric-pill">01 • Guided onboarding</div>
                  <h3>One-click setup</h3>
                  <p>Start with the demo dataset or bring your own CSV and move into the workspace instantly.</p>
                </article>
                <article className="workflow-card">
                  <div className="metric-pill">02 • Transparent evaluation</div>
                  <h3>Live model comparison</h3>
                  <p>Inspect accuracy, feature importance, and model behavior through rich result views.</p>
                </article>
                <article className="workflow-card">
                  <div className="metric-pill">03 • Delivery-ready outputs</div>
                  <h3>Export and share</h3>
                  <p>Download reports and prediction results as polished artifacts for downstream teams.</p>
                </article>
              </div>
            </section>

            <section className="story-section">
              <div className="story-grid">
                <div className="story-nav">
                  {Object.entries(storyItems).map(([key, item]) => (
                    <button key={key} className={`story-card ${activeStory === key ? 'active' : ''}`} onClick={() => setActiveStory(key)}>
                      <strong>{item.title}</strong>
                      <span>{item.accent}</span>
                    </button>
                  ))}
                </div>
                <div className="story-panel">
                  <div className="metric-pill">Interactive workflow</div>
                  <h3>{storyItems[activeStory].title}</h3>
                  <p>{storyItems[activeStory].body}</p>
                </div>
              </div>
            </section>
          </>
        ) : (
          <section className="workspace-stack">
            <div className="studio-shell">
              <aside className="studio-sidebar">
                <div className="studio-sidebar__title">Studio Console</div>
                <div className="sidebar-nav">
                  {['overview', 'data', 'results'].map((item) => {
                    const isResults = item === 'results';
                    return (
                      <button
                        key={item}
                        disabled={isResults && !result}
                        className={`sidebar-nav__item ${activePanel === item ? 'active' : ''}`}
                        onClick={() => {
                          if (isResults && !result) {
                            return;
                          }

                          setActivePanel(item);
                          if (item === 'data') {
                            setQuickView('data');
                          }
                          if (isResults && resultsRef.current) {
                            requestAnimationFrame(() => {
                              resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
                              window.scrollTo({ top: 0, behavior: 'smooth' });
                            });
                          }
                        }}
                      >
                        {item === 'overview' ? 'Overview' : item === 'data' ? 'Inspection tools' : 'Results'}
                      </button>
                    );
                  })}
                </div>
                <div className="studio-sidebar__footer">
                  <div className="metric-pill">Live workflow</div>
                  <p>Built for teams that want an elegant, intelligent ML experience.</p>
                </div>
              </aside>

              <div className="studio-main">
                <div className="panel panel-soft">
                  <div className="panel__header">
                    <div>
                      <h3>Start your AutoML session</h3>
                      <p>Upload a CSV file to preview columns, choose a target, and launch the training pipeline.</p>
                    </div>
                    <button className="secondary-btn" onClick={() => setView('landing')}>Back home</button>
                  </div>

                  <div className={`dropzone ${dragActive ? 'active' : ''}`} onDragEnter={(e) => { e.preventDefault(); setDragActive(true); }} onDragOver={(e) => { e.preventDefault(); setDragActive(true); }} onDragLeave={(e) => { e.preventDefault(); setDragActive(false); }} onDrop={handleDrop}>
                    <div className="dropzone__row">
                      <div>
                        <strong>Drop a CSV here or browse from your machine</strong>
                        <p>The uploaded file is sent to the Django backend preview endpoint.</p>
                      </div>
                      <label className="input-file">
                        <input type="file" accept=".csv" onChange={handleFileChange} />
                        Browse CSV
                      </label>
                    </div>
                  </div>

                  {file ? (
                    <div className="upload-meta">
                      <span className="upload-meta__label">Loaded file</span>
                      <strong>{file.name}</strong>
                    </div>
                  ) : null}
                </div>

                {activePanel === 'overview' ? (
                  <div className="panel panel-soft">
                    <div className="panel__header">
                      <div>
                        <h3>Overview</h3>
                        <p>Upload data, inspect distributions, choose the target, and let AutoML guide the analysis.</p>
                      </div>
                    </div>
                    <div className="result-grid">
                      <div className="metric-card">
                        <span>Dataset</span>
                        <strong>{file ? file.name : 'Not uploaded yet'}</strong>
                      </div>
                      <div className="metric-card">
                        <span>Columns</span>
                        <strong>{previewColumns.length || 0}</strong>
                      </div>
                      <div className="metric-card">
                        <span>Target</span>
                        <strong>{target || 'Not selected'}</strong>
                      </div>
                      <div className="metric-card">
                        <span>Status</span>
                        <strong>{result ? 'Model ready' : 'Waiting for run'}</strong>
                      </div>
                    </div>
                  </div>
                ) : null}

                {activePanel === 'data' ? (
                  <>
                    <div className="workspace-shortcuts">
                      <button
                        className={`workspace-shortcut ${quickView === 'run' ? 'active' : ''}`}
                        onClick={() => setQuickView('run')}
                        type="button"
                      >
                        Run Model
                      </button>
                      <button
                        className={`workspace-shortcut ${quickView === 'data' || quickView === 'histograms' ? 'active' : ''}`}
                        onClick={() => setQuickView('data')}
                        type="button"
                      >
                        Data Preview
                      </button>
                    </div>

                    {quickView === 'run' ? (
                      <div className="panel panel-soft">
                        <div className="panel__header">
                          <div>
                            <h3>Select the target variable</h3>
                            <p>Choose the column the model should predict and then launch the workflow.</p>
                          </div>
                        </div>

                        <div className="target-grid">
                          <div className="target-tools">
                            <input className="target-search" type="search" placeholder="Search columns" value={targetSearch} onChange={(e) => setTargetSearch(e.target.value)} />
                            <div className="target-list">
                              {filteredTargets.length > 0 ? (
                                filteredTargets.map((col) => (
                                  <button key={col} className={`target-chip ${target === col ? 'active' : ''}`} onClick={() => setTarget(col)} type="button">
                                    {col}
                                  </button>
                                ))
                              ) : (
                                <div className="empty-state small">No columns match your search.</div>
                              )}
                            </div>
                          </div>

                          <div className="target-summary">
                            <div className="selected-target-card">
                              <span>Selected target</span>
                              <strong>{target || 'Not selected yet'}</strong>
                              <small>{previewColumns.length ? `${previewColumns.length} available columns` : 'Upload a CSV to continue'}</small>
                            </div>
                            <button className="primary-btn" onClick={handleSubmit} disabled={!file || !target || loading}>
                              {loading ? 'Running...' : 'Run AutoML'}
                            </button>
                          </div>
                        </div>
                      </div>
                    ) : (
                      <>
                        <div className="toolbar-card">
                          <div className="toolbar-card__header">
                            <div>
                              <h3>Inspection tools</h3>
                              <p>{previewColumns.length ? `${previewColumns.length} columns ready` : 'No dataset loaded yet'}</p>
                            </div>
                          </div>
                          <div className="toolbar-actions">
                            <button className={`toolbar-action ${quickView === 'data' ? 'active' : ''}`} onClick={() => setQuickView('data')} disabled={!previewRows.length}>
                              Data preview
                            </button>
                            <button className={`toolbar-action ${quickView === 'histograms' ? 'active' : ''}`} onClick={() => setQuickView('histograms')} disabled={!previewRows.length}>
                              Histograms
                            </button>
                          </div>
                        </div>

                        {previewRows.length > 0 ? (
                          <div className="panel panel-soft">
                            <div className="panel__header">
                              <div>
                                <h3>{quickView === 'histograms' ? 'Feature histograms' : 'Data preview'}</h3>
                                <p>
                                  {quickView === 'histograms'
                                    ? 'Choose a feature to inspect its distribution before running AutoML.'
                                    : 'Review the uploaded dataset before running the pipeline.'}
                                </p>
                              </div>
                            </div>

                            {quickView === 'data' ? (
                              <div className="table-wrapper">
                                <table className="preview-table">
                                  <thead>
                                    <tr>
                                      {previewColumns.map((col) => <th key={col}>{col}</th>)}
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {previewRows.map((row, index) => (
                                      <tr key={`${row[previewColumns[0]] || 'row'}-${index}`}>
                                        {previewColumns.map((col) => <td key={`${col}-${index}`}>{String(row[col] ?? '')}</td>)}
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            ) : (
                              <div className="mini-chart-grid">
                                <label>
                                  <span style={{ display: 'block', marginBottom: '0.5rem', color: '#9aaecb' }}>Select feature</span>
                                  <select
                                    className="feature-select"
                                    value={selectedHistogramFeature}
                                    onChange={(e) => setSelectedHistogramFeature(e.target.value)}
                                    style={{ width: '100%' }}
                                  >
                                    {histogramColumns.map((column) => (
                                      <option key={column} value={column}>{column}</option>
                                    ))}
                                  </select>
                                </label>

                                {selectedHistogramFeature && preRunHistogramData.length ? (
                                  <div className="mini-card" style={{ marginTop: '0.75rem' }}>
                                    <strong>{selectedHistogramFeature}</strong>
                                    <div className="mini-chart-wrap">
                                      <ResponsiveContainer width="100%" height="100%">
                                        <BarChart data={preRunHistogramData} margin={{ top: 8, right: 10, left: 0, bottom: 18 }}>
                                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                                          <XAxis dataKey="name" tick={{ fontSize: 9 }} interval={0} angle={-25} textAnchor="end" height={45} />
                                          <YAxis tick={{ fontSize: 9 }} />
                                          <Tooltip />
                                          <Bar dataKey="value" fill="#8b9cff" radius={[4, 4, 0, 0]} />
                                        </BarChart>
                                      </ResponsiveContainer>
                                    </div>
                                  </div>
                                ) : (
                                  <div className="empty-state">No histogram data available for this feature.</div>
                                )}
                              </div>
                            )}
                          </div>
                        ) : null}
                      </>
                    )}
                  </>
                ) : null}

                {error ? <div className="error-box">{error}</div> : null}

                {result && (activePanel === 'results' || result) ? (
                  <section ref={resultsRef} className="panel panel-soft results-panel">
                    <div className="results-topbar">
                      <div>
                        <div className="metric-pill">Session overview</div>
                        <h3>{result.best_model || 'Workflow analysed'}</h3>
                        <p>Problem type: {result.problem_type} • Session: {sessionId || 'active'}</p>
                      </div>
                      <div className="results-topbar__actions">
                        {htmlReportUrl ? (
                          <>
                            <button type="button" className="secondary-btn" onClick={() => openReportInBrowser(htmlReportUrl)}>
                              Open detailed review
                            </button>
                            <button type="button" className="secondary-btn" onClick={() => downloadReport('html', htmlReportUrl)}>
                              Download HTML
                            </button>
                          </>
                        ) : null}
                      </div>
                    </div>

                    <div className="result-grid">
                      <div className="metric-card">
                        <span>Problem type</span>
                        <strong>{result.problem_type}</strong>
                      </div>
                      <div className="metric-card">
                        <span>Best model</span>
                        <strong>{result.best_model}</strong>
                      </div>
                      <div className="metric-card">
                        <span>Selected features</span>
                        <strong>{result.selected_features?.length || 0}</strong>
                      </div>
                      <div className="metric-card">
                        <span>Report links</span>
                        <strong>{result.files ? Object.keys(result.files).length : 0}</strong>
                      </div>
                    </div>

                    <div className="tabs">
                      {['summary', 'features', 'histograms', 'metrics', 'importance', 'prediction'].map((tab) => (
                        <button key={tab} className={`tab-btn ${activeTab === tab ? 'active' : ''}`} onClick={() => setActiveTab(tab)}>
                          {tab === 'summary' ? 'Summary' : tab === 'features' ? 'Selected Features' : tab === 'histograms' ? 'Histograms' : tab === 'metrics' ? 'Model Scores' : tab === 'importance' ? 'Feature Importance' : 'Prediction Playground'}
                        </button>
                      ))}
                    </div>

                    {activeTab === 'summary' ? (
                      <div className="results-stack">
                        <div className="results-chart-card">
                          <div className="panel__header" style={{ marginBottom: '0.75rem' }}>
                            <div>
                              <h3>Model quality snapshot</h3>
                              <p>Compare the performance of each model in a concise, product-style summary.</p>
                            </div>
                          </div>
                          <div style={{ height: 240 }}>
                            <ResponsiveContainer width="100%" height="100%">
                              <BarChart data={modelComparisonData}>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                                <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                                <YAxis domain={[0, 1]} />
                                <Tooltip />
                                <Bar dataKey="score" fill="#8b9cff" radius={[8, 8, 0, 0]} />
                              </BarChart>
                            </ResponsiveContainer>
                          </div>
                        </div>
                      </div>
                    ) : null}

                    {activeTab === 'features' ? (
                      <div className="feature-list">
                        {result.selected_features?.map((feature) => <div key={feature} className="feature-item">{feature}</div>)}
                      </div>
                    ) : null}

                    {activeTab === 'histograms' ? (
                      <div className="result-grid">
                        {Object.entries(result.histograms || {}).map(([feature, histogram]) => (
                          <div key={feature} className="metric-card chart-card">
                            <strong>{feature}</strong>
                            <div style={{ height: 220 }}>
                              <ResponsiveContainer width="100%" height="100%">
                                <BarChart data={histogramData(histogram)}>
                                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                                  <XAxis dataKey="name" tick={{ fontSize: 10 }} interval={0} angle={-30} textAnchor="end" height={55} />
                                  <YAxis />
                                  <Tooltip />
                                  <Bar dataKey="value" fill="#6366f1" />
                                </BarChart>
                              </ResponsiveContainer>
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : null}

                    {activeTab === 'metrics' ? (
                      <div className="table-wrapper">
                        <table className="preview-table">
                          <thead>
                            <tr>
                              <th>Model</th>
                              {Object.keys(result.results?.[Object.keys(result.results)[0]] || {}).map((metric) => <th key={metric}>{metric}</th>)}
                            </tr>
                          </thead>
                          <tbody>
                            {Object.entries(result.results || {}).map(([modelName, metrics]) => (
                              <tr key={modelName}>
                                <td>{modelName}</td>
                                {Object.keys(result.results[Object.keys(result.results)[0]] || {}).map((metric) => (
                                  <td key={metric}>{Number(metrics[metric]).toFixed(4)}</td>
                                ))}
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    ) : null}

                    {activeTab === 'importance' ? (
                      <div style={{ height: 280 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={importanceData} layout="vertical" margin={{ top: 10, right: 20, left: 30, bottom: 10 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.08)" />
                            <XAxis type="number" tick={{ fontSize: 10 }} />
                            <YAxis dataKey="feature" type="category" width={180} interval={0} tick={{ fontSize: 10 }} />
                            <Tooltip />
                            <Bar dataKey="importance" fill="#22c55e" radius={[0, 8, 8, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    ) : null}

                    {activeTab === 'prediction' ? (
                      <div className="prediction-card">
                        <label>
                          <span>Upload a test CSV</span>
                          <input type="file" accept=".csv" onChange={(e) => setTestFile(e.target.files?.[0] || null)} />
                        </label>
                        <button className="primary-btn" onClick={handlePredict} disabled={!testFile || !sessionId || loading}>
                          {loading ? 'Scoring...' : 'Run predictions'}
                        </button>
                        {testResult ? (
                          <>
                            <div className="metric-card">
                              <span>Score</span>
                              <strong>{testResult.score !== null ? Number(testResult.score).toFixed(4) : 'N/A'}</strong>
                            </div>
                            <div className="table-wrapper">
                              <table className="preview-table">
                                <thead>
                                  <tr>
                                    {testResult.columns.map((col) => <th key={col}>{col}</th>)}
                                  </tr>
                                </thead>
                                <tbody>
                                  {testResult.predictions.map((row, index) => (
                                    <tr key={`${row.index ?? index}`}>
                                      {testResult.columns.map((col) => <td key={`${col}-${index}`}>{String(row[col] ?? '')}</td>)}
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                            <button className="secondary-btn" onClick={downloadCsv}>Download CSV</button>
                          </>
                        ) : null}
                      </div>
                    ) : null}
                  </section>
                ) : null}
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
