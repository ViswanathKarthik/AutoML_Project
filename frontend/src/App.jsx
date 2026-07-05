import { useState } from 'react';
import {
  BarChart,
  Bar,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';

const PREVIEW_URL = 'http://127.0.0.1:8000/api/preview/';
const RUN_URL = 'http://127.0.0.1:8000/api/run/';
const PREDICT_URL = 'http://127.0.0.1:8000/api/predict/';

function App() {
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

  const handleFileChange = async (event) => {
    const selectedFile = event.target.files[0];
    setFile(selectedFile);
    setPreviewColumns([]);
    setPreviewRows([]);
    setTarget('');
    setSessionId('');
    setResult(null);
    setTestResult(null);
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
      setPreviewColumns(data.columns || []);
      setPreviewRows(data.preview || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file || !target) {
      setError('Please upload a CSV file and select the target column.');
      return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('target', target);

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
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
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
      setError(err.message);
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

  const filteredTargets = previewColumns.filter((col) =>
    col.toLowerCase().includes(targetSearch.toLowerCase())
  );

  const maxImportance = result?.feature_importance?.reduce(
    (max, item) => Math.max(max, Number(item.importance || 0)),
    0
  );

  const histogramData = (histogram) =>
    histogram?.bins?.slice(0, histogram.bins.length - 1).map((bin, index) => ({
      name: `${Number(bin).toFixed(1)} - ${Number(histogram.bins[index + 1]).toFixed(1)}`,
      value: histogram.counts[index],
    }));

  const importanceData = result?.feature_importance
    ? result.feature_importance
        .slice(0, 10)
        .map((item) => ({
          feature: item.feature,
          importance: Number(item.importance),
        }))
    : [];

  return (
    <div style={{ maxWidth: 1000, margin: '32px auto', fontFamily: 'Arial, sans-serif', padding: 20 }}>
      <h1 style={{ marginBottom: 8 }}>AutoML Dashboard</h1>
      <p style={{ marginTop: 0, color: '#555' }}>
        Upload a CSV file and choose the target column from the preview. After the pipeline runs, explore selected features, histograms, model scores, and test-file prediction.
      </p>

      <div style={{ display: 'grid', gap: 16, marginBottom: 24 }}>
        <label style={{ display: 'grid', gap: 8 }}>
          <strong>Upload CSV file</strong>
          <input type="file" accept=".csv" onChange={handleFileChange} />
        </label>

        {previewColumns.length > 0 && (
          <div style={{ display: 'grid', gap: 12, padding: 16, border: '1px solid #ddd', borderRadius: 10, background: '#fafafa' }}>
            <div>
              <strong>Select target column</strong>
            </div>
            <input
              type="search"
              placeholder="Search columns..."
              value={targetSearch}
              onChange={(e) => setTargetSearch(e.target.value)}
              style={{ padding: '10px', borderRadius: 6, border: '1px solid #ccc' }}
            />
            <div style={{ maxHeight: 200, overflowY: 'auto', border: '1px solid #eee', borderRadius: 8, padding: 10, background: 'white' }}>
              {filteredTargets.length > 0 ? (
                filteredTargets.map((col) => (
                  <button
                    key={col}
                    type="button"
                    onClick={() => setTarget(col)}
                    style={{
                      display: 'block',
                      width: '100%',
                      textAlign: 'left',
                      padding: '8px 10px',
                      marginBottom: 6,
                      border: '1px solid #ddd',
                      borderRadius: 6,
                      background: target === col ? '#e7f3ff' : '#fff',
                      cursor: 'pointer'
                    }}
                  >
                    {col}
                  </button>
                ))
              ) : (
                <div style={{ color: '#666' }}>No matching columns.</div>
              )}
            </div>

            <div>
              <strong>Selected target:</strong> {target || 'None'}
            </div>
          </div>
        )}

        <button
          onClick={handleSubmit}
          disabled={!file || !target || loading}
          style={{ padding: '12px 18px', border: 'none', borderRadius: 8, background: '#2f80ed', color: 'white', cursor: loading ? 'default' : 'pointer', opacity: !file || !target ? 0.6 : 1 }}
        >
          {loading ? 'Processing...' : 'Run AutoML'}
        </button>

        {error && <div style={{ color: 'crimson', fontWeight: 600 }}>{error}</div>}
      </div>

      {previewRows.length > 0 && (
        <section style={{ marginBottom: 24 }}>
          <h2 style={{ marginBottom: 12 }}>Data Preview</h2>
          <div style={{ overflowX: 'auto', border: '1px solid #ddd', borderRadius: 10 }}>
            <table style={{ width: '100%', borderCollapse: 'collapse' }}>
              <thead>
                <tr>
                  {previewColumns.map((col) => (
                    <th key={col} style={{ padding: 10, borderBottom: '1px solid #eee', background: '#f4f6fb', textAlign: 'left' }}>{col}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {previewRows.map((row, index) => (
                  <tr key={index} style={{ borderBottom: '1px solid #f1f1f1' }}>
                    {previewColumns.map((col) => (
                      <td key={col} style={{ padding: 10 }}>{String(row[col] ?? '')}</td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {result && (
        <section style={{ marginTop: 20 }}>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 18 }}>
            {['summary', 'features', 'histograms', 'metrics', 'importance', 'prediction'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                style={{
                  padding: '10px 16px',
                  borderRadius: 8,
                  border: '1px solid #ccc',
                  background: activeTab === tab ? '#2f80ed' : '#fff',
                  color: activeTab === tab ? '#fff' : '#000',
                  cursor: 'pointer'
                }}
              >
                {tab === 'summary' ? 'Summary' : tab === 'features' ? 'Selected Features' : tab === 'histograms' ? 'Histograms' : tab === 'metrics' ? 'Model Scores' : tab === 'importance' ? 'Feature Importance' : 'Test Prediction'}
              </button>
            ))}
          </div>

          {activeTab === 'summary' && (
            <div style={{ display: 'grid', gap: 18 }}>
              <div style={{ display: 'grid', gap: 8 }}>
                <h2>Pipeline Summary</h2>
                <div><strong>Problem type:</strong> {result.problem_type}</div>
                <div><strong>Best model:</strong> {result.best_model}</div>
                <div><strong>Session ID:</strong> {sessionId}</div>
              </div>

              <div style={{ display: 'grid', gap: 8 }}>
                <h3>Metrics</h3>
                <ul>
                  {Object.entries(result.metrics || {}).map(([k, v]) => (
                    <li key={k}>{k}: {Number(v).toFixed(4)}</li>
                  ))}
                </ul>
              </div>

              <div style={{ display: 'grid', gap: 8 }}>
                <h3>Files & downloads</h3>
                <ul>
                  {Object.entries(result.files || {}).map(([label, url]) => (
                    <li key={label}><a href={url} target="_blank" rel="noreferrer">{label} report</a></li>
                  ))}
                </ul>
                <div>
                  <strong>Model artifacts:</strong>
                  <ul>
                    {Object.entries(result.pickle_files || {}).map(([modelName, url]) => (
                      <li key={modelName}>
                        <a href={url} target="_blank" rel="noreferrer">
                          {modelName} model (.pkl)
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'features' && (
            <div>
              <h2>Selected Features</h2>
              <div style={{ display: 'grid', gap: 8, padding: 16, border: '1px solid #ddd', borderRadius: 10, background: '#fafafa' }}>
                {result.selected_features?.map((feature) => (
                  <div key={feature} style={{ padding: '8px 12px', border: '1px solid #eee', borderRadius: 8, background: '#fff' }}>{feature}</div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'histograms' && (
            <div>
              <h2>Feature Histograms</h2>
              {Object.entries(result.histograms || {}).length === 0 ? (
                <p>No numeric histogram data available.</p>
              ) : (
                <div style={{ display: 'grid', gap: 20 }}>
                  {Object.entries(result.histograms || {}).map(([feature, histogram]) => (
                    <div key={feature} style={{ border: '1px solid #ddd', borderRadius: 10, padding: 16, background: '#fff' }}>
                      <h4 style={{ marginTop: 0 }}>{feature}</h4>
                      <div style={{ height: 280 }}>
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={histogramData(histogram)}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                            <XAxis dataKey="name" tick={{ fontSize: 12 }} interval={0} angle={-45} textAnchor="end" height={60} />
                            <YAxis />
                            <Tooltip formatter={(value) => [value, 'Count']} />
                            <Bar dataKey="value" fill="#2f80ed" />
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}

          {activeTab === 'metrics' && (
            <div>
              <h2>Model Scores</h2>
              {Object.keys(result.results || {}).length === 0 ? (
                <p>No model results available.</p>
              ) : (
                <div style={{ overflowX: 'auto', border: '1px solid #ddd', borderRadius: 10, background: '#fff' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                    <thead>
                      <tr>
                        <th style={{ textAlign: 'left', padding: 10, background: '#f4f6fb', borderBottom: '1px solid #ddd' }}>Model</th>
                        {Object.keys(result.results[Object.keys(result.results)[0]] || {}).map((metric) => (
                          <th key={metric} style={{ textAlign: 'left', padding: 10, background: '#f4f6fb', borderBottom: '1px solid #ddd' }}>{metric}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(result.results).map(([modelName, metrics]) => (
                        <tr key={modelName} style={{ borderBottom: '1px solid #f1f1f1' }}>
                          <td style={{ padding: 10 }}>{modelName}</td>
                          {Object.keys(result.results[Object.keys(result.results)[0]] || {}).map((metric) => (
                            <td key={metric} style={{ padding: 10 }}>{Number(metrics[metric]).toFixed(4)}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          )}

          {activeTab === 'importance' && (
            <div>
              <h2>Feature Importance</h2>
              {importanceData.length === 0 ? (
                <p>No feature importance values available.</p>
              ) : (
                <div style={{ border: '1px solid #ddd', borderRadius: 10, padding: 16, background: '#fff', height: 420 }}>
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={importanceData} layout="vertical" margin={{ top: 10, right: 16, bottom: 10, left: 100 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e0e0e0" />
                      <XAxis type="number" />
                      <YAxis type="category" dataKey="feature" width={160} tick={{ fontSize: 13 }} />
                      <Tooltip formatter={(value) => [Number(value).toFixed(4), 'Importance']} />
                      <Bar dataKey="importance" fill="#2f80ed" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </div>
          )}

          {activeTab === 'prediction' && (
            <div style={{ display: 'grid', gap: 18 }}>
              <h2>Test File Prediction</h2>
              <form onSubmit={handlePredict} style={{ display: 'grid', gap: 12, maxWidth: 520 }}>
                <label style={{ display: 'grid', gap: 8 }}>
                  Select a test CSV file for prediction
                  <input type="file" accept=".csv" onChange={(e) => setTestFile(e.target.files[0])} />
                </label>
                <button type="submit" disabled={!testFile || loading} style={{ padding: '10px 14px', borderRadius: 8, border: 'none', background: '#2f80ed', color: '#fff', cursor: loading ? 'default' : 'pointer' }}>
                  {loading ? 'Predicting...' : 'Run Prediction'}
                </button>
              </form>

              {testResult && (
                <div style={{ padding: 16, border: '1px solid #ddd', borderRadius: 10, background: '#fff' }}>
                  <div><strong>Model:</strong> {testResult.model}</div>
                  {testResult.has_actual && <div><strong>Accuracy / Score:</strong> {Number(testResult.score).toFixed(4)}</div>}
                  <button onClick={downloadCsv} style={{ marginTop: 12, padding: '10px 14px', borderRadius: 8, border: 'none', background: '#27ae60', color: '#fff', cursor: 'pointer' }}>
                    Download Predictions CSV
                  </button>

                  <div style={{ overflowX: 'auto', marginTop: 20 }}>
                    <table style={{ width: '100%', borderCollapse: 'collapse' }}>
                      <thead>
                        <tr>
                          {testResult.columns.map((col) => (
                            <th key={col} style={{ textAlign: 'left', padding: 10, background: '#f4f6fb', borderBottom: '1px solid #ddd' }}>{col}</th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {testResult.predictions.map((row) => (
                          <tr key={row.index} style={{ borderBottom: '1px solid #f1f1f1' }}>
                            {testResult.columns.map((col) => (
                              <td key={col} style={{ padding: 10 }}>{row[col] ?? ''}</td>
                            ))}
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}
        </section>
      )}
    </div>
  );
}

export default App;
