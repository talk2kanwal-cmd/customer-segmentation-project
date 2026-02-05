/**
 * Customer Churn Prediction System - Frontend Application
 * Handles file uploads, API interactions, and UI updates
 */

// === STATE MANAGEMENT ===
const state = {
    uploadedData: null,
    selectedModel: null,
    availableModels: [],
    predictions: null,
    modelMetrics: {}
};

// === DOM ELEMENTS ===
const elements = {
    uploadArea: document.getElementById('upload-area'),
    fileInput: document.getElementById('file-input'),
    fileInfo: document.getElementById('file-info'),
    fileName: document.getElementById('file-name'),
    fileSize: document.getElementById('file-size'),
    clearFileBtn: document.getElementById('clear-file-btn'),
    generateSampleBtn: document.getElementById('generate-sample-btn'),
    
    modelSection: document.getElementById('model-section'),
    modelGrid: document.getElementById('model-grid'),
    predictBtn: document.getElementById('predict-btn'),
    
    resultsSection: document.getElementById('results-section'),
    loadingOverlay: document.getElementById('loading-overlay'),
    
    // Stats
    statModels: document.getElementById('stat-models'),
    statAccuracy: document.getElementById('stat-accuracy'),
    statPredictions: document.getElementById('stat-predictions'),
    
    // Results
    highRiskCount: document.getElementById('high-risk-count'),
    highRiskPercent: document.getElementById('high-risk-percent'),
    mediumRiskCount: document.getElementById('medium-risk-count'),
    mediumRiskPercent: document.getElementById('medium-risk-percent'),
    lowRiskCount: document.getElementById('low-risk-count'),
    lowRiskPercent: document.getElementById('low-risk-percent'),
    avgChurnProb: document.getElementById('avg-churn-prob'),
    totalCustomers: document.getElementById('total-customers'),
    
    predictionsTableBody: document.getElementById('predictions-table-body'),
    downloadResultsBtn: document.getElementById('download-results-btn'),
    
    toastContainer: document.getElementById('toast-container')
};

// === API CLIENT ===
const API = {
    baseURL: '',
    
    async get(endpoint) {
        const response = await fetch(`${this.baseURL}${endpoint}`);
        return await response.json();
    },
    
    async post(endpoint, data, isFormData = false) {
        const options = {
            method: 'POST',
            body: isFormData ? data : JSON.stringify(data)
        };
        
        if (!isFormData) {
            options.headers = { 'Content-Type': 'application/json' };
        }
        
        const response = await fetch(`${this.baseURL}${endpoint}`, options);
        return await response.json();
    }
};

// === INITIALIZATION ===
async function init() {
    console.log('Initializing Customer Churn Prediction System...');
    
    // Setup event listeners
    setupEventListeners();
    
    // Load available models
    await loadModels();
    
    // Load model comparison
    await loadModelComparison();
    
    showToast('System ready! Upload data to get started.', 'info');
}

// === EVENT LISTENERS ===
function setupEventListeners() {
    // File upload
    elements.uploadArea.addEventListener('click', () => elements.fileInput.click());
    elements.fileInput.addEventListener('change', handleFileSelect);
    
    // Drag and drop
    elements.uploadArea.addEventListener('dragover', handleDragOver);
    elements.uploadArea.addEventListener('dragleave', handleDragLeave);
    elements.uploadArea.addEventListener('drop', handleDrop);
    
    // Clear file
    elements.clearFileBtn.addEventListener('click', clearFile);
    
    // Generate sample
    elements.generateSampleBtn.addEventListener('click', generateSampleData);
    
    // Predict button
    elements.predictBtn.addEventListener('click', runPrediction);
    
    // Download results
    elements.downloadResultsBtn.addEventListener('click', downloadResults);
}

// === FILE UPLOAD HANDLERS ===
function handleDragOver(e) {
    e.preventDefault();
    elements.uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    elements.uploadArea.classList.remove('drag-over');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        processFile(files[0]);
    }
}

function handleFileSelect(e) {
    const file = e.target.files[0];
    if (file) {
        processFile(file);
    }
}

async function processFile(file) {
    // Validate file type
    if (!file.name.endsWith('.csv')) {
        showToast('Please upload a CSV file', 'error');
        return;
    }
    
    // Show file info
    elements.fileName.textContent = file.name;
    elements.fileSize.textContent = formatFileSize(file.size);
    elements.fileInfo.style.display = 'flex';
    
    // Read file
    const text = await file.text();
    const rows = text.split('\n').filter(row => row.trim());
    
    showToast(`File uploaded: ${rows.length - 1} customers`, 'success');
    
    // Store file
    state.uploadedData = file;
    
    // Show model selection
    elements.modelSection.style.display = 'block';
    elements.modelSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function clearFile() {
    state.uploadedData = null;
    elements.fileInfo.style.display = 'none';
    elements.fileInput.value = '';
    elements.modelSection.style.display = 'none';
    elements.resultsSection.style.display = 'none';
}

// === SAMPLE DATA GENERATION ===
async function generateSampleData() {
    showLoading('Generating sample data...');
    
    try {
        const response = await API.post('/api/generate-sample', {
            n_samples: 100,
            churn_rate: 0.23
        });
        
        if (response.success) {
            // Convert to CSV
            const csv = jsonToCSV(response.data);
            const blob = new Blob([csv], { type: 'text/csv' });
            const file = new File([blob], 'sample_customers.csv', { type: 'text/csv' });
            
            // Process as uploaded file
            processFile(file);
            
            showToast(`Generated ${response.count} sample customers`, 'success');
        } else {
            showToast(response.error || 'Failed to generate sample data', 'error');
        }
    } catch (error) {
        console.error('Error generating sample:', error);
        showToast('Error generating sample data', 'error');
    } finally {
        hideLoading();
    }
}

// === MODEL MANAGEMENT ===
async function loadModels() {
    try {
        const response = await API.get('/api/models');
        
        if (response.success) {
            state.availableModels = response.models;
            state.modelMetrics = response.metrics;
            
            // Update stats
            elements.statModels.textContent = response.models.length;
            
            // Find best accuracy
            const bestAccuracy = Math.max(...Object.values(response.metrics).map(m => m.roc_auc));
            elements.statAccuracy.textContent = (bestAccuracy * 100).toFixed(1) + '%';
            
            // Render model cards
            renderModelCards();
        }
    } catch (error) {
        console.error('Error loading models:', error);
        showToast('Error loading models', 'error');
    }
}

function renderModelCards() {
    elements.modelGrid.innerHTML = '';
    
    state.availableModels.forEach(modelName => {
        const card = createModelCard(modelName);
        elements.modelGrid.appendChild(card);
    });
}

function createModelCard(modelName) {
    const card = document.createElement('div');
    card.className = 'model-card';
    card.dataset.model = modelName;
    
    const metrics = state.modelMetrics[modelName] || {};
    
    const displayName = modelName
        .split('_')
        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
        .join(' ');
    
    card.innerHTML = `
        <h3 class="model-name">${displayName}</h3>
        <div class="model-metrics">
            <div class="metric">
                <div class="metric-label">ROC-AUC</div>
                <div class="metric-value">${(metrics.roc_auc || 0).toFixed(3)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Precision</div>
                <div class="metric-value">${(metrics.precision || 0).toFixed(3)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Recall</div>
                <div class="metric-value">${(metrics.recall || 0).toFixed(3)}</div>
            </div>
            <div class="metric">
                <div class="metric-label">F1-Score</div>
                <div class="metric-value">${(metrics.f1_score || 0).toFixed(3)}</div>
            </div>
        </div>
    `;
    
    card.addEventListener('click', () => selectModel(modelName, card));
    
    return card;
}

function selectModel(modelName, card) {
    // Remove previous selection
    document.querySelectorAll('.model-card').forEach(c => c.classList.remove('selected'));
    
    // Select new model
    card.classList.add('selected');
    state.selectedModel = modelName;
    
    // Enable predict button
    elements.predictBtn.disabled = false;
    
    showToast(`Selected: ${modelName.replace('_', ' ')}`, 'info');
}

// === PREDICTION ===
async function runPrediction() {
    if (!state.uploadedData || !state.selectedModel) {
        showToast('Please upload data and select a model', 'error');
        return;
    }
    
    showLoading('Running predictions...');
    
    try {
        const formData = new FormData();
        formData.append('file', state.uploadedData);
        formData.append('model', state.selectedModel);
        
        const response = await API.post('/api/predict', formData, true);
        
        if (response.success) {
            state.predictions = response;
            displayResults(response);
            
            // Update prediction count
            elements.statPredictions.textContent = response.summary.total_customers;
            
            showToast('Predictions completed!', 'success');
        } else {
            showToast(response.error || 'Prediction failed', 'error');
        }
    } catch (error) {
        console.error('Prediction error:', error);
        showToast('Error making predictions', 'error');
    } finally {
        hideLoading();
    }
}

// === RESULTS DISPLAY ===
function displayResults(response) {
    const { summary, predictions, risk_distribution } = response;
    
    // Update summary cards
    elements.highRiskCount.textContent = summary.high_risk;
    elements.highRiskPercent.textContent = `${((summary.high_risk / summary.total_customers) * 100).toFixed(1)}%`;
    
    elements.mediumRiskCount.textContent = summary.medium_risk;
    elements.mediumRiskPercent.textContent = `${((summary.medium_risk / summary.total_customers) * 100).toFixed(1)}%`;
    
    elements.lowRiskCount.textContent = summary.low_risk;
    elements.lowRiskPercent.textContent = `${((summary.low_risk / summary.total_customers) * 100).toFixed(1)}%`;
    
    elements.avgChurnProb.textContent = `${(summary.avg_churn_probability * 100).toFixed(1)}%`;
    elements.totalCustomers.textContent = `${summary.total_customers} customers`;
    
    // Render table
    renderPredictionsTable(predictions);
    
    // Render chart
    renderRiskChart(risk_distribution);
    
    // Show results section
    elements.resultsSection.style.display = 'block';
    elements.resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function renderPredictionsTable(predictions) {
    elements.predictionsTableBody.innerHTML = '';
    
    // Show first 50 rows
    const displayPredictions = predictions.slice(0, 50);
    
    displayPredictions.forEach(pred => {
        const row = document.createElement('tr');
        
        const riskClass = pred.risk_category.toLowerCase().replace(' risk', '');
        
        row.innerHTML = `
            <td>${pred.customer_id}</td>
            <td>${(pred.churn_probability * 100).toFixed(2)}%</td>
            <td>${pred.predicted_churn ? 'Yes' : 'No'}</td>
            <td><span class="risk-badge ${riskClass}">${pred.risk_category}</span></td>
        `;
        
        elements.predictionsTableBody.appendChild(row);
    });
    
    if (predictions.length > 50) {
        const row = document.createElement('tr');
        row.innerHTML = `<td colspan="4" style="text-align: center; color: var(--text-muted);">Showing 50 of ${predictions.length} predictions. Download CSV for full results.</td>`;
        elements.predictionsTableBody.appendChild(row);
    }
}

function renderRiskChart(riskDistribution) {
    const ctx = document.getElementById('risk-chart').getContext('2d');
    
    // Destroy existing chart if any
    if (window.riskChart) {
        window.riskChart.destroy();
    }
    
    window.riskChart = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: ['High Risk', 'Medium Risk', 'Low Risk'],
            datasets: [{
                data: [
                    riskDistribution['High Risk'] || 0,
                    riskDistribution['Medium Risk'] || 0,
                    riskDistribution['Low Risk'] || 0
                ],
                backgroundColor: [
                    'rgba(244, 67, 54, 0.8)',
                    'rgba(255, 193, 7, 0.8)',
                    'rgba(76, 175, 80, 0.8)'
                ],
                borderColor: [
                    'rgba(244, 67, 54, 1)',
                    'rgba(255, 193, 7, 1)',
                    'rgba(76, 175, 80, 1)'
                ],
                borderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        color: '#fff',
                        font: { size: 14 },
                        padding: 20
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    borderWidth: 1
                }
            }
        }
    });
}

// === MODEL COMPARISON ===
async function loadModelComparison() {
    try {
        const response = await API.get('/api/model-comparison');
        
        if (response.success) {
            renderComparisonChart(response.data);
        }
    } catch (error) {
        console.error('Error loading model comparison:', error);
    }
}

function renderComparisonChart(data) {
    const ctx = document.getElementById('comparison-chart').getContext('2d');
    
    const models = data.map(d => d.Model.replace('_', ' '));
    const rocAuc = data.map(d => d['ROC-AUC'] * 100);
    const precision = data.map(d => d.Precision * 100);
    const recall = data.map(d => d.Recall * 100);
    const f1 = data.map(d => d['F1-Score'] * 100);
    
    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: models,
            datasets: [
                {
                    label: 'ROC-AUC',
                    data: rocAuc,
                    backgroundColor: 'rgba(170, 100, 255, 0.8)',
                    borderColor: 'rgba(170, 100, 255, 1)',
                    borderWidth: 2
                },
                {
                    label: 'Precision',
                    data: precision,
                    backgroundColor: 'rgba(66, 165, 245, 0.8)',
                    borderColor: 'rgba(66, 165, 245, 1)',
                    borderWidth: 2
                },
                {
                    label: 'Recall',
                    data: recall,
                    backgroundColor: 'rgba(255, 112, 166, 0.8)',
                    borderColor: 'rgba(255, 112, 166, 1)',
                    borderWidth: 2
                },
                {
                    label: 'F1-Score',
                    data: f1,
                    backgroundColor: 'rgba(76, 175, 80, 0.8)',
                    borderColor: 'rgba(76, 175, 80, 1)',
                    borderWidth: 2
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            scales: {
                y: {
                    beginAtZero: true,
                    max: 100,
                    ticks: {
                        color: '#fff',
                        callback: value => value + '%'
                    },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                },
                x: {
                    ticks: { color: '#fff' },
                    grid: {
                        color: 'rgba(255, 255, 255, 0.1)'
                    }
                }
            },
            plugins: {
                legend: {
                    labels: {
                        color: '#fff',
                        font: { size: 14 }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: '#fff',
                    bodyColor: '#fff',
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    borderWidth: 1,
                    callbacks: {
                        label: context => `${context.dataset.label}: ${context.parsed.y.toFixed(2)}%`
                    }
                }
            }
        }
    });
}

// === DOWNLOAD RESULTS ===
function downloadResults() {
    if (!state.predictions) return;
    
    const csv = jsonToCSV(state.predictions.predictions);
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `churn_predictions_${Date.now()}.csv`;
    a.click();
    URL.revokeObjectURL(url);
    
    showToast('Results downloaded!', 'success');
}

// === UTILITY FUNCTIONS ===
function showLoading(message = 'Loading...') {
    elements.loadingOverlay.style.display = 'flex';
    document.querySelector('.loading-text').textContent = message;
}

function hideLoading() {
    elements.loadingOverlay.style.display = 'none';
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    elements.toastContainer.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideInRight 0.3s ease-out reverse';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}

function formatFileSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(2) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(2) + ' MB';
}

function jsonToCSV(data) {
    if (!data || data.length === 0) return '';
    
    const headers = Object.keys(data[0]);
    const rows = data.map(obj => 
        headers.map(header => {
            const value = obj[header];
            return typeof value === 'string' && value.includes(',') 
                ? `"${value}"` 
                : value;
        }).join(',')
    );
    
    return [headers.join(','), ...rows].join('\n');
}

// === START APPLICATION ===
document.addEventListener('DOMContentLoaded', init);
