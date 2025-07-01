from flask import Flask, request, jsonify
from flask_cors import CORS
from main import answer_query, main as setup_embeddings
import threading
import os
import psutil
import time
from collections import deque
import json
import pynvml  # For NVIDIA GPU monitoring

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins for development

# Initialize embeddings in a separate thread
print("Setting up embeddings...")
threading.Thread(target=setup_embeddings).start()

# Initialize NVML
try:
    pynvml.nvmlInit()
    print("NVML initialized successfully")
except Exception as e:
    print(f"Failed to initialize NVML: {e}")

# Store baseline and historical stats
baseline_stats = None
stats_history = deque(maxlen=30)  # Store 30 seconds of data
query_in_progress = False
peak_stats = {
    'cpu': 0,
    'memory': 0,
    'gpu_load': 0,
    'gpu_memory': 0,
    'gpu_temp': 0,
    'gpu_temp_actual': 0  # Store actual temperature
}

def get_system_stats():
    global query_in_progress, peak_stats, baseline_stats
    try:
        # CPU stats - use interval=0.1 for more responsive reading
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_used = memory.used / (1024 * 1024 * 1024)  # Convert to GB
        memory_total = memory.total / (1024 * 1024 * 1024)  # Convert to GB
        
        # GPU stats using NVML
        gpu_stats = []
        try:
            deviceCount = pynvml.nvmlDeviceGetCount()
            for i in range(deviceCount):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # Get device name - handle both bytes and str return types
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                # Get utilization rates
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_util = utilization.gpu  # This is the actual GPU utilization percentage
                
                # Get memory info
                memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_used = memory_info.used / (1024 * 1024)  # Convert to MB
                memory_total = memory_info.total / (1024 * 1024)  # Convert to MB
                
                # Get temperature with better error handling
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                except pynvml.NVMLError as e:
                    print(f"NVML temperature error: {e}")
                    temperature = 0
                except Exception as e:
                    print(f"General temperature error: {e}")
                    temperature = 0
                
                gpu_stats.append({
                    'name': name,
                    'load': gpu_util,
                    'memory_used': round(memory_used, 2),
                    'memory_total': round(memory_total, 2),
                    'temperature': temperature
                })
                
                # Update peak stats if query is in progress
                if query_in_progress:
                    peak_stats['gpu_temp'] = max(peak_stats['gpu_temp'], temperature)
                
        except Exception as e:
            print(f"Error getting GPU stats: {e}")
            gpu_stats = []
            
        stats = {
            'cpu_percent': round(cpu_percent, 2),
            'memory_used': round(memory_used, 2),
            'memory_total': round(memory_total, 2),
            'memory_percent': round(memory.percent, 2),
            'gpu_stats': gpu_stats,
            'timestamp': time.time()
        }

        # Calculate relative changes if baseline exists and we have GPU stats
        if baseline_stats and gpu_stats:
            stats['relative'] = {
                'gpu': [{
                    'load': round(gpu_stats[i]['load'] - baseline_stats['gpu_stats'][i]['load'], 2),
                    'memory': round(gpu_stats[i]['memory_used'] - baseline_stats['gpu_stats'][i]['memory_used'], 2),
                    'temp': round(gpu_stats[i]['temperature'] - baseline_stats['gpu_stats'][i]['temperature'], 2)
                } for i in range(len(gpu_stats))]
            }

            # Update peak stats if query is in progress
            if query_in_progress and len(gpu_stats) > 0:
                peak_stats['cpu'] = max(peak_stats['cpu'], cpu_percent)
                peak_stats['memory'] = max(peak_stats['memory'], memory.percent)
                peak_stats['gpu_load'] = max(peak_stats['gpu_load'], gpu_stats[0]['load'])
                peak_stats['gpu_memory'] = max(peak_stats['gpu_memory'], gpu_stats[0]['memory_used'])
                peak_stats['gpu_temp'] = max(peak_stats['gpu_temp'], gpu_stats[0]['temperature'])
                peak_stats['gpu_temp_actual'] = max(peak_stats['gpu_temp_actual'], gpu_stats[0]['temperature'])

        # Add to history
        stats_history.append(stats)
        
        return stats
    except Exception as e:
        print(f"Error getting system stats: {str(e)}")
        return None

@app.route('/start-monitoring', methods=['POST'])
def start_monitoring():
    global baseline_stats, query_in_progress, peak_stats
    baseline_stats = get_system_stats()
    query_in_progress = True
    # Reset peak stats
    peak_stats = {
        'cpu': 0,
        'memory': 0,
        'gpu_load': 0,
        'gpu_memory': 0,
        'gpu_temp': 0,
        'gpu_temp_actual': 0  # Reset actual temperature
    }
    return jsonify({'status': 'success'})

@app.route('/stop-monitoring', methods=['POST'])
def stop_monitoring():
    global query_in_progress
    query_in_progress = False
    return jsonify({'status': 'success', 'peak_stats': peak_stats})

@app.route('/system-stats')
def system_stats():
    stats = get_system_stats()
    if stats:
        return jsonify(stats)
    return jsonify({'error': 'Failed to get system stats'}), 500

@app.route('/stats-history')
def get_stats_history():
    return jsonify(list(stats_history))

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'question' not in data:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        answer = answer_query(data['question'])
        return jsonify({
            'answer': answer,
            'peak_stats': peak_stats
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/', methods=['GET'])
def home():
    return """
    <html>
        <head>
            <title>NaradMuni ChatBOT</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { 
                    font-family: Arial, sans-serif; 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    padding: 20px;
                    background-color: #f5f5f5;
                }
                .container {
                    display: flex;
                    flex-direction: column;
                    gap: 20px;
                }
                .chat-section {
                    width: 100%;
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .monitor-section {
                    width: 100%;
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                .graphs-container {
                    display: grid;
                    grid-template-columns: repeat(2, 1fr);
                    gap: 20px;
                    margin-bottom: 20px;
                }
                #chat-container { margin-top: 20px; }
                #question { 
                    width: 80%; 
                    padding: 10px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    margin-right: 10px;
                }
                button { 
                    padding: 10px 20px; 
                    background: #007bff; 
                    color: white; 
                    border: none; 
                    border-radius: 5px;
                    cursor: pointer; 
                }
                button:hover { background: #0056b3; }
                #answer { 
                    margin-top: 20px; 
                    padding: 15px; 
                    border: 1px solid #ddd; 
                    border-radius: 5px; 
                    white-space: pre-wrap;
                    min-height: 100px;
                }
                .error { color: red; }
                .loading { color: #666; }
                .stat-card {
                    background: #f8f9fa;
                    padding: 15px;
                    margin: 10px 0;
                    border-radius: 5px;
                    border: 1px solid #dee2e6;
                }
                .peak-stats {
                    margin-top: 20px;
                    padding: 15px;
                    background: #e9ecef;
                    border-radius: 5px;
                }
                .peak-value {
                    color: #dc3545;
                    font-weight: bold;
                }
                .chart {
                    width: 100%;
                    height: 350px;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="chat-section">
                    <h1>NaradMuni ChatBOT</h1>
                    <div id="chat-container">
                        <input type="text" id="question" placeholder="Ask a question about GBU..." onkeypress="handleKeyPress(event)">
                        <button onclick="askQuestion()">Ask</button>
                        <div id="answer"></div>
                    </div>
                </div>
                <div class="monitor-section">
                    <h2>System Monitor</h2>
                    <div class="graphs-container">
                        <div id="cpuChart" class="chart"></div>
                        <div id="gpuChart" class="chart"></div>
                        <div id="tempChart" class="chart"></div>
                    </div>
                    <div id="peak-stats" class="peak-stats">
                        <h3>Peak Usage</h3>
                        <div>Loading...</div>
                    </div>
                </div>
            </div>
            <script>
                let cpuData = {
                    x: [],
                    y: [],
                    name: 'CPU Usage',
                    type: 'scatter',
                    line: {
                        color: '#2196F3',
                        width: 2
                    }
                };

                let gpuData = {
                    x: [],
                    y: [],
                    name: 'GPU Usage',
                    type: 'scatter',
                    line: {
                        color: '#4CAF50',
                        width: 2
                    }
                };

                let tempData = {
                    x: [],
                    y: [],
                    name: 'GPU Temperature',
                    type: 'scatter',
                    line: {
                        color: '#FF5722',
                        width: 2
                    }
                };

                const plotConfig = {
                    displayModeBar: false,
                    responsive: true
                };

                const plotLayout = {
                    height: 350,
                    margin: {t: 30, b: 50, l: 50, r: 30},
                    xaxis: {
                        title: 'Time (seconds)',
                        showgrid: true,
                        zeroline: false,
                        gridcolor: '#E0E0E0'
                    },
                    yaxis: {
                        title: 'Usage (%)',
                        range: [0, 100],
                        showgrid: true,
                        zeroline: true,
                        gridcolor: '#E0E0E0'
                    },
                    showlegend: false,
                    plot_bgcolor: '#FFFFFF',
                    paper_bgcolor: '#FFFFFF'
                };

                const tempLayout = {
                    ...plotLayout,
                    yaxis: {
                        title: 'Temperature (°C)',
                        range: [20, 100],  // Most GPUs operate between 30-90°C
                        showgrid: true,
                        zeroline: true,
                        gridcolor: '#E0E0E0'
                    }
                };

                let startTime;
                let monitoring = false;
                let lastPeakStats = null;
                let updateInterval;

                // Initialize plots
                Plotly.newPlot('cpuChart', [cpuData], {
                    ...plotLayout,
                    title: 'CPU Usage'
                }, plotConfig);

                Plotly.newPlot('gpuChart', [gpuData], {
                    ...plotLayout,
                    title: 'GPU Usage'
                }, plotConfig);

                Plotly.newPlot('tempChart', [tempData], {
                    ...tempLayout,
                    title: 'GPU Temperature'
                }, plotConfig);

                function updatePeakStats(peakStats) {
                    lastPeakStats = peakStats;
                    const peakStatsDiv = document.getElementById('peak-stats');
                    peakStatsDiv.innerHTML = `
                        <h3>Peak Usage</h3>
                        <div>CPU: <span class="peak-value">${peakStats.cpu.toFixed(1)}%</span></div>
                        <div>Memory: <span class="peak-value">${peakStats.memory.toFixed(1)}%</span></div>
                        <div>GPU Load: <span class="peak-value">${peakStats.gpu_load.toFixed(1)}%</span></div>
                        <div>GPU Memory: <span class="peak-value">${peakStats.gpu_memory.toFixed(1)} MB</span></div>
                        <div>GPU Temperature: <span class="peak-value">${peakStats.gpu_temp_actual.toFixed(1)}°C</span></div>
                    `;
                }

                function clearGraphs() {
                    cpuData.x = [];
                    cpuData.y = [];
                    gpuData.x = [];
                    gpuData.y = [];
                    tempData.x = [];
                    tempData.y = [];
                    
                    Plotly.react('cpuChart', [cpuData], {
                        ...plotLayout,
                        title: 'CPU Usage'
                    }, plotConfig);

                    Plotly.react('gpuChart', [gpuData], {
                        ...plotLayout,
                        title: 'GPU Usage'
                    }, plotConfig);

                    Plotly.react('tempChart', [tempData], {
                        ...tempLayout,
                        title: 'GPU Temperature'
                    }, plotConfig);
                }

                function startMonitoring() {
                    if (updateInterval) {
                        clearInterval(updateInterval);
                    }
                    
                    monitoring = true;
                    startTime = Date.now();
                    
                    updateInterval = setInterval(async () => {
                        if (!monitoring) return;
                        
                        try {
                            const response = await fetch('/system-stats');
                            const stats = await response.json();
                            if (!stats.error) {
                                const time = (Date.now() - startTime) / 1000;
                                
                                // Update CPU data
                                cpuData.x.push(time);
                                cpuData.y.push(stats.cpu_percent);
                                
                                // Update GPU data if available
                                if (stats.gpu_stats && stats.gpu_stats.length > 0) {
                                    gpuData.x.push(time);
                                    gpuData.y.push(stats.gpu_stats[0].load);
                                    
                                    // Update temperature data
                                    tempData.x.push(time);
                                    tempData.y.push(stats.gpu_stats[0].temperature);
                                }
                                
                                // Keep only last 30 seconds
                                const cutoff = time - 30;
                                while (cpuData.x[0] < cutoff) {
                                    cpuData.x.shift();
                                    cpuData.y.shift();
                                }
                                while (gpuData.x[0] < cutoff) {
                                    gpuData.x.shift();
                                    gpuData.y.shift();
                                }
                                while (tempData.x[0] < cutoff) {
                                    tempData.x.shift();
                                    tempData.y.shift();
                                }
                                
                                // Update plots
                                Plotly.update('cpuChart', {
                                    x: [cpuData.x],
                                    y: [cpuData.y]
                                });
                                
                                Plotly.update('gpuChart', {
                                    x: [gpuData.x],
                                    y: [gpuData.y]
                                });

                                Plotly.update('tempChart', {
                                    x: [tempData.x],
                                    y: [tempData.y]
                                });
                            }
                        } catch (error) {
                            console.error('Error updating charts:', error);
                        }
                    }, 500);
                }

                function stopMonitoring() {
                    monitoring = false;
                    if (updateInterval) {
                        clearInterval(updateInterval);
                        updateInterval = null;
                    }
                }

                async function askQuestion() {
                    const question = document.getElementById('question').value;
                    if (!question.trim()) return;
                    
                    const answerDiv = document.getElementById('answer');
                    answerDiv.className = 'loading';
                    answerDiv.textContent = 'Thinking...';
                    
                    // Clear graphs but keep last peak stats visible
                    clearGraphs();
                    if (lastPeakStats) {
                        updatePeakStats(lastPeakStats);
                    }
                    
                    // Start monitoring
                    await fetch('/start-monitoring', {method: 'POST'});
                    startMonitoring();
                    
                    try {
                        const response = await fetch('/chat', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({question: question})
                        });
                        
                        const data = await response.json();
                        answerDiv.className = '';
                        
                        if (data.error) {
                            answerDiv.className = 'error';
                            answerDiv.textContent = data.error;
                        } else {
                            answerDiv.textContent = data.answer;
                            if (data.peak_stats) {
                                updatePeakStats(data.peak_stats);
                            }
                        }
                    } catch (error) {
                        answerDiv.className = 'error';
                        answerDiv.textContent = 'Error: Could not connect to the server. Please try again.';
                    }
                    
                    // Stop monitoring
                    stopMonitoring();
                    const stopResponse = await fetch('/stop-monitoring', {method: 'POST'});
                    const stopData = await stopResponse.json();
                    if (stopData.peak_stats) {
                        updatePeakStats(stopData.peak_stats);
                    }
                }

                function handleKeyPress(event) {
                    if (event.key === 'Enter') {
                        askQuestion();
                    }
                }
            </script>
        </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) 