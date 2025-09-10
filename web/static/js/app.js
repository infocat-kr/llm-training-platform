// LLM Training Platform - Main JavaScript

// Global variables
let currentUser = 'default';
let isTraining = false;
let isGenerating = false;

// Initialize app when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeApp();
    setupEventListeners();
    updateSystemStatus();
});

// Initialize application
function initializeApp() {
    console.log('LLM Training Platform initialized');
    
    // Check if model is loaded
    checkModelStatus();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Initialize progress bars
    initializeProgressBars();
    
    // Setup auto-refresh for system status
    setInterval(updateSystemStatus, 5000);
}

// Setup event listeners
function setupEventListeners() {
    // Global keyboard shortcuts
    document.addEventListener('keydown', handleKeyboardShortcuts);
    
    // Window events
    window.addEventListener('beforeunload', handleBeforeUnload);
    window.addEventListener('resize', handleWindowResize);
    
    // Form validation
    setupFormValidation();
}

// Handle keyboard shortcuts
function handleKeyboardShortcuts(e) {
    // Ctrl/Cmd + Enter to submit forms
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        const activeForm = document.querySelector('form:focus-within');
        if (activeForm) {
            e.preventDefault();
            activeForm.dispatchEvent(new Event('submit'));
        }
    }
    
    // Escape to close modals
    if (e.key === 'Escape') {
        const openModal = document.querySelector('.modal.show');
        if (openModal) {
            const modal = bootstrap.Modal.getInstance(openModal);
            if (modal) {
                modal.hide();
            }
        }
    }
}

// Handle before unload
function handleBeforeUnload(e) {
    if (isTraining) {
        e.preventDefault();
        e.returnValue = '훈련이 진행 중입니다. 정말로 페이지를 떠나시겠습니까?';
        return e.returnValue;
    }
}

// Handle window resize
function handleWindowResize() {
    // Adjust chat container height
    const chatContainer = document.getElementById('chat-container');
    if (chatContainer) {
        const windowHeight = window.innerHeight;
        const headerHeight = document.querySelector('.navbar').offsetHeight;
        const footerHeight = document.querySelector('footer').offsetHeight;
        const availableHeight = windowHeight - headerHeight - footerHeight - 200;
        chatContainer.style.height = Math.max(300, availableHeight) + 'px';
    }
}

// Check model status
async function checkModelStatus() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        updateModelStatusIndicator(data.model_loaded);
        
        if (data.model_loaded) {
            loadModelInfo();
        }
    } catch (error) {
        console.error('Error checking model status:', error);
        updateModelStatusIndicator(false);
    }
}

// Update model status indicator
function updateModelStatusIndicator(isLoaded) {
    const statusElement = document.getElementById('model-status');
    if (statusElement) {
        if (isLoaded) {
            statusElement.innerHTML = '<i class="fas fa-circle text-success me-1"></i>모델 로드됨';
        } else {
            statusElement.innerHTML = '<i class="fas fa-circle text-warning me-1"></i>모델 미로드';
        }
    }
}

// Load model information
async function loadModelInfo() {
    try {
        const response = await fetch('/api/model/info');
        const data = await response.json();
        
        if (data.status === 'success') {
            displayModelInfo(data);
        }
    } catch (error) {
        console.error('Error loading model info:', error);
    }
}

// Display model information
function displayModelInfo(modelInfo) {
    // Update model info in UI if elements exist
    const modelTypeElement = document.getElementById('model-type-display');
    if (modelTypeElement) {
        modelTypeElement.textContent = modelInfo.model_type.toUpperCase();
    }
    
    const paramCountElement = document.getElementById('param-count');
    if (paramCountElement) {
        paramCountElement.textContent = formatNumber(modelInfo.total_parameters);
    }
}

// Update system status
async function updateSystemStatus() {
    try {
        const response = await fetch('/api/health');
        const data = await response.json();
        
        // Update connection status
        const connectionStatus = document.getElementById('connection-status');
        if (connectionStatus) {
            if (data.status === 'healthy') {
                connectionStatus.innerHTML = '<i class="fas fa-circle me-1"></i>연결됨';
                connectionStatus.className = 'badge bg-success';
            } else {
                connectionStatus.innerHTML = '<i class="fas fa-circle me-1"></i>연결 끊김';
                connectionStatus.className = 'badge bg-danger';
            }
        }
        
        // Update system metrics (simulated)
        updateSystemMetrics();
        
    } catch (error) {
        console.error('Error updating system status:', error);
        
        const connectionStatus = document.getElementById('connection-status');
        if (connectionStatus) {
            connectionStatus.innerHTML = '<i class="fas fa-circle me-1"></i>연결 끊김';
            connectionStatus.className = 'badge bg-danger';
        }
    }
}

// Update system metrics (simulated)
function updateSystemMetrics() {
    // GPU usage
    const gpuUsage = document.getElementById('gpu-usage');
    if (gpuUsage) {
        const usage = Math.random() * 100;
        gpuUsage.style.width = usage + '%';
        gpuUsage.textContent = Math.round(usage) + '%';
    }
    
    // Memory usage
    const memoryUsage = document.getElementById('memory-usage');
    if (memoryUsage) {
        const usage = Math.random() * 100;
        memoryUsage.style.width = usage + '%';
        memoryUsage.textContent = Math.round(usage) + '%';
    }
    
    // Disk usage
    const diskUsage = document.getElementById('disk-usage');
    if (diskUsage) {
        const usage = Math.random() * 100;
        diskUsage.style.width = usage + '%';
        diskUsage.textContent = Math.round(usage) + '%';
    }
    
    // GPU temperature
    const gpuTemp = document.getElementById('gpu-temp');
    if (gpuTemp) {
        const temp = Math.random() * 20 + 60;
        gpuTemp.style.width = (temp / 80 * 100) + '%';
        gpuTemp.textContent = Math.round(temp) + '°C';
    }
}

// Initialize tooltips
function initializeTooltips() {
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });
}

// Initialize progress bars
function initializeProgressBars() {
    const progressBars = document.querySelectorAll('.progress-bar');
    progressBars.forEach(bar => {
        const width = bar.getAttribute('data-width') || '0';
        bar.style.width = width + '%';
    });
}

// Setup form validation
function setupFormValidation() {
    const forms = document.querySelectorAll('form');
    forms.forEach(form => {
        form.addEventListener('submit', function(e) {
            if (!form.checkValidity()) {
                e.preventDefault();
                e.stopPropagation();
            }
            form.classList.add('was-validated');
        });
    });
}

// Utility functions
function formatNumber(num) {
    if (num >= 1000000000) {
        return (num / 1000000000).toFixed(1) + 'B';
    } else if (num >= 1000000) {
        return (num / 1000000).toFixed(1) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(1) + 'K';
    }
    return num.toString();
}

function formatTime(seconds) {
    const hours = Math.floor(seconds / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);
    
    if (hours > 0) {
        return `${hours}시간 ${minutes}분 ${secs}초`;
    } else if (minutes > 0) {
        return `${minutes}분 ${secs}초`;
    } else {
        return `${secs}초`;
    }
}

function showNotification(message, type = 'info', duration = 3000) {
    const notification = document.createElement('div');
    notification.className = `alert alert-${type} alert-dismissible fade show position-fixed`;
    notification.style.cssText = 'top: 20px; right: 20px; z-index: 9999; min-width: 300px;';
    notification.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    document.body.appendChild(notification);
    
    setTimeout(() => {
        if (notification.parentNode) {
            notification.remove();
        }
    }, duration);
}

function showLoading(element, text = '로딩 중...') {
    const originalContent = element.innerHTML;
    element.innerHTML = `
        <span class="spinner-border spinner-border-sm me-2" role="status"></span>
        ${text}
    `;
    element.disabled = true;
    
    return function hideLoading() {
        element.innerHTML = originalContent;
        element.disabled = false;
    };
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function() {
        const args = arguments;
        const context = this;
        if (!inThrottle) {
            func.apply(context, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

// API helper functions
async function apiRequest(url, options = {}) {
    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };
    
    const mergedOptions = { ...defaultOptions, ...options };
    
    try {
        const response = await fetch(url, mergedOptions);
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.message || 'API request failed');
        }
        
        return data;
    } catch (error) {
        console.error('API request error:', error);
        throw error;
    }
}

// Export functions for global use
window.LLMPlatform = {
    showNotification,
    showLoading,
    formatNumber,
    formatTime,
    apiRequest,
    checkModelStatus,
    updateSystemStatus
};
