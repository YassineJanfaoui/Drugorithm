document.addEventListener('DOMContentLoaded', function() {
    // Handle form submission and show loading spinner
    const uploadForm = document.getElementById('upload-form');
    const loadingIndicator = document.getElementById('loading');
    const analyzeBtn = document.getElementById('analyze-btn');
    
    if (uploadForm) {
        uploadForm.addEventListener('submit', function() {
            // Show loading indicator
            if (loadingIndicator) {
                loadingIndicator.classList.remove('d-none');
            }
            
            // Disable submit button to prevent multiple submissions
            if (analyzeBtn) {
                analyzeBtn.disabled = true;
                analyzeBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            }
        });
    }
    
    // File input validation
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                // Check file type
                const validTypes = ['image/jpeg', 'image/jpg', 'image/png'];
                if (!validTypes.includes(file.type)) {
                    alert('Invalid file type. Please upload a JPG, JPEG, or PNG image.');
                    fileInput.value = '';
                    return;
                }
                
                // Check file size (max 5MB)
                const maxSize = 5 * 1024 * 1024; // 5MB in bytes
                if (file.size > maxSize) {
                    alert('File is too large. Maximum size is 5MB.');
                    fileInput.value = '';
                    return;
                }
            }
        });
    }
    
    // Initialize tooltips if Bootstrap is present
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function(tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
    
    // Auto dismiss alerts after 5 seconds
    const alerts = document.querySelectorAll('.alert:not(.alert-warning, .alert-info)');
    alerts.forEach(function(alert) {
        setTimeout(function() {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });
});
