document.addEventListener('DOMContentLoaded', function() {
    // Hide flash messages after 5 seconds
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(function(alert) {
        setTimeout(function() {
            alert.style.opacity = '0';
            setTimeout(function() {
                alert.style.display = 'none';
            }, 500);
        }, 5000);
    });

    // Initialize footer positioning if needed
    const footer = document.querySelector('footer');
    if (footer) {
        function adjustFooter() {
            const bodyHeight = document.body.offsetHeight;
            const windowHeight = window.innerHeight;
            footer.style.position = bodyHeight < windowHeight ? 'absolute' : 'relative';
            footer.style.bottom = bodyHeight < windowHeight ? '0' : 'auto';
        }
        
        // Adjust footer initially and when window is resized
        adjustFooter();
        window.addEventListener('resize', adjustFooter);
    }
});