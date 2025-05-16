// Share Modal functionality
document.addEventListener('DOMContentLoaded', function() {
    // Only run this code if we're on a page that needs sharing functionality
    const shareButton = document.getElementById('shareButton');
    if (!shareButton) {
        return; // Exit early if no share button is found
    }

    // Initialize share functionality
    shareButton.addEventListener('click', function() {
        // Share functionality will be implemented here
        console.log('Share button clicked');
    });
}); 