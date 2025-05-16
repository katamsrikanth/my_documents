// Utility functions for the application

/**
 * Get the appropriate Bootstrap background color class for a status
 * @param {string} status - The status to get the color for
 * @returns {string} - The Bootstrap background color class
 */
function getStatusColor(status) {
    if (!status) return 'bg-secondary';
    
    status = status.toLowerCase();
    switch(status) {
        case 'scheduled':
        case 'open':
        case 'active':
            return 'bg-success';
        case 'completed':
        case 'closed':
            return 'bg-secondary';
        case 'cancelled':
        case 'inactive':
            return 'bg-danger';
        case 'pending':
        case 'in progress':
            return 'bg-primary';
        case 'on hold':
            return 'bg-warning';
        default:
            return 'bg-info';
    }
} 