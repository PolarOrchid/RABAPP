document.addEventListener('DOMContentLoaded', function() {
    var map = L.map('timeline-map').setView([0, 0], 2);
    L.tileLayer('https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', {
        attribution: 'Benjamin',
        maxZoom: 19
    }).addTo(map);

    fetch('/get_timeline_photos')
        .then(response => response.json())
        .then(data => {
            initializeTimeline(data, map);
        })
        .catch(error => console.error('Error fetching photos:', error));

    function initializeTimeline(locations, map) {
        var rangeInput = document.getElementById('timeline-range');
        var output = document.getElementById('timeline-output');

        // Extract timestamps from all photos
        var allTimestamps = locations.flatMap(loc => loc.photos.map(photo => new Date(photo.timestamp).getTime()));
        var minTimestamp = Math.min(...allTimestamps);
        var maxTimestamp = Math.max(...allTimestamps);

        rangeInput.min = minTimestamp;
        rangeInput.max = maxTimestamp;
        rangeInput.value = minTimestamp;

        var markers = [];
        
        // Iterate through locations and set up markers for each photo
        locations.forEach(function(loc) {
            loc.photos.forEach(function(photo) {
                var timestamp = new Date(photo.timestamp).getTime();
                
                // Create marker for the photo
                var marker = L.marker([photo.latitude, photo.longitude]);

                marker.bindPopup("<a href='" + photo.url + "'><img src='" + photo.filename + "' width='150'></a>");
                
                markers.push({
                    timestamp: timestamp,
                    marker: marker
                });
            });
        });

        var currentMarkers = [];

        rangeInput.oninput = function() {
            var selectedTimestamp = parseInt(this.value);
            output.innerHTML = new Date(selectedTimestamp).toLocaleString();

            // Remove current markers from the map
            currentMarkers.forEach(function(m) {
                map.removeLayer(m);
            });
            currentMarkers = [];

            // Add markers that have timestamps <= selectedTimestamp
            markers.forEach(function(m) {
                if (m.timestamp <= selectedTimestamp) {
                    m.marker.addTo(map);
                    currentMarkers.push(m.marker);
                }
            });

            // Adjust the map view if markers are present
            if (currentMarkers.length > 0) {
                var group = new L.featureGroup(currentMarkers);
                map.fitBounds(group.getBounds());
            } else {
                map.setView([0, 0], 2); // Reset map view if no markers
            }
        };

        // Trigger initial update
        rangeInput.oninput();
    }
});
