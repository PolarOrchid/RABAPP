document.addEventListener('DOMContentLoaded', function() {
    // Log versions to ensure libraries are loaded
    console.log('Leaflet version:', L.version);
    console.log('MarkerClusterGroup exists:', typeof L.markerClusterGroup);

    var map = L.map('mapid').setView([0, 0], 2);

    L.tileLayer('https://{s}.tile.openstreetmap.fr/hot/{z}/{x}/{y}.png', {
        attribution: 'Benjamin',
        maxZoom: 19
    }).addTo(map);

    fetch('/get_photos', { credentials: 'same-origin' })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok: ' + response.statusText);
            }
            return response.json();
        })
        .then(data => {
            console.log('Data received from /get_photos:', data);
            if (data.length === 0) {
                alert('No photos with GPS data found.');
                return;
            }

            // Initialize the marker cluster group
            var markers = L.markerClusterGroup();

            data.forEach(function(photo) {
                console.log('Processing photo:', photo);
                if (photo.latitude != null && photo.longitude != null) {
                    var lat = parseFloat(photo.latitude);
                    var lon = parseFloat(photo.longitude);

                    // Initialize marker here
                    var marker = L.marker([lat, lon]);

                    // Use the filename provided by the server
                    var imageUrl = uploadsUrl + photo.filename;

                    var photoUrl = photoUrlBase + photo.id;
                    console.log('Adding marker at:', lat, lon);
                    console.log('Photo URL:', photoUrl);
                    console.log('Image URL:', imageUrl);

                    // Bind the popup with the image to the marker
                    marker.bindPopup("<a href='" + photoUrl + "'><img src='" + imageUrl + "' width='150'></a>");
                    
                    // Add marker to the cluster group
                    markers.addLayer(marker);
                } else {
                    console.log('Photo has invalid coordinates:', photo);
                }
            });
            
            console.log('Total markers added:', markers.getLayers().length);
            map.addLayer(markers);

            // Fit the map to the bounds of the markers
            if (markers.getLayers().length > 0) {
                map.fitBounds(markers.getBounds());
            } else {
                console.log('No markers to display.');
            }
        })
        .catch(function(error) {
            console.error('Error fetching photos:', error);
            alert('An error occurred while fetching photos: ' + error.message);
        });
});
