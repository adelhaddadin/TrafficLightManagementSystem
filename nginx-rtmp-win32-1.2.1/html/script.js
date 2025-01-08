document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM loaded. Initializing everything...");

    // ---------------------------
    // 1) Video buffering logic
    // ---------------------------
    const video = document.getElementById('video');
    const videoContainer = document.getElementById('videoContainer');
    let lastFrame = null;

    console.log("Setting up video buffering event listeners...");
    video.addEventListener('timeupdate', function() {
        // This fires often while video is playing
        console.log("[video] timeupdate event triggered.");
        if (!video.paused) {
            try {
                const canvas = document.createElement('canvas');
                canvas.width = video.videoWidth;
                canvas.height = video.videoHeight;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0);
                lastFrame = canvas;
                // We won't console.log the entire frame or canvas
            } catch (e) {
                console.error('[video] Error capturing frame:', e);
            }
        }
    });

    video.addEventListener('waiting', function() {
        console.log("[video] waiting event: video is buffering...");
        if (lastFrame) {
            videoContainer.classList.add('buffering');
        }
    });

    video.addEventListener('playing', function() {
        console.log("[video] playing event: video resumed playback.");
        videoContainer.classList.remove('buffering');
    });

    // ---------------------------
    // 2) HLS video playback logic
    // ---------------------------
    console.log("Setting up HLS playback logic...");
    const videoSrc = 'http://localhost:8080/hls/streamkey.m3u8';
    if (Hls.isSupported()) {
        console.log("Hls.js is supported. Creating Hls instance...");
        const hls = new Hls({
            maxBufferLength: 30,
            maxMaxBufferLength: 60,
            maxBufferSize: 60 * 1000 * 1000,
            highBufferWatchdogPeriod: 1,
            lowBufferWatchdogPeriod: 0.5
        });

        hls.loadSource(videoSrc);
        hls.attachMedia(video);

        hls.on(Hls.Events.MANIFEST_PARSED, function() {
            console.log("[HLS] MANIFEST_PARSED event: Attempting to auto-play the video.");
            video.play().catch(function(error) {
                console.log("[HLS] Play failed:", error);
            });
        });

        hls.on(Hls.Events.ERROR, function(event, data) {
            console.log("[HLS] ERROR event:", data);
            if (data.fatal) {
                switch (data.type) {
                    case Hls.ErrorTypes.NETWORK_ERROR:
                        console.error("[HLS] Network error encountered, trying to recover...");
                        hls.startLoad();
                        break;
                    case Hls.ErrorTypes.MEDIA_ERROR:
                        console.error("[HLS] Media error encountered, trying to recover...");
                        hls.recoverMediaError();
                        break;
                    default:
                        console.error("[HLS] Unrecoverable error encountered, destroying Hls instance.");
                        hls.destroy();
                        break;
                }
            }
        });
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
        // Safari supports HLS natively
        console.log("HLS supported natively by Safari. Using direct video.src...");
        video.src = videoSrc;
        video.addEventListener('loadedmetadata', function() {
            console.log("[video] loadedmetadata event: Attempting to play...");
            video.play().catch(function(error) {
                console.log("[video] Play failed:", error);
            });
        });
    } else {
        console.warn("Neither Hls.js nor native HLS is supported. Video might not play.");
    }

    // ---------------------------
    // 3) Chart.js setup
    // ---------------------------
    console.log("Initializing Chart.js charts...");
    const vehicleCtx = document.getElementById('vehicleChart').getContext('2d');
    const vehicleChart = new Chart(vehicleCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'Vehicle Count',
                data: [],
                borderColor: '#8884d8',
                borderWidth: 2,
                fill: false
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    title: { display: true, text: 'Simulation Time (s)' },
                    ticks: { autoSkip: true, maxTicksLimit: 10 }
                },
                y: {
                    title: { display: true, text: 'Count' },
                    beginAtZero: true
                }
            }
        }
    });

    const co2Ctx = document.getElementById('co2Chart').getContext('2d');
    const co2Chart = new Chart(co2Ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'CO2 (g)',
                data: [],
                borderColor: '#8884d8',
                borderWidth: 2,
                fill: false
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    title: { display: true, text: 'Simulation Time (s)' },
                    ticks: { autoSkip: true, maxTicksLimit: 10 }
                },
                y: {
                    title: { display: true, text: 'CO2 (g)' },
                    beginAtZero: true
                }
            }
        }
    });

    const noxCtx = document.getElementById('noxChart').getContext('2d');
    const noxChart = new Chart(noxCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'NOx (g)',
                data: [],
                borderColor: '#82ca9d',
                borderWidth: 2,
                fill: false
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    title: { display: true, text: 'Simulation Time (s)' },
                    ticks: { autoSkip: true, maxTicksLimit: 10 }
                },
                y: {
                    title: { display: true, text: 'NOx (g)' },
                    beginAtZero: true
                }
            }
        }
    });

    const pmxCtx = document.getElementById('pmxChart').getContext('2d');
    const pmxChart = new Chart(pmxCtx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [{
                label: 'PM10 (g)',
                data: [],
                borderColor: '#ffc658',
                borderWidth: 2,
                fill: false
            }]
        },
        options: {
            responsive: true,
            scales: {
                x: {
                    title: { display: true, text: 'Simulation Time (s)' },
                    ticks: { autoSkip: true, maxTicksLimit: 10 }
                },
                y: {
                    title: { display: true, text: 'PM10 (g)' },
                    beginAtZero: true
                }
            }
        }
    });

    // Keep references for easy updating
    const charts = [vehicleChart, co2Chart, noxChart, pmxChart];
    console.log("Charts initialized:", charts);

    // ---------------------------
    // 4) Notification center
    // ---------------------------
    const notifications = document.getElementById('notifications');
    console.log("Notifications container found:", notifications);

    // ---------------------------
    // 5) Data fetching & updates
    // ---------------------------
    async function fetchData() {
        console.log("fetchData() called...");
        try {
            // Fetch emissions, vehicle, and occupancy data in parallel
            console.log("Requesting data from /emissions, /vehicles, and /occupancy endpoints...");
            const [emissionsRes, vehiclesRes, occupancyRes] = await Promise.all([
                fetch("http://localhost:8000/emissions"),
                fetch("http://localhost:8000/vehicles"),
                fetch("http://localhost:8000/occupancy")
            ]);

            console.log("Responses received. Checking statuses...");
            console.log("emissionsRes.ok:", emissionsRes.ok, "status:", emissionsRes.status);
            console.log("vehiclesRes.ok:", vehiclesRes.ok, "status:", vehiclesRes.status);
            console.log("occupancyRes.ok:", occupancyRes.ok, "status:", occupancyRes.status);

            if (!emissionsRes.ok || !vehiclesRes.ok || !occupancyRes.ok) {
                throw new Error('One or more network responses were not OK.');
            }

            // Parse JSON
            console.log("Parsing JSON from responses...");
            const emissionsData = await emissionsRes.json();
            const vehiclesData = await vehiclesRes.json();
            const occupancyData = await occupancyRes.json();

            console.log("emissionsData:", emissionsData);
            console.log("vehiclesData:", vehiclesData);
            console.log("occupancyData:", occupancyData);

            // We assume emissionsData has a numeric 'time' field
            const simTime = parseFloat(emissionsData.time);
            console.log("Parsed simTime =", simTime);

            if (isNaN(simTime)) {
                console.warn("Skipping chart updates because simTime is NaN.");
                return; // skip if we can't parse time
            }

            // -------------- Update CHARTS --------------
            console.log("Updating charts with new data...");

            // 1) Vehicle chart
            vehicleChart.data.labels.push(simTime);
            vehicleChart.data.datasets[0].data.push(vehiclesData.vehicle_count);
            if (vehicleChart.data.labels.length > 20) {
                vehicleChart.data.labels.shift();
                vehicleChart.data.datasets[0].data.shift();
            }
            vehicleChart.update();
            console.log("[vehicleChart] label/data updated -> (time:", simTime, ", value:", vehiclesData.vehicle_count, ")");

            // 2) CO2 chart
            co2Chart.data.labels.push(simTime);
            co2Chart.data.datasets[0].data.push(emissionsData.co2);
            if (co2Chart.data.labels.length > 20) {
                co2Chart.data.labels.shift();
                co2Chart.data.datasets[0].data.shift();
            }
            co2Chart.update();
            console.log("[co2Chart] label/data updated -> (time:", simTime, ", CO2:", emissionsData.co2, ")");

            // 3) NOx chart
            noxChart.data.labels.push(simTime);
            noxChart.data.datasets[0].data.push(emissionsData.nox);
            if (noxChart.data.labels.length > 20) {
                noxChart.data.labels.shift();
                noxChart.data.datasets[0].data.shift();
            }
            noxChart.update();
            console.log("[noxChart] label/data updated -> (time:", simTime, ", NOx:", emissionsData.nox, ")");

            // 4) PMx chart
            pmxChart.data.labels.push(simTime);
            pmxChart.data.datasets[0].data.push(emissionsData.pmx);
            if (pmxChart.data.labels.length > 20) {
                pmxChart.data.labels.shift();
                pmxChart.data.datasets[0].data.shift();
            }
            pmxChart.update();
            console.log("[pmxChart] label/data updated -> (time:", simTime, ", PMx:", emissionsData.pmx, ")");

            // -------------- Notifications --------------
            console.log("Creating new notification entry...");
            // Convert average speed from m/s to km/h (if SUMO default is m/s)
            const averageSpeedKmh = vehiclesData.average_speed * 3.6;
            const timeStr = new Date().toLocaleTimeString();

            // Build occupancy string: e.g. "A=0.12, B=0.50, C=0.03, D=0.00"
            let occupancyString = '';
            for (const lane of Object.keys(occupancyData)) {
                // If occupancyData[lane] is numeric, .toFixed(2) to see 2 decimals
                const val = occupancyData[lane];
                occupancyString += `${lane}=${val.toFixed(2)}, `;
            }
            // remove trailing comma & space
            occupancyString = occupancyString.replace(/,\s*$/, '');

            const notif = document.createElement("div");
            notif.className = "notification";
            notif.textContent = `[${timeStr}] ` +
                `Vehicle Count: ${vehiclesData.vehicle_count}, ` +
                `Avg Speed: ${averageSpeedKmh.toFixed(2)} km/h, ` +
                `Occupancy: ${occupancyString}`;
            notifications.insertBefore(notif, notifications.firstChild);

            console.log("Notification text added:", notif.textContent);

            // Keep only last 5 notifications
            if (notifications.children.length > 5) {
                notifications.removeChild(notifications.lastChild);
            }

        } catch (error) {
            console.error("Error fetching data:", error);

            const errorNotif = document.createElement("div");
            errorNotif.className = "notification";
            errorNotif.style.borderLeftColor = '#e53e3e'; // Red color for errors
            errorNotif.textContent = `[${new Date().toLocaleTimeString()}] Error fetching data: ${error.message}`;
            notifications.insertBefore(errorNotif, notifications.firstChild);

            if (notifications.children.length > 5) {
                notifications.removeChild(notifications.lastChild);
            }
        }
    }

    // Initial fetch
    console.log("Performing initial fetchData() call immediately...");
    fetchData();

    // Fetch updated data every 3 seconds
    console.log("Setting interval to fetch data every 3 seconds...");
    setInterval(fetchData, 3000);
});

