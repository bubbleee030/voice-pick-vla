(function () {
    "use strict";

    function byId(id) {
        return document.getElementById(id);
    }

    var panel = byId("panel-dataset-capture");
    if (!panel || typeof io !== "function") {
        return;
    }

    var elBadge = byId("dataset-capture-badge");
    var elSummary = byId("dataset-capture-summary");
    var elObject = byId("dataset-object-select");
    var elStart = byId("btn-dataset-start");
    var elStop = byId("btn-dataset-stop");
    var elFps = byId("dataset-fps");
    var elTelemetry = byId("dataset-telemetry-hz");
    var elDuration = byId("dataset-duration-s");
    var elNotes = byId("dataset-notes");
    var elProgressFill = byId("dataset-capture-progress-fill");
    var elProgressLabel = byId("dataset-capture-progress-label");

    var socket = io({ reconnection: true, reconnectionDelay: 1000 });

    function checkedStreams() {
        var ids = [
            "cam1-rgb",
            "cam2-rgb",
            "cam1-depth",
            "cam2-depth",
            "claw-rgb"
        ];
        return ids
            .filter(function (key) {
                var el = byId("dataset-stream-" + key);
                return !!(el && el.checked);
            })
            .map(function (key) {
                return key.replace(/-/g, "_");
            });
    }

    function setBadge(text, active) {
        if (!elBadge) return;
        elBadge.textContent = text;
        elBadge.className = "panel-badge" + (active ? " validation-active" : "");
    }

    function setProgress(pct, label) {
        var value = Math.max(0, Math.min(100, Number(pct || 0)));
        if (elProgressFill) {
            elProgressFill.style.width = value + "%";
        }
        if (elProgressLabel) {
            elProgressLabel.textContent = label || (Math.round(value) + "%");
        }
    }

    function updateSummary(state) {
        state = state || {};
        var running = !!state.running;
        var stopping = !!state.stopping;
        if (elStart) elStart.disabled = running;
        if (elStop) {
            elStop.disabled = !running || stopping;
            elStop.textContent = stopping ? "Stopping..." : "Stop";
        }
        if (running) {
            setBadge(stopping ? "Stopping" : "Recording", true);
            setProgress(state.progress_pct || 0, (state.status_message || state.stage || "Recording") + " (" + Math.round(state.progress_pct || 0) + "%)");
            if (elSummary) {
                elSummary.textContent =
                    "Object: " + (state.object || "--")
                    + " | Episode: " + (state.episode_index != null ? state.episode_index : "--")
                    + "\nDir: " + (state.output_dir || "--");
            }
            return;
        }
        setBadge("Idle", false);
        setProgress(state.progress_pct || (state.summary ? 100 : 0), state.summary ? "Completed (100%)" : "Idle");
        if (state.summary && elSummary) {
            elSummary.textContent =
                "Dir: " + (state.output_dir || "--")
                + "\nStop reason: " + ((state.summary && state.summary.stop_reason) || "--");
            return;
        }
        if (elSummary) {
            elSummary.textContent = "No dataset capture running";
        }
    }

    function loadObjects() {
        fetch("/api/objects")
            .then(function (r) { return r.json(); })
            .then(function (objects) {
                if (!elObject) return;
                var current = elObject.value;
                elObject.innerHTML = '<option value="">-- Select Object --</option>';
                Object.keys(objects || {}).forEach(function (key) {
                    var obj = objects[key] || {};
                    var zh = (obj.chinese && obj.chinese[0]) || key;
                    var en = (obj.english && obj.english[0]) || key;
                    var opt = document.createElement("option");
                    opt.value = key;
                    opt.textContent = zh + " / " + en;
                    elObject.appendChild(opt);
                });
                if (current) {
                    elObject.value = current;
                }
            })
            .catch(function () {});
    }

    function loadDefaults() {
        fetch("/api/config")
            .then(function (r) { return r.json(); })
            .then(function (cfg) {
                var ds = (((cfg || {}).teach || {}).dataset_capture) || {};
                if (elFps && ds.image_fps != null) elFps.value = ds.image_fps;
                if (elTelemetry && ds.telemetry_hz != null) elTelemetry.value = ds.telemetry_hz;
                var streams = ds.camera_streams || [];
                [
                    "cam1_rgb",
                    "cam2_rgb",
                    "cam1_depth",
                    "cam2_depth",
                    "claw_rgb"
                ].forEach(function (key) {
                    var el = byId("dataset-stream-" + key.replace(/_/g, "-"));
                    if (el) el.checked = streams.indexOf(key) >= 0;
                });
            })
            .catch(function () {});
    }

    if (elStart) {
        elStart.addEventListener("click", function () {
            var objectName = elObject && elObject.value ? elObject.value : "demo";
            var streams = checkedStreams();
            if (!streams.length) {
                if (elSummary) elSummary.textContent = "Select at least one stream";
                return;
            }
            socket.emit("dataset_capture_start", {
                object: objectName,
                streams: streams,
                fps: parseFloat(elFps && elFps.value ? elFps.value : "6"),
                telemetry_hz: parseFloat(elTelemetry && elTelemetry.value ? elTelemetry.value : "10"),
                duration_s: parseFloat(elDuration && elDuration.value ? elDuration.value : "0"),
                notes: elNotes && elNotes.value ? elNotes.value : ""
            });
        });
    }

    if (elStop) {
        elStop.addEventListener("click", function () {
            updateSummary({ running: true, stopping: true, object: elObject && elObject.value ? elObject.value : "--" });
            socket.emit("dataset_capture_stop");
        });
    }

    socket.on("dataset_capture_state", updateSummary);
    socket.on("config_reloaded", function (payload) {
        var cfg = payload && payload.config ? payload.config : payload;
        var ds = (((cfg || {}).teach || {}).dataset_capture) || {};
        if (elFps && ds.image_fps != null) elFps.value = ds.image_fps;
        if (elTelemetry && ds.telemetry_hz != null) elTelemetry.value = ds.telemetry_hz;
    });
    socket.on("objects_catalog", function (objects) {
        if (!elObject || !objects) return;
        var current = elObject.value;
        elObject.innerHTML = '<option value="">-- Select Object --</option>';
        Object.keys(objects).forEach(function (key) {
            var obj = objects[key] || {};
            var zh = (obj.chinese && obj.chinese[0]) || key;
            var en = (obj.english && obj.english[0]) || key;
            var opt = document.createElement("option");
            opt.value = key;
            opt.textContent = zh + " / " + en;
            elObject.appendChild(opt);
        });
        if (current) elObject.value = current;
    });

    loadObjects();
    loadDefaults();
    updateSummary({ running: false });
})();
