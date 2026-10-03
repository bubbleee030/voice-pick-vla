(function () {
    "use strict";

    var socket = io({ reconnection: true, reconnectionDelay: 1000 });
    var objectsByKey = {};

    function log(message) {
        var box = document.getElementById("legacy-log-box");
        var row = document.createElement("div");
        row.className = "log-row info";
        row.textContent = "[" + new Date().toLocaleTimeString() + "] " + message;
        box.appendChild(row);
        box.scrollTop = box.scrollHeight;
    }

    function fetchJSON(url) {
        return fetch(url).then(function (res) { return res.json(); });
    }

    function renderObjects() {
        var select = document.getElementById("legacy-capture-object");
        select.innerHTML = "";
        Object.keys(objectsByKey).forEach(function (key) {
            var opt = document.createElement("option");
            var info = objectsByKey[key] || {};
            opt.value = key;
            opt.textContent = ((info.chinese && info.chinese[0]) || key) + " / " + key;
            select.appendChild(opt);
        });
    }

    function renderTeachList(items) {
        var select = document.getElementById("legacy-teach-replay-select");
        var list = document.getElementById("legacy-teach-list");
        select.innerHTML = "";
        list.innerHTML = "";
        items.forEach(function (item) {
            var opt = document.createElement("option");
            opt.value = item.id;
            opt.textContent = item.id;
            select.appendChild(opt);

            var row = document.createElement("div");
            row.className = "list-row";
            row.textContent = item.id + " (" + item.count + " wp)";
            list.appendChild(row);
        });
    }

    function selectedStreams() {
        return Array.prototype.slice.call(document.querySelectorAll(".stream-selectors input:checked")).map(function (el) {
            return el.value;
        });
    }

    socket.on("connect", function () { log("Connected"); });
    socket.on("disconnect", function () { log("Disconnected"); });
    socket.on("arm_log", function (data) { log((data.level || "INFO") + " " + (data.message || "")); });
    socket.on("teach_list", renderTeachList);
    socket.on("dataset_capture_state", function (data) {
        document.getElementById("legacy-capture-state").textContent = JSON.stringify(data, null, 2);
    });
    socket.on("system_health", function (data) {
        document.getElementById("legacy-health-box").textContent = JSON.stringify(data, null, 2);
    });

    document.getElementById("legacy-capture-start").addEventListener("click", function () {
        socket.emit("dataset_capture_start", {
            object: document.getElementById("legacy-capture-object").value,
            duration_s: parseFloat(document.getElementById("legacy-capture-duration").value || "0"),
            fps: parseFloat(document.getElementById("legacy-capture-fps").value || "6"),
            telemetry_hz: parseFloat(document.getElementById("legacy-capture-telemetry").value || "10"),
            streams: selectedStreams(),
            notes: "legacy_recorder"
        });
    });
    document.getElementById("legacy-capture-stop").addEventListener("click", function () {
        socket.emit("dataset_capture_stop");
    });
    document.getElementById("legacy-teach-start").addEventListener("click", function () {
        socket.emit("teach_start", { name: document.getElementById("legacy-teach-name").value || ("recording_" + Date.now()) });
    });
    document.getElementById("legacy-teach-waypoint").addEventListener("click", function () {
        socket.emit("teach_waypoint", { gripper: "none", speed: parseInt(document.getElementById("legacy-teach-speed").value || "30", 10) });
    });
    document.getElementById("legacy-teach-open").addEventListener("click", function () {
        socket.emit("teach_waypoint", { gripper: "open", speed: parseInt(document.getElementById("legacy-teach-speed").value || "30", 10) });
    });
    document.getElementById("legacy-teach-close").addEventListener("click", function () {
        socket.emit("teach_waypoint", { gripper: "close", speed: parseInt(document.getElementById("legacy-teach-speed").value || "30", 10) });
    });
    document.getElementById("legacy-teach-stop").addEventListener("click", function () {
        socket.emit("teach_stop");
    });
    document.getElementById("legacy-teach-replay").addEventListener("click", function () {
        socket.emit("teach_replay", { name: document.getElementById("legacy-teach-replay-select").value, replay_mode: "phase_axis_split" });
    });
    document.getElementById("legacy-arm-connect").addEventListener("click", function () { socket.emit("arm_connect"); });
    document.getElementById("legacy-arm-home").addEventListener("click", function () { socket.emit("arm_home"); });
    document.getElementById("legacy-arm-ready").addEventListener("click", function () { socket.emit("arm_ready"); });
    document.getElementById("legacy-arm-stop").addEventListener("click", function () { socket.emit("arm_stop"); });

    Array.prototype.forEach.call(document.querySelectorAll("img"), function (img) {
        img.addEventListener("error", function () {
            setTimeout(function () {
                img.src = img.src.split("?")[0] + "?t=" + Date.now();
            }, 1000);
        });
    });

    fetchJSON("/api/objects").then(function (objects) {
        objectsByKey = objects || {};
        renderObjects();
    }).catch(function () {
        log("Failed to load object list");
    });
    fetchJSON("/api/recordings").then(renderTeachList).catch(function () {});
})();
