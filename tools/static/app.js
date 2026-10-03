(function () {
    "use strict";

    if (!window.io) {
        document.body.innerHTML = '<main style="padding:24px;font-family:monospace;color:#fff;background:#111;min-height:100vh"><h1>UI bootstrap failed</h1><p>socket.io client did not load. Open <a href="/legacy" style="color:#7dd3fc">/legacy</a> or check browser network access.</p></main>';
        return;
    }

    var socket = io({ reconnection: true, reconnectionDelay: 1000 });
    var currentLang = "zh-TW";
    var objectsByKey = {};
    var availableRecordings = {};
    var moduleConfig = {};
    var uiControlConfig = {};
    var fingerStageAction = null;
    var trajectory = { x: [], y: [], z: [] };
    var dashboard = document.getElementById("dashboard");
    var layoutPanels = Array.prototype.slice.call(document.querySelectorAll(".layout-panel"));
    var draggingPanel = null;
    var layoutStorageKey = "voice_pick_layout_v1";
    var draggingCard = null;

    var elProfilePill = document.getElementById("profile-pill");
    var elLauncherPill = document.getElementById("launcher-pill");
    var elCapturePill = document.getElementById("capture-pill");
    var elModuleGrid = document.getElementById("module-grid");
    var elCaptureStateBox = document.getElementById("capture-state-box");
    var elTeachWaypoints = document.getElementById("teach-waypoints");
    var elTeachList = document.getElementById("teach-list");
    var elTeachReplaySelect = document.getElementById("teach-replay-select");
    var elQuickButtons = document.getElementById("quick-buttons");
    var elLogBox = document.getElementById("log-box");
    var elNluBox = document.getElementById("nlu-box");
    var elGripperBox = document.getElementById("gripper-box");
    var elValidationBox = document.getElementById("validation-box");
    var elArmConnectionLabel = document.getElementById("arm-connection-label");
    var elArmConnectionMeta = document.getElementById("arm-connection-meta");
    var elGripperEndpoint = document.getElementById("gripper-endpoint");
    var elGripperMeta = document.getElementById("gripper-meta");
    var elCameraSummary = document.getElementById("camera-summary");
    var elCameraMeta = document.getElementById("camera-meta");
    var elClawSummary = document.getElementById("claw-summary");
    var elClawMeta = document.getElementById("claw-meta");
    var elFingerStageComplete = document.getElementById("btn-finger-stage-complete");

    function saveLayout() {
        var payload = {
            order: Array.prototype.slice.call(dashboard.querySelectorAll(".layout-panel")).map(function (panel) {
                return panel.getAttribute("data-panel");
            }),
            sizes: {}
        };
        layoutPanels.forEach(function (panel) {
            var key = panel.getAttribute("data-panel");
            payload.sizes[key] = {
                width: panel.style.width || "",
                height: panel.style.height || ""
            };
        });
        window.localStorage.setItem(layoutStorageKey, JSON.stringify(payload));
    }

    function groupStorageKey(groupName) {
        return "voice_pick_layout_group_" + groupName;
    }

    function loadLayout() {
        var raw = window.localStorage.getItem(layoutStorageKey);
        if (!raw) return;
        try {
            var payload = JSON.parse(raw);
            (payload.order || []).forEach(function (key) {
                var panel = dashboard.querySelector('.layout-panel[data-panel="' + key + '"]');
                if (panel) {
                    dashboard.appendChild(panel);
                }
            });
            var sizes = payload.sizes || {};
            layoutPanels.forEach(function (panel) {
                var key = panel.getAttribute("data-panel");
                if (sizes[key]) {
                    panel.style.width = sizes[key].width || "";
                    panel.style.height = sizes[key].height || "";
                }
            });
        } catch (err) {
            console.warn("Failed to restore layout", err);
        }
    }

    function saveContainerLayout(container) {
        var groupName = container.getAttribute("data-layout-group");
        if (!groupName) return;
        var cards = Array.prototype.slice.call(container.querySelectorAll(".customizable-card[data-card-id]"));
        var payload = {
            order: cards.map(function (card) {
                return card.getAttribute("data-card-id");
            }),
            sizes: {}
        };
        cards.forEach(function (card) {
            var key = card.getAttribute("data-card-id");
            payload.sizes[key] = {
                width: card.style.width || "",
                height: card.style.height || ""
            };
        });
        window.localStorage.setItem(groupStorageKey(groupName), JSON.stringify(payload));
    }

    function loadContainerLayout(container) {
        var groupName = container.getAttribute("data-layout-group");
        if (!groupName) return;
        var raw = window.localStorage.getItem(groupStorageKey(groupName));
        if (!raw) return;
        try {
            var payload = JSON.parse(raw);
            (payload.order || []).forEach(function (key) {
                var card = container.querySelector('.customizable-card[data-card-id="' + key + '"]');
                if (card) {
                    container.appendChild(card);
                }
            });
            var sizes = payload.sizes || {};
            Array.prototype.slice.call(container.querySelectorAll(".customizable-card[data-card-id]")).forEach(function (card) {
                var key = card.getAttribute("data-card-id");
                if (sizes[key]) {
                    card.style.width = sizes[key].width || "";
                    card.style.height = sizes[key].height || "";
                }
            });
        } catch (err) {
            console.warn("Failed to restore container layout", err);
        }
    }

    function resetLayout() {
        window.localStorage.removeItem(layoutStorageKey);
        window.location.reload();
    }

    function resetCameraLayout() {
        Array.prototype.slice.call(document.querySelectorAll(".layout-container[data-layout-group]")).forEach(function (container) {
            var groupName = container.getAttribute("data-layout-group");
            window.localStorage.removeItem(groupStorageKey(groupName));
        });
        window.location.reload();
    }

    function initLayoutEditing() {
        loadLayout();
        Array.prototype.slice.call(document.querySelectorAll(".layout-container[data-layout-group]")).forEach(loadContainerLayout);
        layoutPanels.forEach(function (panel) {
            panel.addEventListener("dragstart", function () {
                draggingPanel = panel;
                panel.classList.add("dragging");
            });
            panel.addEventListener("dragend", function () {
                panel.classList.remove("dragging");
                draggingPanel = null;
                saveLayout();
            });
            panel.addEventListener("dragover", function (event) {
                event.preventDefault();
            });
            panel.addEventListener("drop", function (event) {
                event.preventDefault();
                if (!draggingPanel || draggingPanel === panel) return;
                var rect = panel.getBoundingClientRect();
                var before = event.clientY < rect.top + rect.height / 2;
                if (before) {
                    dashboard.insertBefore(draggingPanel, panel);
                } else {
                    dashboard.insertBefore(draggingPanel, panel.nextSibling);
                }
                saveLayout();
            });
        });

        if (window.ResizeObserver) {
            var resizeObserver = new ResizeObserver(function () {
                saveLayout();
            });
            layoutPanels.forEach(function (panel) {
                resizeObserver.observe(panel);
            });
        } else {
            window.addEventListener("mouseup", saveLayout);
        }

        document.getElementById("btn-layout-reset").addEventListener("click", resetLayout);
        document.getElementById("btn-camera-layout-reset").addEventListener("click", resetCameraLayout);
        Array.prototype.slice.call(document.querySelectorAll(".customizable-card[data-card-id]")).forEach(function (card) {
            var container = card.closest(".layout-container");
            card.addEventListener("dragstart", function () {
                draggingCard = card;
                card.classList.add("dragging");
            });
            card.addEventListener("dragend", function () {
                card.classList.remove("dragging");
                draggingCard = null;
                if (container) {
                    saveContainerLayout(container);
                }
            });
            card.addEventListener("dragover", function (event) {
                event.preventDefault();
            });
            card.addEventListener("drop", function (event) {
                event.preventDefault();
                if (!draggingCard || draggingCard === card || !container) return;
                if (draggingCard.closest(".layout-container") !== container) return;
                var rect = card.getBoundingClientRect();
                var before = event.clientX < rect.left + rect.width / 2;
                if (before) {
                    container.insertBefore(draggingCard, card);
                } else {
                    container.insertBefore(draggingCard, card.nextSibling);
                }
                saveContainerLayout(container);
            });
        });

        if (window.ResizeObserver) {
            var groupResizeObserver = new ResizeObserver(function (entries) {
                entries.forEach(function (entry) {
                    var container = entry.target.closest(".layout-container");
                    if (container) {
                        saveContainerLayout(container);
                    }
                });
            });
            Array.prototype.slice.call(document.querySelectorAll(".customizable-card[data-card-id]")).forEach(function (card) {
                groupResizeObserver.observe(card);
            });
        }
    }

    function addLog(level, message) {
        var row = document.createElement("div");
        row.className = "log-row " + (level || "info").toLowerCase();
        row.textContent = "[" + new Date().toLocaleTimeString() + "] " + level + "  " + message;
        elLogBox.appendChild(row);
        elLogBox.scrollTop = elLogBox.scrollHeight;
    }

    function renderFingerStageCompletion(data) {
        var action = data && data.finger_stage_action;
        var validActions = [
            "grasp", "release", "release_pose", "release_home"
        ];
        fingerStageAction = validActions.indexOf(action) >= 0 ? action : null;
        if (!elFingerStageComplete) return;
        elFingerStageComplete.hidden = !fingerStageAction;
        if (fingerStageAction === "grasp") {
            elFingerStageComplete.textContent = "Complete Grasp (Enter)";
            elFingerStageComplete.title = "Keep the verified current grip and continue";
        } else if (fingerStageAction === "release") {
            elFingerStageComplete.textContent = "Complete Release + Open (Enter)";
            elFingerStageComplete.title = "Stop LSTM release, open to HOME, verify, then retract";
        } else if (fingerStageAction === "release_pose") {
            elFingerStageComplete.textContent = "Move to Release Pose (Enter)";
            elFingerStageComplete.title = "Stop LSTM and move to the configured release pose";
        } else if (fingerStageAction === "release_home") {
            elFingerStageComplete.textContent = "Return Fingers HOME (Enter)";
            elFingerStageComplete.title = "Release pose verified; return fingers HOME before retract";
        } else {
            elFingerStageComplete.textContent = "Complete Finger Action";
        }
    }

    function completeFingerStage() {
        if (!fingerStageAction) return;
        var requestMessages = {
            grasp: "Requested verified grasp completion...",
            release: "Requested release completion and verified HOME...",
            release_pose: "Requested move to the configured release pose...",
            release_home: "Requested verified finger HOME return..."
        };
        socket.emit("vla_complete_finger_stage");
        addLog("STEP", requestMessages[fingerStageAction]);
    }

    function isFormControlFocused() {
        var active = document.activeElement;
        var tag = active ? active.tagName : "";
        return tag === "INPUT" || tag === "TEXTAREA"
            || tag === "SELECT" || tag === "BUTTON";
    }

    function fetchJSON(url) {
        return fetch(url).then(function (res) {
            var contentType = res.headers.get("content-type") || "";
            if (!res.ok) {
                return res.text().then(function (text) {
                    throw new Error("HTTP " + res.status + " for " + url + ": " + text.slice(0, 120));
                });
            }
            if (contentType.indexOf("application/json") === -1) {
                return res.text().then(function (text) {
                    throw new Error("Non-JSON response for " + url + ": " + text.slice(0, 120));
                });
            }
            return res.json();
        });
    }

    function moduleEntry(name) {
        return moduleConfig[name] || { enabled: true, show_in_ui: true };
    }

    function moduleVisible(name) {
        var entry = moduleEntry(name);
        if (entry.show_in_ui !== undefined) return !!entry.show_in_ui;
        if (entry.enabled !== undefined) return !!entry.enabled;
        return true;
    }

    function uiControlValue(path, defaultValue) {
        var node = uiControlConfig;
        var parts = String(path || "").split(".").filter(Boolean);
        for (var i = 0; i < parts.length; i += 1) {
            if (!node || typeof node !== "object" || !(parts[i] in node)) {
                return defaultValue;
            }
            node = node[parts[i]];
        }
        return node;
    }

    function uiControlVisible(path) {
        return !!uiControlValue(path, true);
    }

    function fixedFallbackAllowlist() {
        var fallback = ["trapezoid", "board", "butter_knife"];
        var configured = uiControlValue(
            "quick_pick.fixed_fallback_objects",
            fallback
        );
        return Array.isArray(configured) ? configured : fallback;
    }

    function applyUiControlVisibility() {
        document.querySelectorAll("[data-ui-control]").forEach(function (node) {
            var visible = uiControlVisible(node.dataset.uiControl);
            if (node.dataset.module) {
                visible = visible && moduleVisible(node.dataset.module);
            }
            node.style.display = visible ? "" : "none";
        });

        document.querySelectorAll("[data-ui-controls-any]").forEach(function (node) {
            var names = String(node.dataset.uiControlsAny || "")
                .split(/\s+/)
                .filter(Boolean);
            node.style.display = names.some(uiControlVisible) ? "" : "none";
        });

        var allowed = fixedFallbackAllowlist();
        document.querySelectorAll("[data-quick-fallback-object]").forEach(function (node) {
            node.style.display = allowed.indexOf(
                node.dataset.quickFallbackObject
            ) >= 0 ? "" : "none";
        });

        document.querySelectorAll("[data-quick-fallback-group]").forEach(function (group) {
            var buttons = Array.prototype.slice.call(
                group.querySelectorAll("[data-quick-fallback-object]")
            );
            var visible = buttons.some(function (button) {
                return button.style.display !== "none";
            });
            if (group.dataset.module) {
                visible = visible && moduleVisible(group.dataset.module);
            }
            group.style.display = visible ? "" : "none";
        });
    }

    function applyModuleVisibility() {
        document.querySelectorAll("[data-module]").forEach(function (node) {
            node.style.display = moduleVisible(node.dataset.module) ? "" : "none";
        });
        document.querySelectorAll("[data-modules-any]").forEach(function (node) {
            var names = String(node.dataset.modulesAny || "")
                .split(/\s+/)
                .filter(Boolean);
            node.style.display = names.some(moduleVisible) ? "" : "none";
        });
    }

    function updateProfileView(cfg) {
        cfg = cfg || {};
        var uiMode = (cfg.ui && cfg.ui.mode) || "modern";
        elProfilePill.textContent = "Profile: " + (cfg.profile || "--") + " / UI: " + uiMode;
        moduleConfig = cfg.modules || {};
        uiControlConfig = ((cfg.ui || {}).controls) || {};
        renderModules();
        applyModuleVisibility();
        applyUiControlVisibility();
    }

    function renderModules() {
        elModuleGrid.innerHTML = "";
        Object.keys(moduleConfig).forEach(function (name) {
            var node = moduleConfig[name] || {};
            var div = document.createElement("div");
            div.className = "module-chip " + (node.enabled ? "enabled" : "disabled");
            div.innerHTML = "<strong>" + name + "</strong><span>" + (node.enabled ? "enabled" : "disabled") + "</span>";
            elModuleGrid.appendChild(div);
        });
    }

    function fillObjectSelects() {
        var selects = ["capture-object", "quick-object"];
        selects.forEach(function (id) {
            var el = document.getElementById(id);
            el.innerHTML = "";
            Object.keys(objectsByKey).forEach(function (key) {
                var opt = document.createElement("option");
                var info = objectsByKey[key] || {};
                var zh = (info.chinese && info.chinese[0]) || key;
                opt.value = key;
                opt.textContent = zh + " / " + key;
                el.appendChild(opt);
            });
        });
    }

    function renderQuickButtons() {
        elQuickButtons.innerHTML = "";
        Object.keys(objectsByKey).forEach(function (key) {
            var btn = document.createElement("button");
            var zh = (objectsByKey[key].chinese && objectsByKey[key].chinese[0]) || key;
            btn.className = "btn ghost";
            btn.textContent = zh;
            btn.addEventListener("click", function () {
                runQuickPick(key);
            });
            elQuickButtons.appendChild(btn);
        });
    }

    function renderTeachList(items) {
        availableRecordings = {};
        elTeachList.innerHTML = "";
        elTeachReplaySelect.innerHTML = "";
        items.forEach(function (item) {
            availableRecordings[item.id] = item;
            var row = document.createElement("div");
            row.className = "list-row";
            row.textContent = item.id + "  (" + item.count + " wp)";
            elTeachList.appendChild(row);

            var opt = document.createElement("option");
            opt.value = item.id;
            opt.textContent = item.id;
            elTeachReplaySelect.appendChild(opt);
        });
    }

    function renderWaypoints(data) {
        var waypoints = data.waypoints || [];
        if (!waypoints.length) {
            elTeachWaypoints.textContent = "No live waypoints.";
            return;
        }
        elTeachWaypoints.innerHTML = "";
        waypoints.forEach(function (wp, idx) {
            var row = document.createElement("div");
            row.className = "list-row";
            row.textContent = "#" + (idx + 1) + "  speed=" + (wp.speed || "--") + "%  gripper=" + (wp.gripper || "none");
            elTeachWaypoints.appendChild(row);
        });
    }

    function selectedStreams() {
        return Array.prototype.slice.call(document.querySelectorAll(".stream-selectors input:checked")).map(function (el) {
            return el.value;
        });
    }

    function runQuickPick(objectKey) {
        var method = document.getElementById("quick-method").value;
        var payload = {
            object_key: objectKey,
            method: method,
            requested_text: objectKey,
            source: "ui",
            speed_preset: document.getElementById("speed-preset").value,
            custom_speed: parseInt(document.getElementById("custom-speed").value || "60", 10),
            replay_mode: document.getElementById("replay-mode").value
        };
        if (method === "teach") {
            payload.recording_name = (objectsByKey[objectKey] || {}).default_teach_recording || "";
        }
        socket.emit("confirm_pick", payload);
        addLog("STEP", "Quick run: " + objectKey + " (" + method + ")");
    }

    function drawTrajectory() {
        if (!window.Plotly) {
            return;
        }
        Plotly.react("trajectory-plot", [{
            x: trajectory.x,
            y: trajectory.y,
            z: trajectory.z,
            type: "scatter3d",
            mode: "lines+markers",
            line: { color: "#ff8c42", width: 5 },
            marker: { size: 2, color: "#ffe1c2" }
        }], {
            margin: { l: 0, r: 0, b: 0, t: 0 },
            paper_bgcolor: "transparent",
            plot_bgcolor: "transparent",
            scene: {
                xaxis: { title: "X" },
                yaxis: { title: "Y" },
                zaxis: { title: "Z" }
            }
        }, { responsive: true, displayModeBar: false });
    }

    socket.on("connect", function () { addLog("STEP", "Connected to backend"); });
    socket.on("disconnect", function () { addLog("ERROR", "Disconnected from backend"); });

    socket.on("vla_status", function (data) {
        renderFingerStageCompletion(data || {});
    });

    socket.on("arm_status", function (data) {
        elArmConnectionLabel.textContent = data.connected ? (data.label || "Connected") : "Disconnected";
        elArmConnectionMeta.textContent = data.connected ? "Ready for motion" : "No arm link";
    });

    socket.on("arm_pose", function (data) {
        var p = data.pose_mm_deg || [];
        if (p.length < 6) return;
        ["x", "y", "z", "rx", "ry", "rz"].forEach(function (key, idx) {
            document.getElementById("mon-" + key).textContent = p[idx].toFixed(2);
        });
        trajectory.x.push(p[0]);
        trajectory.y.push(p[1]);
        trajectory.z.push(p[2]);
        if (trajectory.x.length > 300) {
            trajectory.x.shift();
            trajectory.y.shift();
            trajectory.z.shift();
        }
        drawTrajectory();
    });

    socket.on("camera_status", function (data) {
        var running = [];
        ["cam1", "cam2"].forEach(function (key) {
            if (data[key] && data[key].running) running.push(key);
        });
        elCameraSummary.textContent = running.length ? running.join(" + ") : "offline";
        elCameraMeta.textContent = JSON.stringify(data);
    });

    socket.on("claw_status", function (data) {
        elClawSummary.textContent = data.running ? (data.profile || "live") : "offline";
        elClawMeta.textContent = data.last_error || "No claw error";
    });

    socket.on("arm_log", function (data) {
        addLog(data.level || "INFO", data.message || "");
    });

    socket.on("nlu_result", function (data) {
        elNluBox.textContent = JSON.stringify(data, null, 2);
    });

    socket.on("teach_list", renderTeachList);
    socket.on("teach_data", renderWaypoints);

    socket.on("gripper_state", function (data) {
        var endpoint = data.endpoint || {};
        var state = data.state || {};
        var tactile = state.tactile_data || [];
        var tactileText = Array.isArray(tactile) && tactile.length ? tactile.join(", ") : "--";
        elGripperEndpoint.textContent = data.base_url || "Unavailable";
        elGripperMeta.textContent = data.connected
            ? ((endpoint.label || "connected") + " | tactile: " + tactileText + " | sensor: " + (state.sensor_connected ? "connected" : "disconnected"))
            : (data.last_error || "offline");
        elGripperBox.textContent = [
            "api_version: " + (data.api_version || "--"),
            "base_url: " + (data.base_url || "--"),
            "current_pos: " + JSON.stringify(state.current_pos || []),
            "tactile_data: " + JSON.stringify(tactile),
            "sensor_connected: " + String(!!state.sensor_connected),
            "sample_valid: " + String(!!state.sample_valid),
            "recording_session_id: " + (state.recording_session_id || "--"),
            "last_error: " + (data.last_error || "--")
        ].join("\n");
    });

    socket.on("validation_state", function (data) {
        elValidationBox.textContent = JSON.stringify(data, null, 2);
    });

    socket.on("dataset_capture_state", function (data) {
        elCapturePill.textContent = "Capture: " + (data.running ? "running" : "idle");
        elCaptureStateBox.textContent = JSON.stringify(data, null, 2);
    });

    socket.on("system_health", function (data) {
        elLauncherPill.textContent = "Launcher: " + (data.launcher && data.launcher.running ? "running" : "manual");
    });

    socket.on("config_reloaded", function (data) {
        updateProfileView((data && data.config) || data || {});
        addLog(
            data && data.ok === false ? "ERROR" : "STEP",
            (data && data.message) || "Config reloaded"
        );
    });

    document.getElementById("btn-refresh-health").addEventListener("click", function () {
        fetchJSON("/api/health").then(function (data) {
            elLauncherPill.textContent = "Launcher: " + (data.launcher && data.launcher.running ? "running" : "manual");
            addLog("STEP", "Health refreshed");
        });
    });

    document.getElementById("btn-arm-connect").addEventListener("click", function () { socket.emit("arm_connect"); });
    document.getElementById("btn-arm-home").addEventListener("click", function () { socket.emit("arm_home"); });
    document.getElementById("btn-arm-ready").addEventListener("click", function () { socket.emit("arm_ready"); });
    document.getElementById("btn-arm-stop").addEventListener("click", function () { socket.emit("arm_stop"); });
    if (elFingerStageComplete) {
        elFingerStageComplete.addEventListener("click", completeFingerStage);
    }
    document.addEventListener("keydown", function (e) {
        if (e.key === "Enter" && !e.repeat && fingerStageAction
            && !isFormControlFocused()) {
            e.preventDefault();
            completeFingerStage();
        }
    });
    var btnGripReboot = document.getElementById("btn-gripper-reboot");
    if (btnGripReboot) {
        btnGripReboot.addEventListener("click", function () {
            if (!confirm("Reboot finger motors? Clears an overload latch and homes the fingers open.")) return;
            btnGripReboot.disabled = true;
            var label = btnGripReboot.textContent;
            btnGripReboot.textContent = "Rebooting…";
            socket.emit("gripper_reboot", { home: true });
            setTimeout(function () { btnGripReboot.disabled = false; btnGripReboot.textContent = label; }, 6000);
        });
    }
    document.getElementById("btn-fallback-trapezoid").addEventListener("click", function () { socket.emit("fallback_run", { object_key: "trapezoid" }); });
    document.getElementById("btn-fallback-board").addEventListener("click", function () { socket.emit("fallback_run", { object_key: "board" }); });
    document.getElementById("btn-fallback-butter_knife").addEventListener("click", function () { socket.emit("fallback_run", { object_key: "butter_knife" }); });
    document.getElementById("btn-log-clear").addEventListener("click", function () { elLogBox.innerHTML = ""; });

    document.getElementById("btn-capture-start").addEventListener("click", function () {
        socket.emit("dataset_capture_start", {
            object: document.getElementById("capture-object").value,
            duration_s: parseFloat(document.getElementById("capture-duration").value || "0"),
            fps: parseFloat(document.getElementById("capture-fps").value || "6"),
            telemetry_hz: parseFloat(document.getElementById("capture-telemetry-hz").value || "10"),
            notes: document.getElementById("capture-notes").value,
            streams: selectedStreams()
        });
    });
    document.getElementById("btn-capture-stop").addEventListener("click", function () {
        socket.emit("dataset_capture_stop");
    });

    document.getElementById("btn-teach-start").addEventListener("click", function () {
        socket.emit("teach_start", { name: document.getElementById("teach-name").value || ("recording_" + Date.now()) });
    });
    document.getElementById("btn-teach-waypoint").addEventListener("click", function () {
        socket.emit("teach_waypoint", { gripper: "none", speed: parseInt(document.getElementById("teach-speed").value || "30", 10) });
    });
    document.getElementById("btn-teach-grip-close").addEventListener("click", function () {
        socket.emit("teach_waypoint", { gripper: "close", speed: parseInt(document.getElementById("teach-speed").value || "30", 10) });
    });
    document.getElementById("btn-teach-grip-open").addEventListener("click", function () {
        socket.emit("teach_waypoint", { gripper: "open", speed: parseInt(document.getElementById("teach-speed").value || "30", 10) });
    });
    document.getElementById("btn-teach-stop").addEventListener("click", function () {
        socket.emit("teach_stop");
    });
    document.getElementById("btn-teach-replay").addEventListener("click", function () {
        socket.emit("teach_replay", {
            name: document.getElementById("teach-replay-select").value,
            replay_mode: document.getElementById("replay-mode").value
        });
    });

    document.getElementById("btn-quick-run").addEventListener("click", function () {
        runQuickPick(document.getElementById("quick-object").value);
    });

    document.getElementById("btn-validation-success").addEventListener("click", function () {
        socket.emit("validation_mark", { result: "success" });
    });
    document.getElementById("btn-validation-fail").addEventListener("click", function () {
        socket.emit("validation_mark", { result: "fail" });
    });

    document.getElementById("btn-lang-zh").addEventListener("click", function () { currentLang = "zh-TW"; });
    document.getElementById("btn-lang-en").addEventListener("click", function () { currentLang = "en-US"; });
    document.getElementById("btn-send").addEventListener("click", function () {
        var text = document.getElementById("voice-input").value.trim();
        if (!text) return;
        socket.emit("voice_text", { text: text, lang: currentLang });
    });

    document.getElementById("btn-mic").addEventListener("click", function () {
        var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!SpeechRecognition) {
            addLog("WARN", "SpeechRecognition not available in this browser");
            return;
        }
        var rec = new SpeechRecognition();
        rec.lang = currentLang;
        rec.interimResults = false;
        rec.maxAlternatives = 1;
        rec.onresult = function (event) {
            var text = event.results[0][0].transcript;
            document.getElementById("voice-input").value = text;
            socket.emit("voice_text", { text: text, lang: currentLang });
        };
        rec.start();
    });

    Array.prototype.forEach.call(document.querySelectorAll("img"), function (img) {
        img.addEventListener("error", function () {
            setTimeout(function () {
                img.src = img.src.split("?")[0] + "?t=" + Date.now();
            }, 1000);
        });
    });

    fetchJSON("/api/config").then(updateProfileView).catch(function () {});
    fetchJSON("/api/objects").then(function (objects) {
        objectsByKey = objects || {};
        fillObjectSelects();
        renderQuickButtons();
    }).catch(function () {});
    fetchJSON("/api/recordings").then(renderTeachList).catch(function () {});
    initLayoutEditing();
    drawTrajectory();
})();
