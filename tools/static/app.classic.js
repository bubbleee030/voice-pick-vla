// ============================================================
// Voice Pick Demo - Frontend Logic
// ============================================================

(function () {
    "use strict";

    // -----------------------------------------------------------
    // Socket.IO
    // -----------------------------------------------------------
    var socket = io({ reconnection: true, reconnectionDelay: 1000 });

    // -----------------------------------------------------------
    // State
    // -----------------------------------------------------------
    var currentLang = "zh-TW";
    var lastNluResult = null;
    var vlaRunning = false;   // true while a voice/CLI auto-run is in progress
    var fingerStageAction = null;
    var trajectoryX = [];
    var trajectoryY = [];
    var trajectoryZ = [];
    var trajectoryConfig = {};
    var logAutoScroll = true;
    var teachRecording = false;
    var safetyBoundary = {};
    var runtimeConfig = {};
    var moduleConfigMap = {};
    var uiControlConfig = {};
    var virtualEnvConfig = {};
    var asrConfig = {};
    var sensorConfig = {};
    var nluDebugConfig = {};
    var speechPromptConfig = {};
    var objectsByKey = {};
    var availableTeachRecordings = {};
    var validationState = { active: null, last_completed: null };
    var currentTeachWaypoints = [];
    var sensorSeries1 = [];
    var sensorSeries2 = [];
    var sensorSeries3 = [];
    var sensorTimestamps = [];
    var gripperSeries1 = [];
    var gripperSeries2 = [];
    var gripperSeries3 = [];
    var gripperTimestamps = [];
    var defaultTrajectoryCamera = {
        eye: { x: 1.45, y: 1.45, z: 0.95 },
        center: { x: 0, y: 0, z: 0 },
        up: { x: 0, y: 0, z: 1 }
    };
    var currentTrajectoryCamera = null;
    var trajectoryPlotInitialized = false;
    var trajectoryInteractionActive = false;
    var trajectoryPendingUpdate = false;
    var trajectoryInteractionReleaseTimer = null;
    var trajectoryPointerInside = false;
    var trajectoryLastTapTs = 0;
    var dashboardCols = 24;
    var dashboardRows = 14;
    var dashboardLayoutKey = "voice_pick_dashboard_layout_v1";
    var dashboardPanels = [];
    var dashboardZCounter = 10;
    var plotResizeTimer = null;
    var panelRecordingActive = false;
    var panelRecordingEntries = [];
    var panelRecordingSessionId = "";
    var panelRecordingStartUnix = 0;
    var panelRecordingStartPerfMs = 0;

    // -----------------------------------------------------------
    // DOM references
    // -----------------------------------------------------------
    var elArmDot = document.getElementById("arm-dot");
    var elArmLabel = document.getElementById("arm-label");
    var elYoloBadge = document.getElementById("yolo-badge");
    var elYoloDot = elYoloBadge ? elYoloBadge.querySelector(".status-dot") : null;
    var elYoloText = elYoloBadge ? elYoloBadge.querySelector("span:last-child") : null;
    var elCamDot = document.getElementById("cam-dot");
    var elCamLabel = document.getElementById("cam-label");
    var elMonX = document.getElementById("mon-x");
    var elMonY = document.getElementById("mon-y");
    var elMonZ = document.getElementById("mon-z");
    var elMonRx = document.getElementById("mon-rx");
    var elMonRy = document.getElementById("mon-ry");
    var elMonRz = document.getElementById("mon-rz");
    var elMonUpdate = document.getElementById("mon-update");
    var elMonitorStatus = document.getElementById("monitor-status");
    var elSafety = document.getElementById("safety-indicator");
    var elTranscript = document.getElementById("transcript");
    var elNluCard = document.getElementById("nlu-card");
    var elNluIntent = document.getElementById("nlu-intent");
    var elNluObject = document.getElementById("nlu-object");
    var elNluConfFill = document.getElementById("nlu-conf-fill");
    var elNluConfText = document.getElementById("nlu-conf-text");
    var elNluDisambig = document.getElementById("nlu-disambiguation");
    var elLogContainer = document.getElementById("log-container");
    var elQuickButtons = document.getElementById("quick-buttons");
    var elVoiceInput = document.getElementById("voice-input");
    var elTeachList = document.getElementById("teach-list");
    var elTeachWaypoints = document.getElementById("teach-waypoints");
    var elTeachReplaySelect = document.getElementById("teach-replay-select");
    var elValidationObjectSelect = document.getElementById("validation-object-select");
    var elValidationModeSelect = document.getElementById("validation-mode-select");
    var elValidationStateBadge = document.getElementById("validation-state-badge");
    var elValidationObject = document.getElementById("validation-object");
    var elValidationMode = document.getElementById("validation-mode");
    var elValidationSlot = document.getElementById("validation-slot");
    var elValidationSet = document.getElementById("validation-set");
    var elValidationRequest = document.getElementById("validation-request");
    var elValidationSessionDir = document.getElementById("validation-session-dir");
    var elValidationLastResult = document.getElementById("validation-last-result");
    var elGripperStatus = document.getElementById("gripper-status");
    var elGripperPos1 = document.getElementById("gripper-pos-1");
    var elGripperPos2 = document.getElementById("gripper-pos-2");
    var elGripperPos3 = document.getElementById("gripper-pos-3");
    var elGripperPos = document.getElementById("gripper-pos");
    var elGripperServerTime = document.getElementById("gripper-server-time");
    var elGripperError = document.getElementById("gripper-error");
    var elVirtualEnvStatus = document.getElementById("virtual-env-status");
    var elVirtualEnvBase = document.getElementById("virtual-env-base");
    var elVirtualEnvKind = document.getElementById("virtual-env-kind");
    var elVirtualEnvTarget = document.getElementById("virtual-env-target");
    var elVirtualEnvPhase = document.getElementById("virtual-env-phase");
    var elVirtualEnvTask = document.getElementById("virtual-env-task");
    var elVirtualEnvViews = document.getElementById("virtual-env-views");
    var elVirtualEnvUpdated = document.getElementById("virtual-env-updated");
    var elVirtualEnvSyncState = document.getElementById("virtual-env-sync-state");
    var elVirtualEnvArmSync = document.getElementById("virtual-env-arm-sync");
    var elVirtualEnvGripperSync = document.getElementById("virtual-env-gripper-sync");
    var elVirtualEnvSyncError = document.getElementById("virtual-env-sync-error");
    var elVirtualEnvArmCalibration = document.getElementById("virtual-env-arm-calibration");
    var elVirtualEnvOpenWebrtc = document.getElementById("virtual-env-open-webrtc");
    var elVirtualEnvOpenConsole = document.getElementById("virtual-env-open-console");
    var elVirtualEnvCalibrate = document.getElementById("btn-virtual-env-calibrate");
    var elVirtualEnvResetCalibration = document.getElementById("btn-virtual-env-reset-calibration");
    var elClawDetected = document.getElementById("claw-rgb-detected");
    var elDashboardBoard = document.getElementById("main-grid");
    var elVoiceModeNote = document.getElementById("voice-mode-note");
    var btnAsrOffline = document.getElementById("btn-asr-offline");
    var btnAsrBrowser = document.getElementById("btn-asr-browser");
    var elVoiceMatchTarget = document.getElementById("voice-match-target");
    var elVoiceMatchRate = document.getElementById("voice-match-rate");
    var elVoiceMatchConfidence = document.getElementById("voice-match-confidence");
    var elVoiceMatchStatus = document.getElementById("voice-match-status");
    var elVoiceMatchedPhrase = document.getElementById("voice-matched-phrase");
    var elVoiceResolvedObject = document.getElementById("voice-resolved-object");
    var elVoiceNormalization = document.getElementById("voice-normalization");
    var elVoiceDebugCard = document.getElementById("voice-debug-card");
    var elReloadConfigBtn = document.getElementById("btn-reload-config");
    var elFingerStageComplete = document.getElementById("btn-finger-stage-complete");
    var elRecordPanelsBtn = document.getElementById("btn-record-panels");
    var elSensorStatus = document.getElementById("sensor-status");
    var elSensorValue1 = document.getElementById("sensor-value-1");
    var elSensorValue2 = document.getElementById("sensor-value-2");
    var elSensorValue3 = document.getElementById("sensor-value-3");
    var elSensorError = document.getElementById("sensor-error");

    // -----------------------------------------------------------
    // Load config
    // -----------------------------------------------------------
    function applyRuntimeConfig(cfg) {
        cfg = cfg || {};
        runtimeConfig = cfg;
        safetyBoundary = cfg.safety_boundary || {};
        moduleConfigMap = cfg.modules || {};
        uiControlConfig = ((cfg.ui || {}).controls) || {};
        virtualEnvConfig = cfg.virtual_env || {};
        asrConfig = cfg.asr || {};
        trajectoryConfig = cfg.trajectory || {};
        sensorConfig = cfg.sensor_api || {};
        speechPromptConfig = cfg.speech_prompt || {};
        nluDebugConfig = cfg.nlu_debug || {};
        if (elTeachReplayMode && cfg.teach && cfg.teach.default_replay_mode) {
            elTeachReplayMode.value = cfg.teach.default_replay_mode;
        }
        applySensorPanelDefaults();
        applyModuleVisibility();
        applyUiControlVisibility();
        configureTrajectoryPlot();
        configureSensorPlot();
        configureGripperPlot();
        configureVirtualEnv();
        updateVoiceModeNote();
    }

    function loadRuntimeConfig() {
        return fetch("/api/config")
            .then(function (r) { return r.json(); })
            .then(function (cfg) {
                applyRuntimeConfig(cfg);
                return cfg;
            });
    }

    loadRuntimeConfig().catch(function () {});

    // -----------------------------------------------------------
    // Load objects -> quick pick buttons
    // -----------------------------------------------------------
    function renderObjectControls(objects) {
        objectsByKey = objects || {};
        elQuickButtons.innerHTML = "";
        elValidationObjectSelect.innerHTML = '<option value="">-- Select Object --</option>';
        Object.keys(objectsByKey).forEach(function (key) {
            var obj = objectsByKey[key] || {};
            var zh = (obj.chinese && obj.chinese[0]) || key;
            var en = (obj.english && obj.english[0]) || key;
            var btn = document.createElement("button");
            btn.className = "quick-btn";
            btn.textContent = zh + " / " + en;
            btn.dataset.key = key;
            btn.addEventListener("click", function () {
                quickPick(key, zh);
            });
            elQuickButtons.appendChild(btn);

            var opt = document.createElement("option");
            opt.value = key;
            opt.textContent = zh + " / " + en + " [" + (obj.slot_id || "--") + "]";
            elValidationObjectSelect.appendChild(opt);
        });
    }

    function loadObjectsCatalog() {
        return fetch("/api/objects")
            .then(function (r) { return r.json(); })
            .then(function (objects) {
                renderObjectControls(objects || {});
                return objects;
            });
    }

    loadObjectsCatalog().catch(function () {});

    function loadTeachRecordings() {
        return fetch("/api/teach_recordings")
            .then(function (r) { return r.json(); })
            .then(function (recordings) {
                updateTeachList(Array.isArray(recordings) ? recordings : []);
                return recordings;
            });
    }

    loadTeachRecordings().catch(function () {});

    function quickPick(key, displayName) {
        startValidationPick(key, displayName);
    }

    function primaryChineseName(objectKey) {
        var obj = objectsByKey[objectKey] || {};
        return (obj.chinese && obj.chinese[0]) || objectKey || "--";
    }

    function formatObjectDisplay(objectKey) {
        if (!objectKey) return "--";
        var obj = objectsByKey[objectKey] || {};
        var zh = (obj.chinese && obj.chinese[0]) || objectKey;
        return zh + " (" + objectKey + ")";
    }

    function currentExecutionMode() {
        return elValidationModeSelect.value || "fixed";
    }

    function defaultTeachRecordingName(objectKey) {
        var obj = objectsByKey[objectKey] || {};
        return obj.default_teach_recording || (objectKey + "_pick_v1");
    }

    function resolveBestTeachRecording(objectKey) {
        var prefix = objectKey + "_pick_v";
        var candidates = [];
        Object.keys(availableTeachRecordings).forEach(function (id) {
            var match = id.match(new RegExp("^" + prefix + "(\\d+)(\\.merged)?$"));
            if (!match) return;
            candidates.push({
                id: id,
                version: parseInt(match[1], 10),
                merged: !!match[2]
            });
        });
        candidates.sort(function (a, b) {
            if (a.merged !== b.merged) return a.merged ? -1 : 1;
            return b.version - a.version;
        });
        return candidates.length ? candidates[0].id : "";
    }

    function resolveTeachRecordingName(objectKey) {
        var baseName = defaultTeachRecordingName(objectKey);
        var candidates = baseName.endsWith(".merged")
            ? [baseName]
            : [baseName + ".merged", baseName];
        for (var i = 0; i < candidates.length; i++) {
            if (availableTeachRecordings[candidates[i]]) {
                return candidates[i];
            }
        }
        var best = resolveBestTeachRecording(objectKey);
        if (best) return best;
        addLog("WARN", "Teach recording not found: " + baseName + " or merged variant");
        return "";
    }

    function startValidationPick(objectKey, requestedText, source) {
        if (!objectKey) {
            addLog("WARN", "No object selected");
            return;
        }
        var mode = currentExecutionMode();
        var payload = {
            object_key: objectKey,
            method: mode,
            requested_text: requestedText || objectKey,
            source: source || "ui",
        };
        if (mode === "teach") {
            payload.recording_name = resolveTeachRecordingName(objectKey);
            if (!payload.recording_name) return;
        }
        socket.emit("confirm_pick", payload);
        addLog("STEP", "Validation start: " + objectKey + " (" + mode + ")");
    }

    // -----------------------------------------------------------
    // Socket events
    // -----------------------------------------------------------
    socket.on("connect", function () {
        addLog("STEP", "Connected to server");
    });

    socket.on("disconnect", function () {
        addLog("ERROR", "Disconnected from server");
        setArmStatus(false, "");
    });

    socket.on("arm_status", function (data) {
        setArmStatus(data.connected, data.label || "");
    });

    socket.on("camera_status", function (data) {
        var cam1 = data.cam1 || {};
        var cam2 = data.cam2 || {};
        var claw = data.claw || {};
        var anyOn = cam1.running || cam2.running || claw.running;
        elCamDot.className = "status-dot " + (anyOn ? "on" : "off");
        var parts = [];
        if (cam1.running) parts.push("C1");
        if (cam2.running) parts.push("C2");
        if (claw.running) parts.push("CLAW");
        elCamLabel.textContent = parts.length > 0 ? "CAM: " + parts.join("+") : "CAM: OFF";
        // Update per-camera badges
        ["cam1", "cam2", "claw"].forEach(function (key) {
            var info = data[key] || {};
            var rgbBadge = document.getElementById(key + "-rgb-status");
            var depthBadge = document.getElementById(key + "-depth-status");
            if (rgbBadge) {
                if (key === "claw") {
                    rgbBadge.textContent = info.running ? (info.yolo_ready ? "YOLO" : "RAW") : "OFF";
                } else {
                    rgbBadge.textContent = info.running ? "ON" : "OFF";
                }
                rgbBadge.title = info.last_error || "";
                if (!info.running && info.last_error) rgbBadge.textContent = "ERR";
            }
            if (depthBadge) {
                if (!info.depth_enabled) depthBadge.textContent = "DISABLED";
                else depthBadge.textContent = info.running ? "ON" : "OFF";
                depthBadge.title = info.last_error || "";
                if (info.depth_enabled && !info.running && info.last_error) depthBadge.textContent = "ERR";
            }
        });

        var clawLabels = Array.isArray(claw.last_labels) ? claw.last_labels : [];
        if (elClawDetected) {
            if (claw.running) {
                var clawText = clawLabels.length
                    ? "Detected: " + clawLabels.join(", ")
                    : "Detected: --";
                if (claw.profile) clawText += " | profile: " + claw.profile;
                if (claw.last_error) clawText += " | " + claw.last_error;
                elClawDetected.textContent = clawText;
            } else {
                elClawDetected.textContent = "Detected: --";
            }
        }
        if (elYoloDot && elYoloText) {
            if (claw.running && claw.yolo_ready) {
                elYoloDot.className = "status-dot on";
                elYoloText.textContent = clawLabels.length
                    ? "YOLO: " + clawLabels.slice(0, 2).join(", ")
                    : "YOLO: ON";
            } else if (claw.running) {
                elYoloDot.className = "status-dot warn";
                elYoloText.textContent = "YOLO: RAW";
            } else {
                elYoloDot.className = "status-dot off";
                elYoloText.textContent = "YOLO: OFF";
            }
        }
    });

    socket.on("arm_pose", function (data) {
        var p = data.pose_mm_deg;
        if (!p || p.length < 6) return;
        elMonX.textContent = p[0].toFixed(2);
        elMonY.textContent = p[1].toFixed(2);
        elMonZ.textContent = p[2].toFixed(2);
        elMonRx.textContent = p[3].toFixed(2);
        elMonRy.textContent = p[4].toFixed(2);
        elMonRz.textContent = p[5].toFixed(2);
        elMonUpdate.textContent = "Last: " + new Date().toLocaleTimeString();

        // Safety check
        var raw = data.raw;
        if (raw && safetyBoundary) {
            var safe = checkSafety(raw);
            elSafety.textContent = safe ? "SAFE" : "OUTSIDE";
            elSafety.className = "safety-indicator " + (safe ? "safe" : "unsafe");
        }

        // Trajectory
        trajectoryX.push(p[0]);
        trajectoryY.push(p[1]);
        trajectoryZ.push(p[2]);
        var maxPoints = Math.max(parseInt(trajectoryConfig.max_points || "1800", 10), 50);
        if (trajectoryX.length > maxPoints) {
            trajectoryX.shift();
            trajectoryY.shift();
            trajectoryZ.shift();
        }
        updateTrajectory();
    });

    socket.on("arm_log", function (data) {
        addLog(data.level, data.message, data.timestamp);
    });

    socket.on("nlu_result", function (data) {
        lastNluResult = data;
        showNluResult(data);
    });

    // Auto-run (voice → start_auto, ADR 0003) progress: surface the run's log
    // lines and running state so the operator sees what happens after Confirm.
    socket.on("vla_log", function (data) {
        data = data || {};
        addLog(data.level || "STEP", data.message || "");
    });

    socket.on("vla_status", function (data) {
        data = data || {};
        vlaRunning = !!(data.running || data.status === "running");
        renderFingerStageCompletion(data);
        if (elVoiceMatchStatus && vlaRunning) elVoiceMatchStatus.textContent = "Running";
    });

    socket.on("pick_progress", function (data) {
        var pct = Math.round((data.step / data.total) * 100);
        addLog("STEP", "Progress: " + data.name + " (" + pct + "%)");
    });

    socket.on("teach_list", function (data) {
        updateTeachList(data);
    });

    socket.on("teach_data", function (data) {
        data = data || {};
        currentTeachWaypoints = data.waypoints || [];
        renderTeachWaypoints(currentTeachWaypoints);
        addLog("STEP", "Waypoints: " + (data.count || 0));
    });

    socket.on("validation_state", function (data) {
        updateValidationState(data || {});
    });

    socket.on("gripper_state", function (data) {
        updateGripperState(data || {});
    });

    socket.on("sensor_state", function (data) {
        updateSensorState(data || {});
    });

    socket.on("sensor_data", function (data) {
        updateSensorData(data || {});
    });

    socket.on("virtual_env_sync", function (data) {
        updateVirtualEnvSyncState(data || {});
    });

    socket.on("objects_catalog", function (data) {
        renderObjectControls(data || {});
    });

    socket.on("config_reloaded", function (data) {
        data = data || {};
        if (data.config) {
            applyRuntimeConfig(data.config);
        }
        if (data.ok) {
            addLog("STEP", data.message || "Runtime config reloaded");
            (data.notes || []).forEach(function (note) {
                addLog("STEP", note);
            });
            refreshVirtualEnvStatus();
        } else {
            addLog("ERROR", data.message || "Reload config failed");
        }
    });

    // -----------------------------------------------------------
    // Arm status
    // -----------------------------------------------------------
    function setArmStatus(connected, label) {
        elArmDot.className = "status-dot " + (connected ? "on" : "err");
        elArmLabel.textContent = connected ? "ARM: " + label : "ARM: OFF";
        elMonitorStatus.textContent = connected ? "Connected (" + label + ")" : "Disconnected";
    }

    function updateValidationState(data) {
        validationState = data || { active: null, last_completed: null };
        var active = validationState.active;
        var completed = validationState.last_completed;

        if (active) {
            elValidationStateBadge.textContent = "Active";
            elValidationStateBadge.className = "panel-badge validation-active";
            elValidationObject.textContent = active.object_key || "--";
            elValidationMode.textContent = active.mode || "--";
            elValidationSlot.textContent = active.slot_id || "--";
            elValidationSet.textContent = active.table_set_id || "--";
            elValidationRequest.textContent = active.requested_text || "--";
            elValidationSessionDir.textContent = active.session_dir || "--";
            elValidationObjectSelect.value = active.object_key || "";
            elValidationModeSelect.value = active.mode || "fixed";
        } else {
            elValidationStateBadge.textContent = "Idle";
            elValidationStateBadge.className = "panel-badge";
            elValidationObject.textContent = "--";
            elValidationMode.textContent = "--";
            elValidationSlot.textContent = "--";
            elValidationSet.textContent = "--";
            elValidationRequest.textContent = "--";
            elValidationSessionDir.textContent = "--";
        }

        if (completed) {
            elValidationLastResult.textContent =
                (completed.object_key || "--") + " -> " + (completed.operator_result || "--");
        } else {
            elValidationLastResult.textContent = "--";
        }

        document.getElementById("btn-validation-start").disabled = !!active;
        document.getElementById("btn-validation-success").disabled = !active;
        document.getElementById("btn-validation-fail").disabled = !active;
    }

    function updateGripperState(data) {
        var connected = !!data.connected;
        var state = data.state || {};
        var pos = state.current_pos || [];
        var ts = state.server_time_unix || Date.now() / 1000;
        var pos1 = pos.length > 0 ? Number(pos[0]) : null;
        var pos2 = pos.length > 1 ? Number(pos[1]) : null;
        var pos3 = pos.length > 2 ? Number(pos[2]) : null;
        elGripperStatus.textContent = connected ? "Connected" : "Disconnected";
        elGripperStatus.className = "panel-badge" + (connected ? " validation-active" : "");
        elGripperPos.textContent = pos.length
            ? "Right " + (Number.isFinite(pos2) ? pos2 : "--")
                + ", Left " + (Number.isFinite(pos1) ? pos1 : "--")
                + ", Front " + (Number.isFinite(pos3) ? pos3 : "--")
            : "--";
        elGripperPos1.textContent = Number.isFinite(pos1) ? String(pos1) : "--";
        elGripperPos2.textContent = Number.isFinite(pos2) ? String(pos2) : "--";
        elGripperPos3.textContent = Number.isFinite(pos3) ? String(pos3) : "--";
        if (elGripperServerTime) {
            elGripperServerTime.textContent = state.server_time_unix
                ? Number(state.server_time_unix).toFixed(3)
                : "--";
        }
        if (elGripperError) {
            elGripperError.textContent = data.last_error || "--";
        }

        gripperTimestamps.push(ts);
        gripperSeries1.push(Number.isFinite(pos1) ? pos1 : null);
        gripperSeries2.push(Number.isFinite(pos2) ? pos2 : null);
        gripperSeries3.push(Number.isFinite(pos3) ? pos3 : null);
        var maxPoints = Math.max(parseInt(sensorConfig.history_points || "180", 10), 50);
        if (gripperTimestamps.length > maxPoints) {
            gripperTimestamps.shift();
            gripperSeries1.shift();
            gripperSeries2.shift();
            gripperSeries3.shift();
        }
        configureGripperPlot();
    }

    function updateVirtualEnvSyncState(data) {
        var arm = data.arm || {};
        var gripper = data.gripper || {};
        var syncEnabled = !!data.enabled;
        var connected = !!data.connected;
        var stateText = !syncEnabled
            ? "Disabled"
            : (connected ? "Streaming" : "Waiting");
        var armMode = arm.mode || "direct_ee_move";
        var armText = arm.enabled
            ? (armMode + " | ok " + (arm.ok_count || 0) + " / fail " + (arm.fail_count || 0))
            : "Disabled";
        var gripText = gripper.enabled
            ? ("ok " + (gripper.ok_count || 0) + " / fail " + (gripper.fail_count || 0))
            : "Disabled";

        elVirtualEnvSyncState.textContent = stateText;
        elVirtualEnvArmSync.textContent = armText;
        elVirtualEnvGripperSync.textContent = gripText;
        elVirtualEnvSyncError.textContent = data.last_error || "--";
        if (arm.calibration) {
            elVirtualEnvArmCalibration.textContent =
                "[" + (arm.calibration.position_offsets_m || []).join(", ") + "]"
                + " yaw=" + (arm.calibration.yaw_offset_deg != null ? arm.calibration.yaw_offset_deg : "--");
        } else {
            elVirtualEnvArmCalibration.textContent = "--";
        }
    }

    // -----------------------------------------------------------
    // Safety check (client-side visual only)
    // -----------------------------------------------------------
    function checkSafety(raw) {
        var limits = runtimeConfig.pose_limits || safetyBoundary || {};
        if (!limits || !raw || raw.length < 3) return true;
        var axes = ["x", "y", "z"];
        if (limits.enforce_rotation_limits) {
            axes.push("rx", "ry");
        }
        for (var i = 0; i < axes.length; i++) {
            if (raw[i] === undefined) continue;
            var lo = limits[axes[i] + "_min"];
            var hi = limits[axes[i] + "_max"];
            if (lo !== undefined && raw[i] < lo) return false;
            if (hi !== undefined && raw[i] > hi) return false;
        }
        return true;
    }

    // -----------------------------------------------------------
    // Logs
    // -----------------------------------------------------------
    function addLog(level, message, ts) {
        ts = ts || new Date().toLocaleTimeString();
        var line = document.createElement("div");
        line.className = "log-line " + level;
        line.innerHTML = '<span class="ts">' + ts + '</span> '
            + '<span class="tag">[' + level + ']</span> '
            + '<span class="msg">' + escapeHtml(message) + '</span>';
        elLogContainer.appendChild(line);
        if (logAutoScroll) {
            elLogContainer.scrollTop = elLogContainer.scrollHeight;
        }
        if (socket && socket.connected) {
            socket.emit("frontend_log", {
                level: level,
                message: String(message || ""),
                timestamp: ts
            });
        }
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

    function escapeHtml(s) {
        var d = document.createElement("div");
        d.textContent = s;
        return d.innerHTML;
    }

    elLogContainer.addEventListener("scroll", function () {
        var atBottom = elLogContainer.scrollHeight - elLogContainer.scrollTop
            <= elLogContainer.clientHeight + 30;
        logAutoScroll = atBottom;
    });

    document.getElementById("btn-log-clear").addEventListener("click", function () {
        elLogContainer.innerHTML = "";
        socket.emit("frontend_log_clear");
    });

    document.getElementById("btn-log-export").addEventListener("click", function () {
        var text = elLogContainer.innerText;
        var blob = new Blob([text], { type: "text/plain" });
        var a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "arm_logs_" + Date.now() + ".txt";
        a.click();
    });

    // -----------------------------------------------------------
    // NLU display
    // -----------------------------------------------------------
    function showNluResult(data) {
        elNluCard.style.display = "block";
        elNluCard.style.animation = "none";
        // force reflow
        void elNluCard.offsetHeight;
        elNluCard.style.animation = "slide-in 0.3s ease-out";

        var intent = data.intent || "unknown";
        elNluIntent.textContent = intent.toUpperCase();
        elNluIntent.className = "nlu-intent-badge " + intent;

        var candidates = Array.isArray(data.candidates) ? data.candidates : [];
        var candidateText = candidates.map(formatObjectDisplay).join(" / ");
        var objDisplay = data.object ? formatObjectDisplay(data.object) : (candidateText || "--");
        elNluObject.textContent = objDisplay;

        var conf = Math.round((data.confidence || 0) * 100);
        elNluConfFill.style.width = conf + "%";
        elNluConfText.textContent = "Confidence " + conf + "%";

        var targetText = data.object ? formatObjectDisplay(data.object) : (candidateText || "--");
        var statusText = "No match";
        if (data.fuzzy && data.object) statusText = "Homophone match - check before running";
        else if (data.match_source === "phrase_override" && data.object) statusText = "Phrase override";
        else if (data.match_source === "alias" && data.object) statusText = "Alias matched";
        else if (data.object) statusText = "Matched";
        else if (candidates.length) statusText = "Ambiguous";
        else if (data.need_confirm) statusText = "No match";

        if (elVoiceMatchTarget) elVoiceMatchTarget.textContent = targetText;
        if (elVoiceMatchRate) elVoiceMatchRate.textContent = conf + "%";
        if (elVoiceMatchConfidence) elVoiceMatchConfidence.textContent = conf + "%";
        if (elVoiceMatchStatus) elVoiceMatchStatus.textContent = statusText;
        if (elVoiceMatchedPhrase) elVoiceMatchedPhrase.textContent = data.matched_phrase || "--";
        if (elVoiceResolvedObject) elVoiceResolvedObject.textContent = data.object ? formatObjectDisplay(data.object) : (candidateText || "--");
        if (elVoiceNormalization) {
            var normalizationText = Array.isArray(data.normalization_applied) && data.normalization_applied.length
                ? data.normalization_applied.join(", ")
                : "None";
            elVoiceNormalization.textContent = normalizationText;
        }
        if (elVoiceDebugCard) {
            elVoiceDebugCard.style.display = nluDebugConfig.enabled === false ? "none" : "block";
        }

        // The transcript box previously showed alt0 — the recognizer's loudest
        // guess, which is frequently the wrong homophone (刀子 heard as 包子). Now
        // that the server has resolved the object from ALL alternatives, show
        // what was understood, with the actual heard words as provenance.
        if (elTranscript && data.object) {
            var understood = formatObjectDisplay(data.object);
            var heardWord = data.matched_text || data.heard_top ||
                (Array.isArray(data.heard) ? data.heard[0] : "") || "";
            var mark = data.fuzzy ? "≈ " : "✓ ";   // ≈ for homophone, ✓ for exact
            elTranscript.textContent = mark + understood +
                (heardWord ? "  ⟵ 「" + heardWord + "」" : "");
            elTranscript.className = "transcript active";
        } else if (elTranscript) {
            // Not one of the three demo objects. Do NOT echo the misheard text as
            // if it were a command ("others won't show"): show a muted prompt and
            // let the scope message below name the three. The raw alternatives are
            // still in the ARM LOG for diagnosis.
            elTranscript.textContent = "未辨識（請說 刀子 / pcb / 梯形）";
            elTranscript.className = "transcript";
        }

        var disambiguation = data.disambiguation || "";
        elNluDisambig.textContent = disambiguation;
        elNluDisambig.style.display = disambiguation ? "block" : "none";

        // ADR 0003: only the three demo objects are confirmable, and Confirm
        // runs LIVE. Hide Confirm for out-of-scope/ambiguous input; otherwise
        // label it with the rung it will use and that it's a live run.
        var btnConfirm = document.getElementById("btn-confirm");
        if (btnConfirm) {
            if (data.in_scope && data.object) {
                btnConfirm.style.display = "";
                btnConfirm.textContent = "Confirm ▶ LIVE (" + (data.mode || "vla") + ")";
            } else {
                btnConfirm.style.display = "none";
            }
        }
    }

    function renderFocusKeywords(keywords) {
        return;
    }

    function renderSpeechPrompt(promptCfg) {
        speechPromptConfig = promptCfg || {};
        return;
    }

    // -----------------------------------------------------------
    // NLU confirm/cancel/re-speak
    // -----------------------------------------------------------
    document.getElementById("btn-confirm").addEventListener("click", function () {
        // ADR 0003: voice Confirm drives the LIVE auto-run (start_auto) — the
        // same engine as auto_run_cli.py — NOT the legacy fixed-pose pick. Only
        // the three demo objects are confirmable; the run is always live.
        if (!lastNluResult || !lastNluResult.in_scope || !lastNluResult.object) {
            addLog("WARN", "Not one of the three demo objects — nothing to run");
            return;
        }
        if (vlaRunning) {
            addLog("WARN", "An auto-run is already in progress — stop it first");
            return;
        }
        var key = lastNluResult.object;
        var mode = lastNluResult.mode || "vla";
        socket.emit("vla_auto_start", { object: key, mode: mode, dry_run: false });
        elNluCard.style.display = "none";
        addLog("STEP", "Voice auto-run (LIVE): " + key + " [" + mode + "]");
        lastNluResult = null;
    });

    document.getElementById("btn-cancel").addEventListener("click", function () {
        elNluCard.style.display = "none";
        lastNluResult = null;
        if (elVoiceMatchStatus) elVoiceMatchStatus.textContent = "Cancelled";
        addLog("STEP", "Pick cancelled");
    });

    document.getElementById("btn-respeak").addEventListener("click", function () {
        elNluCard.style.display = "none";
        lastNluResult = null;
        if (elVoiceMatchStatus) elVoiceMatchStatus.textContent = "Listening";
        startListening();
    });

    // -----------------------------------------------------------
    // Voice input - text send
    // -----------------------------------------------------------
    function sendText() {
        var text = elVoiceInput.value.trim();
        if (!text) return;
        elTranscript.textContent = text;
        elTranscript.className = "transcript active";
        socket.emit("voice_text", { text: text, lang: currentLang });
        elVoiceInput.value = "";
    }

    document.getElementById("btn-send").addEventListener("click", sendText);
    elVoiceInput.addEventListener("keydown", function (e) {
        if (e.key === "Enter") sendText();
    });

    // -----------------------------------------------------------
    // Language toggle
    // -----------------------------------------------------------
    document.querySelectorAll(".btn-lang[data-lang]").forEach(function (btn) {
        btn.addEventListener("click", function () {
            document.querySelectorAll(".btn-lang[data-lang]").forEach(function (b) {
                b.classList.remove("active");
            });
            btn.classList.add("active");
            currentLang = btn.dataset.lang;
        });
    });

    // -----------------------------------------------------------
    // Voice input mode: browser online or offline backend ASR
    // -----------------------------------------------------------
    var isListening = false;
    var btnMic = document.getElementById("btn-mic");
    var audioContext = null;
    var mediaStream = null;
    var mediaSource = null;
    var scriptNode = null;
    var audioChunks = [];
    var audioSampleRate = 16000;
    var browserRecognition = null;
    var browserStopRequested = false;
    var browserFinalDelivered = false;
    var browserSpeechCtor = window.SpeechRecognition || window.webkitSpeechRecognition || null;
    var asrModeStorageKey = "voice_pick_asr_mode_v2";
    var asrMode = (function () {
        var saved = "";
        try {
            saved = localStorage.getItem(asrModeStorageKey) || "";
        } catch (e) {}
        if (saved === "browser" && browserSpeechCtor) return "browser";
        if (saved === "offline") return "offline";
        return browserSpeechCtor ? "browser" : "offline";
    })();

    // The offline backend is only real if the server advertises one. Without an
    // `asr` config block /api/asr/transcribe is a 501 stub, and naming an engine
    // it does not have is how operators end up trusting a dead mic.
    var offlineAsrAvailable = true;

    function offlineAsrLabel() {
        if (!asrConfig.mode) return "not enabled in this bundle";
        var pretty = String(asrConfig.mode).replace(/^offline_/, "");
        var fallback = asrConfig.fallback_model ? (" / " + asrConfig.fallback_model) : "";
        return pretty + fallback;
    }

    function markOfflineAsrUnavailable() {
        offlineAsrAvailable = false;
        if (btnAsrOffline) {
            btnAsrOffline.disabled = true;
            btnAsrOffline.title = "Offline ASR is not enabled in this bundle";
        }
    }

    function updateVoiceModeNote(message) {
        if (!elVoiceModeNote) return;
        if (message) {
            elVoiceModeNote.textContent = message;
            return;
        }
        if (asrMode === "browser") {
            elVoiceModeNote.textContent = "ASR Mode: Browser online speech on " + window.location.origin + " | fast, internet-dependent";
        } else {
            elVoiceModeNote.textContent = "ASR Mode: Offline backend (" + offlineAsrLabel() + ") | slower, internet-free";
        }
    }

    function persistAsrMode() {
        try {
            localStorage.setItem(asrModeStorageKey, asrMode);
        } catch (e) {}
    }

    function setAsrMode(mode, note) {
        if (mode === "browser" && !browserSpeechCtor) {
            mode = "offline";
        }
        if (mode === "offline" && !offlineAsrAvailable && browserSpeechCtor) {
            mode = "browser";
        }
        asrMode = mode === "browser" ? "browser" : "offline";
        if (btnAsrOffline) btnAsrOffline.classList.toggle("active", asrMode === "offline");
        if (btnAsrBrowser) {
            btnAsrBrowser.classList.toggle("active", asrMode === "browser");
            btnAsrBrowser.disabled = !browserSpeechCtor;
            btnAsrBrowser.title = browserSpeechCtor ? "Use browser online speech recognition" : "This browser does not expose SpeechRecognition";
        }
        persistAsrMode();
        updateVoiceModeNote(note);
    }

    function browserErrorMessage(errorCode) {
        if (errorCode === "network") {
            return "Browser online speech could not reach its recognition service. This is usually browser-service internet reachability, not port 8090. Try Chrome first.";
        }
        if (errorCode === "not-allowed") {
            return "Browser speech recognition was blocked by microphone or privacy permissions.";
        }
        if (errorCode === "audio-capture") {
            return "Browser speech recognition could not read the microphone.";
        }
        if (errorCode === "no-speech") {
            return "Browser speech recognition heard no clear speech.";
        }
        if (errorCode === "language-not-supported") {
            return "This browser speech service does not support the selected language.";
        }
        return "Browser speech recognition failed: " + errorCode;
    }

    function submitRecognizedText(text, sourceLabel, candidates) {
        var finalText = (text || "").trim();
        if (!finalText) {
            addLog("WARN", sourceLabel + " heard nothing.");
            elTranscript.textContent = "No speech detected.";
            elTranscript.className = "transcript active";
            return;
        }
        elTranscript.textContent = finalText;
        elTranscript.className = "transcript active";
        elVoiceInput.value = finalText;
        var alts = candidates && candidates.length > 1 ? candidates : null;
        addLog("STEP", sourceLabel + ": " + finalText +
            (alts ? "  (alts: " + alts.slice(1).join(" | ") + ")" : ""));
        // Send every alternative, not just the top one: the recognizer ranks by
        // a general language model that has never heard of our three objects.
        socket.emit("voice_text", {
            text: finalText,
            lang: currentLang,
            candidates: candidates || [finalText]
        });
    }

    function ensureBrowserRecognition() {
        if (!browserSpeechCtor) return null;
        if (browserRecognition) return browserRecognition;

        browserRecognition = new browserSpeechCtor();
        browserRecognition.continuous = false;
        browserRecognition.interimResults = true;
        browserRecognition.maxAlternatives = 3;

        browserRecognition.onstart = function () {
            setMicState(true, "Listening in browser speech...");
        };

        browserRecognition.onresult = function (event) {
            var interim = [];
            var finals = [];
            var alternatives = [];
            for (var i = event.resultIndex; i < event.results.length; i++) {
                var result = event.results[i];
                var transcript = result[0] ? result[0].transcript : "";
                if (result.isFinal) {
                    finals.push(transcript);
                    // result is array-like over the alternatives we asked for.
                    var n = Math.min(result.length || 1, browserRecognition.maxAlternatives);
                    for (var a = 0; a < n; a++) {
                        var alt = result[a] && result[a].transcript;
                        if (alt && alternatives.indexOf(alt) === -1) alternatives.push(alt);
                    }
                } else {
                    interim.push(transcript);
                }
            }
            if (finals.length) {
                browserFinalDelivered = true;
                submitRecognizedText(finals.join(" "), "Browser ASR", alternatives);
                return;
            }
            if (interim.length) {
                elTranscript.textContent = interim.join(" ");
                elTranscript.className = "transcript active";
            }
        };

        browserRecognition.onerror = function (event) {
            isListening = false;
            btnMic.classList.remove("listening");
            var msg = browserErrorMessage(event.error);
            addLog("WARN", msg);
            elTranscript.textContent = msg;
            elTranscript.className = "transcript active";
            // Deliberately does NOT fall back to the offline backend: that
            // endpoint is not enabled in this bundle, so switching would trade a
            // transient error for a permanently dead mic (the choice is
            // persisted to localStorage and survives reloads).
            updateVoiceModeNote(msg);
        };

        browserRecognition.onend = function () {
            var hadFinal = browserFinalDelivered;
            isListening = false;
            btnMic.classList.remove("listening");
            if (!hadFinal && browserStopRequested) {
                elTranscript.textContent = "Browser speech stopped.";
                elTranscript.className = "transcript active";
            }
            browserStopRequested = false;
            browserFinalDelivered = false;
            updateVoiceModeNote();
        };

        return browserRecognition;
    }

    function mergeAudioChunks(chunks) {
        var total = 0;
        chunks.forEach(function (chunk) { total += chunk.length; });
        var merged = new Float32Array(total);
        var offset = 0;
        chunks.forEach(function (chunk) {
            merged.set(chunk, offset);
            offset += chunk.length;
        });
        return merged;
    }

    function encodeWav(samples, sampleRate) {
        var buffer = new ArrayBuffer(44 + samples.length * 2);
        var view = new DataView(buffer);

        function writeString(offset, text) {
            for (var i = 0; i < text.length; i++) {
                view.setUint8(offset + i, text.charCodeAt(i));
            }
        }

        writeString(0, "RIFF");
        view.setUint32(4, 36 + samples.length * 2, true);
        writeString(8, "WAVE");
        writeString(12, "fmt ");
        view.setUint32(16, 16, true);
        view.setUint16(20, 1, true);
        view.setUint16(22, 1, true);
        view.setUint32(24, sampleRate, true);
        view.setUint32(28, sampleRate * 2, true);
        view.setUint16(32, 2, true);
        view.setUint16(34, 16, true);
        writeString(36, "data");
        view.setUint32(40, samples.length * 2, true);

        var offset = 44;
        for (var j = 0; j < samples.length; j++) {
            var s = Math.max(-1, Math.min(1, samples[j]));
            view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
            offset += 2;
        }
        return new Blob([buffer], { type: "audio/wav" });
    }

    function cleanupAudioCapture() {
        if (scriptNode) {
            try { scriptNode.disconnect(); } catch (e) {}
            scriptNode.onaudioprocess = null;
            scriptNode = null;
        }
        if (mediaSource) {
            try { mediaSource.disconnect(); } catch (e) {}
            mediaSource = null;
        }
        if (mediaStream) {
            mediaStream.getTracks().forEach(function (track) { track.stop(); });
            mediaStream = null;
        }
        if (audioContext) {
            try { audioContext.close(); } catch (e) {}
            audioContext = null;
        }
    }

    function setMicState(listening, text) {
        isListening = listening;
        btnMic.classList.toggle("listening", listening);
        if (text) {
            elTranscript.textContent = text;
            elTranscript.className = "transcript" + (listening ? "" : " active");
        }
    }

    function submitOfflineAudio(wavBlob) {
        var form = new FormData();
        form.append("audio", wavBlob, "speech.wav");
        form.append("lang", currentLang);

        setMicState(false, "Transcribing locally...");
        fetch("/api/asr/transcribe", {
            method: "POST",
            body: form
        })
            .then(function (r) {
                return r.json().then(function (data) {
                    return { ok: r.ok, status: r.status, data: data };
                }, function () {
                    return { ok: r.ok, status: r.status, data: {} };
                });
            })
            .then(function (result) {
                if (!result.ok || !result.data.ok) {
                    var err = new Error((result.data && result.data.error) || "offline transcription failed");
                    err.httpStatus = result.status;
                    throw err;
                }
                submitRecognizedText(result.data.text || "", "Offline ASR");
            })
            .catch(function (err) {
                if (err.httpStatus === 501) {
                    markOfflineAsrUnavailable();
                    addLog("WARN", "Offline ASR is not enabled in this bundle - using browser speech");
                    setAsrMode("browser", "ASR Mode: Browser online speech | offline backend not enabled in this bundle");
                    return;
                }
                addLog("WARN", "Offline ASR failed: " + err.message);
                elTranscript.textContent = "Offline ASR failed. Type text instead.";
                elTranscript.className = "transcript active";
            });
    }

    function stopOfflineListening() {
        if (!isListening) return;
        setMicState(false, "Processing audio...");

        var samples = mergeAudioChunks(audioChunks);
        audioChunks = [];
        cleanupAudioCapture();

        if (!samples.length || samples.length < Math.floor(audioSampleRate * 0.25)) {
            addLog("WARN", "Audio too short for offline ASR");
            elTranscript.textContent = "Audio too short.";
            elTranscript.className = "transcript active";
            return;
        }

        submitOfflineAudio(encodeWav(samples, audioSampleRate));
    }

    function startOfflineListening() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            addLog("WARN", "Browser microphone capture is unavailable");
            return;
        }

        navigator.mediaDevices.getUserMedia({ audio: true })
            .then(function (stream) {
                mediaStream = stream;
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                audioSampleRate = audioContext.sampleRate || 16000;
                mediaSource = audioContext.createMediaStreamSource(stream);
                scriptNode = audioContext.createScriptProcessor(4096, 1, 1);
                audioChunks = [];
                scriptNode.onaudioprocess = function (event) {
                    var input = event.inputBuffer.getChannelData(0);
                    audioChunks.push(new Float32Array(input));
                };
                mediaSource.connect(scriptNode);
                scriptNode.connect(audioContext.destination);
                setMicState(true, "Listening locally... click mic again to stop");
            })
            .catch(function (err) {
                cleanupAudioCapture();
                addLog("WARN", "Microphone access failed: " + err.message);
            });
    }

    function startBrowserListening() {
        var recognition = ensureBrowserRecognition();
        if (!recognition) {
            setAsrMode("offline", "ASR Mode: Offline backend | browser speech API unavailable in this browser");
            startOfflineListening();
            return;
        }
        browserStopRequested = false;
        browserFinalDelivered = false;
        recognition.lang = currentLang;
        recognition.start();
    }

    function stopBrowserListening() {
        if (!browserRecognition) return;
        browserStopRequested = true;
        try {
            browserRecognition.stop();
        } catch (e) {
            isListening = false;
            btnMic.classList.remove("listening");
        }
    }

    function startListening() {
        if (isListening) {
            if (asrMode === "browser") stopBrowserListening();
            else stopOfflineListening();
            return;
        }
        if (asrMode === "browser") startBrowserListening();
        else startOfflineListening();
    }

    if (btnAsrOffline) {
        btnAsrOffline.addEventListener("click", function () {
            if (isListening) {
                if (asrMode === "browser") stopBrowserListening();
                else stopOfflineListening();
            }
            setAsrMode("offline");
        });
    }

    if (btnAsrBrowser) {
        btnAsrBrowser.addEventListener("click", function () {
            if (isListening) {
                if (asrMode === "browser") stopBrowserListening();
                else stopOfflineListening();
            }
            setAsrMode("browser");
        });
    }

    setAsrMode(asrMode);

    // Mic button: click to toggle
    btnMic.addEventListener("click", function () {
        startListening();
    });

    // Space key to toggle
    document.addEventListener("keydown", function (e) {
        if (e.code === "Space" && document.activeElement.tagName !== "INPUT"
            && document.activeElement.tagName !== "TEXTAREA"
            && document.activeElement.tagName !== "SELECT") {
            e.preventDefault();
            startListening();
        }
    });

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

    // -----------------------------------------------------------
    // Arm connect button
    // -----------------------------------------------------------
    document.getElementById("btn-arm-connect").addEventListener("click", function () {
        socket.emit("arm_connect");
        addLog("STEP", "Connecting to arm...");
    });

    // Ready / Home buttons
    document.getElementById("btn-ready").addEventListener("click", function () {
        socket.emit("arm_ready");
        addLog("STEP", "Sending arm to ready pose...");
    });

    document.getElementById("btn-home").addEventListener("click", function () {
        socket.emit("arm_home");
        addLog("STEP", "Sending arm home...");
    });

    // Fallback (model-free) pick->place buttons
    ["trapezoid", "board", "butter_knife"].forEach(function (obj) {
        var btn = document.getElementById("btn-fallback-" + obj);
        if (btn) {
            btn.addEventListener("click", function () {
                socket.emit("fallback_run", { object_key: obj });
                addLog("STEP", "Fallback pick->place: " + obj);
            });
        }
    });

    if (elReloadConfigBtn) {
        elReloadConfigBtn.addEventListener("click", function () {
            var previousLabel = elReloadConfigBtn.textContent;
            elReloadConfigBtn.disabled = true;
            elReloadConfigBtn.textContent = "Reloading...";
            fetch("/api/reload_config", { method: "POST" })
                .then(function (r) { return r.json(); })
                .then(function (data) {
                    if (data && data.config) {
                        applyRuntimeConfig(data.config);
                    }
                    if (!data || !data.ok) {
                        throw new Error((data && data.message) || "Reload config failed");
                    }
                })
                .catch(function (err) {
                    addLog("ERROR", "Reload config failed: " + err.message);
                })
                .finally(function () {
                    elReloadConfigBtn.disabled = false;
                    elReloadConfigBtn.textContent = previousLabel;
                });
        });
    }

    function recordingMimeType() {
        var candidates = [
            "video/webm;codecs=vp9",
            "video/webm;codecs=vp8",
            "video/webm",
            "video/mp4;codecs=avc1.42E01E",
            "video/mp4"
        ];
        for (var i = 0; i < candidates.length; i++) {
            if (window.MediaRecorder && MediaRecorder.isTypeSupported(candidates[i])) {
                return candidates[i];
            }
        }
        return "";
    }

    function recordingExtension(mimeType) {
        return mimeType.indexOf("mp4") >= 0 ? "mp4" : "webm";
    }

    function visibleDashboardPanels() {
        return dashboardPanels.filter(function (panel) {
            if (!panel || panel.style.display === "none") return false;
            var rect = panel.getBoundingClientRect();
            return rect.width >= 40 && rect.height >= 40;
        });
    }

    function downloadPanelRecording(panelId, blob, extension) {
        if (!blob || !blob.size) return;
        var safeId = String(panelId || "panel").replace(/[^a-z0-9_-]+/gi, "_");
        var stamp = panelRecordingSessionId || new Date().toISOString().replace(/[:.]/g, "-");
        var a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "voice_pick_" + safeId + "_" + stamp + "." + extension;
        document.body.appendChild(a);
        a.click();
        setTimeout(function () {
            URL.revokeObjectURL(a.href);
            a.remove();
        }, 1000);
    }

    function uploadPanelRecording(panelId, blob, mimeType, extension, width, height) {
        if (!blob || !blob.size) {
            addLog("WARN", "Panel recording empty: " + panelId);
            return Promise.resolve(null);
        }
        var safeId = String(panelId || "panel").replace(/[^a-z0-9_-]+/gi, "_");
        var formData = new FormData();
        formData.append("session_id", panelRecordingSessionId || "panel_recording");
        formData.append("panel_id", safeId);
        formData.append("width", String(width || ""));
        formData.append("height", String(height || ""));
        formData.append("file", blob, safeId + "." + extension);
        return fetch("/api/panel_recordings/upload", {
            method: "POST",
            body: formData
        }).then(function (r) {
            return r.json().then(function (data) {
                if (!r.ok || !data.ok) {
                    throw new Error((data && data.error) || "upload failed");
                }
                addLog("STEP", "Panel recording saved: " + data.path);
                return data;
            });
        }).catch(function (err) {
            addLog("ERROR", "Panel recording save failed for " + safeId + ": " + err.message);
            return null;
        });
    }

    function canvasToPngBlob(canvas) {
        return new Promise(function (resolve, reject) {
            canvas.toBlob(function (blob) {
                if (blob && blob.size) resolve(blob);
                else reject(new Error("empty canvas frame"));
            }, "image/png");
        });
    }

    function uploadPanelFrame(entry, blob) {
        var nowPerf = (window.performance && typeof performance.now === "function")
            ? performance.now()
            : Date.now();
        var nowUnix = Date.now() / 1000;
        var elapsedMs = Math.max(0, nowPerf - panelRecordingStartPerfMs);
        var formData = new FormData();
        formData.append("session_id", panelRecordingSessionId);
        formData.append("panel_id", entry.panelId);
        formData.append("frame_index", String(entry.frameIndex));
        formData.append("width", String(entry.width));
        formData.append("height", String(entry.height));
        formData.append("timestamp_unix", nowUnix.toFixed(6));
        formData.append("elapsed_ms", elapsedMs.toFixed(3));
        formData.append("perf_ms", nowPerf.toFixed(3));
        formData.append("file", blob, entry.panelId + "_" + String(entry.frameIndex).padStart(6, "0") + ".png");
        entry.frameIndex += 1;
        return fetch("/api/panel_recordings/frame", {
            method: "POST",
            body: formData
        }).then(function (r) {
            if (!r.ok) {
                return r.text().then(function (text) {
                    throw new Error(text || ("frame upload failed: " + r.status));
                });
            }
            return r.json();
        }).catch(function (err) {
            addLog("WARN", "Panel frame upload failed for " + entry.panelId + ": " + err.message);
            return null;
        });
    }

    function finishPanelRecording() {
        var panels = panelRecordingEntries.map(function (entry) {
            return {
                panel_id: entry.panelId,
                width: entry.width,
                height: entry.height,
                frames: entry.frameIndex
            };
        });
        return fetch("/api/panel_recordings/finish", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: panelRecordingSessionId,
                fps: 60,
                panels: panels
            })
        }).then(function (r) {
            return r.json().then(function (data) {
                if (!r.ok || !data.ok) {
                    throw new Error((data && data.error) || "finish failed");
                }
                (data.outputs || []).forEach(function (out) {
                    if (out.path) {
                        var sourceInfo = out.source_frames ? ", source " + out.source_frames : "";
                        var fpsInfo = out.effective_fps ? ", " + Number(out.effective_fps).toFixed(1) + "fps" : "";
                        addLog("STEP", "Panel MP4 saved: " + out.path + " (" + out.frames + " frames" + sourceInfo + fpsInfo + ")");
                    } else if (out.frames_dir) {
                        addLog("STEP", "Panel frames saved: " + out.frames_dir + " (" + out.frames + " frames)");
                        if (out.frame_manifest) {
                            addLog("STEP", "Panel manifest: " + out.frame_manifest);
                        }
                    }
                });
                (data.errors || []).forEach(function (err) {
                    addLog("WARN", "Panel capture issue: " + err);
                });
                if (data.genlock && data.genlock.path) {
                    addLog("STEP", "Genlock manifest: " + data.genlock.path + " (" + (data.genlock.frames || 0) + " rows)");
                }
                return data;
            });
        }).catch(function (err) {
            addLog("ERROR", "Panel recording finish failed: " + err.message);
            return null;
        });
    }

    function startPanelRecording() {
        if (panelRecordingActive) return;

        var panels = visibleDashboardPanels();
        if (!panels.length) {
            addLog("WARN", "No visible panels to record");
            return;
        }

        panelRecordingEntries = [];
        panelRecordingSessionId = new Date().toISOString().replace(/[:.]/g, "-");
        panelRecordingStartUnix = Date.now() / 1000;
        panelRecordingStartPerfMs = (window.performance && typeof performance.now === "function")
            ? performance.now()
            : Date.now();

        fetch("/api/panel_recordings/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                session_id: panelRecordingSessionId,
                fps: 60,
                genlock: {
                    client_start_unix: panelRecordingStartUnix,
                    client_start_perf_ms: panelRecordingStartPerfMs,
                    capture_interval_ms: 250
                },
                panels: panels.map(function (panel) {
                    var rect = panel.getBoundingClientRect();
                    return {
                        panel_id: panel.dataset.panelId || panel.id || "panel",
                        width: Math.max(1, Math.round(rect.width)),
                        height: Math.max(1, Math.round(rect.height))
                    };
                })
            })
        }).then(function (r) {
            return r.json().then(function (data) {
                if (!r.ok || !data.ok) {
                    throw new Error((data && data.error) || "start failed");
                }
                panelRecordingActive = true;
                if (elRecordPanelsBtn) {
                    elRecordPanelsBtn.textContent = "Stop Recording";
                    elRecordPanelsBtn.classList.add("recording");
                }
                panels.forEach(function (panel) {
                    var rect = panel.getBoundingClientRect();
                    var entry = {
                        panel: panel,
                        panelId: String(panel.dataset.panelId || panel.id || "panel").replace(/[^a-z0-9_-]+/gi, "_"),
                        width: Math.max(1, Math.round(rect.width)),
                        height: Math.max(1, Math.round(rect.height)),
                        frameIndex: 0
                    };
                    panelRecordingEntries.push(entry);
                });
                addLog("STEP", "Panel recording started: " + panelRecordingEntries.length + " visible panels");
                addLog("STEP", "Panel recording session: data/panel_recordings/" + panelRecordingSessionId);
                addLog("STEP", "Panel recorder mode: backend MP4");
            });
        }).catch(function (err) {
            panelRecordingActive = false;
            addLog("ERROR", "Panel recording start failed: " + err.message);
        });
    }

    function stopPanelRecording() {
        if (!panelRecordingActive) return;
        panelRecordingActive = false;
        if (elRecordPanelsBtn) {
            elRecordPanelsBtn.disabled = true;
            elRecordPanelsBtn.textContent = "Record Panels";
            elRecordPanelsBtn.classList.remove("recording");
        }
        addLog("STEP", "Panel recording stopped; finalizing backend MP4 files...");
        finishPanelRecording().finally(function () {
            panelRecordingEntries = [];
            if (elRecordPanelsBtn) {
                elRecordPanelsBtn.disabled = false;
            }
        });
    }

    if (elRecordPanelsBtn) {
        elRecordPanelsBtn.addEventListener("click", function () {
            if (panelRecordingActive) stopPanelRecording();
            else startPanelRecording();
        });
    }

    document.getElementById("btn-reset-alarms").addEventListener("click", function () {
        socket.emit("arm_reset_alarms");
        addLog("STEP", "Sending reset alarm command...");
    });

    document.getElementById("btn-emergency-stop").addEventListener("click", function () {
        socket.emit("arm_emergency_stop");
        addLog("ERROR", "EMERGENCY STOP sent");
    });

    document.getElementById("btn-validation-start").addEventListener("click", function () {
        var key = elValidationObjectSelect.value;
        var obj = objectsByKey[key] || {};
        var requestedText = (obj.chinese && obj.chinese[0]) || key;
        startValidationPick(key, requestedText, "ui");
    });

    document.getElementById("btn-validation-success").addEventListener("click", function () {
        socket.emit("validation_mark", { result: "success" });
    });

    document.getElementById("btn-validation-fail").addEventListener("click", function () {
        socket.emit("validation_mark", { result: "fail" });
    });

    elVirtualEnvCalibrate.addEventListener("click", function () {
        postLocalVirtualEnv("/api/virtual_env/calibrate_current")
            .then(function (data) {
                addLog("STEP", "Digital twin calibration updated: " + JSON.stringify(data.position_offsets_m || []));
                refreshVirtualEnvStatus();
            })
            .catch(function (err) {
                addLog("WARN", "Digital twin calibration failed: " + err.message);
            });
    });

    elVirtualEnvResetCalibration.addEventListener("click", function () {
        postLocalVirtualEnv("/api/virtual_env/reset_calibration")
            .then(function () {
                addLog("STEP", "Digital twin calibration reset");
                refreshVirtualEnvStatus();
            })
            .catch(function (err) {
                addLog("WARN", "Reset calibration failed: " + err.message);
            });
    });

    // -----------------------------------------------------------
    // 3D Trajectory Plot
    // -----------------------------------------------------------
    var trajLayout = {
        paper_bgcolor: "#0d1520",
        plot_bgcolor: "#0d1520",
        margin: { l: 40, r: 20, t: 20, b: 30 },
        uirevision: "trajectory-ui",
        scene: {
            dragmode: "orbit",
            uirevision: "trajectory-scene",
            xaxis: { title: "X (mm)", color: "#5a6a7e", gridcolor: "#1a2332" },
            yaxis: { title: "Y (mm)", color: "#5a6a7e", gridcolor: "#1a2332" },
            zaxis: { title: "Z (mm)", color: "#5a6a7e", gridcolor: "#1a2332" },
            bgcolor: "#0d1520",
        },
        font: { family: "Inter, sans-serif", size: 10, color: "#8b9bb4" },
        showlegend: false,
    };

    function buildTrajectoryTraces() {
        var len = trajectoryX.length;
        var colors = [];
        for (var i = 0; i < len; i++) {
            colors.push(len <= 1 ? 1 : i / (len - 1));
        }
        return [
            {
                x: trajectoryX,
                y: trajectoryY,
                z: trajectoryZ,
                type: "scatter3d",
                mode: "lines",
                line: {
                    color: "#448aff",
                    width: parseInt(trajectoryConfig.line_width || "2", 10)
                },
                opacity: 0.45,
                name: "Path",
            },
            {
                x: trajectoryX,
                y: trajectoryY,
                z: trajectoryZ,
                type: "scatter3d",
                mode: "markers",
                marker: {
                    size: 3,
                    color: colors,
                    colorscale: trajectoryConfig.colorscale || "Turbo",
                    cmin: 0,
                    cmax: 1,
                    opacity: trajectoryConfig.fade_by_time === false ? 0.85 : 0.95,
                },
                hoverinfo: "skip",
                name: "Order",
            },
            {
                x: len ? [trajectoryX[len - 1]] : [],
                y: len ? [trajectoryY[len - 1]] : [],
                z: len ? [trajectoryZ[len - 1]] : [],
                type: "scatter3d",
                mode: "markers",
                marker: { color: "#ff5252", size: trajectoryConfig.show_head_marker === false ? 0 : 6 },
                name: "Current",
                visible: trajectoryConfig.show_head_marker === false ? "legendonly" : true,
            }
        ];
    }

    function cloneTrajectoryCamera(camera) {
        if (!camera) return null;
        return {
            eye: {
                x: Number(camera.eye && camera.eye.x != null ? camera.eye.x : defaultTrajectoryCamera.eye.x),
                y: Number(camera.eye && camera.eye.y != null ? camera.eye.y : defaultTrajectoryCamera.eye.y),
                z: Number(camera.eye && camera.eye.z != null ? camera.eye.z : defaultTrajectoryCamera.eye.z),
            },
            center: {
                x: Number(camera.center && camera.center.x != null ? camera.center.x : defaultTrajectoryCamera.center.x),
                y: Number(camera.center && camera.center.y != null ? camera.center.y : defaultTrajectoryCamera.center.y),
                z: Number(camera.center && camera.center.z != null ? camera.center.z : defaultTrajectoryCamera.center.z),
            },
            up: {
                x: Number(camera.up && camera.up.x != null ? camera.up.x : defaultTrajectoryCamera.up.x),
                y: Number(camera.up && camera.up.y != null ? camera.up.y : defaultTrajectoryCamera.up.y),
                z: Number(camera.up && camera.up.z != null ? camera.up.z : defaultTrajectoryCamera.up.z),
            }
        };
    }

    function markTrajectoryInteraction(durationMs) {
        trajectoryInteractionActive = true;
        if (trajectoryInteractionReleaseTimer) {
            clearTimeout(trajectoryInteractionReleaseTimer);
        }
        trajectoryInteractionReleaseTimer = setTimeout(function () {
            if (trajectoryPointerInside) {
                markTrajectoryInteraction(durationMs || 220);
                return;
            }
            trajectoryInteractionActive = false;
            trajectoryInteractionReleaseTimer = null;
            if (trajectoryPendingUpdate) {
                trajectoryPendingUpdate = false;
                updateTrajectory();
            }
        }, Math.max(120, durationMs || 220));
    }

    function attachTrajectoryHandlers() {
        var plot = document.getElementById("trajectory-plot");
        if (!plot || plot.__trajectoryHandlersAttached) return;
        function syncCameraFromEvent(eventData) {
            if (!eventData) return;
            if (eventData["scene.camera"]) {
                currentTrajectoryCamera = cloneTrajectoryCamera(eventData["scene.camera"]);
                return;
            }
            if (
                eventData["scene.camera.eye.x"] === undefined &&
                eventData["scene.camera.eye.y"] === undefined &&
                eventData["scene.camera.eye.z"] === undefined &&
                eventData["scene.camera.center.x"] === undefined &&
                eventData["scene.camera.center.y"] === undefined &&
                eventData["scene.camera.center.z"] === undefined &&
                eventData["scene.camera.up.x"] === undefined &&
                eventData["scene.camera.up.y"] === undefined &&
                eventData["scene.camera.up.z"] === undefined
            ) {
                return;
            }
            var base = cloneTrajectoryCamera(currentTrajectoryCamera || defaultTrajectoryCamera);
            if (eventData["scene.camera.eye.x"] !== undefined) base.eye.x = Number(eventData["scene.camera.eye.x"]);
            if (eventData["scene.camera.eye.y"] !== undefined) base.eye.y = Number(eventData["scene.camera.eye.y"]);
            if (eventData["scene.camera.eye.z"] !== undefined) base.eye.z = Number(eventData["scene.camera.eye.z"]);
            if (eventData["scene.camera.center.x"] !== undefined) base.center.x = Number(eventData["scene.camera.center.x"]);
            if (eventData["scene.camera.center.y"] !== undefined) base.center.y = Number(eventData["scene.camera.center.y"]);
            if (eventData["scene.camera.center.z"] !== undefined) base.center.z = Number(eventData["scene.camera.center.z"]);
            if (eventData["scene.camera.up.x"] !== undefined) base.up.x = Number(eventData["scene.camera.up.x"]);
            if (eventData["scene.camera.up.y"] !== undefined) base.up.y = Number(eventData["scene.camera.up.y"]);
            if (eventData["scene.camera.up.z"] !== undefined) base.up.z = Number(eventData["scene.camera.up.z"]);
            currentTrajectoryCamera = base;
        }

        plot.addEventListener("pointerenter", function () {
            trajectoryPointerInside = true;
            markTrajectoryInteraction(1000);
        }, true);
        plot.addEventListener("pointerleave", function () {
            trajectoryPointerInside = false;
            markTrajectoryInteraction(80);
        }, true);
        plot.addEventListener("pointerdown", function () {
            markTrajectoryInteraction(500);
        }, true);
        plot.addEventListener("wheel", function () {
            markTrajectoryInteraction(500);
        }, { passive: true, capture: true });
        window.addEventListener("pointerup", function () {
            if (trajectoryPointerInside) {
                var now = Date.now();
                if (now - trajectoryLastTapTs < 320) {
                    trajectoryPendingUpdate = false;
                    trajectoryInteractionActive = false;
                    resetTrajectoryCamera();
                }
                trajectoryLastTapTs = now;
            }
            markTrajectoryInteraction(140);
        }, true);

        if (plot.on) {
            plot.on("plotly_relayouting", function (eventData) {
                markTrajectoryInteraction(500);
                syncCameraFromEvent(eventData);
            });
        }
        plot.on("plotly_relayout", function (eventData) {
            markTrajectoryInteraction(180);
            syncCameraFromEvent(eventData);
        });
        if (plot.on) {
            plot.on("plotly_doubleclick", function () {
                trajectoryPendingUpdate = false;
                trajectoryInteractionActive = false;
                resetTrajectoryCamera();
            });
        }
        plot.addEventListener("dblclick", function () {
            trajectoryPendingUpdate = false;
            trajectoryInteractionActive = false;
            resetTrajectoryCamera();
        }, true);
        plot.__trajectoryHandlersAttached = true;
    }

    function resetTrajectoryCamera() {
        var plot = document.getElementById("trajectory-plot");
        if (!plot) return;
        currentTrajectoryCamera = cloneTrajectoryCamera(defaultTrajectoryCamera);
        try {
            Plotly.relayout(plot, { "scene.camera": cloneTrajectoryCamera(currentTrajectoryCamera) });
        } catch (e) {}
    }

    function configureTrajectoryPlot(resetCamera) {
        var plot = document.getElementById("trajectory-plot");
        if (!plot) return;
        if (resetCamera || !currentTrajectoryCamera) {
            currentTrajectoryCamera = cloneTrajectoryCamera(defaultTrajectoryCamera);
        }
        var layout = JSON.parse(JSON.stringify(trajLayout));
        layout.scene.camera = cloneTrajectoryCamera(currentTrajectoryCamera);
        var config = {
            displayModeBar: false,
            responsive: true,
            scrollZoom: true,
            doubleClick: "reset",
        };
        if (!trajectoryPlotInitialized) {
            Plotly.newPlot("trajectory-plot", buildTrajectoryTraces(), layout, config);
            trajectoryPlotInitialized = true;
            attachTrajectoryHandlers();
            return;
        }
        Plotly.react("trajectory-plot", buildTrajectoryTraces(), layout, config);
    }

    function schedulePlotResize() {
        if (plotResizeTimer) {
            clearTimeout(plotResizeTimer);
        }
        plotResizeTimer = setTimeout(function () {
            try {
                Plotly.Plots.resize(document.getElementById("trajectory-plot"));
            } catch (e) {}
            try {
                Plotly.Plots.resize(document.getElementById("sensor-chart-plot"));
            } catch (e) {}
            try {
                Plotly.Plots.resize(document.getElementById("gripper-chart-plot"));
            } catch (e) {}
        }, 40);
    }

    function updateTrajectory() {
        var plot = document.getElementById("trajectory-plot");
        if (!plot || !trajectoryPlotInitialized || trajectoryX.length === 0) return;
        if (trajectoryInteractionActive || trajectoryPointerInside) {
            trajectoryPendingUpdate = true;
            return;
        }
        var len = trajectoryX.length;
        var colors = [];
        for (var i = 0; i < len; i++) {
            colors.push(len <= 1 ? 1 : i / (len - 1));
        }
        trajectoryPendingUpdate = false;
        Plotly.update("trajectory-plot", {
            x: [trajectoryX, trajectoryX, [trajectoryX[len - 1]]],
            y: [trajectoryY, trajectoryY, [trajectoryY[len - 1]]],
            z: [trajectoryZ, trajectoryZ, [trajectoryZ[len - 1]]],
            "marker.color": [null, colors, null]
        }, {}, [0, 1, 2]);
    }

    document.getElementById("btn-traj-clear").addEventListener("click", function () {
        trajectoryX = [];
        trajectoryY = [];
        trajectoryZ = [];
        configureTrajectoryPlot(true);
        resetTrajectoryCamera();
    });

    document.getElementById("btn-traj-export").addEventListener("click", function () {
        var csv = "x_mm,y_mm,z_mm\n";
        for (var i = 0; i < trajectoryX.length; i++) {
            csv += trajectoryX[i].toFixed(2) + ","
                + trajectoryY[i].toFixed(2) + ","
                + trajectoryZ[i].toFixed(2) + "\n";
        }
        var blob = new Blob([csv], { type: "text/csv" });
        var a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = "trajectory_" + Date.now() + ".csv";
        a.click();
    });

    function configureSensorPlot() {
        if (!document.getElementById("sensor-chart-plot")) return;
        var yMin = parseFloat(sensorConfig.y_min != null ? sensorConfig.y_min : 0);
        var yMax = parseFloat(sensorConfig.y_max != null ? sensorConfig.y_max : 1100);
        if (sensorConfig.auto_range !== false) {
            var values = sensorSeries1.concat(sensorSeries2, sensorSeries3).filter(function (v) {
                return Number.isFinite(v);
            });
            if (values.length) {
                var dataMin = Math.min.apply(null, values);
                var dataMax = Math.max.apply(null, values);
                var span = dataMax - dataMin;
                var minSpan = Math.max(parseFloat(sensorConfig.min_range_span || 300), 1);
                var effectiveSpan = Math.max(span, minSpan);
                var padRatio = parseFloat(sensorConfig.auto_range_padding_ratio || 0.08);
                var pad = Math.max(effectiveSpan * padRatio, minSpan * 0.04);
                var center = (dataMin + dataMax) / 2;
                var half = (effectiveSpan / 2) + pad;
                yMin = center - half;
                yMax = center + half;
            }
        }
        Plotly.react("sensor-chart-plot", [
            {
                x: sensorTimestamps,
                y: sensorSeries1,
                type: "scatter",
                mode: "lines",
                line: { color: "#ff5252", width: 2 },
                name: "Left",
            },
            {
                x: sensorTimestamps,
                y: sensorSeries2,
                type: "scatter",
                mode: "lines",
                line: { color: "#5b8dff", width: 2 },
                name: "Right",
            },
            {
                x: sensorTimestamps,
                y: sensorSeries3,
                type: "scatter",
                mode: "lines",
                line: { color: "#45f2a1", width: 2 },
                name: "Front",
            }
        ], {
            paper_bgcolor: "#0d1520",
            plot_bgcolor: "#0d1520",
            margin: { l: 40, r: 10, t: 12, b: 22 },
            xaxis: { color: "#5a6a7e", gridcolor: "#1a2332", showticklabels: false },
            yaxis: {
                color: "#5a6a7e",
                gridcolor: "#1a2332",
                range: [yMin, yMax]
            },
            font: { family: "Inter, sans-serif", size: 10, color: "#8b9bb4" },
            showlegend: false,
        }, {
            displayModeBar: false,
            responsive: true,
        });
    }

    function configureGripperPlot() {
        if (!document.getElementById("gripper-chart-plot")) return;
        var values = gripperSeries1.concat(gripperSeries2, gripperSeries3).filter(function (v) {
            return Number.isFinite(v);
        });
        var yMin = 0;
        var yMax = 3200;
        if (values.length) {
            var dataMin = Math.min.apply(null, values);
            var dataMax = Math.max.apply(null, values);
            var span = Math.max(dataMax - dataMin, 120);
            var pad = Math.max(span * 0.08, 24);
            yMin = Math.max(0, dataMin - pad);
            yMax = dataMax + pad;
        }
        var plotX = gripperTimestamps.slice();
        var plotYRight = gripperSeries2.slice();
        var plotYLeft = gripperSeries1.slice();
        var plotYFront = gripperSeries3.slice();
        if (plotX.length === 1) {
            var ts = plotX[0];
            plotX = [ts - 1, ts];
            plotYLeft = [plotYLeft[0], plotYLeft[0]];
            plotYRight = [plotYRight[0], plotYRight[0]];
            plotYFront = [plotYFront[0], plotYFront[0]];
        }
        Plotly.react("gripper-chart-plot", [
            {
                x: plotX,
                y: plotYLeft,
                type: "scatter",
                mode: "lines",
                line: { color: "#ff5252", width: 2 },
                connectgaps: true,
                name: "Left",
            },
            {
                x: plotX,
                y: plotYRight,
                type: "scatter",
                mode: "lines",
                line: { color: "#5b8dff", width: 2 },
                connectgaps: true,
                name: "Right",
            },
            {
                x: plotX,
                y: plotYFront,
                type: "scatter",
                mode: "lines",
                line: { color: "#45f2a1", width: 2 },
                connectgaps: true,
                name: "Front",
            }
        ], {
            paper_bgcolor: "#0d1520",
            plot_bgcolor: "#0d1520",
            margin: { l: 40, r: 10, t: 12, b: 22 },
            xaxis: { color: "#5a6a7e", gridcolor: "#1a2332", showticklabels: false },
            yaxis: {
                color: "#5a6a7e",
                gridcolor: "#1a2332",
                range: [yMin, yMax]
            },
            font: { family: "Inter, sans-serif", size: 10, color: "#8b9bb4" },
            showlegend: false,
        }, {
            displayModeBar: false,
            responsive: true,
        });
    }

    function updateSensorState(data) {
        if (!elSensorStatus) return;
        var connected = !!data.connected;
        var status = String(data.status || (connected ? "success" : "error")).toLowerCase();
        var badgeText = "DISABLED";
        var badgeClass = "panel-badge";
        if (data.enabled) {
            if (status === "degraded") {
                badgeText = "DEGRADED";
                badgeClass += " btn-warning";
            } else if (connected) {
                badgeText = "LIVE";
                badgeClass += " validation-active";
            } else {
                badgeText = "OFFLINE";
            }
        }
        elSensorStatus.textContent = badgeText;
        elSensorStatus.className = badgeClass;
        var sample = data.last_sample || {};
        var channels = Array.isArray(sample.channels) ? sample.channels : [];
        var channelErrors = channels
            .map(function (channel, idx) {
                if (!channel || channel.ok !== false || !channel.last_error) return "";
                return "S" + (idx + 1) + ": " + channel.last_error;
            })
            .filter(Boolean);
        elSensorError.textContent = channelErrors.join(" | ") || data.last_error || sample.last_error || "--";
        if (sample) {
            elSensorValue1.textContent = channels[0] && channels[0].ok === false ? "--" : (sample.analog_value1 != null ? String(sample.analog_value1) : "--");
            elSensorValue2.textContent = channels[1] && channels[1].ok === false ? "--" : (sample.analog_value2 != null ? String(sample.analog_value2) : "--");
            elSensorValue3.textContent = channels[2] && channels[2].ok === false ? "--" : (sample.analog_value3 != null ? String(sample.analog_value3) : "--");
        }
    }

    function updateSensorData(data) {
        function channelValue(row, index, key) {
            var channels = Array.isArray(row.channels) ? row.channels : [];
            if (channels[index] && channels[index].ok === false) {
                return null;
            }
            if (row[key] == null) return null;
            var value = Number(row[key]);
            return Number.isFinite(value) ? value : null;
        }

        var history = Array.isArray(data.history) ? data.history : [];
        if (history.length) {
            sensorTimestamps = history.map(function (row, idx) {
                return idx;
            });
            sensorSeries1 = history.map(function (row) { return channelValue(row, 0, "analog_value1"); });
            sensorSeries2 = history.map(function (row) { return channelValue(row, 1, "analog_value2"); });
            sensorSeries3 = history.map(function (row) { return channelValue(row, 2, "analog_value3"); });
        } else if (data.sample) {
            var limit = Math.max(parseInt(sensorConfig.history_points || "180", 10), 20);
            sensorTimestamps.push(sensorTimestamps.length);
            sensorSeries1.push(channelValue(data.sample, 0, "analog_value1"));
            sensorSeries2.push(channelValue(data.sample, 1, "analog_value2"));
            sensorSeries3.push(channelValue(data.sample, 2, "analog_value3"));
            if (sensorTimestamps.length > limit) {
                sensorTimestamps.shift();
                sensorSeries1.shift();
                sensorSeries2.shift();
                sensorSeries3.shift();
            }
        }
        if (data.sample) {
            var channels = Array.isArray(data.sample.channels) ? data.sample.channels : [];
            elSensorValue1.textContent = channels[0] && channels[0].ok === false ? "--" : String(data.sample.analog_value1);
            elSensorValue2.textContent = channels[1] && channels[1].ok === false ? "--" : String(data.sample.analog_value2);
            elSensorValue3.textContent = channels[2] && channels[2].ok === false ? "--" : String(data.sample.analog_value3);
            var channelErrors = channels
                .map(function (channel, idx) {
                    if (!channel || channel.ok !== false || !channel.last_error) return "";
                    return "S" + (idx + 1) + ": " + channel.last_error;
                })
                .filter(Boolean);
            if (elSensorError && channelErrors.length) {
                elSensorError.textContent = channelErrors.join(" | ");
            }
        }
        configureSensorPlot();
    }

    // -----------------------------------------------------------
    // Camera retry
    // -----------------------------------------------------------
    window.retryCamera = function (camKey, stream) {
        addLog("STEP", "Restarting camera: " + camKey);
        fetch("/api/cameras/" + camKey + "/restart", { method: "POST" })
            .then(function (resp) {
                return resp.json().then(function (data) {
                    if (!resp.ok || !data.ok) {
                        throw new Error((data && data.error) || "restart failed");
                    }
                    return data;
                });
            })
            .then(function () {
                addLog("STEP", "Camera restarted: " + camKey);
            })
            .catch(function (err) {
                addLog("ERROR", "Camera restart failed (" + camKey + "): " + err.message);
            });
        var img = document.getElementById(camKey + "-" + stream + "-feed");
        if (img) {
            img.src = "/stream/" + camKey + "/" + stream + "?t=" + Date.now();
        }
        var overlay = document.getElementById(camKey + "-" + stream + "-overlay");
        if (overlay) overlay.style.display = "none";
    };

    // Camera error handling for all 4 feeds
    ["cam1-rgb", "cam1-depth", "cam2-rgb", "cam2-depth", "claw-rgb"].forEach(function (id) {
        var img = document.getElementById(id + "-feed");
        if (img) {
            img.addEventListener("error", function () {
                var overlay = document.getElementById(id + "-overlay");
                if (overlay) overlay.style.display = "flex";
            });
        }
    });

    // -----------------------------------------------------------
    // Teach Mode
    // -----------------------------------------------------------
    var btnTeachStart = document.getElementById("btn-teach-start");
    var btnTeachWaypoint = document.getElementById("btn-teach-waypoint");
    var btnTeachStop = document.getElementById("btn-teach-stop");
    var btnTeachReplay = document.getElementById("btn-teach-replay");
    var btnTeachRegeneratePhase = document.getElementById("btn-teach-regenerate-phase");
    var elTeachReplayMode = document.getElementById("teach-replay-mode");

    btnTeachStart.addEventListener("click", function () {
        var name = document.getElementById("teach-name").value.trim();
        if (!name) {
            name = "recording_" + Date.now();
            document.getElementById("teach-name").value = name;
        }
        socket.emit("teach_start", { name: name });
        teachRecording = true;
        btnTeachStart.disabled = true;
        btnTeachWaypoint.disabled = false;
        btnTeachStop.disabled = false;
    });

    btnTeachWaypoint.addEventListener("click", function () {
        var speed = parseInt(document.getElementById("teach-speed").value) || 30;
        socket.emit("teach_waypoint", { gripper: "none", speed: speed });
    });

    btnTeachStop.addEventListener("click", function () {
        socket.emit("teach_stop");
        teachRecording = false;
        btnTeachStart.disabled = false;
        btnTeachWaypoint.disabled = true;
        btnTeachStop.disabled = true;
    });

    btnTeachReplay.addEventListener("click", function () {
        var name = elTeachReplaySelect.value;
        if (!name) {
            addLog("WARN", "No recording selected");
            return;
        }
        socket.emit("teach_replay", {
            name: name,
            replay_mode: elTeachReplayMode ? elTeachReplayMode.value : "phase"
        });
    });

    btnTeachRegeneratePhase.addEventListener("click", function () {
        var name = elTeachReplaySelect.value;
        if (!name) {
            addLog("WARN", "No recording selected");
            return;
        }
        socket.emit("teach_regenerate_phase", { name: name });
        addLog("STEP", "Regenerating phase YAML for " + name);
    });

    function updateTeachList(recordings) {
        availableTeachRecordings = {};
        // Update select
        elTeachReplaySelect.innerHTML = '<option value="">-- Select Recording --</option>';
        var listHtml = "";
        recordings.forEach(function (r) {
            var id = r.id || r.name;
            var label = r.name || id;
            availableTeachRecordings[id] = true;
            var opt = document.createElement("option");
            opt.value = id;
            opt.textContent = label + " (" + r.count + " pts"
                + (r.has_external_timeline ? ", timeline" : "")
                + (r.has_phase_yaml ? ", phase" : ", no-phase")
                + (r.has_episode_data ? ", episode" : "")
                + ")";
            elTeachReplaySelect.appendChild(opt);
            listHtml += label + " [" + id + "] (" + r.count + " pts"
                + (r.has_external_timeline ? ", timeline" : "")
                + ", phase=" + (r.has_phase_yaml ? "yes" : "no")
                + ", episode=" + (r.has_episode_data ? "yes" : "no")
                + ")  ";
        });
        elTeachList.textContent = recordings.length > 0
            ? "Saved: " + listHtml
            : "No recordings yet";
    }

    function renderTeachWaypoints(waypoints) {
        if (!waypoints || !waypoints.length) {
            elTeachWaypoints.textContent = "Current waypoints: none";
            return;
        }
        var lines = waypoints.map(function (wp, idx) {
            var pose = wp.pose || [];
            return [
                "#" + (idx + 1),
                "t=" + (wp.t_ms || 0) + "ms",
                "pose=[" + pose.slice(0, 3).join(", ") + "]",
                "speed=" + (wp.speed || "--") + "%"
            ].join("  ");
        });
        elTeachWaypoints.textContent = lines.join("\n");
    }

    function virtualEnvBaseUrl() {
        return (virtualEnvConfig.base_url || "").replace(/\/+$/, "");
    }

    function virtualEnvEmbedUrl() {
        var base = virtualEnvBaseUrl();
        if (!base) return "";
        return base + (virtualEnvConfig.embed_path || "/webrtc/");
    }

    function virtualEnvDesiredViews() {
        var views = virtualEnvConfig.desired_views || ["d435_left", "d435_right", "wrist"];
        return views.slice(0, 3);
    }

    function virtualEnvDisplayUrl(view) {
        var base = virtualEnvBaseUrl();
        if (!base) return "";
        var template = virtualEnvConfig.display_path_template || "/display/{view}.jpg";
        return base + template.replace("{view}", view);
    }

    function virtualEnvConsoleUrl() {
        var base = virtualEnvBaseUrl();
        if (!base) return "";
        return base + (virtualEnvConfig.console_path || "/ui");
    }

    function virtualEnvWebrtcUrl() {
        var base = virtualEnvBaseUrl();
        if (!base) return "";
        return base + (virtualEnvConfig.webrtc_path || "/webrtc/");
    }

    function configureVirtualEnv() {
        var base = virtualEnvBaseUrl();
        var enabled = !!virtualEnvConfig.enabled && !!base;
        elVirtualEnvBase.textContent = base || "--";
        elVirtualEnvOpenWebrtc.href = enabled ? virtualEnvWebrtcUrl() : "#";
        elVirtualEnvOpenConsole.href = enabled ? virtualEnvConsoleUrl() : "#";
        if (enabled) {
            refreshVirtualEnvFrames(true);
            refreshVirtualEnvStatus();
        } else {
            elVirtualEnvStatus.textContent = "Disabled";
            virtualEnvDesiredViews().forEach(function (view) {
                var img = document.getElementById("virtual-env-view-" + view);
                if (img) img.removeAttribute("src");
                var badge = document.getElementById("virtual-env-view-status-" + view);
                if (badge) badge.textContent = "--";
            });
        }
    }

    function refreshVirtualEnvFrames(force) {
        if (!virtualEnvConfig.enabled) return;
        var stamp = force ? ("t=" + Date.now()) : "";
        virtualEnvDesiredViews().forEach(function (view) {
            var img = document.getElementById("virtual-env-view-" + view);
            if (!img) return;
            var url = virtualEnvDisplayUrl(view);
            img.src = stamp ? (url + "?" + stamp) : (url + "?t=" + Date.now());
        });
    }

    function updateVirtualEnvStatus(status) {
        var meta = (status && status.web_display_meta) || {};
        var connected = !!status;
        elVirtualEnvStatus.textContent = connected ? "Connected" : "Unavailable";
        elVirtualEnvStatus.className = "panel-badge" + (connected ? " validation-active" : "");
        elVirtualEnvKind.textContent = status.current_kind || "--";
        elVirtualEnvTarget.textContent = status.monitor_target || meta.target || "--";
        elVirtualEnvPhase.textContent = meta.phase || meta.state || "--";
        elVirtualEnvTask.textContent = status.current_task ? JSON.stringify(status.current_task) : "--";
        elVirtualEnvViews.textContent = (meta.available_views || []).join(", ") || "--";
        elVirtualEnvUpdated.textContent = meta.updated_at
            ? Number(meta.updated_at).toFixed(3)
            : new Date().toLocaleTimeString();
        virtualEnvDesiredViews().forEach(function (view) {
            var badge = document.getElementById("virtual-env-view-status-" + view);
            if (!badge) return;
            badge.textContent = (meta.available_views || []).indexOf(view) >= 0 ? "Live" : "Missing";
            badge.className = "panel-badge" + ((meta.available_views || []).indexOf(view) >= 0 ? " validation-active" : "");
        });
    }

    function refreshVirtualEnvStatus() {
        var base = virtualEnvBaseUrl();
        if (!virtualEnvConfig.enabled || !base) return;
        fetch(base + "/status")
            .then(function (r) { return r.json(); })
            .then(function (data) { updateVirtualEnvStatus(data || {}); })
            .catch(function (err) {
                elVirtualEnvStatus.textContent = "Error";
                elVirtualEnvStatus.className = "panel-badge";
                elVirtualEnvKind.textContent = "--";
                elVirtualEnvTarget.textContent = "--";
                elVirtualEnvPhase.textContent = "--";
                elVirtualEnvTask.textContent = String(err);
                elVirtualEnvViews.textContent = "--";
                elVirtualEnvUpdated.textContent = "--";
            });
    }

    function postLocalVirtualEnv(path) {
        return fetch(path, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: "{}"
        }).then(function (r) {
            return r.json().then(function (data) {
                if (!r.ok || data.ok === false) {
                    throw new Error((data && (data.error || data.message)) || ("request failed: " + path));
                }
                return data;
            });
        });
    }

    function moduleEntry(name) {
        return moduleConfigMap[name] || { enabled: true, show_in_ui: true };
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

    function setPanelVisible(panelSuffix, visible) {
        var panel = document.getElementById("panel-" + panelSuffix);
        if (!panel) return;
        panel.style.display = visible ? "" : "none";
    }

    function applySensorPanelDefaults() {
        var panel = document.getElementById("panel-sensor-chart");
        if (!panel || !sensorConfig.panel || !sensorConfig.panel.default_rect) return;
        var rect = sensorConfig.panel.default_rect;
        panel.dataset.defaultX = String(rect.x != null ? rect.x : panel.dataset.defaultX);
        panel.dataset.defaultY = String(rect.y != null ? rect.y : panel.dataset.defaultY);
        panel.dataset.defaultW = String(rect.w != null ? rect.w : panel.dataset.defaultW);
        panel.dataset.defaultH = String(rect.h != null ? rect.h : panel.dataset.defaultH);
        var savedLayout = loadDashboardLayout();
        if (!savedLayout["sensor-chart"]) {
            applyPanelRect(panel, {
                x: parseInt(panel.dataset.defaultX || "0", 10),
                y: parseInt(panel.dataset.defaultY || "0", 10),
                w: parseInt(panel.dataset.defaultW || "4", 10),
                h: parseInt(panel.dataset.defaultH || "4", 10),
            });
        }
    }

    function applyModuleVisibility() {
        setPanelVisible("voice", moduleVisible("voice"));
        setPanelVisible("cam1-rgb", moduleVisible("cam1"));
        setPanelVisible("cam1-depth", moduleVisible("cam1"));
        setPanelVisible("cam2-rgb", moduleVisible("cam2"));
        setPanelVisible("cam2-depth", moduleVisible("cam2"));
        setPanelVisible("claw-rgb", moduleVisible("claw_cam"));
        setPanelVisible("sensor-chart", moduleVisible("sensor_chart"));
        setPanelVisible("monitor", moduleVisible("arm_monitor"));
        setPanelVisible("gripper", moduleVisible("gripper"));
        setPanelVisible("validation", moduleVisible("validation"));
        setPanelVisible("quick", moduleVisible("quick_pick"));
        setPanelVisible("vla", moduleVisible("vla"));
        setPanelVisible("dataset-capture", moduleVisible("dataset_capture"));
        setPanelVisible("trajectory", moduleVisible("trajectory"));
        setPanelVisible("teach", moduleVisible("teach"));
        setPanelVisible("logs", moduleVisible("logs"));

        var virtualVisible = moduleVisible("virtual_env");
        var virtualDataVisible = virtualVisible && moduleVisible("virtual_env_data");
        setPanelVisible("virtual-env-left", virtualVisible);
        setPanelVisible("virtual-env-right", virtualVisible);
        setPanelVisible("virtual-env-wrist", virtualVisible);
        setPanelVisible("virtual-env-data", virtualDataVisible);

        var teachOption = elValidationModeSelect
            ? elValidationModeSelect.querySelector('option[value="teach"]')
            : null;
        if (teachOption) {
            teachOption.hidden = !moduleVisible("teach");
            teachOption.disabled = !moduleVisible("teach");
            if (!moduleVisible("teach") && elValidationModeSelect.value === "teach") {
                elValidationModeSelect.value = "fixed";
            }
        }
        schedulePlotResize();
    }

    function defaultPanelRect(panel) {
        return {
            x: parseInt(panel.dataset.defaultX || "0", 10),
            y: parseInt(panel.dataset.defaultY || "0", 10),
            w: parseInt(panel.dataset.defaultW || "6", 10),
            h: parseInt(panel.dataset.defaultH || "3", 10),
        };
    }

    function minPanelRect(panel) {
        return {
            w: parseInt(panel.dataset.minW || "2", 10),
            h: parseInt(panel.dataset.minH || "2", 10),
        };
    }

    function clampPanelRect(panel, rect) {
        var minRect = minPanelRect(panel);
        var w = Math.max(minRect.w, Math.min(dashboardCols, rect.w));
        var h = Math.max(minRect.h, Math.min(dashboardRows, rect.h));
        var x = Math.max(0, Math.min(dashboardCols - w, rect.x));
        var y = Math.max(0, Math.min(dashboardRows - h, rect.y));
        return { x: x, y: y, w: w, h: h };
    }

    function loadDashboardLayout() {
        try {
            return JSON.parse(localStorage.getItem(dashboardLayoutKey) || "{}");
        } catch (e) {
            return {};
        }
    }

    function saveDashboardLayout() {
        var payload = {};
        dashboardPanels.forEach(function (panel) {
            payload[panel.dataset.panelId] = {
                x: parseInt(panel.dataset.x || "0", 10),
                y: parseInt(panel.dataset.y || "0", 10),
                w: parseInt(panel.dataset.w || "1", 10),
                h: parseInt(panel.dataset.h || "1", 10),
                z: parseInt(panel.dataset.z || "1", 10),
            };
        });
        try {
            localStorage.setItem(dashboardLayoutKey, JSON.stringify(payload));
        } catch (e) {}
    }

    function bringPanelToFront(panel) {
        dashboardZCounter += 1;
        panel.dataset.z = String(dashboardZCounter);
        panel.style.zIndex = String(dashboardZCounter);
    }

    function applyPanelRect(panel, rect) {
        var safeRect = clampPanelRect(panel, rect);
        panel.dataset.x = String(safeRect.x);
        panel.dataset.y = String(safeRect.y);
        panel.dataset.w = String(safeRect.w);
        panel.dataset.h = String(safeRect.h);
        panel.style.left = (safeRect.x * 100 / dashboardCols) + "%";
        panel.style.top = (safeRect.y * 100 / dashboardRows) + "%";
        panel.style.width = (safeRect.w * 100 / dashboardCols) + "%";
        panel.style.height = (safeRect.h * 100 / dashboardRows) + "%";
    }

    function setupPanelInteraction(panel) {
        var header = panel.querySelector(".panel-header");
        if (!header) return;

        var resizeHandle = document.createElement("div");
        resizeHandle.className = "panel-resize-handle";
        panel.appendChild(resizeHandle);

        function startPointerAction(event, mode) {
            if (event.button !== 0) return;
            if (mode === "drag") {
                var tag = event.target.tagName;
                if (/(BUTTON|A|INPUT|SELECT|TEXTAREA|SVG|PATH|LINE|SPAN)/.test(tag) && event.target !== header) {
                    if (event.target.closest(".panel-actions, .lang-toggle, button, a, input, select, textarea")) {
                        return;
                    }
                }
            }

            event.preventDefault();
            bringPanelToFront(panel);

            var boardRect = elDashboardBoard.getBoundingClientRect();
            var startX = event.clientX;
            var startY = event.clientY;
            var startRect = {
                x: parseInt(panel.dataset.x || "0", 10),
                y: parseInt(panel.dataset.y || "0", 10),
                w: parseInt(panel.dataset.w || "1", 10),
                h: parseInt(panel.dataset.h || "1", 10),
            };
            var className = mode === "drag" ? "dragging" : "resizing";
            panel.classList.add(className);

            function onMove(moveEvent) {
                var dxPx = moveEvent.clientX - startX;
                var dyPx = moveEvent.clientY - startY;
                var dx = Math.round(dxPx / (boardRect.width / dashboardCols));
                var dy = Math.round(dyPx / (boardRect.height / dashboardRows));
                if (mode === "drag") {
                    applyPanelRect(panel, {
                        x: startRect.x + dx,
                        y: startRect.y + dy,
                        w: startRect.w,
                        h: startRect.h,
                    });
                } else {
                    applyPanelRect(panel, {
                        x: startRect.x,
                        y: startRect.y,
                        w: startRect.w + dx,
                        h: startRect.h + dy,
                    });
                    schedulePlotResize();
                }
            }

            function onUp() {
                panel.classList.remove(className);
                window.removeEventListener("pointermove", onMove);
                window.removeEventListener("pointerup", onUp);
                saveDashboardLayout();
                schedulePlotResize();
            }

            window.addEventListener("pointermove", onMove);
            window.addEventListener("pointerup", onUp);
        }

        header.addEventListener("pointerdown", function (event) {
            if (event.target.closest(".panel-actions, .lang-toggle, button, a, input, select, textarea")) {
                return;
            }
            startPointerAction(event, "drag");
        });

        resizeHandle.addEventListener("pointerdown", function (event) {
            startPointerAction(event, "resize");
        });

        panel.addEventListener("pointerdown", function () {
            bringPanelToFront(panel);
            saveDashboardLayout();
        });
    }

    function initDashboard() {
        dashboardPanels = Array.prototype.slice.call(
            document.querySelectorAll(".dashboard-panel")
        );
        var savedLayout = loadDashboardLayout();

        dashboardPanels.forEach(function (panel, index) {
            var panelId = panel.dataset.panelId || ("panel_" + index);
            var saved = savedLayout[panelId] || {};
            var rect = {
                x: saved.x !== undefined ? saved.x : defaultPanelRect(panel).x,
                y: saved.y !== undefined ? saved.y : defaultPanelRect(panel).y,
                w: saved.w !== undefined ? saved.w : defaultPanelRect(panel).w,
                h: saved.h !== undefined ? saved.h : defaultPanelRect(panel).h,
            };
            dashboardZCounter = Math.max(dashboardZCounter, saved.z || index + 10);
            panel.dataset.z = String(saved.z || index + 10);
            panel.style.zIndex = panel.dataset.z;
            applyPanelRect(panel, rect);
            setupPanelInteraction(panel);
        });

        window.addEventListener("resize", function () {
            dashboardPanels.forEach(function (panel) {
                applyPanelRect(panel, {
                    x: parseInt(panel.dataset.x || "0", 10),
                    y: parseInt(panel.dataset.y || "0", 10),
                    w: parseInt(panel.dataset.w || "1", 10),
                    h: parseInt(panel.dataset.h || "1", 10),
                });
            });
            schedulePlotResize();
        });
        schedulePlotResize();
    }

    function resetDashboardLayout() {
        try {
            localStorage.removeItem(dashboardLayoutKey);
        } catch (e) {}
        dashboardPanels.forEach(function (panel, index) {
            var rect = defaultPanelRect(panel);
            panel.dataset.z = String(index + 10);
            panel.style.zIndex = panel.dataset.z;
            applyPanelRect(panel, rect);
        });
        dashboardZCounter = 10 + dashboardPanels.length;
        saveDashboardLayout();
        schedulePlotResize();
    }

    // -----------------------------------------------------------
    // Init log
    // -----------------------------------------------------------
    initDashboard();
    configureTrajectoryPlot(true);
    configureSensorPlot();
    document.getElementById("btn-layout-reset").addEventListener("click", function () {
        resetDashboardLayout();
        addLog("STEP", "Dashboard layout reset");
    });
    addLog("STEP", "Voice Pick Demo loaded");
    addLog("STEP", "Language: " + currentLang + " | Voice input mode: " + asrMode);
    setInterval(refreshVirtualEnvStatus, 2000);
    setInterval(function () {
        refreshVirtualEnvFrames(false);
    }, 700);

})();
