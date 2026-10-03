(function () {
    "use strict";

    // ── Camera zoom ───────────────────────────────────────────────────────────
    // Click any camera feed image to open a half-screen live overlay.
    var zoomOverlay = null;

    function openZoom(src, label) {
        if (zoomOverlay) zoomOverlay.remove();

        var overlay = document.createElement("div");
        overlay.style.cssText = [
            "position:fixed", "top:0", "left:0", "width:100%", "height:100%",
            "background:rgba(0,0,0,0.88)", "z-index:9999",
            "display:flex", "flex-direction:column",
            "align-items:center", "justify-content:center",
            "cursor:zoom-out",
        ].join(";");

        var title = document.createElement("div");
        title.textContent = label || "";
        title.style.cssText = "color:#ddd;font-size:13px;margin-bottom:8px;font-family:monospace;";

        var img = document.createElement("img");
        img.src = src;
        img.alt = label || "";
        img.style.cssText = [
            "max-width:50vw", "max-height:80vh",
            "object-fit:contain",
            "border:2px solid #444", "border-radius:4px",
            "cursor:default",
        ].join(";");

        var closeBtn = document.createElement("button");
        closeBtn.textContent = "×";  // ×
        closeBtn.style.cssText = [
            "position:absolute", "top:10px", "right:16px",
            "background:none", "border:none", "color:#aaa",
            "font-size:30px", "line-height:1", "cursor:pointer", "padding:4px 8px",
        ].join(";");

        overlay.appendChild(title);
        overlay.appendChild(img);
        overlay.appendChild(closeBtn);
        document.body.appendChild(overlay);
        zoomOverlay = overlay;

        function dismiss() {
            overlay.remove();
            if (zoomOverlay === overlay) zoomOverlay = null;
            document.removeEventListener("keydown", onKey);
        }
        function onKey(e) { if (e.key === "Escape") dismiss(); }

        overlay.addEventListener("click", dismiss);
        img.addEventListener("click", function (e) { e.stopPropagation(); });
        closeBtn.addEventListener("click", dismiss);
        document.addEventListener("keydown", onKey);
    }

    // Attach to all existing camera container images
    var camImgs = document.querySelectorAll(".camera-container img");
    for (var i = 0; i < camImgs.length; i++) {
        (function (el) {
            el.style.cursor = "zoom-in";
            el.addEventListener("click", function () {
                openZoom(el.src, el.alt);
            });
        })(camImgs[i]);
    }

    // ── VLA panel ────────────────────────────────────────────────────────────
    var panel = document.getElementById("panel-vla");
    if (!panel || typeof io !== "function") return;

    var VLA_CKPT  = "data/datasets/checkpoints/combined_nchc_20260529_120309";
    // Combined model is language-conditioned: pick the precomputed English embedding by instruction.
    var VLA_EMBEDS = {
        "拿起梯形": "data/vla_embed_trapezoid_en.pt",
        "拿起筷子": "data/vla_embed_chopsticks_en.pt",
    };
    var VLA_EMBED_DEFAULT = "data/vla_embed_trapezoid_en.pt";

    var elBadge = document.getElementById("vla-badge");
    var elLog   = document.getElementById("vla-log-container");

    var socket = io({ reconnection: true, reconnectionDelay: 1000 });

    function addVlaLog(level, message, ts) {
        if (!elLog) return;
        ts = ts || new Date().toLocaleTimeString();
        var line    = document.createElement("div");
        var tsSpan  = document.createElement("span");
        var tagSpan = document.createElement("span");
        var msgSpan = document.createElement("span");
        line.className    = "log-line " + String(level);
        tsSpan.className  = "ts";  tsSpan.textContent  = ts;
        tagSpan.className = "tag"; tagSpan.textContent = "[" + level + "]";
        msgSpan.className = "msg"; msgSpan.textContent = String(message || "");
        line.appendChild(tsSpan);
        line.appendChild(document.createTextNode(" "));
        line.appendChild(tagSpan);
        line.appendChild(document.createTextNode(" "));
        line.appendChild(msgSpan);
        elLog.appendChild(line);
        elLog.scrollTop = elLog.scrollHeight;
    }

    var btnGate = document.getElementById("btn-vla-gate");

    socket.on("vla_status", function (d) {
        if (elBadge) {
            var txt = (d.status || "?") + " (" + (d.step || 0) + ")";
            if (d.auto_stage) txt = (d.auto_object || "") + " " + d.auto_stage + " (" + (d.step || 0) + ")";
            elBadge.textContent = txt;
            elBadge.style.background =
                d.status === "error"                ? "#ef4444" :
                d.awaiting_gate                     ? "#f59e0b" :
                d.status === "loading"              ? "#f59e0b" :
                (d.running || d.status === "running") ? "#22c55e" : "";
        }
        // Gate button appears only while the auto tail is waiting on the AGX
        // handoff; its label names the gate it acknowledges.
        if (btnGate) {
            if (d.awaiting_gate) {
                var gateLabels = { descend_ok: "DESCEND ✓", pick_done: "PICK DONE ✓", released: "RELEASED ✓" };
                btnGate.style.display = "";
                btnGate.textContent = gateLabels[d.awaiting_gate] || (d.awaiting_gate + " ✓");
                btnGate.dataset.gate = d.awaiting_gate;
            } else {
                btnGate.style.display = "none";
                btnGate.dataset.gate = "";
            }
        }
    });

    socket.on("vla_log", function (d) {
        addVlaLog(d.level, d.message, d.timestamp);
    });

    var btnStart = document.getElementById("btn-vla-start");
    var btnStop  = document.getElementById("btn-vla-stop");

    if (btnStart) {
        btnStart.addEventListener("click", function () {
            var instr = document.getElementById("vla-instruction");
            var steps = document.getElementById("vla-max-steps");
            var dry   = document.getElementById("vla-dry-run");
            var instruction = instr ? (instr.value || "拿起梯形") : "拿起梯形";
            socket.emit("vla_start", {
                checkpoint: VLA_CKPT,
                embed_path: VLA_EMBEDS[instruction] || VLA_EMBED_DEFAULT,
                instruction: instruction,
                max_steps:   steps ? parseInt(steps.value || "200", 10) : 200,
                exec_steps:  8,
                dry_run:     dry ? dry.checked : true,
            });
        });
    }

    if (btnStop) {
        btnStop.addEventListener("click", function () {
            socket.emit("vla_stop");
        });
    }

    // ── Auto-run (three-object demo §3) ──────────────────────────────────────
    var btnAuto = document.getElementById("btn-vla-auto");
    if (btnAuto) {
        btnAuto.addEventListener("click", function () {
            var obj  = document.getElementById("vla-auto-object");
            var mode = document.getElementById("vla-auto-mode");
            var dry  = document.getElementById("vla-dry-run");
            socket.emit("vla_auto_start", {
                object:  obj ? obj.value : "trapezoid",
                mode:    mode ? mode.value : "vla",
                dry_run: dry ? dry.checked : true,
            });
        });
    }

    if (btnGate) {
        btnGate.addEventListener("click", function () {
            if (btnGate.dataset.gate) socket.emit("vla_gate", { name: btnGate.dataset.gate });
        });
    }

    var btnTune = document.getElementById("btn-vla-tune");
    if (btnTune) {
        btnTune.addEventListener("click", function () {
            var knobs = ["conf_floor", "arrival_tol_mm", "stay_mm", "servo_max_step_mm"];
            for (var k = 0; k < knobs.length; k++) {
                var el = document.getElementById("tune-" + knobs[k]);
                if (el && el.value !== "") {
                    socket.emit("vla_tune", { key: knobs[k], value: parseFloat(el.value) });
                }
            }
        });
    }

})();
