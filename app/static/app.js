document.addEventListener('DOMContentLoaded', async () => {
    /* =========================================================================
       0. Fetch runtime config from backend (pitch dimensions, thresholds)
          Falls back to safe defaults so the dashboard works without a server.
       ========================================================================= */
    let pitchConfig = {
        pitch_length_m: 105.0,
        pitch_width_m: 68.0,
        fatigue_high_threshold: 0.75,
    };
    try {
        const cfgRes = await fetch('/api/v1/config');
        if (cfgRes.ok) pitchConfig = await cfgRes.json();
    } catch (_) { /* use defaults */ }

    /* =========================================================================
       1. Base Squad Data & Pitch Mapping
       ========================================================================= */
    // Initializing starting XI mapping exactly matching backend definitions
    const squad = [
        { id: "mci_gk1",  name: "Ederson",  number: 31, pos: "GK",  basePos: [5, 50] },
        { id: "mci_cb1",  name: "Dias",     number: 3,  pos: "CB",  basePos: [25, 30] },
        { id: "mci_cb2",  name: "Akanji",   number: 25, pos: "CB",  basePos: [25, 70] },
        { id: "mci_rb1",  name: "Walker",   number: 2,  pos: "RB",  basePos: [30, 90] },
        { id: "mci_lb1",  name: "Gvardiol", number: 24, pos: "LB",  basePos: [30, 10] },
        { id: "mci_cdm1", name: "Rodri",    number: 16, pos: "CDM", basePos: [45, 50] },
        { id: "mci_cm1",  name: "De Bruyne",number: 17, pos: "CM",  basePos: [65, 35] },
        { id: "mci_cm2",  name: "Silva",    number: 20, pos: "CM",  basePos: [65, 65] },
        { id: "mci_rw1",  name: "Doku",     number: 11, pos: "RW",  basePos: [75, 85] },
        { id: "mci_lw1",  name: "Grealish", number: 10, pos: "LW",  basePos: [75, 15] },
        { id: "mci_st1",  name: "Haaland",  number: 9,  pos: "ST",  basePos: [85, 50] }
    ];

    let currentTelemetry = [];
    let matchMinute = 65;

    /* =========================================================================
       2. Telemetry Simulation Engine
       ========================================================================= */
    function generateTelemetry() {
        let maxSpeed = 0;
        let sumDist = 0;
        let highFatigueCount = 0;

        currentTelemetry = squad.map(p => {
            // Apply a bit of random movement (-5 to +5 on X and Y)
            const rx = (Math.random() - 0.5) * 10;
            const ry = (Math.random() - 0.5) * 15;
            
            // X goes from 0 to 100 (left goal to right goal)
            // Y goes from 0 to 100 (top sideline to bottom sideline)
            const x = Math.max(2, Math.min(98, p.basePos[0] + rx));
            const y = Math.max(2, Math.min(98, p.basePos[1] + ry));
            
            // Generate Speed & Fatigue
            const speed = Math.random() * (p.pos === 'GK' ? 8 : (p.pos === 'RW' || p.pos === 'LW' ? 34 : 26));
            if (speed > maxSpeed) maxSpeed = speed;
            
            const dist = 0.5 + (Math.random() * 0.8);
            sumDist += dist;
            
            const fatigue = 0.4 + (Math.random() * 0.4);
            if (fatigue > pitchConfig.fatigue_high_threshold) highFatigueCount++;

            return {
                player_id: p.id,
                _name: p.name,
                _number: p.number,
                _pos: p.pos,
                speed: parseFloat(speed.toFixed(1)),
                distance_covered: parseFloat(dist.toFixed(2)),
                heart_rate: 140 + Math.floor(Math.random() * 40),
                position_x: parseFloat(Math.max(0, Math.min(pitchConfig.pitch_length_m, x * (pitchConfig.pitch_length_m / 100))).toFixed(1)),
                position_y: parseFloat(Math.max(0, Math.min(pitchConfig.pitch_width_m,  y * (pitchConfig.pitch_width_m  / 100))).toFixed(1)),
                fatigue_index: parseFloat(fatigue.toFixed(2)),
                
                // Keep raw % for rendering
                _render_x: x, 
                _render_y: y
            };
        });

        // Update UI Stats
        document.getElementById('stat-dist').innerText = (sumDist / 11).toFixed(2) + " km";
        document.getElementById('stat-speed').innerText = maxSpeed.toFixed(1) + " km/h";
        
        const fEl = document.getElementById('stat-fatigue');
        const fBox = document.getElementById('fatigue-warning');
        fEl.innerText = `${highFatigueCount} Players > ${pitchConfig.fatigue_high_threshold}`;
        if (highFatigueCount >= 3) {
            fBox.classList.add('warning');
            addSystemMessage("⚠️ Alert: Several players are showing critical fatigue indexes.");
        } else {
            fBox.classList.remove('warning');
        }

        renderPitch();
    }

    function renderPitch() {
        const layer = document.getElementById('players-layer');
        layer.innerHTML = ''; // Fast DOM clear

        currentTelemetry.forEach(t => {
            const el = document.createElement('div');
            el.className = 'player-node';
            if (t.fatigue_index > pitchConfig.fatigue_high_threshold) el.classList.add('fatigue-warning');
            
            // X is left-to-right. Y is top-to-bottom.
            el.style.left = `${t._render_x}%`;
            el.style.top = `${t._render_y}%`;
            el.innerText = t._number;
            
            el.setAttribute('data-tooltip', `${t._name} | HR: ${t.heart_rate} | F: ${t.fatigue_index}`);
            
            layer.appendChild(el);
        });
    }

    /* =========================================================================
       3. Agent Chat Interface
       ========================================================================= */
    const chatContainer = document.getElementById('chat-container');
    const input = document.getElementById('query-input');
    const btnSend = document.getElementById('send-btn');
    const routeLabel = document.getElementById('active-route');

    function appendMessage(sender, text, routeUsed = null) {
        const msg = document.createElement('div');
        msg.className = `message ${sender}`;
        
        let header = sender === 'user' ? 'Coach' : 'Orchestrator';
        if (routeUsed) {
            header = `Orchestrator [${routeUsed}]`;
        }

        msg.innerHTML = `
            <div class="avatar">${sender === 'user' ? '⚽' : '🤖'}</div>
            <div class="content">
                <strong>${header}</strong>
                <div>${text}</div>
            </div>
        `;
        chatContainer.appendChild(msg);
        chatContainer.scrollTop = chatContainer.scrollHeight;
    }

    function addSystemMessage(text) {
        appendMessage('system', `<p style="color:var(--text-secondary); font-size:0.85rem;">[System] ${text}</p>`);
    }

    async function sendQuery(queryText) {
        if (!queryText.trim()) return;
        
        appendMessage('user', queryText);
        input.value = '';
        input.disabled = true;
        btnSend.disabled = true;

        routeLabel.innerText = 'Analyzing...';
        routeLabel.style.color = 'var(--text-accent)';

        // Remove private UI fields before sending to API
        const payloadTelemetry = currentTelemetry.map(t => {
            const { _name, _number, _pos, _render_x, _render_y, ...apiData } = t;
            return apiData;
        });

        try {
            const response = await fetch('/api/v1/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    query: queryText,
                    telemetry_context: payloadTelemetry
                })
            });

            if (!response.ok) {
                const err = await response.json();
                throw new Error(err.detail || `Server error ${response.status}`);
            }

            const data = await response.json();
            const route = data.route_used;
            routeLabel.innerText = `Route: ${route}`;
            routeLabel.style.color = 'white';

            // Convert simple line breaks to HTML
            const formattedInsight = data.insight.replace(/\n/g, '<br>');
            appendMessage('system', formattedInsight, route);

        } catch (error) {
            appendMessage('system', `<span style="color:var(--danger)"><strong>Error:</strong> ${error.message}</span>`);
            routeLabel.innerText = 'Error';
            routeLabel.style.color = 'var(--danger)';
        } finally {
            input.disabled = false;
            btnSend.disabled = false;
            input.focus();
        }
    }

    /* =========================================================================
       4. Event Listeners
       ========================================================================= */
    document.getElementById('simulate-btn').addEventListener('click', () => {
        // Just increment match time a bit
        matchMinute += 1;
        document.getElementById('match-time').innerText = `${matchMinute}:00`;
        
        generateTelemetry();
        addSystemMessage(`Match state advanced to ${matchMinute} min. Pushed new coordinates.`);
    });

    btnSend.addEventListener('click', () => sendQuery(input.value));
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendQuery(input.value);
    });

    // Handle suggested queries
    document.querySelectorAll('.suggestion').forEach(btn => {
        btn.addEventListener('click', (e) => {
            const text = e.target.innerText;
            sendQuery(text);
        });
    });

    // Initialize the visual pitch immediately
    generateTelemetry();
});
