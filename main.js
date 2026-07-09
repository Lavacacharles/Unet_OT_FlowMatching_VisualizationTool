const cifar10_labels_es = {
    0: 'avión',
    1: 'automóvil',
    2: 'pájaro',
    3: 'gato',
    4: 'ciervo',
    5: 'perro',
    6: 'rana',
    7: 'caballo',
    8: 'barco',
    9: 'camión'
};
// --- 1. State Manager ---
const State = {
    workspace: 'evolution', // 'evolution', 'generation', 'compare'
    epoch: 1500,
    step: 5,
    compareMode: false,
    selectedSample: null,
    listeners: [],
    
    update(changes) {
        Object.assign(this, changes);
        this.notify();
    },
    subscribe(listener) { this.listeners.push(listener); },
    notify() { this.listeners.forEach(fn => fn(this)); }
};

let dataset = [];
let globalXDomain = [-10, 10];
let globalYDomain = [-10, 10];
const colorScale = d3.scaleOrdinal(d3.schemeCategory10);

// --- 2. Definición de Workspaces (Configuración Contextual) ---
// --- 2. Definición de Workspaces (Configuración Contextual) ---
const WORKSPACES = {
    'evolution': {
        title: "Training Evolution",
        desc: "Acciones:<br>1) Desplace el control 'Training Epoch' para observar el movimiento de los puntos.<br>2) Haga clic en un punto para inspeccionar su imagen generada.",
        help: "Contexto Analítico: Explore cómo el modelo aprende a organizar y separar las clases en el espacio latente a través de las distintas etapas del entrenamiento.",
        emptyState: "Seleccione un punto en el espacio latente para identificar su clase y visualizar la imagen generada en la época actual.",
        defaults: { epoch: 1500, step: 5, compareMode: false },
        minimize: ['control-step']
    },
    'generation': {
        title: "Generation Dynamics",
        desc: "Acciones:<br>1) Seleccione un punto en el gráfico.<br>2) Mueva el control 'Generation Step' para observar cómo el ruido se transforma en una imagen.",
        help: "Contexto Analítico: Analice la trayectoria de generación de una muestra y correlacione cómo la atención espacial guía los cambios visuales paso a paso.",
        emptyState: "La dinámica de generación se evalúa a nivel de muestra. Seleccione un punto para desplegar su trayectoria completa de probabilidad.",
        defaults: { epoch: 1500, step: 0, compareMode: false },
        minimize: ['control-epoch']
    },
    'compare': {
        title: "Structural Comparison",
        desc: "Acciones:<br>1) Modifique la época de entrenamiento.<br>2) Haga clic en cualquier punto para contrastar sus resultados en el panel derecho.",
        help: "Contexto Analítico: Compare simultáneamente las representaciones latentes, mapas de atención y resultados de un estado inmaduro (Época 250) frente al estado actual.",
        emptyState: "Seleccione una muestra para comparar directamente su representación visual en la época 250 versus la época seleccionada.",
        defaults: { epoch: 1500, step: 5, compareMode: true },
        minimize: ['control-step']
    }
};

// --- 3. Inicialización ---
d3.json('assets/dataset.json').then(data => {
    dataset = data;
    // Escalas globales fijas para determinismo espacial
    globalXDomain = d3.extent(dataset, d => d.x);
    globalYDomain = d3.extent(dataset, d => d.y);
    
    setupUI();
    State.notify(); 
});

function setupUI() {
    // Sliders
    d3.select("#epoch-slider").on("input", function() { State.update({ epoch: +this.value }); });
    d3.select("#step-slider").on("input", function() { State.update({ step: +this.value }); });

    // Workspace Buttons
    d3.selectAll(".ws-btn").on("click", function() {
        const wsId = this.getAttribute("data-ws");
        d3.selectAll(".ws-btn").classed("active", false);
        d3.select(this).classed("active", true);
        
        const config = WORKSPACES[wsId];
        // Aplicar "soft defaults" sin bloquear la herramienta
        State.update({ workspace: wsId, ...config.defaults });
    });

    State.subscribe(updateContextualUI);
    State.subscribe(renderScatterplots);
    State.subscribe(renderDetailPanel);
}

// --- 4. Actualización de Interfaz (Progressive Disclosure) ---
function updateContextualUI(state) {
    const config = WORKSPACES[state.workspace];
    
    // Status Bar
    d3.select("#hud-workspace").text(config.title);
    d3.select("#hud-epoch").text(state.compareMode ? `250 vs ${state.epoch}` : state.epoch);
    d3.select("#hud-step").text(state.step);
    d3.select("#hud-selected").text(state.selectedSample !== null ? state.selectedSample : "None");

    // Sliders Sync
    d3.select("#epoch-slider").property("value", state.epoch);
    d3.select("#step-slider").property("value", state.step);

    // Progressive Disclosure (CSS Classes)
    d3.select("#control-epoch").classed("minimized-control", config.minimize.includes('control-epoch'));
    d3.select("#control-step").classed("minimized-control", config.minimize.includes('control-step'));

    // Task Banner & Help
    d3.select("#task-title").text(config.title);
    d3.select("#task-desc").html(config.desc);
    d3.select("#contextual-help").text(config.help);
}

// --- 5. Motor de Renderizado (Estricto determinismo espacial) ---
function renderScatterplots(state) {
    const container = d3.select("#view-primary").node().getBoundingClientRect();
    const margin = {top: 20, right: 20, bottom: 20, left: 20};
    const width = container.width - margin.left - margin.right;
    const height = container.height - margin.top - margin.bottom;

    const xScale = d3.scaleLinear().domain(globalXDomain).nice().range([0, width]);
    const yScale = d3.scaleLinear().domain(globalYDomain).nice().range([height, 0]);

    if (state.compareMode) {
        // QUICK COMPARE: Izquierda (250) fijo, Derecha (state.epoch) fijo.
        d3.select("#view-secondary").classed("hidden", false);
        d3.select("#title-primary").text(`Epoch 250 (Step ${state.step})`);
        d3.select("#title-secondary").text(`Epoch ${state.epoch} (Step ${state.step})`);

        const dataLeft = dataset.filter(d => d.epoch === 250 && d.generation_step === state.step);
        const dataRight = dataset.filter(d => d.epoch === state.epoch && d.generation_step === state.step);
        
        drawScatter("#svg-primary", dataLeft, state, xScale, yScale);
        drawScatter("#svg-secondary", dataRight, state, xScale, yScale);
    } else {
        d3.select("#view-secondary").classed("hidden", true);
        d3.select("#title-primary").text(`Epoch ${state.epoch} (Step ${state.step})`);

        const data = dataset.filter(d => d.epoch === state.epoch && d.generation_step === state.step);
        drawScatter("#svg-primary", data, state, xScale, yScale);
    }
}

function drawScatter(svgId, data, state, xScale, yScale) {
    const svg = d3.select(svgId);
    if (svg.empty()) return;

    let g = svg.select("g.main-group");
    if (g.empty()) g = svg.append("g").attr("class", "main-group").attr("transform", "translate(20,20)");

    const dots = g.selectAll(".dot").data(data, d => d.sample_id);

    dots.join(
        enter => enter.append("circle")
            .attr("class", "dot")
            .attr("r", d => d.sample_id === state.selectedSample ? 6 : 4)
            .attr("cx", d => xScale(d.x))
            .attr("cy", d => yScale(d.y))
            .attr("fill", d => colorScale(d.label))
            .attr("stroke", d => d.sample_id === state.selectedSample ? "black" : "none")
            .attr("stroke-width", d => d.sample_id === state.selectedSample ? 2 : 0)
            .call(attachInteractions),
        update => update
            .attr("cx", d => xScale(d.x))
            .attr("cy", d => yScale(d.y))
            .attr("r", d => d.sample_id === state.selectedSample ? 6 : 4)
            .attr("stroke", d => d.sample_id === state.selectedSample ? "black" : "none")
            .attr("stroke-width", d => d.sample_id === state.selectedSample ? 2 : 0),
        exit => exit.remove()
    );
}

function attachInteractions(selection) {
    const tooltip = d3.select("#tooltip");
    selection.on("mouseover", function(event, d) {
        d3.select(this).raise().attr("r", 7).attr("stroke", "black").attr("stroke-width", 2);
        d3.select("#tt-id").text(d.sample_id);
        d3.select("#tt-class").text(cifar10_labels_es[d.label]);
        d3.select("#tt-img").attr("src", d.image_path);
        tooltip.classed("hidden", false).style("left", (event.pageX -250) + "px").style("top", (event.pageY - 15) + "px");
    })
    .on("mouseout", function(event, d) {
        const isSel = d.sample_id === State.selectedSample;
        d3.select(this).attr("r", isSel ? 6 : 4).attr("stroke", isSel ? "black" : "none").attr("stroke-width", isSel ? 2 : 0);
        tooltip.classed("hidden", true);
    })
    .on("click", function(event, d) {
        State.update({ selectedSample: d.sample_id });
    });
}

// --- 6. Details on Demand (Panel Derecho Limpio y Estático) ---
function renderDetailPanel(state) {
    const panel = d3.select("#detail-content");
    panel.selectAll("*").remove();
    
    // Estado vacío contextual
    if (state.selectedSample === null) {
        panel.append("div")
             .attr("class", "empty-state")
             .text(WORKSPACES[state.workspace].emptyState);
        return;
    }

    const sampleMetadata = dataset.find(d => d.sample_id === state.selectedSample);
    if (!sampleMetadata) return;

    panel.append("div").attr("class", "detail-meta")
         .html(`<b>Sample ID:</b> ${state.selectedSample} <br> <b>Class:</b> ${cifar10_labels_es[sampleMetadata.label]}`);

    // Lógica de renderizado según Workspace (sin código muerto de animación)
    if (state.workspace === 'evolution') {
        // Muestra únicamente el resultado de la época actual, paso actual
        const currentData = dataset.find(d => d.sample_id === state.selectedSample && d.epoch === state.epoch && d.generation_step === state.step);
        if(currentData) renderImagePair(panel, currentData, `Epoch ${state.epoch} - Step ${state.step}`);
    } 
    else if (state.workspace === 'generation') {
        // Muestra la matriz completa de pasos de la época actual
        const pathData = dataset.filter(d => d.sample_id === state.selectedSample && d.epoch === state.epoch).sort((a, b) => a.generation_step - b.generation_step);
        
        pathData.forEach(stepData => {
            const isCurrent = stepData.generation_step === state.step;
            
            // Creamos el bloque con un ID único y estilos dinámicos de resaltado
            const block = panel.append("div")
                .attr("id", `gen-step-${stepData.generation_step}`) // ID necesario para el scroll
                .style("margin-bottom", "15px")
                .style("opacity", isCurrent ? "1" : "0.3") // Atenuamos más (0.3) los no seleccionados
                .style("padding", isCurrent ? "10px" : "0px")
                .style("background-color", isCurrent ? "#f0f7fb" : "transparent")
                .style("border-left", isCurrent ? "4px solid var(--highlight)" : "4px solid transparent")
                .style("border-radius", "0 4px 4px 0")
                .style("transition", "all 0.3s ease"); // Transición suave para el cambio
                
            renderImagePair(block, stepData, `Step ${stepData.generation_step}`);
        });

        // Lógica de Auto-Scroll
        // Usamos un pequeño timeout para asegurar que D3 haya terminado de pintar el DOM
        setTimeout(() => {
            const currentBlock = document.getElementById(`gen-step-${state.step}`);
            if (currentBlock) {
                // Hacemos scroll suave para que el elemento seleccionado quede en el centro del panel
                currentBlock.scrollIntoView({ behavior: "smooth", block: "center" });
            }
        }, 50);
    }
    else if (state.workspace === 'compare') {
        // Muestra comparación estática Izquierda (250) vs Derecha (state.epoch)
        const dataLeft = dataset.find(d => d.sample_id === state.selectedSample && d.epoch === 250 && d.generation_step === state.step);
        const dataRight = dataset.find(d => d.sample_id === state.selectedSample && d.epoch === state.epoch && d.generation_step === state.step);
        
        if(dataLeft) renderImagePair(panel, dataLeft, `Epoch 250`);
        panel.append("hr").style("margin", "15px 0").style("border", "0").style("border-top", "1px dashed #ddd");
        if(dataRight) renderImagePair(panel, dataRight, `Epoch ${state.epoch}`);
    }
}

// Función auxiliar para renderizar el par (Prob Path | Attention)
function renderImagePair(container, dataRecord, title) {
    container.append("strong").style("font-size", "12px").style("display", "block").text(title);
    const grid = container.append("div").attr("class", "detail-grid");
    
    const probBox = grid.append("div").attr("class", "detail-img-box");
    probBox.append("span").text("Generación");
    probBox.append("img").attr("src", dataRecord.image_path);
    
    const attnBox = grid.append("div").attr("class", "detail-img-box");
    attnBox.append("span").text("Atención Espacial");
    attnBox.append("img").attr("src", dataRecord.attention_path);
}