// Interactividad de la app. Todo es una mejora: sin JavaScript el formulario funciona igual.
// - filtra las ciudades segun el pais
// - muestra el valor de los deslizadores y la antiguedad de la vivienda
// - calcula credito, mensualidad y proporcion sobre el ingreso mientras se escribe
// - pide el resultado al servidor en vivo y anima el medidor y el porcentaje
(function () {
  "use strict";
  var form = document.getElementById("form");
  var panel = document.getElementById("resultado");
  if (!form || !panel || !window.fetch) return;

  var sinMovimiento = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var TASA = 0.065 / 12, LARGO = 251.33, ANIO = 2026;
  var ultimaProb = 0, temporizador = null, peticion = 0;

  document.querySelector("main").classList.add("en-vivo");

  function miles(n) { return Math.round(n).toString().replace(/\B(?=(\d{3})+(?!\d))/g, " "); }
  function num(nombre) {
    var campo = form.elements[nombre];
    var v = campo && campo.value !== undefined ? campo.value : "";
    return v === "" ? NaN : Number(v);
  }

  // ---------- ciudades segun el pais ----------
  var selPais = form.elements.pais, selCiudad = form.elements.ciudad;
  var todas = Array.prototype.map.call(selCiudad.querySelectorAll("option[data-pais]"), function (o) {
    return { valor: o.value, pais: o.dataset.pais };
  });
  function filtrarCiudades() {
    var pais = selPais.value, actual = selCiudad.value;
    selCiudad.innerHTML = "";
    var vacia = new Option(pais ? "Elige una ciudad" : "Primero elige un país", "");
    vacia.disabled = true; selCiudad.add(vacia);
    var hay = false;
    todas.forEach(function (c) {
      if (c.pais === pais) { selCiudad.add(new Option(c.valor, c.valor, false, c.valor === actual)); hay = hay || c.valor === actual; }
    });
    if (!hay) selCiudad.value = "";
    // un pais con una sola ciudad (Singapur) se elige solo
    if (!hay && selCiudad.options.length === 2) selCiudad.selectedIndex = 1;
  }
  selPais.addEventListener("change", filtrarCiudades);
  filtrarCiudades();

  // ---------- deslizadores y antiguedad ----------
  form.querySelectorAll(".deslizador input").forEach(function (r) {
    var salida = r.parentNode.querySelector("output");
    r.addEventListener("input", function () { salida.textContent = r.value; });
  });
  var ayudaAnio = form.querySelector("[data-antiguedad]");
  function antiguedad() {
    var a = num("anio");
    ayudaAnio.textContent = a >= 1960 && a <= 2023 ? "Tiene " + (ANIO - a) + " años." : "De 1960 a 2023.";
  }

  // ---------- calculo del credito ----------
  var caja = form.querySelector("[data-calc-caja=proporcion]");
  function calcular() {
    var precio = num("precio"), enganche = num("enganche"), ingreso = num("ingreso");
    var plazo = Number((form.querySelector("input[name=plazo]:checked") || {}).value || 20);
    var credito = precio - enganche, n = plazo * 12;
    var factor = TASA * Math.pow(1 + TASA, n) / (Math.pow(1 + TASA, n) - 1);
    var ok = precio > 0 && enganche >= 0 && credito > 0;
    form.querySelector("[data-calc=credito]").textContent = ok ? miles(credito) : "—";
    form.querySelector("[data-calc=mensualidad]").textContent = ok ? miles(credito * factor) : "—";
    var prop = ok && ingreso > 0 ? credito * factor / ingreso : NaN;
    form.querySelector("[data-calc=proporcion]").textContent = isNaN(prop) ? "—" : Math.round(prop * 100) + " %";
    caja.classList.toggle("alerta", prop > 0.39);
  }

  // ---------- animaciones del resultado ----------
  function animar(desde) {
    var svg = panel.querySelector(".medidor");
    if (!svg) { ultimaProb = 0; return; }
    var hasta = Number(svg.dataset.prob), arco = svg.querySelector(".arco"), texto = svg.querySelector("[data-numero]");
    if (sinMovimiento) { ultimaProb = hasta; return; }
    // el arco parte del valor anterior y la transicion CSS lo lleva al nuevo
    arco.style.transition = "none";
    arco.setAttribute("stroke-dashoffset", (LARGO * (1 - desde)).toFixed(2));
    arco.getBoundingClientRect();
    arco.style.transition = "";
    requestAnimationFrame(function () { arco.setAttribute("stroke-dashoffset", (LARGO * (1 - hasta)).toFixed(2)); });
    // el porcentaje cuenta desde el valor anterior
    var inicio = null, dur = 1000;
    function paso(t) {
      if (inicio === null) inicio = t;
      var x = Math.min(1, (t - inicio) / dur), suave = 1 - Math.pow(1 - x, 3);
      texto.textContent = Math.round((desde + (hasta - desde) * suave) * 100) + " %";
      if (x < 1) requestAnimationFrame(paso);
    }
    requestAnimationFrame(paso);
    ultimaProb = hasta;
  }

  // ---------- resultado en vivo ----------
  function pedirResultado() {
    var id = ++peticion;
    panel.classList.add("cargando");
    fetch("/", { method: "POST", body: new FormData(form), headers: { "X-Parcial": "1" } })
      .then(function (r) { return r.ok ? r.text() : Promise.reject(r.status); })
      .then(function (html) {
        if (id !== peticion) return;  // llego una respuesta vieja; ya hay otra en camino
        var desde = ultimaProb;
        panel.innerHTML = html;
        animar(desde);
      })
      .catch(function () { /* sin conexion: el boton de enviar sigue disponible */ document.querySelector("main").classList.remove("en-vivo"); })
      .then(function () { if (id === peticion) panel.classList.remove("cargando"); });
  }
  function alCambiar() {
    calcular(); antiguedad();
    clearTimeout(temporizador);
    temporizador = setTimeout(pedirResultado, 450);
  }
  form.addEventListener("input", alCambiar);
  form.addEventListener("change", alCambiar);
  form.addEventListener("submit", function (e) { e.preventDefault(); clearTimeout(temporizador); pedirResultado(); });

  calcular(); antiguedad();
  animar(0);  // si la pagina ya trae un resultado (ejemplo o envio sin JS), se anima desde 0
})();
