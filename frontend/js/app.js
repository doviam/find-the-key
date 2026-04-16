(function () {
  const meta = document.querySelector('meta[name="api-base"]');
  const apiBase = (() => {
    const c = meta?.getAttribute("content")?.trim();
    if (c) return c.replace(/\/$/, "");
    const API_LOCAL = "http://127.0.0.1:8000";
    if (window.location.protocol === "file:") {
      return API_LOCAL;
    }
    const loc = window.location;
    const port = parseInt(loc.port, 10) || (loc.protocol === "https:" ? 443 : 80);
    const staticDevPorts = [5500, 5501, 8080];
    const host = loc.hostname;
    if (
      (host === "127.0.0.1" || host === "localhost") &&
      staticDevPorts.includes(port)
    ) {
      return API_LOCAL;
    }
    return loc.origin.replace(/\/$/, "");
  })();

  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("file-input");
  const btnBrowse = document.getElementById("btn-browse");
  const loading = document.getElementById("loading");
  const results = document.getElementById("results");
  const errEl = document.getElementById("error");
  const valKey = document.getElementById("val-key");

  function showError(msg) {
    errEl.textContent = msg;
    errEl.hidden = false;
  }

  function clearError() {
    errEl.hidden = true;
    errEl.textContent = "";
  }

  function setLoading(on) {
    loading.hidden = !on;
    results.hidden = on ? true : results.hidden;
  }

  async function analyzeFile(file) {
    clearError();
    results.hidden = true;
    setLoading(true);

    const fd = new FormData();
    fd.append("file", file);

    try {
      const res = await fetch(`${apiBase}/api/analyze`, {
        method: "POST",
        body: fd,
      });
      const text = await res.text();
      let data;
      try {
        data = JSON.parse(text);
      } catch {
        throw new Error(text.slice(0, 200) || "Respuesta no válida");
      }
      if (!res.ok) {
        const detail = data.detail;
        const detailMsg =
          typeof detail === "string"
            ? detail
            : Array.isArray(detail)
              ? detail.map((d) => d.msg || JSON.stringify(d)).join("; ")
              : data.message || `Error ${res.status}`;
        throw new Error(detailMsg);
      }
      render(data);
      results.hidden = false;
    } catch (e) {
      console.error(e);
      showError(e.message || "No se pudo analizar el archivo.");
    } finally {
      setLoading(false);
    }
  }

  function render(data) {
    valKey.textContent = data.key ?? "—";
  }

  function pickFile() {
    fileInput.click();
  }

  btnBrowse.addEventListener("click", (e) => {
    e.stopPropagation();
    pickFile();
  });
  fileInput.addEventListener("change", () => {
    const f = fileInput.files?.[0];
    if (f) analyzeFile(f);
    fileInput.value = "";
  });

  ["dragenter", "dragover", "dragleave", "drop"].forEach((ev) => {
    dropzone.addEventListener(ev, (e) => {
      e.preventDefault();
      e.stopPropagation();
    });
  });

  dropzone.addEventListener("dragenter", () => dropzone.classList.add("dragover"));
  dropzone.addEventListener("dragover", () => dropzone.classList.add("dragover"));
  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));
  dropzone.addEventListener("drop", (e) => {
    dropzone.classList.remove("dragover");
    const f = e.dataTransfer?.files?.[0];
    if (f) analyzeFile(f);
  });

  dropzone.addEventListener("click", (e) => {
    if (e.target.closest("button")) return;
    pickFile();
  });
})();
