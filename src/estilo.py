"""
estilo.py
~~~~~~~~~

Estilo comun de las figuras, para que todas las graficas del repositorio
se vean igual y los colores sean distinguibles tambien con daltonismo.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

AZUL = "#2a78d6"      # solucion de la red
NARANJA = "#eb6834"   # funcion objetivo / solucion analitica
VERDE = "#1baf7a"
VIOLETA = "#4a3aa7"
TINTA = "#0b0b0b"
TINTA_SUAVE = "#52514e"
REJILLA = "#d8d7d2"
FONDO = "#fcfcfb"


def marco(ax, titulo=None, subtitulo=None, xlabel=None, ylabel=None,
          leyenda=True):
    """Aplica el estilo a unos ejes: rejilla suave, sin marco derecho ni
    superior, tipografia discreta."""
    if titulo:
        ax.set_title(titulo, fontsize=12, color=TINTA, fontweight="bold",
                     loc="left", pad=18 if subtitulo else 10)
    if subtitulo:
        ax.text(0, 1.02, subtitulo, transform=ax.transAxes, fontsize=9,
                color=TINTA_SUAVE, va="bottom")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=10, color=TINTA_SUAVE)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=10, color=TINTA_SUAVE)
    ax.grid(True, color=REJILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(REJILLA)
    ax.tick_params(colors=TINTA_SUAVE, labelsize=9)
    ax.set_facecolor(FONDO)
    if leyenda:
        leg = ax.legend(frameon=False, fontsize=9.5)
        if leg:
            for t in leg.get_texts():
                t.set_color(TINTA)


def figura(ancho=8.4, alto=4.6):
    fig, ax = plt.subplots(figsize=(ancho, alto))
    fig.patch.set_facecolor(FONDO)
    return fig, ax


def guardar(fig, ruta):
    fig.tight_layout()
    fig.savefig(ruta, dpi=200, facecolor=FONDO)
    plt.close(fig)
    print("figura ->", ruta)
